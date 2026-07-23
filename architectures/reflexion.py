"""
Episodic Reflexion and its blind-retry controls.

Up to ``max_attempts = 3`` attempts per task. On a failure the model writes a
reflection — one extra LLM call that sees the failed trajectory plus the
sanitized failure signal — and the task restarts **fresh-episode**: the failed
conversation never enters the retry, only the reflection and the signal do, as
a user preamble before turn 0. The attempt loop lives in the runner
(`run_benchmark.run_one_multi`), which owns grading and instance resets; this
file holds only the cognition.

Why the controls exist: temperature-0 is not deterministic here (~7.8% of
failures flip on a plain re-roll), so a retry is a lottery even when the
reflection says nothing. The controls are rungs on a ladder, and **all of them
are reported** — never one picked after seeing which flatters the result:

    @1 → plain re-run → blind_retry_lite → blind_retry → reflexion
          retry luck     minimal advice     generic advice   task-specific

    Reflexion@k − BlindRetry@k = value of the reflection content
    BlindRetry@k − @1          = retry luck + signal + framing

The blind arms are *active* controls, not placebos: their sham is form-matched
to a real reflection and therefore carries real generic advice.
``blind_retry_lite`` strips that advice, so the pair bounds how much of the
control's performance came from the advice rather than from the retry.

Fairness invariants: attempt 1 is identical to the baseline (no preamble — and
when seeded it literally *is* the baseline run), so every arm retries the same
failure set; the preamble is a user message, never a system-prompt change and
never injected mid-task; its wording says "previous, separate attempt" that
"restarts from the beginning", so the model cannot misread the failure as
something it did earlier in the current conversation.
"""

from architectures.architecture import Architecture
from utils.config import MODEL, client
from utils.logging import TrajectoryLogger
from utils.retry import call_with_retry

# Sham reflection for `blind_retry`. Form-matched to REFLECTION_PROMPT's output
# — same three headings, comparable length — so the arms differ only in whether
# the text says anything about the actual failure. It *replaces* the reflection
# in the control arm; it is never stacked on top of a real one.
#
# FROZEN: this exact string produced results/blind_retry/ (run 22 Jul 2026).
# Never edit in place — add a new constant plus a subclass, so committed
# results stay reproducible from committed source.
SHAM_REFLECTION = (
    "What I tried: I attempted the task using the tools provided.\n"
    "What went wrong: Some part of my approach did not match what the task "
    "required.\n"
    "Lesson: I should re-read the request carefully, verify preconditions "
    "before acting, and double-check tool arguments."
)

# Sham for `blind_retry_lite`: same headings and same first two lines, with the
# two imperative clauses dropped from the Lesson. Those clauses are plausibly
# corrective for miss_param's premature-guess failures, so this arm ablates
# them. Still a constant, so still carries nothing about the actual failure.
SHAM_REFLECTION_LITE = (
    "What I tried: I attempted the task using the tools provided.\n"
    "What went wrong: Some part of my approach did not match what the task "
    "required.\n"
    "Lesson: I should re-read the request carefully."
)

# Input to the reflection call: the failed transcript plus the sanitized
# signal. The wording is deliberately variant-neutral — no "specific to this
# task", no "next attempt" — so any arm reusing it gets the same prompt. The
# closing line is load-bearing: whoever reads the lesson later never sees the
# transcript, so the Lesson has to stand on its own.
REFLECTION_PROMPT = (
    "Below is the full transcript of your previous attempt at a multi-turn "
    "tool-calling task. The attempt was graded as FAILED.\n\n"
    "{attempt_text}\n\n"
    "Grader failure signal: {signal}\n\n"
    "Reflect on this failed attempt. Reply in exactly this format:\n"
    "What I tried: <one or two sentences>\n"
    "What went wrong: <one or two sentences — your best diagnosis, using the "
    "failure signal>\n"
    "Lesson: <one or two sentences of concrete, actionable guidance>\n\n"
    "Whoever reads this reflection later will not see this transcript — only "
    "the reflection itself. Make the Lesson self-contained."
)


def _retry_preamble(memory: list[dict]) -> str:
    """The user message a retry opens with — identical template in every arm,
    only the reflection *content* differs. ``memory`` holds one
    ``{attempt, signal, reflection}`` per failed attempt so far."""
    n = len(memory)
    header = (
        "Note: a previous, separate attempt at this task failed."
        if n == 1
        else f"Note: {n} previous, separate attempts at this task failed."
    )
    lines = [header]
    for m in memory:
        lines.append(f"Attempt {m['attempt']} failure signal: {m['signal']}")
        lines.append(f"Attempt {m['attempt']} reflection: {m['reflection']}")
    lines.append("The task now restarts from the beginning.")
    return "\n".join(lines)


class ReflexionEpisodic(Architecture):
    name = "reflexion"

    # The runner switches to its multi-attempt path when this is > 1.
    max_attempts = 3
    # True → reflect() makes an LLM call over the failed trajectory.
    # The blind controls override this to use their fixed sham text.
    writes_reflections = True

    def __init__(self):
        self._memory: list[dict] = []  # set per attempt via run_attempt

    @staticmethod
    def _init_stats() -> dict:
        stats = Architecture._init_stats()
        stats["n_reflections"] = 0  # reflection LLM calls (0 in the blind arms)
        return stats

    def run_attempt(
        self, task: dict, tlog: TrajectoryLogger, memory: list[dict]
    ) -> tuple[list[list[list[str]]], dict, list[dict]]:
        """One fresh-episode attempt. ``memory`` is this task's accumulated
        ``{attempt, signal, reflection}`` entries — empty on attempt 1, which
        makes it byte-identical to a baseline run. Returns
        ``(all_turns_calls, stats, messages)``; the message list feeds the
        reflection if this attempt fails. Memory is episodic: scoped to one
        task, passed in explicitly, never persisted across tasks."""
        self._memory = memory
        try:
            return self._run_fc_loop(task, tlog)
        finally:
            self._memory = []

    def _initial_messages(
        self, system_prompt: str, tlog: TrajectoryLogger
    ) -> list[dict]:
        """On retries, open with the preamble user message before turn 0."""
        messages = super()._initial_messages(system_prompt, tlog)
        if self._memory:
            preamble = _retry_preamble(self._memory)
            messages.append({"role": "user", "content": preamble})
            tlog.event("retry_context", attempt=len(self._memory) + 1, text=preamble)
        return messages

    def reflect(
        self,
        attempt_text: str | None,
        signal: str,
        attempt: int,
        tlog: TrajectoryLogger,
        stats: dict,
    ) -> str:
        """One reflection LLM call over the failed transcript. No tools; same
        temperature/max_tokens as the loop. Cost accumulates into ``stats`` —
        the cost axis has to include reflection overhead. Not an action step,
        so it never consumes MAX_STEPS_PER_TURN budget."""
        prompt = REFLECTION_PROMPT.format(attempt_text=attempt_text, signal=signal)
        response, latency = call_with_retry(
            client,
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=8192,
        )
        text = (response.choices[0].message.content or "").strip()

        stats["n_llm_calls"] += 1
        stats["n_reflections"] += 1
        stats["latency_s"] += latency
        usage = response.usage
        if usage:
            stats["input_tokens"] += usage.prompt_tokens or 0
            stats["output_tokens"] += usage.completion_tokens or 0
            stats["peak_context"] = max(stats["peak_context"], usage.prompt_tokens or 0)
        if response.choices[0].finish_reason == "length":
            stats["truncations"] += 1
            tlog.event("truncated", turn_idx=None, step=None, where="reflection")

        tlog.event(
            "reflection",
            attempt=attempt,
            signal=signal,
            text=text,
            sham=False,
            # The rendered input, verbatim: the transcript arrives through a
            # rebuild path (log_to_messages → messages_to_text) nothing else
            # exercises, so the log must show what the reflector actually saw.
            prompt=prompt,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            latency_s=round(latency, 3),
        )
        return text


class BlindRetry(ReflexionEpisodic):
    """Control arm: same loop, preamble and sanitized signal, but the
    reflection slot holds a fixed sham written by no one. An *active* control —
    the sham carries generic advice — so `BlindRetryLite` ablates that advice
    and both are reported."""

    name = "blind_retry"
    writes_reflections = False
    # Which constant fills the reflection slot. Subclass and override to add a
    # ladder rung — never edit a frozen constant in place.
    sham_text = SHAM_REFLECTION

    def reflect(self, attempt_text, signal, attempt, tlog, stats):
        # No LLM call: the sham is a constant, so it costs nothing and cannot
        # contain task-specific information.
        tlog.event(
            "reflection",
            attempt=attempt,
            signal=signal,
            text=self.sham_text,
            sham=True,
            # Names the rung, so a trajectory log identifies which control
            # produced it without reference to its directory.
            sham_variant=self.name,
        )
        return self.sham_text


class BlindRetryLite(BlindRetry):
    """Advice-stripped control. Identical to `BlindRetry` except the sham's
    Lesson line, so `BlindRetry` − `BlindRetryLite` bounds how much of the
    control's performance came from generic advice rather than the retry."""

    name = "blind_retry_lite"
    sham_text = SHAM_REFLECTION_LITE
