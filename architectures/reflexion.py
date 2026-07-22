"""
Episodic Reflexion + its blind-retry (sham-reflection) control.

Within-task attempt loop, up to ``max_attempts = 3``: on a failed attempt the
model writes a reflection (one extra LLM call that *does* see the failed
trajectory plus the sanitized failure signal), and the task is retried
**fresh-episode** — the failed conversation never leaks into the retry; only
the reflection (and the signal) carry over, as a user preamble before turn 0.
The attempt loop itself lives in the runner (`run_benchmark.run_one_multi`),
which owns grading and instance resets; this file holds only the cognition.

The resampling confound and its control: temperature-0 on
this endpoint is not deterministic (~7.8% of failures flip on a pure re-roll),
so retrying is a lottery even with an empty reflection. ``BlindRetry`` is the
placebo arm: identical loop, identical preamble template, identical sanitized
signal — but the reflection slot holds a fixed, generic, reflection-*shaped*
sham text instead of a model-written one. The only difference between the arms
is whether the lesson contains task-specific information distilled from the
actual failure:

    Reflexion@3 − BlindRetry@3 = value of the reflection content
    BlindRetry@3 − @1          = retry luck + failure-signal + framing effects

Fairness invariants: attempt 1 is *identical to the baseline* (empty
``reasoning_scaffold``, no preamble — and when seeded, it literally *is* the
baseline run); the preamble is a user message, never a system-prompt change,
never injected mid-task; wording says "previous, separate attempt" and "the
task now restarts from the beginning" so the model cannot misattribute the
failure to steps inside the current conversation.
"""

from architectures.architecture import Architecture
from utils.config import MODEL, client
from utils.logging import TrajectoryLogger
from utils.retry import call_with_retry

# The blind arm's fixed, reflection-shaped placebo text a bare "you failed"
# could baffle the model in a way the treatment arm never experiences; a sham
# lesson of the same *form* keeps the arms symmetric).
#
# Form-matched to REFLECTION_PROMPT's output — same three headings, comparable
# length — so the arms differ *only* in whether the text carries information
# about the actual failure, not in shape or verbosity. It is a constant, so
# it cannot: the placebo is the same pill, with nothing in it. Never given to
# the treatment arm — the arms substitute this text for a real reflection,
# they don't stack both.
SHAM_REFLECTION = (
    "What I tried: I attempted the task using the tools provided.\n"
    "What went wrong: Some part of my approach did not match what the task "
    "required.\n"
    "Lesson: I should re-read the request carefully, verify preconditions "
    "before acting, and double-check tool arguments."
)

# Input to the reflection call: the failed attempt's transcript + the
# sanitized signal. Fixed output format (NEXT_STEPS Phase 3). Shared verbatim
# with the vector-DB variant (Phase 4), so the wording is variant-neutral: no
# "specific to this task" (poison when the lesson is retrieved for a
# *different* task) and no "next attempt" (the vector arm has none) —
# REVIEW.md §3.2.2, decision 4.
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
    """
    The user message a retry attempt opens with — identical template in both
    arms; only the reflection *content* differs. ``memory`` holds one
    ``{attempt, signal, reflection}`` per failed attempt so far.
    """
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
    # The blind control overrides this to use the fixed sham text.
    writes_reflections = True

    def __init__(self):
        self._memory: list[dict] = []  # set per attempt via run_attempt

    @staticmethod
    def _init_stats() -> dict:
        stats = Architecture._init_stats()
        stats["n_reflections"] = 0  # reflection LLM calls (0 in the blind arm)
        return stats

    def run_attempt(
        self, task: dict, tlog: TrajectoryLogger, memory: list[dict]
    ) -> tuple[list[list[list[str]]], dict, list[dict]]:
        """
        One fresh-episode attempt. ``memory`` is this task's accumulated
        ``{attempt, signal, reflection}`` entries (empty on attempt 1 — which
        makes it byte-identical to a baseline run). Returns
        ``(all_turns_calls, stats, messages)``; the final message list feeds
        the reflection if this attempt fails. Memory is episodic: scoped to
        one task, passed in explicitly, never persisted across tasks.
        """
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
        """
        One reflection LLM call over the failed attempt's transcript. No
        tools; same temperature/max_tokens as the loop. Cost is accumulated
        into ``stats`` (the runner folds it into the task's totals — the cost
        axis must include reflection overhead). Not part of the action step
        loop, so it never consumes MAX_STEPS_PER_TURN budget.
        """
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
            # The rendered input, verbatim: the transcript reaches here through
            # a rebuild path (log_to_messages → messages_to_text) that
            # nothing else exercises, so the log must show what the reflector
            # actually saw
            prompt=prompt,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            latency_s=round(latency, 3),
        )
        return text


class BlindRetry(ReflexionEpisodic):
    """
    Placebo arm: same attempt loop, same preamble, same sanitized signal —
    the reflection slot holds the fixed sham text, written by no one.
    """

    name = "blind_retry"
    writes_reflections = False

    def reflect(self, attempt_text, signal, attempt, tlog, stats):
        # No LLM call: the sham is a constant, so it costs nothing and cannot
        # contain task-specific information.
        tlog.event(
            "reflection",
            attempt=attempt,
            signal=signal,
            text=SHAM_REFLECTION,
            sham=True,
        )
        return SHAM_REFLECTION
