"""
Base class for all architectures under comparison.

The shared multi-turn function-calling loop lives here exactly once, so the
architectures stay comparable: each one differs only in its `system_prompt`
(and, later, optional hooks for reflection). The loop does **no printing** — it
only logs via `TrajectoryLogger`; the runner renders the log. It returns the
per-turn/per-step call strings plus a small `stats` dict for the results table.

Fairness invariants enforced here (see README.md):
  - actions go through the *native FC channel* only; reasoning is free text;
  - zero-shot (no exemplars);
  - `temperature=0`.
"""

import json
from abc import ABC

# utils.config (imported above) has already put the BFCL repo on sys.path.
from bfcl_eval.constants.default_prompts import (
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC as _ADDITIONAL_FUNCTION_PROMPT,
)
from utils.config import MAX_STEPS_PER_TURN, client, MODEL, DOCS_DIR
from utils.executor import tool_call_to_python_string, execute_call_locally
from utils.logging import TrajectoryLogger
from utils.retry import call_with_retry
from utils.schema import load_tools_for_classes

"""
Shared task-instruction core, held IDENTICAL across all five architectures
(fairness invariant: only the reasoning/reflection scaffold may differ).
Two semantic pieces are adapted from BFCL's intended task instruction: 
(1) recognise when no tool fits or a required parameter is missing instead of 
guessing — this is what ``multi_turn_irrelevance_checker`` rewards; (2) the turn-completion
protocol that matches how the loop ends a turn (a reply with no tool call).
"""
BASE_TASK_INSTRUCTION = (
    "You are an agent that completes user tasks by calling the provided tools. "
    "Call tools whenever they are needed to satisfy the user's request.\n"
    "If none of the available tools can fulfil the request, say so instead of "
    "guessing. If the request is missing a value that a tool requires, ask the "
    "user for it.\n"
    "Keep calling tools until the current request is fully handled. When you "
    "have nothing left to call, reply in natural language summarising what you "
    "did — a message with no tool call ends the current turn."
)


class Architecture(ABC):
    # short identifier used for log/results directories — override per subclass
    name: str = "architecture"

    # architecture-specific reasoning scaffold, appended to the shared task
    # core. Empty for the baseline; reasoning architectures override it.
    reasoning_scaffold: str = ""

    def system_prompt(self) -> str:
        """
        Shared task-instruction core plus this architecture's reasoning scaffold.
        The core (``BASE_TASK_INSTRUCTION``) is identical for all architectures;
        only ``reasoning_scaffold`` varies — that is the fairness invariant in
        code. Subclasses normally set ``reasoning_scaffold`` rather than override
        this method.
        """
        if self.reasoning_scaffold:
            return f"{BASE_TASK_INSTRUCTION}\n\n{self.reasoning_scaffold}"
        return BASE_TASK_INSTRUCTION

    def run_task(
        self, task: dict, tlog: TrajectoryLogger
    ) -> tuple[list[list[list[str]]], dict]:
        """
        Run one BFCL multi-turn task. Returns ``(all_turns_calls, stats)`` where
        ``all_turns_calls`` has the shape ``list[list[list[str]]]``
        (turn → step → call strings) that ``multi_turn_checker`` expects.
        """
        return self._run_fc_loop(task, tlog)

    # ------------------------------------------------------------------
    # Shared loop
    # ------------------------------------------------------------------

    def _run_fc_loop(
        self, task: dict, tlog: TrajectoryLogger
    ) -> tuple[list[list[list[str]]], dict]:
        active_tools, held_out_tools, missed = self._setup_tools(task)
        # long_context category must match the grader's simulator scaffold.
        long_context = "long_context" in task["id"]
        tlog.task_start(task)

        messages = [{"role": "system", "content": self.system_prompt()}]
        all_turns_calls: list[list[list[str]]] = []
        stats = self._init_stats()

        for turn_idx, turn_messages in enumerate(task["question"]):
            turn_messages = self._reveal_held_out(
                turn_idx, missed, held_out_tools, active_tools, turn_messages
            )
            messages.extend(turn_messages)
            tlog.user_turn(turn_idx, turn_messages)

            this_turn_calls = self._run_turn(
                turn_idx, messages, active_tools, task, long_context, tlog, stats
            )
            tlog.turn_end(turn_idx, n_steps=len(this_turn_calls))
            all_turns_calls.append(this_turn_calls)

        return all_turns_calls, stats

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _setup_tools(task: dict) -> tuple[list[dict], dict[str, dict], dict]:
        """
        Split the task's tools into those active from turn 0 and those held out.

        ``miss_func`` tasks hide some functions until a later "holdout" turn via
        ``missed_function``, which maps a 0-based turn index (string key, as JSON
        keys always are) to the names revealed then — e.g. ``{"2": ["mv"]}`` holds
        ``mv`` out until turn 2. ``_reveal_held_out`` re-adds them at that turn.

        Returns ``(active_tools, held_out_tools, missed)``: ``active_tools``
        excludes every held-out name, ``held_out_tools`` is keyed by name for
        re-adding, and ``missed`` is the raw mapping.
        """
        all_tools = load_tools_for_classes(task["involved_classes"], DOCS_DIR)
        missed = task.get("missed_function", {})
        held_out_names = {n for names in missed.values() for n in names}
        active_tools = [
            t for t in all_tools if t["function"]["name"] not in held_out_names
        ]
        held_out_tools = {
            t["function"]["name"]: t
            for t in all_tools
            if t["function"]["name"] in held_out_names
        }
        return active_tools, held_out_tools, missed

    @staticmethod
    def _init_stats() -> dict:
        return {
            "n_llm_calls": 0,
            "n_tool_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "peak_context": 0,
            "latency_s": 0.0,
            "max_steps_hits": 0,
            # Fairness guards: a reply truncated at max_tokens can cut the
            # tool-call JSON at the end of a long Thought/reflection.
            # Counted so the results table shows a per-architecture
            # truncation/parse rate; verify it stays ~0.
            "truncations": 0,
            "parse_errors": 0,
        }

    @staticmethod
    def _reveal_held_out(
        turn_idx: int,
        missed: dict,
        held_out_tools: dict[str, dict],
        active_tools: list[dict],
        turn_messages: list[dict],
    ) -> list[dict]:
        """
        At a ``miss_func`` holdout turn, reveal the held-out function(s) (mutating
        ``active_tools`` in place) and swap the empty question turn for BFCL's
        synthetic "more functions available" prompt. Otherwise a no-op. See
        ``_setup_tools`` for the ``missed`` mapping format - ``{"2": ["mv"]}`` holds
        ``mv`` out until turn 2.
        """
        if str(turn_idx) not in missed:
            return turn_messages
        for name in missed[str(turn_idx)]:
            active_tools.append(held_out_tools[name])
        return [{"role": "user", "content": _ADDITIONAL_FUNCTION_PROMPT}]

    # ------------------------------------------------------------------
    # Per-turn / per-step loop
    # ------------------------------------------------------------------

    def _run_turn(
        self,
        turn_idx: int,
        messages: list[dict],
        active_tools: list[dict],
        task: dict,
        long_context: bool,
        tlog: TrajectoryLogger,
        stats: dict,
    ) -> list[list[str]]:
        """Run one turn's step loop until the model replies without a tool call."""
        this_turn_calls: list[list[str]] = []
        for step in range(MAX_STEPS_PER_TURN):
            msg = self._call_model(turn_idx, step, messages, active_tools, tlog, stats)

            if not msg.tool_calls:
                break  # model is done with this turn

            step_call_strings, parse_failed = self._execute_calls(
                turn_idx, step, msg, messages, task, long_context, tlog, stats
            )
            if step_call_strings:
                this_turn_calls.append(step_call_strings)
            if parse_failed:
                break  # truncated tool call: can't reliably continue the turn
        else:
            stats["max_steps_hits"] += 1
            tlog.event("max_steps_reached", turn_idx=turn_idx)
        return this_turn_calls

    def _call_model(
        self,
        turn_idx: int,
        step: int,
        messages: list[dict],
        active_tools: list[dict],
        tlog: TrajectoryLogger,
        stats: dict,
    ):
        """
        One LLM call: log the request, accumulate stats, append the assistant
        message to ``messages``, and return the response message.
        """
        tlog.llm_request(turn_idx, step, MODEL, messages, active_tools)
        # latency is the successful-call time only (retry backoff excluded)
        response, latency = call_with_retry(
            client,
            model=MODEL,
            messages=messages,
            tools=active_tools,
            tool_choice="auto",
            temperature=0,
            # Generous ceiling, not a target: we pay only for tokens the model
            # actually emits, so a high cap costs nothing on normal replies but
            # prevents ever truncating a Thought/reflection + tool_call. Same value
            # across all five architectures. `stats["truncations"]` must stay ~0
            max_tokens=8192,
        )
        tlog.llm_response(turn_idx, step, response, latency)

        stats["n_llm_calls"] += 1
        stats["latency_s"] += latency
        usage = response.usage
        if usage:
            stats["input_tokens"] += usage.prompt_tokens or 0
            stats["output_tokens"] += usage.completion_tokens or 0
            stats["peak_context"] = max(stats["peak_context"], usage.prompt_tokens or 0)
        if response.choices[0].finish_reason == "length":
            # Truncated response: tool-call JSON may be malformed downstream.
            stats["truncations"] += 1
            tlog.event("truncated", turn_idx=turn_idx, step=step)

        msg = response.choices[0].message
        messages.append(self._assistant_message(msg))
        return msg

    @staticmethod
    def _assistant_message(msg) -> dict:
        """Rebuild the assistant message (content + native tool_calls) for replay."""
        assistant_msg = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return assistant_msg

    def _execute_calls(
        self,
        turn_idx: int,
        step: int,
        msg,
        messages: list[dict],
        task: dict,
        long_context: bool,
        tlog: TrajectoryLogger,
        stats: dict,
    ) -> tuple[list[str], bool]:
        """
        Execute one step's tool calls in order, appending each tool result to
        ``messages``. Returns ``(step_call_strings, parse_failed)``; calls made
        before a malformed (truncated) one are kept, and ``parse_failed`` signals
        the caller to end the turn.
        """
        step_call_strings: list[str] = []
        for call_idx, tc in enumerate(msg.tool_calls):
            try:
                args_dict = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                # Malformed (usually truncated) tool-call arguments —
                # don't crash the run; log and end the turn.
                stats["parse_errors"] += 1
                tlog.event(
                    "tool_call_parse_error",
                    turn_idx=turn_idx,
                    step=step,
                    call_idx=call_idx,
                    name=tc.function.name,
                    raw_arguments=tc.function.arguments,
                )
                # Every tool_call in the assistant message needs a tool response
                # or the next request is malformed; stub the unanswered ones
                # (this call onward).
                for rem in msg.tool_calls[call_idx:]:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": rem.id,
                            "content": "Error: malformed (truncated) tool call.",
                        }
                    )
                return step_call_strings, True

            call_str = tool_call_to_python_string(
                tc.function.name, tc.function.arguments
            )
            tlog.tool_call(
                turn_idx, step, call_idx, tc.function.name, args_dict, call_str
            )
            step_call_strings.append(call_str)

            result = execute_call_locally(
                call_str,
                task["initial_config"],
                task["involved_classes"],
                task["id"],
                long_context,
            )
            tlog.tool_result(turn_idx, step, call_idx, result)
            stats["n_tool_calls"] += 1

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                }
            )

        return step_call_strings, False
