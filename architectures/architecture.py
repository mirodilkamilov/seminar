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
import time
from abc import ABC

from utils.config import MAX_STEPS_PER_TURN, client, MODEL, DOCS_DIR
from utils.executor import tool_call_to_python_string, execute_call_locally
from utils.logging import TrajectoryLogger
from utils.retry import call_with_retry
from utils.schema import load_tools_for_classes

# utils.config (imported above) has already put the BFCL repo on sys.path.
from bfcl_eval.constants.default_prompts import (
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC as _ADDITIONAL_FUNCTION_PROMPT,
)


class Architecture(ABC):
    #: short identifier used for log/results directories — override per subclass
    name: str = "architecture"

    def system_prompt(self) -> str:
        """The system prompt. Subclasses override to vary the reasoning scaffold."""
        raise NotImplementedError

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
        all_tools = load_tools_for_classes(task["involved_classes"], DOCS_DIR)
        # miss_func: some functions are held out of the toolset until a later
        # "holdout" turn. `missed_function` maps {turn_idx_str: [func names]}.
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
        # long_context category must match the grader's simulator scaffold.
        long_context = "long_context" in task["id"]
        tlog.task_start(task)

        messages = [{"role": "system", "content": self.system_prompt()}]
        all_turns_calls: list[list[list[str]]] = []
        stats = {
            "n_llm_calls": 0,
            "n_tool_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "peak_context": 0,
            "latency_s": 0.0,
            "max_steps_hits": 0,
        }

        for turn_idx, turn_messages in enumerate(task["question"]):
            if str(turn_idx) in missed:
                # Holdout turn: reveal the held-out function(s) and inject BFCL's
                # synthetic "more functions available" prompt. The question turn
                # itself is empty here, so we replace it.
                for name in missed[str(turn_idx)]:
                    active_tools.append(held_out_tools[name])
                turn_messages = [
                    {"role": "user", "content": _ADDITIONAL_FUNCTION_PROMPT}
                ]
            for m in turn_messages:
                messages.append(m)
            tlog.user_turn(turn_idx, turn_messages)

            this_turn_calls: list[list[str]] = []

            for step in range(MAX_STEPS_PER_TURN):
                tlog.llm_request(turn_idx, step, MODEL, messages, active_tools)
                t0 = time.monotonic()
                response = call_with_retry(
                    client,
                    model=MODEL,
                    messages=messages,
                    tools=active_tools,
                    tool_choice="auto",
                    temperature=0,
                    max_tokens=2048,
                )
                latency = time.monotonic() - t0
                tlog.llm_response(turn_idx, step, response, latency)

                # --- stats ---
                stats["n_llm_calls"] += 1
                stats["latency_s"] += latency
                usage = response.usage
                if usage:
                    stats["input_tokens"] += usage.prompt_tokens or 0
                    stats["output_tokens"] += usage.completion_tokens or 0
                    stats["peak_context"] = max(
                        stats["peak_context"], usage.prompt_tokens or 0
                    )
                if response.choices[0].finish_reason == "length":
                    # Truncated response: tool-call JSON may be malformed below.
                    tlog.event("truncated", turn_idx=turn_idx, step=step)

                msg = response.choices[0].message

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
                messages.append(assistant_msg)

                if not msg.tool_calls:
                    break  # model is done with this turn

                step_call_strings = []
                parse_failed = False
                for call_idx, tc in enumerate(msg.tool_calls):
                    try:
                        args_dict = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        # Malformed (usually truncated) tool-call arguments —
                        # don't crash the run; log and end the turn.
                        tlog.event(
                            "tool_call_parse_error",
                            turn_idx=turn_idx,
                            step=step,
                            call_idx=call_idx,
                            name=tc.function.name,
                            raw_arguments=tc.function.arguments,
                        )
                        # Every tool_call in the assistant message needs a tool
                        # response or the next request is malformed; stub the
                        # unanswered ones (this call onward).
                        for rem in msg.tool_calls[call_idx:]:
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": rem.id,
                                    "content": "Error: malformed (truncated) tool call.",
                                }
                            )
                        parse_failed = True
                        break

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

                if step_call_strings:
                    this_turn_calls.append(step_call_strings)
                if parse_failed:
                    break  # truncated tool call: can't reliably continue the turn
            else:
                stats["max_steps_hits"] += 1
                tlog.event("max_steps_reached", turn_idx=turn_idx)

            tlog.turn_end(turn_idx, n_steps=len(this_turn_calls))
            all_turns_calls.append(this_turn_calls)

        return all_turns_calls, stats
