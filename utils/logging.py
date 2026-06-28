"""
Trajectory logging for all architectures.

Every run writes a JSONL file at:
    logs/{architecture}/{task_id}.jsonl

Each line is a JSON object with at minimum:
    {
        "ts":   <float seconds since task start>,
        "type": <event type string>,
        ...event-specific fields...
    }

Standard event types
--------------------
task_start      task_id, architecture, involved_classes, n_turns,
                missed_function ({turn_idx_str: [held-out names]}, miss_func only)
user_turn       turn_idx, messages (list of {role, content})
llm_request     turn_idx, step, model, n_messages, n_tools
tools_revealed  turn_idx, names, n_active_tools (miss_func holdout turn only)
llm_response    turn_idx, step, finish_reason, content, tool_calls,
                input_tokens, output_tokens, latency_s
tool_call       turn_idx, step, call_idx, name, arguments (dict), call_str
tool_result     turn_idx, step, call_idx, result
turn_end        turn_idx, n_steps
task_end        result ("pass"/"fail"/"error"), error_message, elapsed_s

Usage (context manager)
-----------------------
    with TrajectoryLogger("baseline", task["id"]) as tlog:
        tlog.task_start(task)
        ...
        tlog.llm_request(turn_idx=0, step=0, model=MODEL, messages=msgs, tools=tools)
        response = call_with_retry(client, ...)
        tlog.llm_response(turn_idx=0, step=0, response=response, latency_s=elapsed)
        ...
        tlog.task_end(passed=True)
"""

import json
import time
from pathlib import Path


class TrajectoryLogger:
    """
    Context-manager that writes a JSONL trajectory log for one task run.

    Parameters
    ----------
    architecture : str
        Short name of the architecture, e.g. ``"baseline"`` or ``"react"``.
    task_id : str
        BFCL task ID, e.g. ``"multi_turn_base_1"``.
    logs_root : str or Path
        Root directory for all logs (default ``"logs"`` relative to CWD).
    """

    def __init__(
        self,
        architecture: str,
        task_id: str,
        logs_root: str | Path = "logs",
    ) -> None:
        self.architecture = architecture
        self.task_id = task_id
        self.path = Path(logs_root) / architecture / f"{task_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
        self._start = time.monotonic()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, event_type: str, **fields) -> None:
        record = {
            "ts": round(time.monotonic() - self._start, 3),
            "type": event_type,
            **fields,
        }
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    # ------------------------------------------------------------------
    # Structured event methods — call these from your agent loop
    # ------------------------------------------------------------------

    def task_start(self, task: dict) -> None:
        self._write(
            "task_start",
            task_id=task["id"],
            architecture=self.architecture,
            involved_classes=task.get("involved_classes", []),
            n_turns=len(task.get("question", [])),
            # miss_func only: which functions are held out and at which turn they
            # are revealed (empty for other categories). See _setup_tools.
            missed_function=task.get("missed_function", {}),
        )

    def user_turn(self, turn_idx: int, messages: list[dict]) -> None:
        self._write("user_turn", turn_idx=turn_idx, messages=messages)

    def llm_request(
        self,
        turn_idx: int,
        step: int,
        model: str,
        messages: list[dict],
        tools: list[dict],
    ) -> None:
        self._write(
            "llm_request",
            turn_idx=turn_idx,
            step=step,
            model=model,
            n_messages=len(messages),
            n_tools=len(tools),
        )

    def llm_response(
        self,
        turn_idx: int,
        step: int,
        response,  # openai.types.chat.ChatCompletion
        latency_s: float,
    ) -> None:
        msg = response.choices[0].message
        usage = response.usage
        tool_calls_log = None
        if msg.tool_calls:
            tool_calls_log = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in msg.tool_calls
            ]
        self._write(
            "llm_response",
            turn_idx=turn_idx,
            step=step,
            finish_reason=response.choices[0].finish_reason,
            content=msg.content,
            tool_calls=tool_calls_log,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            latency_s=round(latency_s, 3),
        )

    def tool_call(
        self,
        turn_idx: int,
        step: int,
        call_idx: int,
        name: str,
        arguments: dict,
        call_str: str,
    ) -> None:
        self._write(
            "tool_call",
            turn_idx=turn_idx,
            step=step,
            call_idx=call_idx,
            name=name,
            arguments=arguments,
            call_str=call_str,
        )

    def tool_result(
        self,
        turn_idx: int,
        step: int,
        call_idx: int,
        result: str,
    ) -> None:
        self._write(
            "tool_result",
            turn_idx=turn_idx,
            step=step,
            call_idx=call_idx,
            result=result,
        )

    def turn_end(self, turn_idx: int, n_steps: int) -> None:
        self._write("turn_end", turn_idx=turn_idx, n_steps=n_steps)

    def task_end(
        self,
        passed: bool,
        error_message: str | None = None,
        error_type: str | None = None,
        stats: dict | None = None,
    ) -> None:
        self._write(
            "task_end",
            result="pass" if passed else "fail",
            error_message=error_message,
            error_type=error_type,  # e.g. multi_turn:instance_state_mismatch
            stats=stats or {},
            elapsed_s=round(time.monotonic() - self._start, 3),
        )

    # Allow free-form extra events (e.g. for reflection steps)
    def event(self, event_type: str, **fields) -> None:
        self._write(event_type, **fields)

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._write("task_end", result="error", error=str(exc_val))
        self.close()
        return False  # don't suppress exceptions


def pretty_print_log(path, printer=print) -> None:
    """
    Render a trajectory JSONL (written by `TrajectoryLogger`) as readable text.

    Single source of truth for console output: architectures only *log*; the
    runner calls this to print. Also handy for eyeballing failures during the
    qualitative analysis.
    """
    with open(path, encoding="utf-8") as fh:
        events = [json.loads(line) for line in fh if line.strip()]

    for ev in events:
        t = ev["type"]
        if t == "user_turn":
            for m in ev.get("messages", []):
                printer(f"\n--- Turn {ev['turn_idx'] + 1} ---")
                printer(f"[user] {m.get('content')}")
        elif t == "llm_response" and ev.get("content"):
            # Reasoning / final answer text the model produced.
            printer(f"[assistant] {ev['content']}")
        elif t == "tool_call":
            printer(f"[tool_call] {ev['call_str']}")
        elif t == "tool_result":
            printer(f"[tool_result] {ev['result']}")
        elif t == "tools_revealed":
            names = ", ".join(ev.get("names", []))
            printer(
                f"[tools_revealed] {names} "
                f"(now {ev.get('n_active_tools')} tools available)"
            )
        elif t == "reflection":
            printer(f"[reflection] {ev.get('text', '')}")
        elif t == "max_steps_reached":
            printer(f"⚠ MAX_STEPS_PER_TURN reached on turn {ev['turn_idx'] + 1}")
