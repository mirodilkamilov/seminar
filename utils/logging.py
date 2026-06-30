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
                missed_function ({turn_idx_str: [held-out names]}, miss_func only),
                system_prompt (task core + architecture reasoning scaffold)
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

import ast
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

    def task_start(self, task: dict, system_prompt: str | None = None) -> None:
        self._write(
            "task_start",
            task_id=task["id"],
            architecture=self.architecture,
            involved_classes=task.get("involved_classes", []),
            n_turns=len(task.get("question", [])),
            # miss_func only: which functions are held out and at which turn they
            # are revealed (empty for other categories). See _setup_tools.
            missed_function=task.get("missed_function", {}),
            # Exact system prompt for this run (shared task core + this
            # architecture's reasoning scaffold). Logged per task so every
            # trajectory is a self-contained, reproducible record of the prompt
            # that produced it — important while the scaffold is being tuned.
            system_prompt=system_prompt,
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


def _norm_val(v):
    """Normalize a literal so equal-but-differently-written values compare equal
    (e.g. 750 vs 750.0). bool is kept distinct from int."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return tuple(_norm_val(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _norm_val(x)) for k, x in v.items()))
    return v


def _call_name(call_str):
    """Function name of a call string (the text before the first '(')."""
    return call_str.split("(", 1)[0].strip()


def _canon_call(call_str):
    """Order-/whitespace-/number-insensitive key for a call string, so model
    calls can be matched against ground-truth calls. Falls back to the stripped
    string when the call doesn't parse as a simple literal call."""
    try:
        node = ast.parse(call_str.strip(), mode="eval").body
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            return call_str.strip()
        pos = tuple(_norm_val(ast.literal_eval(a)) for a in node.args)
        kw = frozenset(
            (k.arg, _norm_val(ast.literal_eval(k.value))) for k in node.keywords
        )
        return (node.func.id, pos, kw)
    except Exception:
        return call_str.strip()


def _print_unmatched_gt(printer, gt_calls, made, turn_model_calls) -> None:
    """Flag the turn's GT calls the model didn't satisfy.

    `made` is the global set of canonical model calls; `turn_model_calls` are the
    raw call strings the model made *this* turn. A GT call the model made exactly
    (anywhere) is skipped. Otherwise it's reported as never-called, or — if the
    model called the same function this turn with other arguments — as a diff.
    """
    by_name = {}
    for c in turn_model_calls:
        by_name.setdefault(_call_name(c), []).append(c)

    lines = []
    for gt in gt_calls:
        if _canon_call(gt) in made:
            continue
        same_fn = by_name.get(_call_name(gt))
        lines.append((gt, same_fn))
    if not lines:
        return

    printer("❌ ground truth not satisfied here:")
    for gt, same_fn in lines:
        if same_fn:
            printer(f"     expected:         {gt}")
            for actual in same_fn:
                printer(f"     model called:     {actual}")
        else:
            printer(f"     never called:     {gt}")


def pretty_print_log(path, printer=print, ground_truth=None) -> None:
    """
    Render a trajectory JSONL (written by `TrajectoryLogger`) as readable text.

    Single source of truth for console output: architectures only *log*; the
    runner calls this to print. Also handy for eyeballing failures during the
    qualitative analysis.

    `ground_truth`, if given, is the BFCL answer for this task — a list indexed
    by turn, each a list of expected call strings. On a FAIL, each GT call the
    model didn't satisfy is flagged inline at its turn (matched
    whitespace/order/number-insensitively): either as *never called*, or — when
    the model called the same function with different arguments — as an
    expected-vs-actual diff. GT calls the model did make exactly aren't shown;
    they're not the problem. (Loaded by the analysis CLI; the runner passes none.)
    """
    with open(path, encoding="utf-8") as fh:
        events = [json.loads(line) for line in fh if line.strip()]

    # Surface the final grade up front (scanned ahead of the trajectory).
    end = next((e for e in events if e["type"] == "task_end"), None)
    failed = end is not None and end.get("result") != "pass"
    if end is not None:
        if not failed:
            printer("✅ PASS")
        else:
            etype = end.get("error_type") or "unknown"
            emsg = end.get("error_message") or ""
            printer(f"❌ FAIL — {etype}: {emsg}".rstrip())

    # Model calls: a global exact set (did the model make this call *anywhere*?)
    # and a per-turn name->calls map (to show same-function arg diffs in place).
    made = set()
    calls_by_turn = {}
    for e in events:
        if e["type"] == "tool_call":
            made.add(_canon_call(e["call_str"]))
            calls_by_turn.setdefault(e["turn_idx"], []).append(e["call_str"])

    # System prompt (scanned ahead) to be rendered inside turn 0, right before that turn's user message.
    system_prompt = next(
        (e.get("system_prompt") for e in events if e["type"] == "task_start"), None
    )

    for ev in events:
        t = ev["type"]
        if t == "user_turn":
            turn_idx = ev["turn_idx"]
            printer(f"\n--- Turn {turn_idx} ---")
            # Guarded so logs predating this field (e.g. earlier baseline runs)
            # still render cleanly — they just omit the [system] line.
            if turn_idx == 0 and system_prompt:
                printer(f"[system] {system_prompt}")
            for m in ev.get("messages", []):
                printer(f"[user] {m.get('content')}")
            if failed and ground_truth and turn_idx < len(ground_truth):
                _print_unmatched_gt(
                    printer, ground_truth[turn_idx], made, calls_by_turn.get(turn_idx, [])
                )
        elif t == "llm_response":
            # Reasoning / final answer text the model produced. Pure tool-call
            # turns carry no text — still surface them so the turn is visible.
            content = (ev.get("content") or "").rstrip()
            if content:
                printer(f"[assistant] {content}")
            elif ev.get("tool_calls"):
                printer("[assistant] (tool call, no text)")
        elif t == "tool_call":
            printer(f"[tool_call] {ev['call_str']}")
        elif t == "tool_result":
            printer(f"[tool_result] {ev['result']}")
        elif t == "tools_revealed":
            names = ", ".join(ev.get("names", []))
            printer(f"[tools_revealed] {names}")
        elif t == "reflection":
            printer(f"[reflection] {ev.get('text', '')}")
        elif t == "max_steps_reached":
            printer(f"⚠ MAX_STEPS_PER_TURN reached on turn {ev['turn_idx']}")
        elif t == "truncated":
            printer(f"⚠ truncated (max_tokens) on turn {ev['turn_idx']}")
        elif t == "tool_call_parse_error":
            printer(
                f"⚠ tool-call parse error on turn {ev['turn_idx']}: "
                f"{ev.get('name')}({ev.get('raw_arguments')})"
            )
