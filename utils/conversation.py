"""
Rebuild a conversation from a trajectory log, and render a conversation as
plain text.

Both feed Reflexion's reflection call: a live attempt hands over its final
message list directly, while a *seeded* attempt 1 (reused from the baseline
run — REVIEW.md §3.2) is first reconstructed from its log with
`log_to_messages`. Either way the reflection sees the same canonical
`messages_to_text` rendering, so reflection inputs are identical in form
across attempts.
"""

import json


def log_to_messages(log_path, default_system_prompt: str | None = None) -> list[dict]:
    """
    Reconstruct the message list of a *single-attempt* run from its
    TrajectoryLogger jsonl. Returns OpenAI-format (JSON) chat messages —
    exactly what `_run_fc_loop` appended live. Example return::

        {"role": "system",    "content": "<system prompt>"}
        {"role": "user",      "content": "I am alex. Check ..."}
        {"role": "assistant", "content": "I'll check ...",   # None if no preamble
         "tool_calls": [{"id": "chatcmpl-tool-b944...", "type": "function",
                         "function": {"name": "pwd", "arguments": "{}"}}]}
        {"role": "tool",      "tool_call_id": "chatcmpl-tool-b944...",
         "content": "{\\"current_working_directory\\": \\"/alex\\"}"}

    `arguments` and a tool message's `content` are both strings. One assistant
    message may carry several `tool_calls`, each followed by its own `tool`
    message (paired via the `call_idx` order of the preceding `llm_response`);
    content without `tool_calls` ends the turn.

    ``default_system_prompt`` covers logs written before `task_start` recorded
    the system prompt (the baseline run-1 logs). Multi-attempt logs (more than
    one `task_start`) are rejected — several fresh conversations are
    interleaved in one file there.
    """
    with open(log_path, encoding="utf-8") as fh:
        events = [json.loads(line) for line in fh if line.strip()]

    if sum(e["type"] == "task_start" for e in events) > 1:
        raise ValueError(
            f"{log_path} is a multi-attempt log; cannot reconstruct one conversation"
        )

    messages: list[dict] = []
    system_prompt = next(
        (e.get("system_prompt") for e in events if e["type"] == "task_start"), None
    ) or default_system_prompt
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    last_tool_calls: list[dict] = []  # tool_calls of the latest assistant message
    for event in events:
        task = event["type"]
        if task == "user_turn":
            messages.extend(event["messages"])
        elif task == "llm_response":
            msg = {"role": "assistant", "content": event.get("content")}
            tcs = event.get("tool_calls") or []
            if tcs:
                msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in tcs
                ]
            messages.append(msg)
            last_tool_calls = tcs
        elif task == "tool_result":
            tc = last_tool_calls[event["call_idx"]]
            messages.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": str(event["result"])}
            )
        elif task == "tool_call_parse_error":
            # _execute_calls stubs the malformed call and every later one in
            # the same step so each tool_call has a tool response.
            for tc in last_tool_calls[event["call_idx"]:]:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": "Error: malformed (truncated) tool call.",
                    }
                )
    return messages


def log_to_calls(log_path) -> list[list[list[str]]]:
    """
    Reconstruct the graded call list of a *single-attempt* run from its
    TrajectoryLogger jsonl, in the ``list[list[list[str]]]`` (turn → step →
    call strings) shape ``multi_turn_checker`` expects — the same object
    ``_run_fc_loop`` accumulates live as ``all_turns_calls``.

    Used to re-grade the seeded attempt-1 failure in ``richer_reflexion``: the
    runner skips ``grade()`` for a seeded attempt, so there are no ``_eval``
    simulator instances to dump. Rebuilding the baseline call list and grading
    it repopulates them (zero LLM cost) and reproduces the seed verdict.

    One entry per turn (empty list for turns with no calls), so per-turn
    alignment with the ground truth is preserved. Multi-attempt logs (more than
    one ``task_start``) are rejected, matching ``log_to_messages``.
    """
    with open(log_path, encoding="utf-8") as fh:
        events = [json.loads(line) for line in fh if line.strip()]

    if sum(e["type"] == "task_start" for e in events) > 1:
        raise ValueError(
            f"{log_path} is a multi-attempt log; cannot reconstruct one call list"
        )

    # turn_idx -> step -> ordered call strings, then flattened per turn.
    per_turn: dict[int, dict[int, list[str]]] = {}
    n_turns = 0
    for event in events:
        t = event["type"]
        if t == "turn_end":
            n_turns = max(n_turns, event["turn_idx"] + 1)
        elif t == "tool_call":
            turn = per_turn.setdefault(event["turn_idx"], {})
            turn.setdefault(event["step"], []).append(event["call_str"])
            n_turns = max(n_turns, event["turn_idx"] + 1)

    calls: list[list[list[str]]] = []
    for turn_idx in range(n_turns):
        steps = per_turn.get(turn_idx, {})
        calls.append([steps[s] for s in sorted(steps)])
    return calls


def messages_to_text(messages: list[dict]) -> str:
    """Plain-text rendering of a message list, for the reflection call."""
    lines = []
    for m in messages:
        role = m.get("role")
        if role in ("system", "user"):
            lines.append(f"[{role}] {m.get('content')}")
        elif role == "assistant":
            if m.get("content"):
                lines.append(f"[assistant] {m['content']}")
            for tc in m.get("tool_calls") or []:
                fn = tc["function"]
                lines.append(f"[tool_call] {fn['name']}({fn['arguments']})")
        elif role == "tool":
            lines.append(f"[tool_result] {m.get('content')}")
    return "\n".join(lines)
