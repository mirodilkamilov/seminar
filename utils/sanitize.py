"""
Sanitize the BFCL grader verdict into a failure signal that leaks no ground
truth.

Reflexion retries because an oracle (the grader) says it failed. The grader's
raw output leaks the expected answer — the `details` dict carries
`ground_truth_instance_state` / `missing_items` — so the reflection call and
the retry preamble may only ever see *this* sanitized signal. Both arms
(reflexion and blind_retry) receive the identical signal (REVIEW.md §3.2,
design decision 2), so any value it adds cancels in the
Reflexion@3 − BlindRetry@3 subtraction.

Granularity (locked): class + turn, never values, never which call was
expected. The state gate's message carries no turn index (the vendored checker
doesn't record which turn fired), so state failures are class-level; the other
gates include a 1-based turn number. Works from `error_type`/`error_message`
alone — verified value-free for all four gates active at tag v1.3
(`method_invoke_order_checker`, whose message would leak GT method names, is
commented out upstream).
"""

import re


def _turn_phrase(error_message: str | None) -> str:
    """Human 1-based turn reference from the checker's 0-based message."""
    m = re.search(r"turn (\d+)", error_message or "")
    if not m:
        return "one of the task's turns"
    return f"user turn {int(m.group(1)) + 1} (counting from 1)"


def sanitize_failure_signal(error_type: str | None, error_message: str | None) -> str:
    et = error_type if error_type is not None else ""
    turn = _turn_phrase(error_message)

    if "instance_state_mismatch" in et:
        m = re.search(r"Model instance for (\w+)", error_message or "")
        env = m.group(1) if m else "one of the environments"
        return (
            f"After your attempt, the {env} environment was left in a state "
            "that does not match what the task required."
        )
    if "execution_response_mismatch" in et:
        return (
            f"By the end of {turn}, the output of one or more required tool "
            "calls was missing: a call the task required was never made, or "
            "was made with different arguments."
        )
    if "empty_turn_model_response" in et:
        return (
            f"You ended {turn} without making any tool calls, but that turn "
            "required action."
        )
    if "irrelevance_error" in et:
        return (
            f"You made tool calls in {turn}, but the correct behavior there "
            "was to make none (e.g. ask the user for missing information, or "
            "say that no available tool fits)."
        )
    # Unknown gate (defensive): binary signal rather than risk leaking
    # anything from an unrecognized message format.
    return "The attempt failed."
