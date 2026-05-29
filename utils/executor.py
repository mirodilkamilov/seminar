import json

from utils.config import MODEL


def tool_call_to_python_string(name: str, json_args: str) -> str:
    """
    Convert a model's tool call into a Python call string the BFCL executor
    can `eval`. Example:
        name='mv', json_args='{"source":"log.txt","destination":"archive"}'
        -> "mv(source='log.txt', destination='archive')"
    """
    args = json.loads(json_args)
    parts = []
    for k, v in args.items():
        # repr() handles strings, bools, ints, lists, dicts, None correctly
        parts.append(f"{k}={repr(v)}")
    return f"{name}({', '.join(parts)})"


def execute_call_locally(
    call_string: str,
    initial_config: dict,
    involved_classes: list,
    test_entry_id: str,
    long_context: bool = False,
) -> str:
    """
    Run one call string against the *live* simulator so we can feed the
    result back to the model as a tool observation.

    Uses BFCL's own executor so state is consistent with grading.

    `long_context` must match what the grader uses for the category
    (True for the `long_context` category) — otherwise the model observes a
    different simulator state at runtime than the grader rebuilds.

    State persistence note: execute_multi_turn_func_call stores the simulator
    instance in globals() on first call and reuses it on subsequent calls,
    so state accumulates correctly across multiple calls to this function
    within the same task.
    """
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
    )

    results, _ = execute_multi_turn_func_call(
        func_call_list=[call_string],
        initial_config=initial_config,
        involved_classes=involved_classes,
        model_name=MODEL,
        test_entry_id=test_entry_id,
        long_context=long_context,
        is_evaL_run=False,
    )

    # We always pass exactly one call, so there is always exactly one result.
    assert len(results) == 1, f"expected 1 result, got {len(results)}"
    return results[0]


def reset_bfcl_instances() -> None:
    """
    Clear every cached simulator instance from BFCL's module globals.

    `execute_multi_turn_func_call` caches instances in the `multi_turn_utils`
    module namespace keyed `{model}_{task_id}_{class}_instance`, with no
    run/trial number, and never deletes them. Re-running the same task id in one
    process would otherwise resume from the previous run's end-state. Call this
    at the start of every task (cheap; also bounds memory).
    """
    import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils as mtu

    for key in [k for k in mtu.__dict__ if k.endswith("_instance")]:
        del mtu.__dict__[key]

