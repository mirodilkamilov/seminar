"""
Leak-guarded state-diff signal for the richer-signal Reflexion arm.
"""

import json

# Uniform diff-size cap. Non-long_context failures max out at ~1.3k chars and
# long_context problem tasks start at ~4k, so any value in [1.3k, 4k] behaves
# identically
STATE_DIFF_CHAR_CAP = 2000

# Cap for the per-turn timeline (`state_timeline_string`). Set by the same rule
# as above — above where ordinary failures top out, below where the genuinely
# huge `long_context` filesystem rewrites start — but measured for the timeline,
# which repeats a changed attribute once per turn that touched it and so runs larger.
STATE_TIMELINE_CHAR_CAP = 3500

# Fixed strings for the two non-diff cases. Constants, so they leak nothing.
_EMPTY_DIFF = "Your actions made no changes to the environment state."
_OVERSIZED_DIFF = "[state diff omitted: too large to include]"

# Private instance namespaces. Both are ours: BFCL keys cached simulators
# ``{model_name}_{task_id}_{class}_instance`` and appends ``_eval`` only when
# ``is_evaL_run=True``, so these can never collide with the live run
# (``MODEL``), the graded replay (``..._eval``) or — the one that matters — the
# answer key (``..._ground_truth_eval``). Asserted below rather than trusted.
_INIT_MODEL = "richer_init"  # clean state from initial_config
_REPLAY_MODEL = "richer_replay"  # the model's own trajectory, replayed in full

for _name in (_INIT_MODEL, _REPLAY_MODEL):
    assert "eval" not in _name and "ground_truth" not in _name, (
        f"state-dump namespace {_name!r} could alias a graded or ground-truth "
        f"instance key — the whole leak argument rests on it not doing so"
    )


def _public_attrs(inst) -> dict:
    """Public attributes of a simulator instance — mirrors the grader's own
    filter (`multi_turn_checker.py:172-181`) so the diff reflects graded state."""
    return {k: v for k, v in vars(inst).items() if not k.startswith("_")}


def _long_context(task: dict) -> bool:
    # Matches what the grader rebuilds (`"long_context" in test_category`).
    return "long_context" in task["id"]


def _clear_namespace(model_name: str) -> None:
    """Drop only *our* cached instances, so each build starts from scratch.

    Deliberately not ``reset_bfcl_instances()``: that clears every namespace,
    including the live run's instances, which are still in use mid-task.
    """
    import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils as mtu

    prefix = f"{model_name}_"
    for key in [
        k for k in mtu.__dict__ if k.startswith(prefix) and k.endswith("_instance")
    ]:
        del mtu.__dict__[key]


def _snapshot(involved: dict) -> dict[str, dict[str, str]]:
    """Freeze a set of live instances into comparable JSON strings.

    Serializing *at snapshot time* is load-bearing, not a convenience: BFCL
    mutates the same instance objects in place across turns, so keeping the raw
    values would leave every snapshot aliasing the final state and every
    turn-to-turn diff would come out empty.
    """
    return {
        cls: {k: json.dumps(v, default=str) for k, v in _public_attrs(inst).items()}
        for cls, inst in involved.items()
    }


def _delta(before: dict[str, dict[str, str]], after: dict[str, dict[str, str]]) -> dict:
    """Attributes that differ between two snapshots, keyed by class."""
    out = {}
    for cls, attrs in after.items():
        base = before.get(cls, {})
        changed = {k: v for k, v in attrs.items() if base.get(k) != v}
        if changed:
            out[cls] = changed
    return out


def state_timeline_string(
        task: dict, model_calls: list[list[list[str]]], cap: int = STATE_TIMELINE_CHAR_CAP
) -> str:
    """Per-turn state *timeline*: what each turn changed, relative to the turn
    before it.

    The turnwise counterpart to ``state_diff_string``. Motivation: the net
    end-of-episode diff is always true but folds in everything downstream of
    the mistake, and a mistake the model later self-corrects shows up as nothing.
    Attributing changes to turns restores that resolution.

    Leak-free on the same grounds as the net diff, and for one extra reason
    worth being explicit about: the turn boundaries come from ``task["question"]``
    — the *task's* structure — never from the ground truth or from where the
    grader stopped. Contrast the first (defective) implementation, whose cut
    point was oracle-derived.

    Turns that changed nothing are listed as such rather than omitted: "turn 3
    persisted nothing" is exactly the under-acting signal, and dropping those
    lines would make an inactive turn indistinguishable from a missing one.
    """
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
    )

    def _run(calls: list[str]):
        _, involved = execute_multi_turn_func_call(
            func_call_list=calls,
            initial_config=task["initial_config"],
            involved_classes=task["involved_classes"],
            model_name=_REPLAY_MODEL,
            test_entry_id=task["id"],
            long_context=_long_context(task),
            is_evaL_run=False,
        )
        return involved

    _clear_namespace(_REPLAY_MODEL)
    prev = _snapshot(_run([]))  # clean state from initial_config
    involved_names = set(task["involved_classes"])
    assert set(prev) <= involved_names, (
        f"state timeline read unexpected classes for {task['id']}: "
        f"{set(prev) - involved_names}"
    )

    lines: list[str] = []
    any_change = False
    for turn_idx, turn in enumerate(model_calls):
        cur = _snapshot(_run([call for step in turn for call in step]))
        delta = _delta(prev, cur)
        if delta:
            any_change = True
            lines.append(f"Turn {turn_idx + 1}:")
            for cls in sorted(delta):
                lines.append(f"  {cls}:")
                for attr in sorted(delta[cls]):
                    lines.append(f"    {attr} = {delta[cls][attr]}")
        else:
            lines.append(f"Turn {turn_idx + 1}: no state changes")
        prev = cur

    if not any_change:
        return _EMPTY_DIFF
    rendered = "\n".join(lines)
    return _OVERSIZED_DIFF if len(rendered) > cap else rendered

# def _replay(task: dict, model_name: str, calls: list[str]) -> dict[str, dict]:
#     """Instantiate the task's classes from ``initial_config`` and apply ``calls``.
#
#     An empty list is meaningful and supported: BFCL instantiates the involved
#     classes before executing anything, so ``calls=[]`` yields the clean starting
#     state. Execution errors are swallowed per-call by BFCL itself (it appends an
#     error string rather than raising), which is what we want — a call the model
#     got wrong should affect the state exactly as much as it did at run time.
#     """
#     from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
#         execute_multi_turn_func_call,
#     )
#
#     _clear_namespace(model_name)
#     _, involved = execute_multi_turn_func_call(
#         func_call_list=calls,
#         initial_config=task["initial_config"],
#         involved_classes=task["involved_classes"],
#         model_name=model_name,
#         test_entry_id=task["id"],
#         long_context=_long_context(task),
#         is_evaL_run=False,
#     )
#     state = {cls: _public_attrs(inst) for cls, inst in involved.items()}
#     # Never MORE classes than the task involves — that would mean a wrong-task
#     # or wrong-namespace read.
#     involved_names = set(task["involved_classes"])
#     assert set(state) <= involved_names, (
#         f"state dump read unexpected classes for {task['id']}: "
#         f"{set(state) - involved_names}"
#     )
#     return state
#
# def initial_public_state(task: dict) -> dict[str, dict]:
#     """Clean baseline state from ``initial_config``, keyed by class name."""
#     return _replay(task, _INIT_MODEL, [])
#
#
# def model_final_state(task: dict, model_calls: list[list[list[str]]]) -> dict[str, dict]:
#     """The model's **end-of-episode** state, keyed by class name.
#
#     ``model_calls`` is the ``turn → step → call strings`` list the architecture
#     accumulated (or, for a seeded attempt, the one rebuilt from its log by
#     ``utils.conversation.log_to_calls``) — i.e. exactly the trajectory that gets
#     graded. It is flattened and replayed in one go: BFCL caches instances in its
#     module globals, so executing the calls in one batch or step-by-step reaches
#     the same state.
#
#     Unlike the graded ``_eval`` instances this covers the **whole** episode; see
#     the module docstring for why that distinction cost 19 false signals.
#     """
#     flat = [call for turn in model_calls for step in turn for call in step]
#     return _replay(task, _REPLAY_MODEL, flat)

# def _render_diff(diff: dict[str, dict]) -> str:
#     """Compact, deterministic rendering of a per-class attribute diff."""
#     lines = []
#     for cls in sorted(diff):
#         lines.append(f"{cls}:")
#         for attr in sorted(diff[cls]):
#             lines.append(f"  {attr} = {json.dumps(diff[cls][attr], default=str)}")
#     return "\n".join(lines)
