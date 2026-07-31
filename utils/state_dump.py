"""
Leak-guarded state signal for the richer-signal Reflexion arm
(``richer_reflexion_turnwise``).

The arm's whole validity rests on this block being the **agent's own product**:
the model's trajectory is replayed from ``initial_config`` in a private
namespace, and nothing here ever reads the grader's instances. Two independent
guards, both structural rather than advisory — see ``_REPLAY_MODEL`` below, and
the turn-boundary note in ``state_timeline_string``.
"""

import json

# Size cap for the per-turn timeline. Chosen by a bimodal gap in the measured
# distribution: ordinary failures top out around 1.3k chars while the genuinely
# huge `long_context` filesystem rewrites start around 4k, so any value in
# between behaves identically. Set above a plain net diff's cap because a
# timeline repeats a changed attribute once per turn that touched it.
# As delivered over the 218 reflection calls of the committed run: 175 real
# timelines (median 510, max 3,156 chars), 28 empty, 15 → the oversized
# sentinel, all of them `long_context`.
STATE_TIMELINE_CHAR_CAP = 3500

# Fixed strings for the two non-diff cases. Constants, so they leak nothing.
_EMPTY_DIFF = "Your actions made no changes to the environment state."
_OVERSIZED_DIFF = "[state diff omitted: too large to include]"

# Our private instance namespace — the load-bearing leak guard. BFCL keys
# cached simulators ``{model_name}_{task_id}_{class}_instance`` and appends
# ``_eval`` only when ``is_evaL_run=True``, so this name can never collide with
# the live run (``MODEL``), the graded replay (``..._eval``) or — the one that
# matters — the answer key (``..._ground_truth_eval``). Asserted at import
# rather than trusted: a filter can be forgotten at a call site, a namespace
# that cannot alias in the first place cannot be.
_REPLAY_MODEL = "richer_replay"  # the model's own trajectory, replayed in full

assert "eval" not in _REPLAY_MODEL and "ground_truth" not in _REPLAY_MODEL, (
    f"state-dump namespace {_REPLAY_MODEL!r} could alias a graded or "
    f"ground-truth instance key — the whole leak argument rests on it not doing so"
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

    Why turn-attributed rather than one net diff against ``initial_config``: a
    net end-of-episode diff is always true but folds in everything downstream of
    the mistake, and a mistake the model later self-corrects shows up as
    nothing. Attributing changes to turns restores that resolution.

    **Leak-free, by two independent structural properties.** (1) The replay runs
    under ``_REPLAY_MODEL``, a namespace that cannot alias the live run, the
    graded ``_eval`` instances, or the ``_ground_truth_eval`` answer key — this
    is asserted at import, not left to a string filter. (2) The turn boundaries
    come from ``task["question"]`` — the *task's* own structure — never from the
    ground truth or from where the grader stopped.

    Property (2) is not decoration. The first implementation of this arm read
    the grader's ``_eval`` instances after ``grade()``; since
    ``multi_turn_checker`` returns at the first failing turn, its cut point was
    decided by the ground-truth comparison, silently carrying turn-localization
    that ``utils.sanitize`` deliberately withholds for state-gate failures and
    that no control arm received. That build was discarded.

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
