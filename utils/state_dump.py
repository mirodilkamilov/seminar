"""
Leak-guarded state-diff signal for the richer-signal Reflexion arm.

`richer_reflexion` adds the model's **own** final environment state to the
reflection call. The state is the agent's product, never the ground truth, so
it sidesteps the oracle-leak constraint that `utils/sanitize.py` guards — but
only if we read the right instances.

The dump is a **diff against ``initial_config``**, not the absolute state:
measured over the frozen subset, 71-77% of public attributes are never touched
by the task, and the inert surface is an active distractor (auth-shaped flags
in 16/115 failures). The diff removes it by construction. A uniform character cap
replaces oversized diffs with a sentinel, so the already-large ``long_context``
reflection call is not diluted.
"""

import json

# Uniform diff-size cap. Non-long_context failures max out at ~1.3k chars and
# long_context problem tasks start at ~4k, so any value in [1.3k, 4k] behaves
# identically
STATE_DIFF_CHAR_CAP = 2000

# Fixed strings for the two non-diff cases. Constants, so they leak nothing.
_EMPTY_DIFF = "Your actions made no changes to the environment state."
_OVERSIZED_DIFF = "[state diff omitted: too large to include]"


def _public_attrs(inst) -> dict:
    """Public attributes of a simulator instance — mirrors the grader's own
    filter (`multi_turn_checker.py:172-181`) so the diff reflects graded state."""
    return {k: v for k, v in vars(inst).items() if not k.startswith("_")}


def _long_context(task: dict) -> bool:
    # Matches what the grader rebuilds (`"long_context" in test_category`).
    return "long_context" in task["id"]


def initial_public_state(task: dict) -> dict[str, dict]:
    """Clean baseline state from ``initial_config``, keyed by class name.

    Instantiates the involved classes with an empty call list under a distinct
    ``model_name`` so its cache keys never collide with the graded ``_eval``
    instances. Cleared like any other instance by ``reset_bfcl_instances()``.
    """
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
    )

    _, involved = execute_multi_turn_func_call(
        func_call_list=[],
        initial_config=task["initial_config"],
        involved_classes=task["involved_classes"],
        model_name="richer_init",
        test_entry_id=task["id"],
        long_context=_long_context(task),
        is_evaL_run=False,
    )
    return {cls: _public_attrs(inst) for cls, inst in involved.items()}


def model_eval_state(task: dict) -> dict[str, dict]:
    """The model's post-``grade()`` state, read from the BFCL module globals.

    Reads only the model replay instances (``_eval``), never the ground-truth
    ones (``_ground_truth_eval``). Must be called after ``grade()`` and before
    the next ``reset_bfcl_instances()``.

    Returns an **empty** dict when the model made no tool calls at all — BFCL
    instantiates every involved class on the *first* executed call, so the model
    set is all-or-nothing: either all involved classes or none. An empty result
    is legitimate (the model changed nothing) and downstream becomes the "no
    changes" signal — which is exactly the right diagnosis for an under-acting
    (empty-turn) failure.
    """
    import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils as mtu

    marker = f"_eval_{task['id']}_"
    # A ground-truth key proves grade() ran; guard against reading a stale/empty
    # diff caused by a plumbing bug that skipped grading. Existence only — never
    # read GT contents.
    grade_ran = any(
        marker in k and "_ground_truth" in k and k.endswith("_instance")
        for k in mtu.__dict__
    )
    assert grade_ran, (
        f"no graded instances for {task['id']} — grade() must run before "
        f"model_eval_state()"
    )

    state: dict[str, dict] = {}
    for key, inst in mtu.__dict__.items():
        if not key.endswith("_instance") or marker not in key:
            continue
        # Ground-truth instances share the marker but carry `_ground_truth` in
        # the key — that is the answer key, so skip them.
        if "_ground_truth" in key:
            continue
        state[type(inst).__name__] = _public_attrs(inst)

    # The model set is a subset of the involved classes (all of them, or none if
    # the model never acted) — never MORE, which would mean a wrong-task read.
    involved = set(task["involved_classes"])
    assert set(state) <= involved, (
        f"model_eval_state read unexpected classes for {task['id']}: "
        f"{set(state) - involved}"
    )
    return state


def _render_diff(diff: dict[str, dict]) -> str:
    """Compact, deterministic rendering of a per-class attribute diff."""
    lines = []
    for cls in sorted(diff):
        lines.append(f"{cls}:")
        for attr in sorted(diff[cls]):
            lines.append(f"  {attr} = {json.dumps(diff[cls][attr], default=str)}")
    return "\n".join(lines)


def state_diff_string(task: dict, cap: int = STATE_DIFF_CHAR_CAP) -> str:
    """The state signal fed to the reflection call: what the model's actions
    changed relative to ``initial_config``.

    Returns a fixed sentinel for the empty and oversized cases (both constants,
    so they carry no failure-specific information). Deterministic given the
    model's post-attempt state, so reproducible.
    """
    before = initial_public_state(task)
    after = model_eval_state(task)

    diff: dict[str, dict] = {}
    for cls, attrs in after.items():
        base = before.get(cls, {})
        changed = {
            k: v
            for k, v in attrs.items()
            if json.dumps(base.get(k), default=str) != json.dumps(v, default=str)
        }
        if changed:
            diff[cls] = changed

    if not diff:
        return _EMPTY_DIFF
    # Cap on the rendered form — that is exactly what reaches the prompt, and it
    # avoids sorting nested dicts whose keys mix int and str (e.g. order ids).
    rendered = _render_diff(diff)
    if len(rendered) > cap:
        return _OVERSIZED_DIFF
    return rendered
