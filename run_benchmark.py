"""
Run an architecture over the (frozen) BFCL multi-turn task subset and grade it.

Orchestration lives here — architectures only run the loop and log. This module
handles: subset selection, per-task simulator reset, grading, console rendering
(via the shared trajectory pretty-printer), and the machine-readable results
table (results/{label}/{category}.jsonl, one row per task).

Examples
--------
    python run_benchmark.py --make-subset            # freeze task_subset.json
    python run_benchmark.py --arch baseline          # full frozen subset
    python run_benchmark.py --arch react --category base
    python run_benchmark.py --arch baseline --sample 5   # quick smoke (no frozen subset)
    python run_benchmark.py --arch baseline --tag run2   # noise re-run (separate output)
"""
import argparse
import json
from pathlib import Path

from architectures.baseline import Baseline
from architectures.react import ReAct
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
    multi_turn_checker,
    multi_turn_irrelevance_checker,
)
from utils.config import CATEGORIES, MODEL, task_paths
from utils.executor import reset_bfcl_instances
from utils.logging import TrajectoryLogger, pretty_print_log
from utils.sampling import freeze_subset, load_subset, stratified_sample

# Architecture registry: --arch <key>
ARCHITECTURES = {cls.name: cls for cls in (Baseline, ReAct)}

RESULTS_ROOT = Path("results")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def grade(model_calls, answer, task, category):
    """Grade a task: state + response gate AND the irrelevance gate.

    `multi_turn_checker` only `continue`s on empty-GT turns; the check that the
    model stayed silent when it should have lives in
    `multi_turn_irrelevance_checker`. Both must pass (the latter is a no-op for
    `base`/`long_context`). Returns (passed, error_type, error_message).
    """
    result = multi_turn_checker(
        multi_turn_model_result_list_decoded=model_calls,
        multi_turn_ground_truth_list=answer["ground_truth"],
        test_entry=task,
        test_category=f"multi_turn_{category}",
        model_name=MODEL,
    )
    if not result.get("valid"):
        return False, result.get("error_type"), result.get("error_message")

    irr = multi_turn_irrelevance_checker(model_calls, answer["ground_truth"])
    if not irr.get("valid"):
        return False, irr.get("error_type"), irr.get("error_message")

    return True, None, None


def run_one(arch, task, answer, category, results_fh, label):
    print(f"\n{'=' * 70}\nRunning task: {task['id']}  ({label})")
    print(f"Involved classes: {task['involved_classes']} | "
          f"turns: {len(task['question'])}\n{'=' * 70}")

    reset_bfcl_instances()  # never resume from a previous run's end-state

    with TrajectoryLogger(label, task["id"]) as tlog:
        model_calls, stats = arch.run_task(task, tlog)
        pretty_print_log(tlog.path)  # render what the architecture logged

        # Model vs ground truth, per turn
        print(f"\n{'=' * 70}\nMODEL CALLS vs GROUND TRUTH\n{'=' * 70}")
        for i, (mt, gt) in enumerate(zip(model_calls, answer["ground_truth"])):
            print(f"Turn {i + 1}:\n  model: {mt}\n  gt:    {gt}")

        passed, error_type, error_message = grade(
            model_calls, answer, task, category
        )
        tlog.task_end(passed, error_message, error_type, stats)

    print(f"\n{'=' * 70}")
    print("✓ PASS" if passed else f"✗ FAIL: {error_type}: {error_message}")
    print(f"{'=' * 70}\n  Log: {tlog.path}")

    results_fh.write(json.dumps({
        "task_id": task["id"],
        "category": category,
        "passed": passed,
        "error_type": error_type,
        "error_message": error_message,
        "n_turns": len(task["question"]),
        **stats,
    }) + "\n")
    results_fh.flush()
    return passed


def select_tasks(category, args):
    """Return aligned (task, answer) pairs for the chosen tasks in a category."""
    task_file, answer_file = task_paths(category)
    tasks = {t["id"]: t for t in load_jsonl(task_file)}
    answers = {a["id"]: a for a in load_jsonl(answer_file)}

    if args.sample is not None:  # dev/smoke: random stratified, no frozen file
        ids = stratified_sample(list(tasks.values()), args.sample, args.seed)
    else:
        ids = load_subset()[category]
    if args.limit:
        ids = ids[: args.limit]
    return [(tasks[i], answers[i]) for i in ids]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", choices=list(ARCHITECTURES), default="baseline")
    p.add_argument("--category", choices=CATEGORIES + ["all"], default="all")
    p.add_argument("--sample", type=int, default=None,
                   help="Ignore frozen subset; randomly sample N tasks/category.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap tasks actually run (for quick tests).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default=None,
                   help="Output label suffix, e.g. --tag run2. Results and logs "
                        "go to {arch}__{tag}/ so re-runs (e.g. the baseline noise "
                        "re-run) don't overwrite each other.")
    p.add_argument("--make-subset", action="store_true",
                   help="Freeze task_subset.json (50/category, stratified) and exit.")
    args = p.parse_args()

    if args.make_subset:
        path = freeze_subset(seed=args.seed)
        print(f"Wrote {path}")
        return

    arch = ARCHITECTURES[args.arch]()  # instantiate ONCE (persists across tasks)
    label = f"{arch.name}__{args.tag}" if args.tag else arch.name
    categories = CATEGORIES if args.category == "all" else [args.category]

    out_dir = RESULTS_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for category in categories:
        pairs = select_tasks(category, args)
        results_path = out_dir / f"{category}.jsonl"
        with open(results_path, "w") as fh:
            n_pass = sum(run_one(arch, t, a, category, fh, label) for t, a in pairs)
        summary[category] = (n_pass, len(pairs))

    print(f"\n{'#' * 70}\nSUMMARY — {label}\n{'#' * 70}")
    for cat, (n_pass, total) in summary.items():
        rate = n_pass / total if total else 0.0
        print(f"  {cat:<14} {n_pass}/{total}  ({rate:.0%})")


if __name__ == "__main__":
    main()
