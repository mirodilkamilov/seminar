"""
Flatten results/<label>/<category>.jsonl into analysis-friendly CSVs.

Scans every architecture run under results/ (each subdirectory is one run
label, e.g. `baseline`, `baseline__run2`, `react`) and produces:

  results/results.csv   one row per (architecture, category, task) — the tidy
                        long table to load into pandas/Excel for paired
                        analysis. Enriched with `involved_classes` (joined from
                        the BFCL task files) so per-class breakdowns are easy.

  results/summary.csv   one row per (architecture, category): pass rate plus
                        median cost columns.

Offline-only: imports nothing that needs the FIM API key. Run any time.

    python results_to_csv.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results"
DATA_DIR = (
    PROJECT_ROOT / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"
)

# Per-task columns, in output order. Stats keys come straight from the result
# rows; the rest are added/derived here.
TASK_COLUMNS = [
    "architecture",
    "category",
    "task_id",
    "passed",
    "n_turns",
    "involved_classes",
    "n_llm_calls",
    "n_tool_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "peak_context",
    "latency_s",
    "max_steps_hits",
    "error_type",
    "error_message",
]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def involved_classes_map(category):
    """task_id -> '|'-joined involved_classes, from the BFCL task file."""
    task_file = DATA_DIR / f"BFCL_v3_multi_turn_{category}.json"
    if not task_file.exists():
        return {}
    return {
        t["id"]: "|".join(t.get("involved_classes", []))
        for t in load_jsonl(task_file)
    }


def collect_rows():
    """Yield one flat dict per task across every run label and category."""
    class_cache = {}
    for label_dir in sorted(p for p in RESULTS_ROOT.iterdir() if p.is_dir()):
        for results_file in sorted(label_dir.glob("*.jsonl")):
            category = results_file.stem
            if category not in class_cache:
                class_cache[category] = involved_classes_map(category)
            classes = class_cache[category]
            for r in load_jsonl(results_file):
                yield {
                    "architecture": label_dir.name,
                    "category": r.get("category", category),
                    "task_id": r["task_id"],
                    "passed": r["passed"],
                    "n_turns": r.get("n_turns"),
                    "involved_classes": classes.get(r["task_id"], ""),
                    "n_llm_calls": r.get("n_llm_calls"),
                    "n_tool_calls": r.get("n_tool_calls"),
                    "input_tokens": r.get("input_tokens"),
                    "output_tokens": r.get("output_tokens"),
                    "total_tokens": (r.get("input_tokens") or 0)
                    + (r.get("output_tokens") or 0),
                    "peak_context": r.get("peak_context"),
                    "latency_s": round(r["latency_s"], 2) if "latency_s" in r else None,
                    "max_steps_hits": r.get("max_steps_hits"),
                    "error_type": r.get("error_type"),
                    "error_message": r.get("error_message"),
                }


def write_tasks(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TASK_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def write_summary(rows, path):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["architecture"], r["category"])].append(r)

    def med(vals):
        vals = [v for v in vals if v is not None]
        return round(median(vals), 2) if vals else ""

    cols = [
        "architecture",
        "category",
        "n_tasks",
        "n_pass",
        "pass_rate",
        "median_latency_s",
        "median_input_tokens",
        "median_total_tokens",
        "median_peak_context",
        "median_llm_calls",
        "max_steps_hits",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for (arch, cat), g in sorted(groups.items()):
            n = len(g)
            n_pass = sum(bool(r["passed"]) for r in g)
            w.writerow(
                {
                    "architecture": arch,
                    "category": cat,
                    "n_tasks": n,
                    "n_pass": n_pass,
                    "pass_rate": round(n_pass / n, 4) if n else "",
                    "median_latency_s": med([r["latency_s"] for r in g]),
                    "median_input_tokens": med([r["input_tokens"] for r in g]),
                    "median_total_tokens": med([r["total_tokens"] for r in g]),
                    "median_peak_context": med([r["peak_context"] for r in g]),
                    "median_llm_calls": med([r["n_llm_calls"] for r in g]),
                    "max_steps_hits": sum((r["max_steps_hits"] or 0) for r in g),
                }
            )


def main():
    rows = list(collect_rows())
    if not rows:
        print(f"No results found under {RESULTS_ROOT}/")
        return

    tasks_csv = RESULTS_ROOT / "results.csv"
    summary_csv = RESULTS_ROOT / "summary.csv"
    write_tasks(rows, tasks_csv)
    write_summary(rows, summary_csv)

    # Console digest so a bare run is informative.
    groups = defaultdict(list)
    for r in rows:
        groups[(r["architecture"], r["category"])].append(r)
    print(f"{len(rows)} task rows  ->  {tasks_csv}")
    print(f"{len(groups)} (arch, category) groups  ->  {summary_csv}\n")
    print(f"{'architecture':<18}{'category':<14}{'pass':>8}   {'rate':>5}")
    print("-" * 50)
    for (arch, cat), g in sorted(groups.items()):
        n_pass = sum(bool(r["passed"]) for r in g)
        print(f"{arch:<18}{cat:<14}{n_pass:>4}/{len(g):<3}   {n_pass / len(g):>4.0%}")


if __name__ == "__main__":
    main()
