"""
Freeze the evaluation task subset, stratified by `involved_classes`.

We sample a fixed subset once and commit it to `task_subset.json` so every
architecture is scored on the exact same tasks. Stratifying by the set of
involved simulator classes keeps each category a representative spread instead
of, say, all file-system tasks.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

from utils.config import CATEGORIES, PROJECT_ROOT, task_paths

SUBSET_FILE = PROJECT_ROOT / "task_subset.json"


def _load_tasks(category: str) -> list[dict]:
    task_file, _ = task_paths(category)
    with open(task_file) as f:
        return [json.loads(line) for line in f if line.strip()]


def _stratum_key(task: dict) -> tuple[str, ...]:
    return tuple(sorted(task["involved_classes"]))


def stratified_sample(tasks: list[dict], n: int, seed: int) -> list[str]:
    """Return `n` task IDs, allocated across involved_classes strata
    proportionally to stratum size (deterministic for a given seed)."""
    rng = random.Random(seed)
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for t in tasks:
        strata[_stratum_key(t)].append(t)

    total = len(tasks)
    n = min(n, total)

    # Proportional allocation with largest-remainder rounding.
    keys = sorted(strata)  # deterministic order
    raw = {k: n * len(strata[k]) / total for k in keys}
    alloc = {k: min(int(raw[k]), len(strata[k])) for k in keys}
    remainder = n - sum(alloc.values())
    # Hand out leftover slots by largest fractional part, skipping full strata.
    for k in sorted(keys, key=lambda k: raw[k] - int(raw[k]), reverse=True):
        if remainder <= 0:
            break
        if alloc[k] < len(strata[k]):
            alloc[k] += 1
            remainder -= 1

    chosen: list[str] = []
    for k in keys:
        picked = rng.sample(strata[k], alloc[k])
        chosen.extend(t["id"] for t in picked)
    return sorted(chosen)


def make_subset(n_per_category: int = 50, seed: int = 42) -> dict[str, list[str]]:
    return {
        cat: stratified_sample(_load_tasks(cat), n_per_category, seed)
        for cat in CATEGORIES
    }


def freeze_subset(n_per_category: int = 50, seed: int = 42) -> Path:
    subset = make_subset(n_per_category, seed)
    payload = {"seed": seed, "n_per_category": n_per_category, "subset": subset}
    SUBSET_FILE.write_text(json.dumps(payload, indent=2) + "\n")
    return SUBSET_FILE


def load_subset() -> dict[str, list[str]]:
    if not SUBSET_FILE.exists():
        raise FileNotFoundError(
            f"{SUBSET_FILE} not found — run `python run_benchmark.py --make-subset` first."
        )
    return json.loads(SUBSET_FILE.read_text())["subset"]


if __name__ == "__main__":
    path = freeze_subset()
    data = json.loads(path.read_text())
    print(f"Wrote {path}")
    for cat, ids in data["subset"].items():
        print(f"  {cat}: {len(ids)} tasks")
