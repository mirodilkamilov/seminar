"""
Pretty-print BFCL trajectory logs from the command line.

Quick viewer for the JSONL trajectories written by `TrajectoryLogger`. Resolves
short task ids against a run-label directory under logs/ so you can eyeball a
failure without typing the full path. Rendering itself lives in
`utils.logging.pretty_print_log` (shared with the runner) — this is just a CLI.

    python print_log.py 117                  # logs/baseline/multi_turn_base_117.jsonl
    python print_log.py base_117 miss_param_3
    python print_log.py 117 --arch baseline__run2
    python print_log.py 42 --category miss_func
    python print_log.py 117 120 28 -o trace.txt   # concatenate into ONE file
    python print_log.py 117 120 -s   # -> pretty_logs/baseline_base_117.txt, baseline_base_120.txt

Offline-only; no FIM API key needed.
"""

import argparse
import json
import sys
from pathlib import Path

from utils.logging import pretty_print_log

PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_ROOT = PROJECT_ROOT / "logs"
DEFAULT_SAVE_DIR = LOGS_ROOT / "pretty_logs"
ANSWER_DIR = (
    PROJECT_ROOT / "gorilla" / "berkeley-function-call-leaderboard"
    / "bfcl_eval" / "data" / "possible_answer"
)
CATEGORIES = ("base", "miss_param", "miss_func", "long_context")


def load_ground_truth(path: Path):
    """BFCL ground-truth turn list for a trajectory log, or None.

    The log is named after its task id (multi_turn_<category>_<n>), so the
    answer file follows from it. Best-effort: None if the file or id is missing.
    """
    task_id = path.stem
    core = task_id[len("multi_turn_") :] if task_id.startswith("multi_turn_") else task_id
    category = next((c for c in CATEGORIES if core.startswith(c + "_")), None)
    if category is None:
        return None
    answer_file = ANSWER_DIR / f"BFCL_v3_multi_turn_{category}.json"
    if not answer_file.exists():
        return None
    for line in answer_file.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("id") == task_id:
            return row.get("ground_truth")
    return None


def out_name(path: Path) -> str:
    """Filename stem for a saved trajectory: `<category>_<task_id>`.

    Drops the `multi_turn_` prefix so `multi_turn_base_117.jsonl` saves as
    `base_117.txt`.
    """
    stem = path.stem
    prefix = "multi_turn_"
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


def resolve_path(token: str, arch: str, category: str) -> Path:
    """Map a user token to a trajectory JSONL path under logs/<arch>/.

    Accepts a full path, a full id (`multi_turn_base_117`), a
    `<category>_<n>` id (`base_117`), or a bare number (`117`, expanded with
    --category).
    """
    p = Path(token)
    if p.suffix == ".jsonl" or p.exists():
        return p

    stem = token
    if stem.isdigit():
        stem = f"multi_turn_{category}_{stem}"
    elif not stem.startswith("multi_turn_"):
        stem = f"multi_turn_{stem}"
    return LOGS_ROOT / arch / f"{stem}.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("tasks", nargs="+", help="task ids (117, base_117, ...) or paths")
    ap.add_argument(
        "--arch",
        default="baseline",
        help="run-label dir under logs/ (default: baseline)",
    )
    ap.add_argument(
        "--category",
        default="base",
        help="category used to expand bare-number ids (default: base)",
    )
    dest = ap.add_mutually_exclusive_group()
    dest.add_argument(
        "-o", "--out", help="concatenate all output into this single file"
    )
    dest.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="save each task to its own <category>_<task_id>.txt under --save-dir",
    )
    ap.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help=f"folder for --save (default: {DEFAULT_SAVE_DIR.name}/)",
    )
    args = ap.parse_args()

    paths = [(t, resolve_path(t, args.arch, args.category)) for t in args.tasks]
    found = [(t, p) for t, p in paths if p.exists()]
    missing = [(t, p) for t, p in paths if not p.exists()]

    def render(path: Path, printer, header: bool) -> None:
        if header:
            printer(f"{'=' * 70}\n# {path}\n{'=' * 70}")
        pretty_print_log(path, printer=printer, ground_truth=load_ground_truth(path))

    if args.save:
        # One self-describing file per task, named <run_label>_<category>_<id>
        # so the same task id from different runs (baseline vs baseline__run2)
        # doesn't collide.
        args.save_dir.mkdir(parents=True, exist_ok=True)
        for _, path in found:
            name = f"{path.parent.name}_{out_name(path)}"
            out_path = args.save_dir / f"{name}.txt"
            with open(out_path, "w", encoding="utf-8") as fh:
                render(path, lambda s="": print(s, file=fh), header=True)
            print(f"Wrote {name} -> {out_path}")
    elif args.out:
        # All trajectories concatenated into one file.
        with open(args.out, "w", encoding="utf-8") as fh:

            def printer(s: str = "") -> None:
                print(s, file=fh)

            for i, (_, path) in enumerate(found):
                if i:
                    printer()
                render(path, printer, header=True)
        print(f"Wrote {len(found)} trajectory(ies) -> {args.out}")
    else:
        # Stdout. Header only when there's more than one trajectory.
        for i, (_, path) in enumerate(found):
            if i:
                print()
            render(path, print, header=len(found) > 1)

    for token, path in missing:
        print(f"!! not found: {token} -> {path}", file=sys.stderr)
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
