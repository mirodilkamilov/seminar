"""
Central configuration: paths, model, client, and BFCL category mapping.

Importing this module sets up `sys.path` so the cloned BFCL repo is importable
(this has to happen early — `run_benchmark.py` imports `bfcl_eval` at module
top). Constructing the OpenAI client is cheap (it does not open a connection
until the first request), so it stays at import time too.
"""
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import OpenAI

# --- Paths -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BFCL_REPO = PROJECT_ROOT / "gorilla" / "berkeley-function-call-leaderboard"
DATA_DIR = BFCL_REPO / "bfcl_eval" / "data"
DOCS_DIR = DATA_DIR / "multi_turn_func_doc"


def setup_bfcl_path() -> None:
    """Make the cloned BFCL repo importable. Idempotent."""
    p = str(BFCL_REPO)
    if p not in sys.path:
        sys.path.insert(0, p)


setup_bfcl_path()

# --- Model / client --------------------------------------------------------

MODEL = "qwen3-next-80b-a3b-instruct"
BASE_URL = "https://llms.innkube.fim.uni-passau.de/v1"
MAX_STEPS_PER_TURN = 30  # caps runaway loops only; cap-hits are logged


def get_client() -> OpenAI:
    """Construct an OpenAI client pointed at the FIM endpoint.

    An explicit per-request timeout is essential: without one a dead/stalled
    connection to the shared university endpoint blocks the whole run forever
    (observed 26 Jul — a 38-min hang mid-run with the client stuck reading one
    socket).
    """
    load_dotenv()
    return OpenAI(
        api_key=os.environ["FIM_API_KEY"],
        base_url=BASE_URL,
        timeout=httpx.Timeout(180.0, connect=10.0),
        max_retries=0,
    )


client = get_client()

# --- BFCL multi-turn categories --------------------------------------------

# The four categories present at tag v1.3 (no `composite`).
CATEGORIES = ["base", "miss_param", "miss_func", "long_context"]


def task_paths(category: str) -> tuple[Path, Path]:
    """(task_file, possible_answer_file) for a multi-turn category."""
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category {category!r}; expected one of {CATEGORIES}")
    stem = f"BFCL_v3_multi_turn_{category}.json"
    return DATA_DIR / stem, DATA_DIR / "possible_answer" / stem


# Back-compat aliases (the original single-category constants).
TASK_FILE, ANSWER_FILE = task_paths("base")
