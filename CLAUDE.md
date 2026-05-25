# Seminar Project: Reasoning-Action Loop Architectures on BFCL v3

## TL;DR

Comparative evaluation of five LLM agent architectures on the BFCL v3
multi-turn tool-calling benchmark. Course: Agentic AI seminar, University of
Passau (WS 2025/2026). Presentation delivered 22 May 2026; report due July 2026.

Research question: **Where in the reason-act loop does adding reflection
actually help on multi-turn tool calling?**

Failure analysis (per architecture, tied to the five failure modes) is
used as supporting evidence to interpret the quantitative results, not
as a separate research goal.

## Architectures under comparison

1. **Function calling baseline** — model's native FC, no reasoning prompts.
2. **ReAct** — text-format Thought/Action/Observation prompting; actions still
   go through the native FC channel.
3. **Reflexion (episodic)** — original: on failure, reflect verbally, retry
   with reflection in context. Reflection is scoped to the current task only.
4. **Reflexion (vector DB)** — **own contribution**. Reflections are embedded
   (Octen-Embedding-8B) and stored persistently. New tasks retrieve top-k
   semantically similar past reflections and inject them into context.
5. **REBACT** — reflect-before-act: after each tool result, run a reflect step
   to detect if the last action should be revised.

## Stack

- **Model**: `qwen3-next-80b-a3b-instruct` via FIM API
  (`https://llms.innkube.fim.uni-passau.de/v1`)
- **API key**: in `.env` as `FIM_API_KEY` (gitignored)
- **Embedder**: `octen-embedding-8b` (FIM)
- **Reranker**: `qwen3-reranker-4b` (FIM, optional)
- **Vector DB**: TBD (likely Chroma or FAISS, local; nothing exotic needed)
- **Benchmark**: BFCL v3 (tag `v1.3` of github.com/ShishirPatil/gorilla,
  cloned to `./gorilla/`). Multi-turn data lives in
  `gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/`.

## Current state (as of 2026-05-25)

- **Phase 0 complete.** All de-risking tasks done; pipeline is production-ready.
- Baseline pipeline works end-to-end: `multi_turn_base_1` → PASS, with full
  trajectory log written to `logs/baseline/multi_turn_base_1.jsonl`.
- Smoke tests: `test_api.py`, `test_function_calling.py` (both pass).
- Shared utils in `utils/`: retry, schema conversion, trajectory logging.
- Next: Phase 1 — scale baseline to 5+ tasks, then pick the fixed task subset.

## Important implementation details

### Schema conversion (`utils/schema.py`)

BFCL function docs use two type names that are **not valid JSON Schema**:
- `"dict"` → must become `"object"` (top-level AND in nested properties)
- `"float"` → must become `"number"` (42 occurrences across all 8 doc files)

Both are fixed recursively in `utils/schema.py::_normalize_schema()`. The old
`bfcl_schema_to_openai()` in `run_one_bfcl_task.py` only patched the
top-level type and missed `float` entirely — it would have silently broken on
MathAPI, TradingBot, TravelAPI, and TicketAPI tasks.

Always use `utils.schema.load_tools_for_classes()` — never the old helper.

### Retry (`utils/retry.py`)

All LLM calls go through `call_with_retry(client, **kwargs)`.
Catches: `APIError`, `APIConnectionError`, `APITimeoutError`, `RateLimitError`,
`httpx.ReadTimeout`, `httpx.ConnectTimeout`, `httpx.RemoteProtocolError`.
Exponential backoff: `2^attempt` seconds, default 5 attempts.

### Trajectory logging (`utils/logging.py`)

Every run writes `logs/{architecture}/{task_id}.jsonl`.  Use `TrajectoryLogger`
as a context manager. Events: `task_start`, `user_turn`, `llm_request`,
`llm_response` (with latency + token counts), `tool_call`, `tool_result`,
`turn_end`, `task_end`. Free-form events via `tlog.event(type, **fields)` for
reflection steps in later architectures.

### FIM endpoint

- Use `/v1` suffix: `https://llms.innkube.fim.uni-passau.de/v1`
- Occasional 5xx blips — always use `call_with_retry`.

### BFCL grading mechanics (confirmed, important)

- Grading is **state-based, not trajectory-based**: the simulator's final
  state after executing the model's calls must match the final state the
  ground-truth trajectory would have produced.
- **Extra read-only calls never cause failure.** `pwd()`, `ls()`,
  `get_account_info()` etc. produce output but change no state — the grader
  sees the same final state whether or not they were called. The model should
  feel free to make orientation calls.
- **Mutating extra calls can cause failure** if they change state in a way the
  GT didn't account for (e.g., buying extra stock before selling).
- Ground-truth trajectories are the *minimal sufficient* call sequence —
  they reflect what a designer would write, not the only valid path.
- `multi_turn_checker` is the entry point; pass `list[list[list[str]]]`.

### State persistence in BFCL executor (confirmed)

`execute_multi_turn_func_call` stores simulator instances in `globals()` keyed
as `{model_name}_{task_id}_{class_name}_instance`. On the first call for a
(model, task, class) triple it creates the instance and loads `initial_config`;
on subsequent calls it reuses the existing instance. **State IS preserved
across calls within the same Python process.**

The `is_evaL_run=True` flag appends `_eval` to the key — the grader uses
separate instances, so live-run state and grading state never contaminate
each other.

`is_evaL_run` has a **capital L** — this is intentional in BFCL source
(`multi_turn_utils.py:31`). Our call-site (`is_evaL_run=False`) is correct.

### REBACT rewind (forward note)

For REBACT Option B (state rewind): since simulator state lives in `globals()`,
rewinding is `del globals()[instance_key]` and re-calling
`execute_multi_turn_func_call` — it rebuilds from `initial_config` cleanly.
No elaborate rollback needed.

## Resolved questions

- ~~**State persistence**~~ → Confirmed from source. See above.
- ~~**`is_evaL_run` spelling**~~ → Capital-L confirmed. Our code is correct.
- ~~**Schema `dict`/`float` types**~~ → Fixed in `utils/schema.py`.

## Open questions / TBD

- **Reflexion adaptation to BFCL**: BFCL grades single trajectories;
  Reflexion is multi-attempt. Plan: report Reflexion@1 (first attempt) and
  Reflexion@k (any attempt within k tries). Gap between them = value of
  reflection mechanism.
- **REBACT irreversibility**: most BFCL tools mutate state. Two options:
  (a) restrict REBACT's "modify" to read-only tools only; (b) rewind by
  deleting the global instance (see above). Lean toward (a) as primary,
  (b) as upper-bound comparison.
- **Vector DB choice**: Chroma vs FAISS. Both work; Chroma has a simpler
  persistent-collection API. Decide when starting Phase 4.

## Evaluation plan

- 20-30 tasks per BFCL category × 4 categories (base, missing_param,
  missing_func, long_context).
- 3 trials per task → report `pass@1` and `pass^3`.
- Same model and same task subset across all architectures.
- Qualitative analysis: 2-3 failure trajectories per architecture, tied to
  the five failure modes (loops, goal drift, error propagation,
  plan/execute conflation, inconsistency).

## File structure

```
├── CLAUDE.md
├── NEXT_STEPS.md               # ordered task list; source of truth for "what next"
├── .env                        # FIM_API_KEY (gitignored)
├── utils/                      # shared helpers — import these, don't reimplement
│   ├── __init__.py
│   ├── retry.py                # call_with_retry() — all LLM calls go through this
│   ├── schema.py               # bfcl_func_to_openai_tool(), load_tools_for_classes()
│   └── logging.py              # TrajectoryLogger context manager
├── logs/                       # trajectory logs — gitignore large runs
│   └── baseline/
│       └── multi_turn_base_1.jsonl
├── gorilla/                    # cloned BFCL repo at tag v1.3
│   └── berkeley-function-call-leaderboard/
│       └── bfcl_eval/
│           ├── data/           # task JSONLs + possible_answer/ + multi_turn_func_doc/
│           └── eval_checker/multi_turn_eval/
│               ├── multi_turn_checker.py   # entry point for grading
│               └── multi_turn_utils.py     # execute_multi_turn_func_call, state mgmt
├── run_one_bfcl_task.py        # FC baseline: one task end-to-end (reference impl)
├── test_api.py                 # smoke test 1: endpoint reachable
└── test_function_calling.py    # smoke test 2: native FC works
```

## Conventions for Claude Code

- All scripts use `python-dotenv` to load `FIM_API_KEY`.
- Always use `temperature=0` for the agent; record any deviation.
- When adding a new architecture, factor the system prompt and loop
  structure into separate functions so the architectures stay comparable.
- Log full trajectories via `TrajectoryLogger` — qualitative analysis needs them.
- Don't install BFCL as a pip package. Use the cloned repo's source by
  adding it to `sys.path` (see `run_one_bfcl_task.py`).
- Shared code belongs in `utils/`. Never copy `call_with_retry` or schema
  helpers into a new file — import from `utils`.
- New architectures live in their own files (e.g., `run_react.py`,
  `run_reflexion_episodic.py`) and import the shared utils.