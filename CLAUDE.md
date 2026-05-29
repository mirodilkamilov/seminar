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

### Cross-architecture fairness invariants (must hold for all five)

These are what make the comparison valid — violating any one confounds the
result. See README.md "Methodology" for the rationale.

- **Action channel is identical.** Every architecture emits *actions* through
  the **native FC channel** (structured `tool_calls`). Only the *added
  cognition* (Thought, reflection) is free text. Never parse actions out of
  text in one architecture and use native FC in another — that introduces
  differential decoding-failure rates that masquerade as reasoning quality.
- **Zero-shot, always.** No few-shot exemplars (and never exemplars borrowed
  from another benchmark). The only text that differs between conditions is the
  reasoning/reflection scaffold; the task-instruction core of the system prompt
  is held constant.
- **Same model, same fixed task subset, same `temperature=0`** across all five.

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

## Current state (as of 2026-05-29)

- **Phases 0, 0.5, and 1-scaffolding complete** (see `NEXT_STEPS.md`). All
  correctness fixes from the design review (`REVIEW.md`) are applied and each
  verified on a live task in its category: per-task instance reset, irrelevance
  gate ANDed into grading, `miss_func` held-out functions, `long_context` live
  flag, `error_type` capture, `finish_reason=="length"` handling, no-retry-on-4xx.
- **Methodology decisions locked** (see "Methodology decisions (locked)" below
  and `README.md` for rationale).
- **Harness refactored for comparability**: shared FC loop in
  `architectures/architecture.py`; `baseline.py` / `react.py` only set `name` +
  `system_prompt`. Runner has a registry + argparse CLI; results table at
  `results/{arch}/{category}.jsonl` with cost fields.
- **Subset frozen** to `task_subset.json` (seed 42, 50/category, stratified by
  `involved_classes`). **ReAct is implemented** alongside the baseline.
- Smoke tests pass (`test_api.py`, `test_function_calling.py`); a 6-task baseline
  pilot is in `logs/baseline/` (`results/` not yet created).
- **Next (immediate):** run the FC baseline over the full 200-task subset — the
  first real result — then the baseline noise re-run, then ReAct over the subset.
  No production run has happened yet; Reflexion (episodic + vector DB) and REBACT
  are unstarted.

## Important implementation details

### Schema conversion (`utils/schema.py`)

BFCL function docs use two type names that are **not valid JSON Schema**:

- `"dict"` → must become `"object"` (top-level AND in nested properties)
- `"float"` → must become `"number"` (42 occurrences across all 8 doc files)

Both are fixed recursively in `utils/schema.py::_normalize_schema()`. An earlier
helper only patched the top-level type and missed `float` entirely — it would
have silently broken on MathAPI, TradingBot, TravelAPI, and TicketAPI tasks.

Always use `utils.schema.load_tools_for_classes()` — never the old helper.

### Retry (`utils/retry.py`)

All LLM calls go through `call_with_retry(client, **kwargs)`.
Catches: `APIError`, `APIConnectionError`, `APITimeoutError`, `RateLimitError`,
`httpx.ReadTimeout`, `httpx.ConnectTimeout`, `httpx.RemoteProtocolError`.
Exponential backoff: `2^attempt` seconds, default 5 attempts.

### Trajectory logging (`utils/logging.py`)

Every run writes `logs/{architecture}/{task_id}.jsonl`. Use `TrajectoryLogger`
as a context manager. Events: `task_start`, `user_turn`, `llm_request`,
`llm_response` (with latency + token counts), `tool_call`, `tool_result`,
`turn_end`, `task_end`. Free-form events via `tlog.event(type, **fields)` for
reflection steps in later architectures.

### FIM endpoint

- Use `/v1` suffix: `https://llms.innkube.fim.uni-passau.de/v1`
- Occasional 5xx blips — always use `call_with_retry`.

### BFCL grading mechanics (confirmed from source, important)

`multi_turn_checker` runs **two gates per turn** — both must pass. The earlier
"grading is purely state-based" note was wrong and caused a confusing FAIL
(`multi_turn_base_28`). The real picture:

1. **`state_checker`** — after replaying the turn's calls on fresh `_eval`
   instances, every involved simulator's public attributes must equal the
   ground-truth instance's. This is the state-based part.
2. **`response_checker`** — the GT turn's **execution outputs** must be a
   *multiset subset* of the model's accumulated outputs across all turns so far
   (`_is_subsequence_unordered`).

Consequences (state the relevant ones in the report):

- **Extra read-only calls are free.** The model can have *more* outputs than GT
  (subset only requires GT ⊆ model). Orientation calls (`pwd`, `ls`, …) never
  hurt — neither gate penalizes them.
- **Missing a GT output call can fail you even when final state matches.** If GT
  calls a state-neutral function that returns a value (e.g. a redundant
  `touch()` returning `None`) and the model skips it, `response_checker` fails
  on the missing output. This is `base_28`: state matched, but the model lacked
  one `None`. Faithful BFCL behavior — not a harness bug. Expect a small class
  of "the model did the right thing but skipped a GT call" failures.
- **Mutating extra calls can fail** if they leave state GT didn't produce.
- GT trajectories are the *minimal sufficient* sequence, not the only valid path.
- `multi_turn_checker` is the entry point; pass `list[list[list[str]]]`
  (turn → step → call strings).

**Empty-GT turns + irrelevance check (DONE — `run_benchmark.py::grade`).**
In `miss_param` and `miss_func`, some turns have an **empty GT list** — the
correct behavior is to make *no* call (ask the user / recognize a held-out
function). `multi_turn_checker` only `continue`s on these turns; it does **not**
penalize a model that wrongly calls. The "should have stayed silent" check lives
in a **separate** function, `multi_turn_irrelevance_checker`. `grade()` ANDs the
two; it's a no-op for `base`/`long_context`, so it's applied everywhere. Without
it those two categories' scores inflate.

### `miss_func` held-out functions (DONE — `architecture.py::_run_fc_loop`)

`miss_func` entries carry a `missed_function` field, e.g. `{"2": ["mv"]}`: `mv`
is **held out of the toolset until turn 2**. The shared loop replicates BFCL's
own pipeline (`_llm_response_generation.py`, `base_handler.py`):

1. At load, **remove** every `missed_function` name from the tools given to the
   model (so it does *not* have them on early turns).
2. The held-out turn has an **empty user message**; at that turn the loop
   **re-adds** the held-out tool(s) and **injects the synthetic user prompt**
   `DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC` ("I have updated some more
   functions you can choose from. What about now?").
3. GT typically has an empty turn just before, where the model should recognize
   the function is missing and not hallucinate (caught by the irrelevance check).

Verified live: tool count goes 17→18 exactly at the holdout turn. (Before this,
`load_tools_for_classes` handed the model every function from turn 0, so the
category measured nothing.)

### `long_context` flag threaded to live execution (DONE)

`execute_call_locally(..., long_context)` now takes the flag; the loop derives
`long_context = "long_context" in task["id"]`, matching what the grader rebuilds
(`"long_context" in test_category`). Otherwise the model would observe a
different simulator state at runtime than the grader.

### Reset BFCL simulator instances between runs (DONE — `executor.py`)

`execute_multi_turn_func_call` caches instances in the *module globals* of
`multi_turn_utils`, keyed `{model}_{task_id}_{class}_instance` — **no run/trial
number, never cleared**. Distinct task IDs don't collide, but **re-running the
same task id in one process reuses the prior end-state** (verified empirically) —
which would corrupt the baseline noise re-run. `reset_bfcl_instances()` clears
them and is called at the start of every task in `run_one` (cheap; also bounds
memory):

```python
import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils as mtu


def reset_bfcl_instances():
    for k in [k for k in mtu.__dict__ if k.endswith("_instance")]:
        del mtu.__dict__[k]
```

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

### REBACT rewind — DECISION: rewind-with-record-editing (primary)

We go with state rewind (the old "Option B") as REBACT's primary mechanism, not
the read-only restriction. Rationale: the RQ is about the *value of
self-reflection*, and the read-only restriction would neuter reflection on
exactly the mutating tasks where revision matters. We accept that rewind is
**not faithful to a real deployment** (a real agent cannot un-send a committed
mutation) and state this as an explicit limitation. Framing for the report:
if reflection-with-rewind proves valuable, a natural follow-up is to distill the
successful rewound trajectories into a model that operates *without* rewind in
production — i.e. rewind is a training-signal generator, not a deployment claim.

**Critical implementation coupling:** the grader replays our *recorded* call
list (`all_turns_calls`), not the live execution. So a rewind must edit **both**:

1. the live simulator — `del mtu.__dict__[instance_key]`, then re-run the kept
   calls via `execute_multi_turn_func_call` (rebuilds from `initial_config`);
2. the **recorded call list** — drop the retracted calls so what gets graded is
   exactly the corrected trajectory. If the recorded list still contains the
   retracted mutating call, the grader replays it and the state check fails for
   a trajectory the agent didn't actually end in. Live state and recorded list
   must stay in lockstep at all times.

## Resolved questions

- ~~**State persistence**~~ → Confirmed from source. See above.
- ~~**`is_evaL_run` spelling**~~ → Capital-L confirmed. Our code is correct.
- ~~**Schema `dict`/`float` types**~~ → Fixed in `utils/schema.py`.

## Methodology decisions (locked — see README.md for rationale)

- **One run per task, `temperature=0`. No trials, no `pass^k`.** Report a single
  pass *rate* per (architecture, category). NB: this concerns repeated *trials*
  of the whole benchmark — it is **distinct** from Reflexion's within-task
  retries (`@k`, below), which stay.
- **Measurement-noise re-run**: run the **baseline twice** over the frozen
  subset (resetting instances between passes), report the task-level flip rate
  as a noise band. This is the only repeat we do — not pass^k.
- **Sampling**: **50 tasks per category × 4 categories = 200 tasks**
  (`base`, `miss_param`, `miss_func`, `long_context`). **No `composite`** —
  BFCL scaffolds a Composite category (README description, grader support in
  `multi_turn_checker.py`, and a **commented-out** `category_mapping.py` entry)
  but **never shipped the data file**. Confirmed disabled-in-code — not just a
  missing file — at our pinned tag `v1.3` and on current upstream `main`; no
  GitHub issue needed (it's by design). Only these four multi-turn files exist
  (each has **200 tasks**, ids 0–199; note `wc -l` reports 199 because the last
  line lacks a trailing newline — count `"id":` occurrences instead).
  Sample **stratified by `involved_classes`** (optionally also turn count), with
  a fixed seed, and **freeze the chosen IDs to a committed `task_subset.json`**
  shared by every architecture.
- **Analysis is paired**: same subset across architectures → compare per-task
  with **McNemar / bootstrap over tasks**, report CIs. (At n=50 unpaired
  proportions are too coarse; paired is what gives the comparison power.)
- **Cost is a reported axis**: from the trajectory logs, report total tokens,
  peak context, # LLM calls, and latency per task — pass-rate-vs-cost. We
  **measure** context growth; we do **not** add context management
  (summarization/truncation) — that would be another confound.
- **Single model** (`qwen3-next-80b-a3b-instruct`) → results are model-specific;
  state external validity as a limitation.
- **`MAX_STEPS_PER_TURN = 30`, and log every time the cap is hit.** The cap only
  guards against runaway loops; it must never fire on a legitimate trajectory.
  Preferably do not count reflection sub-calls (REBACT/Reflexion) against the
  budget — it caps *actions*, not *thoughts*. After the pilot, check the
  hit-rate is ~0 for every architecture.

### Reflexion adaptation to BFCL (within-task retries — distinct from trials)

- **Episodic**: up to `k=3` sequential attempts per task; on a failed attempt,
  reflect, prepend the reflection, retry. Report **Reflexion@1** (= the baseline,
  definitionally — make attempt-1 prompt identical to baseline) and
  **Reflexion@k**. The @1→@k gap = value of the reflection mechanism.
- **Failure signal is an oracle and MUST be sanitized.** Reflexion retries
  because it is told it failed — that signal is the BFCL grader. Two
  non-negotiables: (1) state the asymmetry (baseline never sees this signal);
  (2) **the grader's error message leaks the ground truth** (e.g. `base_70`
  prints `ground_truth: "456 Oakwood Avenue…"`). Feed back only *which*
  instance/attribute mismatched (or a binary fail) — **never the expected
  value** — or Reflexion's result is invalid.
- **Vector-DB variant (the contribution)**: keep it a **single attempt per
  task** plus top-k reflections retrieved from *past* tasks. Compare against
  episodic **Reflexion@1** (single attempt, no memory) to isolate the value of
  persistent cross-task memory. Fix and document task order; report the
  learning curve (pass rate vs. position in the sequence as memory fills).

## Open questions / TBD

- **Vector DB choice**: Chroma vs FAISS. Both work; Chroma has a simpler
  persistent-collection API. Decide when starting Phase 4.

## Evaluation plan

The quantitative design is in "Methodology decisions (locked)" above
(200-task stratified subset, single run + noise re-run, paired analysis,
cost axis). Qualitative analysis:

- 2-3 failure trajectories per architecture, tied to the five failure modes
  (loops, goal drift, error propagation, plan/execute conflation,
  inconsistency). Taxonomy from the ReAct/Reflect overview
  (emergentmind.com); **define each mode operationally** so tagging is
  reproducible — auto-detect what you can from the logs (e.g. "loop" =
  identical call string ≥3× in a turn; "error propagation" = a tool result
  containing `"error"` followed by an action that ignores it) and hand-code the
  rest against a written rubric.

## File structure

```
├── CLAUDE.md
├── README.md                   # human-facing overview + locked methodology + rationale
├── NEXT_STEPS.md               # ordered task list; source of truth for "what next"
├── REVIEW.md                   # design review findings (correctness fixes to apply)
├── .env                        # FIM_API_KEY (gitignored)
├── architectures/              # one file per architecture; all share utils/
│   ├── __init__.py
│   ├── architecture.py         # base class + shared FC loop (run_task)
│   ├── baseline.py             # FC baseline (name + system_prompt)
│   └── react.py                # ReAct (name + system_prompt)
├── utils/                      # shared helpers — import these, don't reimplement
│   ├── __init__.py
│   ├── retry.py                # call_with_retry() — all LLM calls go through this
│   ├── schema.py               # bfcl_func_to_openai_tool(), load_tools_for_classes()
│   ├── executor.py             # execute_call_locally(), reset_bfcl_instances()
│   ├── sampling.py             # stratified subset → task_subset.json
│   └── logging.py              # TrajectoryLogger + pretty_print_log()
├── logs/                       # trajectory logs — gitignore large runs
│   └── baseline/
│       └── *.jsonl
├── results/                    # one-row-per-task grading + cost; {arch}/{category}.jsonl
├── task_subset.json            # frozen 50/category stratified subset (committed)
├── gorilla/                    # cloned BFCL repo at tag v1.3
│   └── berkeley-function-call-leaderboard/
│       └── bfcl_eval/
│           ├── data/           # task JSONLs + possible_answer/ + multi_turn_func_doc/
│           └── eval_checker/multi_turn_eval/
│               ├── multi_turn_checker.py   # entry point for grading
│               └── multi_turn_utils.py     # execute_multi_turn_func_call, state mgmt
├── run_benchmark.py            # runs an architecture over the subset + grades
├── test_api.py                 # smoke test 1: endpoint reachable
└── test_function_calling.py    # smoke test 2: native FC works
```

## Conventions for Claude Code

- All scripts use `python-dotenv` to load `FIM_API_KEY`.
- Always use `temperature=0` for the agent; record any deviation.
- **Honor the fairness invariants** (see "Architectures under comparison"):
  actions via native FC only, zero-shot, shared system-prompt core. The only
  thing a new architecture adds is reasoning/reflection cognition + loop shape.
- New architectures live in `architectures/<name>.py` and subclass the base in
  `architectures/function_calling_abstract.py`; give each an explicit `name`
  used for log/results dirs (don't rely on `__class__.__name__`).
- Keep orchestration (printing, grading, logging, instance reset) in the runner
  / utils, **not** inside `run_task`, so architectures stay minimal and
  comparable.
- Log full trajectories via `TrajectoryLogger`; capture `error_type` (not just
  `error_message`) and the failing turn — the failure-mode analysis needs them.
  Use `tlog.event("reflection", ...)` with a fixed schema for reflection steps.
- Don't install BFCL as a pip package. Use the cloned repo's source by adding it
  to `sys.path` (see `utils/config.py`).
- Shared code belongs in `utils/`. Never copy `call_with_retry`, schema, or
  executor helpers into a new file — import from `utils`.
- Always grade with `multi_turn_checker(...) AND multi_turn_irrelevance_checker(...)`,
  and `reset_bfcl_instances()` at the start of each task.