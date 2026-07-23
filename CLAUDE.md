# Seminar Project: Reasoning-Action Loop Architectures on BFCL v3

## TL;DR

Comparative evaluation of LLM agent architectures on the BFCL v3 multi-turn
tool-calling benchmark. Course: Agentic AI seminar, University of Passau
(WS 2025/2026). Presentation delivered 22 May 2026; report due July 2026.

Research question: **Where in the reason-act loop does adding reflection
actually help on multi-turn tool calling?**

**Final scope (revised 23 Jul 2026):** four architectures — `baseline`,
`react`, `reflexion`, `richer_reflexion` (the contribution) — plus two control
arms (`blind_retry`, `blind_retry_lite`) and a baseline noise re-run. The
vector-DB Reflexion and REBACT are dropped and reported as limitations; see
REVIEW.md §3.2.4. Headline result so far: **the entire +7.5pp Reflexion gain is
retry luck; reflection content is worth −1.5pp (CI [−5.5, +2.0])** — but it is
not a flat null, because reflection reliably shifts the model's *act/ask
threshold*, helping where failures are under-acting (miss_func +8pp) and
hurting where they over-hesitate (miss_param −10pp).

Failure analysis (per architecture, tied to the five failure modes) is
used as supporting evidence to interpret the quantitative results, not
as a separate research goal.

## Architectures under comparison

*Scope revised 23 Jul 2026 after the Phase-3 audit — see REVIEW.md §3.2.4.*

1. **Function calling baseline** (`baseline`) — model's native FC, no reasoning
   prompts. Also run twice (`baseline__run2`) as the measurement-noise band.
2. **ReAct** (`react`) — text-format Thought/Action/Observation prompting;
   actions still go through the native FC channel.
3. **Reflexion (episodic)** (`reflexion`) — original: on failure, reflect
   verbally, retry with the reflection in context. Reflection is scoped to the
   current task only.
4. **Reflexion (richer signal)** (`richer_reflexion`) — **own contribution**.
   Identical to (3) except the reflection call additionally sees **the model's
   own final environment state** (public attrs of its own simulator instances),
   not just the class+turn signal. Leak-free by construction: it is the agent's
   own product, never the ground-truth instances. Motivated by the Phase-3
   finding that episodic reflections are *information-starved* — they guess at
   diagnoses — rather than useless.
5. ~~**Reflexion (vector DB)**~~ — **dropped** (REVIEW.md §3.2.4 Decision 4).
   Replaced by (4) as the contribution.
6. ~~**REBACT**~~ — **dropped** for time (§3.2.4 Decision 5); reported as a
   scoped limitation with the forward-looking framing given there. This is the
   costlier cut on RQ grounds — REBACT is the *mid-loop* position, whereas the
   vector DB was about memory persistence, orthogonal to "where in the loop".

### Control arms (not architectures — they isolate what a result is made of)

Retrying is a lottery at temperature 0 (7.8% of failures flip on a pure
re-roll), so the `@1 → @k` gap conflates reflection value with retry luck. The
controls decompose it into a **ladder**, and **every rung is reported** — they
are not alternatives to choose between (§3.2.4 Decision 1):

| rung            | arm                | adds                                         |
|-----------------|--------------------|----------------------------------------------|
| `@1`            | —                  | no retry                                     |
| plain re-run    | `baseline__run2`   | retry luck                                   |
| minimal advice  | `blind_retry_lite` | + sanitized signal + reflection-shaped slot  |
| specific advice | `blind_retry`      | + generic corrective imperatives             |
| task-specific   | `reflexion`        | + a lesson distilled from the actual failure |

`blind_retry` is the **primary** control (only it and `reflexion` have the full
200-task history); `blind_retry_lite` bounds how much of its performance came
from the advice content rather than the retry. Both are *active* controls, not
placebos — the sham contains real generic advice — so never call them placebos
in the report.

### Cross-architecture fairness invariants (must hold for every arm)

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
- **Same model, same fixed task subset, same `temperature=0`** across every arm.

## Stack

- **Model**: `qwen3-next-80b-a3b-instruct` via FIM API
  (`https://llms.innkube.fim.uni-passau.de/v1`)
- **API key**: in `.env` as `FIM_API_KEY` (gitignored)
- ~~**Embedder**: `octen-embedding-8b`~~ / ~~**Reranker**: `qwen3-reranker-4b`~~
  / ~~**Vector DB**~~ — all three were only needed by the vector-DB variant,
  **dropped** 23 Jul 2026 (REVIEW.md §3.2.4 Decision 4). No embedding or
  retrieval stack is used anywhere in the final setup.
- **Benchmark**: BFCL v3 (tag `v1.3` of github.com/ShishirPatil/gorilla,
  cloned to `./gorilla/`). Multi-turn data lives in
  `gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/`.

## Important implementation details

### Schema conversion (`utils/schema.py`)

BFCL function docs use two type names that are **not valid JSON Schema**:

- `"dict"` → must become `"object"` (top-level AND in nested properties)
- `"float"` → must become `"number"` (42 occurrences across all 8 doc files)

Both are fixed recursively in `utils/schema.py::_normalize_schema()`.

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

> **NOT IMPLEMENTED — REBACT was dropped 23 Jul 2026** (REVIEW.md §3.2.4
> Decision 5) for time. This section is kept as the design record: it is the
> spec to build from if the work is ever continued, and the rewind/record
> coupling below is the reason the build was too large to fit the deadline.

We go with state rewind as REBACT's primary mechanism, not
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
- **`max_tokens = 8192`, identical across every arm.** Part of the fixed setup, so
  a shared token budget can't penalize the talkative (thinking) architectures
  whose reply carries a Thought/reflection *plus* the tool call — a truncated
  reply can cut the tool-call JSON and fail for a formatting reason. Set generous
  (a ceiling, not a target: you pay only for tokens emitted) so it never fires;
  `stats["truncations"]`/`["parse_errors"]` count when it does — verify ~0 per
  architecture, bump higher if not.

### Reflexion adaptation to BFCL (within-task retries — distinct from trials)

Implemented (`architectures/reflexion.py` + `run_benchmark.py::run_one_multi`);
design per REVIEW.md §3.2. Key mechanics:

- **Episodic, `k=3`, fresh-episode retries.** The failed conversation never
  enters the retry's context. A separate reflection call sees the failed
  trajectory + sanitized signal and writes `What I tried / What went wrong /
  Lesson`; the retry is a new conversation opening with a `[user]` preamble
  **before turn 0** (never mid-task, never a system-prompt change) that says a
  *previous, separate* attempt failed and the task *restarts from the
  beginning*. System prompt = baseline's on every attempt.
- **Blind-retry control arms** (`--arch blind_retry`, `--arch
  blind_retry_lite`): temp-0 is not deterministic here (7.8% of failures flip
  on identical re-run), so @1→@k alone conflates reflection value with retry
  luck. Identical loop, preamble template and sanitized signal; the reflection
  slot holds a fixed **sham** reflection-shaped text, no reflection LLM call.
  Reflexion@k − BlindRetry@k = reflection content; BlindRetry@k − @1 = the
  retry+framing term. **They are *active* controls, not placebos** — the sham
  carries real generic advice, so the subtraction estimates "task-specific
  reflection vs generic advice". `blind_retry_lite` strips the advice clauses to
  bound that; report **both rungs**, never one selected on outcome.
- **`k=2` is the primary contrast, `k=3` secondary and caveated** (REVIEW.md
  §3.2.4 Decision 3). At attempt 3 `_retry_preamble` gives the blind arms the
  *verbatim identical* sham twice while Reflexion gets two distinct lessons, so
  the arms differ in novelty and quantity there, not only content.
- **Attempt 1 seeded from baseline run 1** (always — hardcoded `SEED_LABEL`
  in `run_benchmark.py`, deliberately not configurable):
  verdict/stats from `results/baseline/`, trajectory rebuilt from
  `logs/baseline/*.jsonl` (`utils/conversation.py` — baseline run-1 logs predate
  the `system_prompt` field, so the caller passes the known prompt as a
  fallback). Reflexion@1 ≡ baseline by identity; both arms retry the same
  failure set.
- **Failure signal is an oracle and MUST be sanitized**
  (`utils/sanitize.py`). The leak is in the checker's `details` dict (e.g.
  `ground_truth_instance_state`), which `grade()` already discards; the
  sanitizer maps `error_type`/`error_message` to class + turn only (state gate
  carries no turn index → class-level there), never values. Both arms get the
  identical signal; state the baseline-never-sees-it asymmetry in the report.
- **Reset per ATTEMPT, not per task** — `reset_bfcl_instances()` before every
  live attempt: the grader caches `_eval`/ground-truth simulator instances in
  module globals, so a second in-process `grade()` would otherwise resume
  stale state.
- **Cost accounting**: per-task totals sum all attempts *plus* reflection calls
  (`n_reflections` counted; `peak_context` is a max across attempts, not a
  sum). Results rows add `passed_at_1`, `n_attempts`, `attempts[]`.
- ~~**Vector-DB variant (the contribution)**~~ — dropped 23 Jul 2026
  (REVIEW.md §3.2.4 Decision 4). Replaced by the richer-signal arm below.

### Richer-signal Reflexion (`richer_reflexion`) — the contribution

Same architecture as episodic Reflexion in every respect (k=3, fresh-episode,
seeded attempt 1, same preamble, same `REFLECTION_PROMPT`) **except** that the
reflection call also receives the model's **own final environment state**.

Why: Phase 3 found the reflections are *information-starved*, not useless —
with a class+turn signal the model guesses at diagnoses, and a guessed
diagnosis is behaviourally sham-like. The state dump is **not oracle-derived**,
so it sidesteps the leak constraint entirely: it is the agent's own product.

`richer_reflexion − reflexion` is a clean contrast needing **no new control** —
both arms retry, both reflect, they differ only in signal richness, so the
retry lottery is present in both and cancels.

Two hazards, both confirmed against BFCL source:

1. **The leak is one string-match away.** After `grade()`, `mtu.__dict__` holds
   *both* state sets: model replay as
   `{model}_eval_{task_id}_{class}_instance` (`multi_turn_checker.py:41`) and
   **ground truth** as `{model}_ground_truth_eval_{task_id}_{class}_instance`
   (`:58`). Filter on `"_ground_truth" not in key` and **assert it** — getting
   this wrong pipes the answer key into the prompt and would look like a
   spectacular result. Mirror `state_checker`'s filter (public attrs via
   `vars()`), and cap serialized length or `long_context` will blow up the
   prompt.
2. **Seeded attempt 1 has no `_eval` instances** — `run_one_multi` reads the
   seed row and skips `grade()`, so there is no state to dump for the very
   failure most worth reflecting on. Fix: add `log_to_calls(log_path)` beside
   `log_to_messages` (same event-walk over `tool_call` events, grouped
   turn → step), rebuild the baseline call list, and run `grade()` on the
   seeded attempt too. Zero LLM cost, and it yields a free integrity check:
   assert the replayed verdict equals the seed row's `passed`.

## Open questions / TBD

*(none open — the vector-DB choice was the only entry and the variant is
dropped.)*

## Evaluation plan

The quantitative design is in "Methodology decisions (locked)" above
(200-task stratified subset, single run + noise re-run, paired analysis,
cost axis). For the retry arms specifically: **report the control ladder, k=2
as the primary contrast**, and state next to every `@k` figure that retries
fire on a graded FAIL — so `@k ≥ @1` holds by construction and no `@k` number
is deployment-realistic.

Qualitative analysis:

- 2-3 failure trajectories per architecture, tied to the five failure modes
  (loops, goal drift, error propagation, plan/execute conflation,
  inconsistency). Taxonomy from the ReAct/Reflect overview
  (emergentmind.com); **define each mode operationally** so tagging is
  reproducible — auto-detect what you can from the logs (e.g. "loop" =
  identical call string ≥3× in a turn; "error propagation" = a tool result
  containing `"error"` followed by an action that ignores it) and hand-code the
  rest against a written rubric.
- **Hand-code ~30 reflection lessons** (correct / partial / wrong diagnosis)
  against ground truth. Originally planned for the vector arm; it survives that
  cut and now applies to the episodic arm, where it is the evidence for the
  null's scope statement (*reflection over a **minimal** signal adds nothing* —
  not "reflection is worthless") and the motivation for the richer-signal arm.

## File structure

```
├── CLAUDE.md
├── README.md                   # human-facing overview + locked methodology + rationale
├── NEXT_STEPS.md               # ordered task list; source of truth for "what next"
├── .env                        # FIM_API_KEY (gitignored)
├── architectures/              # one file per architecture; all share utils/
│   ├── __init__.py
│   ├── architecture.py         # base class + shared FC loop (run_task)
│   ├── baseline.py             # FC baseline (name + system_prompt)
│   ├── react.py                # ReAct (name + system_prompt)
│   └── reflexion.py            # ReflexionEpisodic + BlindRetry + BlindRetryLite
├── utils/                      # shared helpers — import these, don't reimplement
│   ├── __init__.py
│   ├── config.py               # MODEL, client, CATEGORIES, task_paths(), sys.path setup
│   ├── retry.py                # call_with_retry() — all LLM calls go through this
│   ├── schema.py               # bfcl_func_to_openai_tool(), load_tools_for_classes()
│   ├── executor.py             # execute_call_locally(), reset_bfcl_instances()
│   ├── sampling.py             # stratified subset → task_subset.json
│   ├── sanitize.py             # sanitize_failure_signal() — the oracle-leak guard
│   ├── conversation.py         # log_to_messages(), messages_to_text() — reflection input
│   └── logging.py              # TrajectoryLogger + pretty_print_log()
├── analysis/                   # one-off probes; each prints its own numbers
├── docs/
│   └── REVIEW.md               # dated critical review + locked design decisions
├── logs/                       # trajectory logs — gitignore large runs
│   └── {baseline,baseline__run2,react,reflexion,blind_retry,...}/*.jsonl
├── results/                    # one-row-per-task grading + cost; {arch}/{category}.jsonl
│                               # + results.csv / summary.csv (results_to_csv.py)
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
- New architectures live in `architectures/<name>.py` and subclass the base
  class `Architecture` in `architectures/architecture.py`; give each an explicit
  `name` used for log/results dirs (don't rely on `__class__.__name__`). The
  shared loop is decomposed into override points (`_run_turn`, `_call_model`,
  `_execute_calls`, …) — hook those rather than copying `_run_fc_loop`.
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