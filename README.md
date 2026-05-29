# Reasoning–Action Loop Architectures on BFCL v3

Comparative evaluation of five LLM agent architectures on the **BFCL v3
multi-turn** tool-calling benchmark.

> **Course:** Agentic AI seminar, University of Passau (WS 2025/2026).
> **Presentation:** 22 May 2026. **Report:** due July 2026.

**Research question:** *Where in the reason–act loop does adding reflection
actually help on multi-turn tool calling?*

Failure analysis (per architecture, tied to five failure modes) is supporting
evidence for interpreting the quantitative results — not a separate goal.

---

## Architectures

| # | Architecture                                   | What it adds to the loop                                                                                                             |
|---|------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| 1 | **Function-calling baseline**                  | nothing — native FC, no reasoning prompt                                                                                             |
| 2 | **ReAct**                                      | an explicit `Thought:` before each action (action still via native FC)                                                               |
| 3 | **Reflexion (episodic)**                       | on failure, reflect, retry with the reflection in context (scoped to the current task)                                               |
| 4 | **Reflexion (vector DB)** — *own contribution* | reflections embedded (Octen-Embedding-8B) and stored persistently; new tasks retrieve top-k similar past reflections and inject them |
| 5 | **REBACT**                                     | reflect-before-act: after each tool result, a reflect step decides whether to revise the last action                                 |

These form a deliberate progression: no reasoning → reasoning → within-task
reflection → cross-task reflection → per-step reflection. The comparison is only
meaningful if everything *except* the reasoning scaffold is held constant — see
**Methodology → fairness invariants**.

---

## Benchmark

- **BFCL v3 multi-turn**, tag `v1.3` of `github.com/ShishirPatil/gorilla`,
  cloned to `./gorilla/`. Data in
  `gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/`.
- **Four categories** (200 tasks each at this tag): `base`, `miss_param`,
  `miss_func`, `long_context`. **No `composite`** — it was added to BFCL after
  v1.3 and is not present here.
- **Grading is state + response based** (`multi_turn_checker`), not a string
  match against the ground-truth calls. After each turn, two gates must pass:
  (1) the simulator state matches the ground truth, and (2) the ground truth's
  execution outputs are a multiset-subset of the model's. Implications and the
  category-specific mechanics (held-out functions, empty-GT turns) are in
  `CLAUDE.md → BFCL grading mechanics`.

---

## Methodology

### Sampling

- **50 tasks per category × 4 categories = 200 tasks.**
- **Stratified by `involved_classes`** (optionally also by turn count) rather
  than pure random, so each category is a representative spread instead of, say,
  20 file-system tasks. Fixed seed.
- The chosen IDs are **frozen to a committed `task_subset.json`** and reused by
  every architecture, so all five are scored on the *exact* same tasks.

### Metrics

- **One run per task at `temperature=0`. No trials, no `pass^k`.** We report a
  single **pass rate** per (architecture, category).
- **Measurement-noise band:** because the FIM endpoint serves an MoE model with
  request batching, `temperature=0` is not perfectly deterministic — the same
  task can occasionally flip PASS↔FAIL run-to-run. We run the **baseline twice**
  over the frozen subset and report the **task-level flip rate** as a noise band,
  so small differences between architectures aren't mistaken for signal. (This
  is the *only* repeat — it is not a return to `pass^k`.)
- **Paired analysis:** since all architectures share the subset, we compare
  **per task** (McNemar / bootstrap over tasks) and report confidence intervals.
  At n=50 this is what gives the comparison statistical power; unpaired
  proportion tests would be too coarse.
- **Cost is a first-class axis:** from the trajectory logs we report total
  tokens, peak context, number of LLM calls, and latency per task, and present
  **pass-rate-vs-cost**. Reflection isn't free — REBACT roughly doubles the LLM
  calls per action; Reflexion adds retries; the vector-DB variant adds retrieved
  context. We **measure** how each architecture grows the context window; we do
  **not** add context management (summarization/truncation) — that would be
  another confound.

### Fairness invariants (the comparison is invalid if any is violated)

- **Identical action channel.** Every architecture emits *actions* through the
  **native FC channel** (structured `tool_calls`); only the added *cognition*
  (Thought, reflection) is free text. Parsing actions from text in one
  architecture and using native FC in another would give them different
  decoding-failure rates that masquerade as reasoning differences.
- **Zero-shot, always.** No few-shot exemplars, and never exemplars borrowed
  from another benchmark (GSM8K, ALFWorld, …). Modern instruction-tuned models
  follow format instructions without them, and exemplars would inject
  task-specific knowledge that confounds "does reasoning help". The only text
  that differs across conditions is the reasoning/reflection scaffold; the
  task-instruction core of the system prompt is shared.
- **Same model, same subset, same `temperature=0`, same step budget**
  (`MAX_STEPS_PER_TURN = 30`, with every cap-hit logged — the cap only guards
  against runaway loops and should never fire on a legitimate trajectory;
  reflection sub-calls preferably don't count against it).

### Architecture-specific adaptations

- **Reflexion (episodic)** does up to `k=3` *within-task* attempts (this is the
  architecture's mechanism — distinct from the dropped cross-run trials). We
  report **Reflexion@1** (= baseline, by definition) and **Reflexion@k**; the
  gap is the value of the reflection mechanism. The failure signal between
  attempts comes from the grader and is an **oracle the baseline never sees** —
  we state this asymmetry, and we **sanitize** the feedback: the grader's error
  message leaks the ground-truth value (e.g. an expected address), so we feed
  back only *which* instance/attribute mismatched, never the expected value.
- **Reflexion (vector DB)** stays a **single attempt per task** plus retrieved
  cross-task reflections, compared against episodic Reflexion@1 to isolate the
  value of *persistent* memory. Task order is fixed and we plot the learning
  curve (pass rate vs. position as memory fills).
- **REBACT** uses **state rewind** to revise mutating actions, implemented by
  rebuilding the simulator from its initial config and replaying only the kept
  calls. **This is not faithful to a real deployment** — a deployed agent cannot
  un-send a committed mutation — and we say so explicitly. We accept it because
  the RQ is about the *value of self-reflection* independent of deployment
  constraints; the read-only-only alternative would neuter reflection on exactly
  the tasks where it matters. (Framing: if reflection-with-rewind proves
  valuable, a follow-up is to distill the successful rewound trajectories into a
  model that operates without rewind in production.)

### Limitations

- **Single model** (`qwen3-next-80b-a3b-instruct`): conclusions are
  model-specific — "where reflection helps *this* model". A second model on a
  small subset is a stretch goal to check the ordering is robust.
- BFCL grades a single trajectory per turn against a *minimal sufficient* ground
  truth; a model that "did the right thing" but skipped a state-neutral
  ground-truth call can still fail the response gate (see `CLAUDE.md`).

---

## Stack

- **Model:** `qwen3-next-80b-a3b-instruct` via the FIM API
  (`https://llms.innkube.fim.uni-passau.de/v1`). Key in `.env` as `FIM_API_KEY`
  (gitignored).
- **Embedder:** `octen-embedding-8b` (FIM). **Reranker:** `qwen3-reranker-4b`
  (FIM, optional).
- **Vector DB:** Chroma or FAISS (local, persistent) — decided at Phase 4.
- All LLM calls go through `utils.retry.call_with_retry` (the FIM endpoint has
  occasional 5xx blips).

## Repository layout

```
architectures/   one file per architecture, all subclass a shared base
utils/            shared helpers: retry, schema, executor, logging, config
run_benchmark.py  runs an architecture over the subset and grades it
gorilla/          cloned BFCL repo at tag v1.3 (do not edit; third-party)
logs/             per-task trajectory JSONL
CLAUDE.md         implementation details + conventions
NEXT_STEPS.md     ordered task list ("what to work on next")
REVIEW.md         design-review findings / correctness fixes to apply
```

## Running

```bash
# .env must contain FIM_API_KEY
python run_benchmark.py
```

Each run writes a full trajectory log to `logs/{architecture}/{task_id}.jsonl`.

