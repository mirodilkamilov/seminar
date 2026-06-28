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
- **Cost is a first-class axis:** from the logs we report total tokens, peak
  context, LLM-call count, and latency per task, and plot **pass-rate-vs-cost**.
  Reflection isn't free (REBACT ~doubles calls, Reflexion adds retries, vector-DB
  adds retrieved context). We **measure** context growth but **don't** add context
  management (summarization/truncation) — that would be another confound.

### Fairness invariants (the comparison is invalid if any is violated)

- **Identical action channel.** Every architecture emits *actions* through the
  **native FC channel** (structured `tool_calls`); only the added *cognition*
  (Thought, reflection) is free text. Parsing actions from text in one
  architecture and using native FC in another would give them different
  decoding-failure rates that masquerade as reasoning differences.
- **Zero-shot, always.** No few-shot exemplars, and never any borrowed from
  another benchmark (GSM8K, ALFWorld, …) — they'd inject task-specific knowledge
  that confounds "does reasoning help". The only text that differs across
  conditions is the reasoning scaffold; the task-instruction core of the prompt
  is shared.
- **Shared task core (`BASE_TASK_INSTRUCTION`).** All five build their system
  prompt from one identical core, adding only their reasoning scaffold on top
  (enforced in code, not by convention). The core is a *simplified, FC-adapted*
  version of BFCL's intended task instruction — we keep its two behavioural
  cues (flag when no tool fits or a required parameter is missing rather than
  guessing; keep calling tools until done, a no-call reply ends the turn) but
  drop its text-action format directives, since actions go through the native FC
  channel. Including the missing-tool/parameter cue makes the **baseline a fair,
  non-strawman control** — reflection is then credited only with gains *beyond* a
  competent plain prompt, not with recovering from one we deliberately weakened.
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
- **REBACT** uses **state rewind** to revise mutating actions (rebuild the
  simulator from its initial config, replay only the kept calls). **This is not
  faithful to a real deployment** — an agent cannot un-send a committed mutation —
  and we say so. We accept it because the RQ is about the *value of
  self-reflection*, not deployability; a read-only-only variant would neuter
  reflection on exactly the mutating tasks where it matters. (If rewind proves
  valuable, a natural follow-up distills the rewound trajectories into a model
  that runs without it.)

### What each architecture adds beyond cognition

Beyond the reasoning scaffold, two architectures quietly gain something else.
These are **named confounds**, not features of "reflection":

- **Reflexion gets an oracle** — the grader tells it the task failed before it
  retries; the others get one shot, no answer key. (Sanitized as above, but the
  fail/retry bit is itself information they lack.)
- **REBACT gets an undo** — the state rewind (above). It does **not** get the
  oracle, so it has the **same information as ReAct**; the only difference is the
  undo.

So the five pass rates are **not a single ladder** where more reflection = higher
score — that would credit reflection with gains from the oracle or the undo. We
lead Results with the clean within-family contrasts and treat the absolute
five-way ranking as *illustrative*:

| Compare…                                  | …isolates                                                                       | Clean?                                                                                         |
|-------------------------------------------|---------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| Baseline → ReAct                          | pure added thinking, nothing else changes                                       | ✅ fully clean                                                                                  |
| Reflexion@1 → Reflexion@k                 | value of reflecting-after-failure                                               | ✅ clean (same architecture)                                                                    |
| Reflexion-episodic@1 → Reflexion-vectorDB | value of *persistent* cross-task memory                                         | ✅ clean                                                                                        |
| ReAct → REBACT                            | *where* in the loop reflection sits (reason-before-act vs reflect-after-result) | ◐ same info, no oracle; only confound is the undo — we **measure the rewind rate** as its size |
| ReAct vs REBACT vs Reflexion (absolute)   | "which idea wins overall"                                                       | ⚠️ confounded by oracle / undo                                                                 |

The `ReAct → REBACT` pair is the most direct answer to the research question
("*where* in the loop does reflection help"), and its only confound is the undo —
whose magnitude we quantify from the REBACT logs (how often a rewind actually
fires). On read-only-heavy turns nothing gets un-done, so the confound is bounded
by the measured rewind rate rather than assumed.

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
```

## Running

```bash
# .env must contain FIM_API_KEY
python run_benchmark.py                              # baseline, full frozen subset

python run_benchmark.py --make-subset                # freeze task_subset.json and exit
python run_benchmark.py --arch react --category base # one architecture, one category
python run_benchmark.py --arch baseline --sample 5   # quick smoke (random, no frozen subset)
python run_benchmark.py --arch baseline --tag run2   # noise re-run → separate output dir

# Re-run specific task id(s) — category is inferred from the id, and the fresh
# row(s) are MERGED into the existing category file (matching rows replaced,
# all others kept). Handy for debugging or re-measuring a single task.
python run_benchmark.py --task-id multi_turn_base_90
python run_benchmark.py --task-id multi_turn_base_90 multi_turn_miss_param_1
```

Each run writes a full trajectory log to `logs/{architecture}/{task_id}.jsonl`,
a per-task results table to `results/{label}/{category}.jsonl`, and regenerates
the analysis CSVs (`results/results.csv`, `results/summary.csv`) at the end.

