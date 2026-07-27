# Reasoning–Action Loop Architectures on BFCL v3

Comparative evaluation of LLM agent architectures on the **BFCL v3 multi-turn**
tool-calling benchmark.

> **Course:** Agentic AI seminar, University of Passau (WS 2025/2026).
> **Presentation:** 22 May 2026. **Report:** due July 2026.
> **Status:** all runs complete (27 Jul 2026); writing is what remains.

**Research question** *(restated 27 Jul 2026 — see `docs/REVIEW.md` §3.6)*:
*Does the information content of the reflection signal change **how many**
failures reflection recovers, or only **which ones**?*

Secondary axis: *where* in the reason–act loop reflection sits. Two of three
positions are tested (before-act, after-episode); per-step reflection is not —
stated as a scoped limitation, not implied coverage.

**Answer: only which ones.** Across six rungs of increasing signal content the
pass count never leaves a 3-task band, because every arm draws from the same
28-of-115 pool of reachable failures; the signal relocates the model's act/ask
threshold, trading one category's conversions for another's.

Failure analysis (per architecture, tied to five failure modes) is supporting
evidence for interpreting the quantitative results — not a separate goal.

---

## Architectures

*Scope revised 23 Jul 2026 after the Phase-3 audit (`docs/REVIEW.md` §3.2.4);
final 27 Jul.*

| # | Architecture                                                                     | What it adds to the loop                                                                             |
|---|----------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| 1 | **Function-calling baseline** (`baseline`)                                       | nothing — native FC, no reasoning prompt. Run twice (`baseline__run2`) as the measurement-noise band |
| 2 | **ReAct** (`react`)                                                              | an explicit `Thought:` before each action (action still via native FC)                               |
| 3 | **Reflexion (episodic)** (`reflexion`)                                           | on failure, reflect, retry with the reflection in context (scoped to the current task)               |
| 4 | **Reflexion (richer signal)** (`richer_reflexion_turnwise`) — *own contribution* | as (3), plus the model's **own per-turn state timeline** in the reflection call                      |
| — | ~~Reflexion (vector DB)~~                                                        | **dropped** — replaced by (4) as the contribution (§3.2.4 Decision 4)                                |
| — | ~~REBACT~~                                                                       | **dropped** for time; reported as a scoped limitation (§3.2.4 Decision 5)                            |

Plus two **sham-reflection control arms** (`blind_retry`, `blind_retry_lite`)
that are not architectures — they exist to decompose what a retry result is made
of. See *Methodology → control ladder*.

The comparison is only meaningful if everything *except* the reasoning scaffold
is held constant — see **Methodology → fairness invariants**.

Both dropped arms: REBACT is the costlier loss on the secondary axis
(it is the mid-loop position), whereas the vector DB concerned memory
*persistence*, which is orthogonal to the signal-content question
the project ended up answering.

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
  context and LLM-call count per task, and plot **pass-rate-vs-cost**.
  (Latency is logged but *not* comparable across runs — see Limitations.)
  Reflection isn't free: the retry arms cost ≈2.1× baseline input tokens.
  The result cuts against intuition — the **retries** are the entire bill and
  the part that worked, while the reflection calls are ~2% of the arm total and
  the part that didn't. We **measure** context growth but **don't** add context
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
  architecture's mechanism — distinct from the dropped cross-run trials).
  Design, following REVIEW.md §3.2:
  - **Retries are fresh-episode.** The failed conversation never enters the
    retry's context. A separate reflection LLM call — which *does* see the full
    failed trajectory plus the sanitized failure signal — distills a portable
    lesson; the retry opens with a `[user]` preamble (before turn 0, never
    mid-task, never a system-prompt change) carrying the signal + lesson and
    stating that a *previous, separate* attempt failed and the task *restarts
    from the beginning*.
  - **Attempt 1 is seeded from baseline run 1** (verdict, cost and — on
    failure — the trajectory are reused from `results/` + `logs/`), so
    **Reflexion@1 ≡ baseline by identity** and the compute goes to the retries.
  - The failure signal between attempts comes from the grader and is an
    **oracle the baseline never sees** — we state this asymmetry, and we
    **sanitize** it (`utils/sanitize.py`): the grader's raw output leaks the
    ground truth, so every arm sees only *which* environment class mismatched
    or *which* turn went wrong (class + turn; never values, never which call
    was expected) — identical signal in all retry arms.
- **Reflexion (richer signal)** (`richer_reflexion_turnwise`) is identical to
  the episodic arm except that the reflection call also receives the model's
  **own per-turn state timeline** — what each turn changed relative to the turn
  before (`utils/state_dump.py`). It is *not* oracle-derived: the trajectory is
  replayed in a private namespace that cannot alias the grader's `_eval` or
  `_ground_truth_eval` instances (asserted at import), and the turn boundaries
  come from `task["question"]`, i.e. the task's own structure. So
  `richer_reflexion_turnwise − reflexion` needs **no new control**: both arms
  retry, both reflect, the retry lottery is present in both and cancels.

### The control ladder

Retrying is a lottery even with an empty reflection: temperature-0 is not
deterministic on this endpoint (7.8% of baseline failures flipped to pass on an
identical re-run), so the raw @1→@k gap conflates reflection value with retry
luck. `blind_retry` runs the identical loop, preamble template and sanitized
signal, but fills the reflection slot with a fixed, reflection-*shaped* **sham**;
`blind_retry_lite` strips the sham's two imperative advice clauses. The
subtractions separate the ingredients:

> **Reflexion@k − BlindRetry@k** = value of the reflection *content*
> **BlindRetry@k − @1** = retry luck + failure-signal + framing

**Every rung is reported** — they are not alternatives to choose between:

| rung            | arm                         | adds                                         | pass@3  |
|-----------------|-----------------------------|----------------------------------------------|---------|
| `@1`            | —                           | no retry                                     | 85/200  |
| plain re-run    | `baseline__run2`            | retry luck (7.8%/retry)                      | —       |
| minimal advice  | `blind_retry_lite`          | + sanitized signal + reflection-shaped slot  | 102/200 |
| specific advice | `blind_retry`               | + generic corrective imperatives             | 103/200 |
| task-specific   | `reflexion`                 | + a lesson distilled from the actual failure | 100/200 |
| + own state     | `richer_reflexion_turnwise` | + the model's own per-turn state timeline    | 100/200 |

Call these **sham-reflection controls**, not placebos: the sham arms genuinely
beat a plain re-run (12.2% vs 7.8% flip rate), so the slot is not inert even
though its *advice content* is — `blind_retry` − `blind_retry_lite` is +0.5pp
(2 v 1), with three of four categories identical task-for-task. `blind_retry`
is the primary control; the headline is unchanged whichever is used.

`blind_retry_lite` doubles as a **contemporaneous minimal-perturbation noise
band**: two conditions differing by 69 prompt characters disagree on 3/200
tasks. Use it, not the 30 Jun baseline re-run, as the ruler for the retry arms.

**Every `@k` figure carries a caveat:** retries fire only on a graded FAIL, so
`@k ≥ @1` holds by construction and no `@k` number is deployment-realistic — a
real agent does not know it failed.

### What each architecture adds beyond cognition

Beyond the reasoning scaffold, the retry arms quietly gain something else. This
is a **named confound**, not a feature of "reflection":

- **Every retry arm gets an oracle** — the grader tells it the task failed
  before it retries; `baseline` and `react` get one shot, no answer key.
  (Sanitized as above, but the fail/retry bit is itself information they lack.)
  All four retry arms share it identically, so it cancels in every
  within-family contrast and lands, correctly labelled, in `BlindRetry@k − @1`.

So the pass rates are **not a single ladder** where more reflection = higher
score — that would credit reflection with gains from the oracle. We lead Results
with the clean within-family contrasts and treat the absolute ranking as
*illustrative*:

| Compare…                                | …isolates                                                     | Clean?                                                                                   |
|-----------------------------------------|---------------------------------------------------------------|------------------------------------------------------------------------------------------|
| Baseline → ReAct                        | *instructed* vs *spontaneous* explicit reasoning              | ✅ clean as an experiment — but the manipulation is near-saturated for this model (below) |
| BlindRetryLite@k → BlindRetry@k         | value of generic corrective advice                            | ✅ clean — a 69-character prompt difference, constant in both                             |
| BlindRetry@k → Reflexion@k              | value of the reflection *content*                             | ✅ clean — the sham arm differs only in whether the lesson is task-specific               |
| Reflexion@k → RicherReflexionTurnwise@k | value of a *richer, non-oracle* signal in the reflection call | ✅ clean — both retry, both reflect, retry lottery cancels                                |
| Reflexion@1 → BlindRetry@k              | retry luck + failure-signal + framing                         | ✅ clean (its luck floor is the measured 7.8% noise flip rate)                            |
| absolute ranking across all arms        | "which idea wins overall"                                     | ⚠️ confounded by the oracle                                                              |

**The `Baseline → ReAct` row needs its caveat stated, not buried.** The baseline
is *not* a no-thinking control: with zero reasoning scaffold the model already
emits free-text deliberation before **80.2%** of its tool calls. The ReAct prompt
moves that to 83.0% (+3.1pp, CI [−0.3, +6.4]) and adds ~20 chars per action, and
the literal `Thought:` label is picked up on only 21.6% of action-steps. A null
pass-rate difference is therefore the *expected* outcome of a near-saturated
manipulation — and the manipulation check is what lets us say so rather than
guess. Report it as a dose-response measurement (tiny dose, null response), and
name the condition precisely: **zero-shot ReAct-style thought injection over
native function calling**, not "ReAct" (the original used few-shot exemplars and
text-parsed actions, both ruled out by our fairness invariants).

### Limitations

- **Single model** (`qwen3-next-80b-a3b-instruct`): conclusions are
  model-specific. A second model on a small subset was a stretch goal, not run.
- **Two of three loop positions tested.** Per-step reflection (REBACT) was cut
  for time. Since our central finding is that episodic reflection shifts the
  act/ask threshold *globally* — which is why opposite-signed category effects
  cancel — per-step reflection is the most promising untested position: it
  operates at a finer grain and could apply the correction locally, escaping
  exactly the cancellation we observe.
- **The richer signal manipulated salience, not information.** 47.6% of the
  state block is already verbatim in the transcript the reflector reads and the
  rest is derivable from it. A signal genuinely *new* to the agent (an
  environment probe, a validator, an execution trace it could not observe)
  remains untested. This is a lexical measurement and is stated as one.
- **No `@k` number is deployment-realistic** — retries fire on a graded FAIL,
  which a real agent does not have.
- **Arms ran sequentially, not interleaved**, over 30 Jun – 27 Jul. The two
  shams, run on different days, flipped exactly 14/115 each, which bounds
  day-to-day variation; future runs should interleave at task level.
- **Latency is not comparable across runs.** The shared university endpoint's
  load dominates it (ReAct is "faster" than baseline while emitting more
  tokens). Tokens and LLM-call counts are the cost axis.
- BFCL grades a single trajectory per turn against a *minimal sufficient* ground
  truth; a model that "did the right thing" but skipped a state-neutral
  ground-truth call can still fail the response gate (see `CLAUDE.md`).

---

## Stack

- **Model:** `qwen3-next-80b-a3b-instruct` via the FIM API
  (`https://llms.innkube.fim.uni-passau.de/v1`). Key in `.env` as `FIM_API_KEY`
  (gitignored).
- ~~**Embedder** / **Reranker** / **Vector DB**~~ — all three were needed only
  by the dropped vector-DB variant. **No embedding or retrieval stack is used
  anywhere in the final setup.**
- All LLM calls go through `utils.retry.call_with_retry` (the FIM endpoint has
  occasional 5xx blips).

## Repository layout

```
architectures/    one file per architecture, all subclass a shared base
utils/            shared helpers: retry, schema, executor, sanitize,
                  state_dump, conversation, logging, config
run_benchmark.py  runs an architecture over the subset and grades it
results/          per-task grading + cost; {arch}/{category}.jsonl, plus
                  results.csv / summary.csv
gorilla/          cloned BFCL repo at tag v1.3 (do not edit; third-party)
logs/             per-task trajectory JSONL
docs/REVIEW.md    dated critical review + locked design decisions
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

# Retry arms. Attempt 1 is always seeded from results/logs of the `baseline`
# run, so these need a completed baseline on this machine and cost only the
# retries.
python run_benchmark.py --arch reflexion
python run_benchmark.py --arch blind_retry
python run_benchmark.py --arch blind_retry_lite
python run_benchmark.py --arch richer_reflexion_turnwise

# Re-run specific task id(s) — category is inferred from the id, and the fresh
# row(s) are MERGED into the existing category file (matching rows replaced,
# all others kept). Handy for debugging or re-measuring a single task.
python run_benchmark.py --task-id multi_turn_base_90
python run_benchmark.py --task-id multi_turn_base_90 multi_turn_miss_param_1
```

Each run writes a full trajectory log to `logs/{architecture}/{task_id}.jsonl`,
a per-task results table to `results/{label}/{category}.jsonl`, and regenerates
the analysis CSVs (`results/results.csv`, `results/summary.csv`) at the end.

