# Next Steps

Ordered by priority. Cross items off as completed. Keep this in sync with
reality — it's the source of truth for "what should I work on right now?"
Methodology is locked in `README.md`.

## Phase 0: De-risk the pipeline — DONE

- [x] **Verify state persistence in BFCL executor.** Instances cached in the
  `multi_turn_utils` module globals, keyed `{model}_{task_id}_{class}_instance`;
  reused across calls within a process. `is_evaL_run=True` appends `_eval` so
  grading uses separate instances. (Caveat → Phase 0.5: keys have no run/trial
  number and are never cleared.)
- [x] **Verify `is_evaL_run`** — capital-L confirmed; our call-site is correct.
- [x] **Broaden retry exceptions** — `utils/retry.py`.
- [x] **Add full trajectory logging** — `utils/logging.py` (`TrajectoryLogger`).
- [x] **Fix schema conversion** — recursive `dict→object`, `float→number` in
  `utils/schema.py::_normalize_schema()`.
- [x] **Run baseline end-to-end** — `run_benchmark.py` grades a 5-task sample.

## Phase 1: Baseline + harness scaffolding

- [x] **Freeze the task subset.** `python run_benchmark.py --make-subset` →
  `task_subset.json` (50/category, stratified by `involved_classes`, seed 42,
  verified 50 unique each). `utils/sampling.py`.
- [ ] **Run the FC baseline over the full 200-task subset.** First real result.
- [ ] **Baseline noise re-run.** Run the baseline a second time over the subset
  (instances reset between passes); report the task-level flip rate as the
  noise band. (This is the only repeat — no `pass^k`.)

## Phase 2: ReAct

- [x] **Base `Architecture` abstraction** — done as part of Phase 1
  (`architecture.py`; `react.py`). Architectures vary only in system prompt
  (loop hooks to be added for Reflexion/REBACT).
- [x] **Implement ReAct** — zero-shot `Thought:` before each action via the
  shared loop; action through native FC. No exemplars. (`architectures/react.py`)
- [ ] **Run ReAct over the subset**; paired comparison vs baseline per category.

## Phase 3: Reflexion (episodic)

- [ ] **Within-task multi-attempt loop**, `k=3`. On a failed attempt: reflect
  (one LLM call), prepend reflection, retry. Attempt-1 prompt **identical to
  baseline** so Reflexion@1 == baseline by construction.
- [ ] **Sanitize the failure signal.** Feed back only which instance/attribute
  mismatched (or binary fail) — **never** the grader's expected value (it leaks
  ground truth). Note the oracle asymmetry explicitly.
- [ ] **Fix the reflection prompt format** (e.g. `What I tried / What went wrong
  / Lesson`); reuse for the vector-DB variant.
- [ ] **Report Reflexion@1 and Reflexion@k.** @1→@k gap = value of reflection.

## Phase 4: Reflexion (vector DB) — main contribution

- [ ] **Local persistent vector store** (Chroma or FAISS) at `./reflections.db`.
- [ ] **Wire Octen embedder.** Embed `task_description + reflection_text`; store
  with metadata (task_id, category, success flag).
- [ ] **Single attempt per task + retrieve top-k (k=3) past reflections**,
  injected as "lessons from similar past tasks". (Single attempt keeps the
  contribution clean: the only new variable vs episodic@1 is persistent memory.)
- [ ] **Fixed task order; plot the learning curve** (pass rate vs. position as
  the DB fills).
- [ ] **Compare to episodic Reflexion@1** — does persistent memory beat
  episodic, and where?

## Phase 5: REBACT

- [ ] **Reflect step after each tool result**: one LLM call → CONTINUE or
  MODIFY `<revised_call>`.
- [ ] **State rewind on MODIFY** (decided primary): rebuild the simulator from
  initial config + replay kept calls, **and edit the recorded call list** to
  drop the retracted call (the grader replays the recorded list, so live state
  and record must stay in lockstep — `CLAUDE.md → REBACT rewind`).
- [ ] **Run over the subset.** Document the rewind-vs-real-deployment caveat.

## Phase 6: Analysis & report

- [ ] **Aggregate results table.** Architecture × Category × **pass rate**
  (single run; + noise band). Paired tests (McNemar / bootstrap) for
  architecture differences; report CIs.
- [ ] **Cost analysis.** Tokens / LLM calls / peak context / latency per
  architecture; pass-rate-vs-cost. How each architecture grows the context.
- [ ] **Qualitative failure analysis.** 2–3 failed trajectories per
  architecture, tagged by the five failure modes using **operational
  definitions** (auto-detect what you can from the logs; rubric for the rest).
  Cite the emergentmind taxonomy.
- [ ] **Vector-DB learning curve plot.**
- [ ] **Write report.** 1 Intro+taxonomy · 2 Architectures · 3 Benchmark &
  methodology · 4 Results · 5 Qualitative analysis · 6 Discussion & limitations
  (incl. single-model external validity + REBACT rewind caveat).

## Stretch / nice-to-have

- [ ] Second model (e.g. `qwen35-397b`) on a small subset to show the
  architectural ordering is robust across models.
- [ ] One or two τ-bench tasks as an illustrative "harder problems" appendix.
- [ ] Trajectory pretty-printer (jsonl → readable markdown) for the qualitative
  pass.
- [ ] Combine REBACT + Reflexion-vector (in-trajectory + cross-trajectory).
  Probably out of scope.

## What NOT to do

- No `pass^k` / repeated trials (other than the single baseline noise re-run).
- No `composite` category — absent at tag `v1.3`.
- Don't extend to τ-bench full scale.
- Don't switch models mid-project — fixed model is essential for clean
  comparison.
- Don't implement CoT/ToT on BFCL (they can't act; not meaningful here).
- Don't add context-window management (summarization/truncation) — measure
  growth, don't manage it; managing it is another confound.
- No few-shot exemplars in any architecture; don't parse actions from text
  (native FC only). These are fairness invariants, not preferences.
