# Next Steps

Ordered by priority. Cross items off as completed. Add new tasks as they
emerge. Keep this in sync with reality — it's the source of truth for
"what should I work on right now?"

## Phase 0: De-risk the pipeline (a few hours)

These are the things that, if broken, will silently corrupt every result
that follows. Do them before scaling up.

- [x] **Verify state persistence in BFCL executor.**
  Confirmed by reading source (`multi_turn_utils.py:47-58`).
  `execute_multi_turn_func_call` stores instances in `globals()` keyed as
  `{model}_{task_id}_{class}_instance`.  On first call it creates and
  configures the instance; on subsequent calls it reuses it — state IS
  preserved across tool calls within the same process.
  `multi_turn_checker` passes `is_evaL_run=True` which appends `_eval` to
  the key, so grading uses a completely separate instance.  No refactoring
  needed; no state contamination between live run and grading.

- [x] **Verify the `is_evaL_run` parameter.**
  Confirmed capital-L in `multi_turn_utils.py:31`.
  Our call-site `is_evaL_run=False` is correct.

- [x] **Broaden retry exceptions.**
  `utils/retry.py` now catches `APIError`, `APIConnectionError`,
  `APITimeoutError`, `RateLimitError`, `httpx.ReadTimeout`,
  `httpx.ConnectTimeout`, `httpx.RemoteProtocolError`.
  All architectures share this via `from utils.retry import call_with_retry`.

- [x] **Add full trajectory logging.**
  `utils/logging.py` provides `TrajectoryLogger` context manager.
  Writes JSONL to `logs/{architecture}/{task_id}.jsonl`.
  Structured events: task_start, user_turn, llm_request, llm_response,
  tool_call, tool_result, turn_end, task_end (+ free-form `event()` for
  reflection steps).  `run_one_bfcl_task.py` updated to use it.

- [x] **Fix schema conversion bug (found during Phase 0 analysis).**
  `bfcl_schema_to_openai` only converted top-level `"dict"→"object"`.
  Bugs: (1) nested `dict` properties (e.g. `edit_ticket.updates`) not
  converted; (2) `"float"` is not a valid JSON Schema type — must be
  `"number"` (42 occurrences across all doc files).
  Fixed in `utils/schema.py` with a recursive `_normalize_schema()`.
  All 129 tool schemas across all 8 simulator classes now pass cleanly.

## Phase 1: Scale the baseline (1-2 days)

- [ ] **Run the FC baseline on 5 tasks** from `multi_turn_base` to confirm
  the pipeline is stable. Print pass/fail per task and aggregate.

- [ ] **Add task-iteration loop.**
  Refactor `run_one_bfcl_task.py` → `run_baseline.py` that loops over a
  configurable list of task IDs and emits a summary CSV: task_id, passed,
  num_steps, total_tokens, elapsed_seconds.

- [ ] **Pick the 20-30 task subset per category.**
  Don't run on all 1000 tasks. Pick a fixed subset, document the
  selection method (random with fixed seed, or curated by length/difficulty).
  Save the list as `task_subsets.json` so all architectures evaluate on
  the exact same tasks.

- [ ] **Run FC baseline on full subset** (4 categories × 20-30 tasks).
  This is the first real result. Save the summary CSV.

## Phase 2: ReAct architecture (2-3 days)

- [ ] **Factor architecture into a clean abstraction.**
  Define `class Architecture` with `run_task(task) -> calls_per_turn`.
  Both FC baseline and ReAct inherit from this. Different architectures
  vary only in: system prompt, optional pre/post-step hooks, optional
  reflection logic.

- [ ] **Implement ReAct.**
  System prompt instructs explicit `Thought:` reasoning before each
  action. Action still uses native function calling — we're testing the
  *thought injection*, not the *action format*. Output should look like:
  `Thought: ... \n [tool_call via native FC]`

- [ ] **Run ReAct on the full subset.** Compare to FC baseline,
  per-category.

## Phase 3: Reflexion (episodic) (2-3 days)

- [ ] **Implement single-task multi-attempt loop.**
  If trajectory fails state check, generate reflection (one LLM call),
  prepend reflection to the next attempt's system prompt, retry. Cap at
  k=3 attempts.

- [ ] **Decide reflection prompt format.**
  Suggested: structured (`What I tried / What went wrong / Lesson`). Fix
  this once and reuse for vector DB variant.

- [ ] **Report both Reflexion@1 and Reflexion@3.**
  @1 is directly comparable to ReAct@1. @3 is the gain from the
  reflection mechanism.

## Phase 4: Reflexion (vector DB) — the main contribution (3-5 days)

- [ ] **Set up local vector store.**
  Chroma or FAISS, persistent on disk at `./reflections.db`.

- [ ] **Wire Octen embedder.**
  Embed `task_description + reflection_text` (both, concatenated). Store
  with metadata: task_id, task_category, success_after_k_attempts.

- [ ] **Retrieve top-k on new task.**
  Query embedding = new task's description. k=3 to start. Inject
  retrieved reflections as system-prompt context: "Lessons from past
  similar tasks: ..."

- [ ] **Run with cold-start ordering.**
  Don't randomize task order — run tasks in fixed sequence so the DB
  grows naturally. Plot accuracy vs DB size to show the learning curve.

- [ ] **Compare to Reflexion-episodic.**
  Main empirical question: does persistent memory beat episodic? Where?

## Phase 5: REBACT (2-3 days)

- [ ] **Implement reflect step between observation and next action.**
  After each tool result, one extra LLM call: "Was that action right?
  CONTINUE or MODIFY <revised_call>." If MODIFY, replace last action
  (Option A: only for read-only tools; Option B: rewind simulator state).

- [ ] **Start with Option A.** Run on full subset.

- [ ] **If time: also run Option B** as upper-bound comparison.

## Phase 6: Analysis and report (1-2 weeks)

- [ ] **Aggregate results table.**
  Architecture × Category × (pass@1, pass^3). One main table for the report.

- [ ] **Qualitative failure analysis.**
  Pick 2-3 failed trajectories per architecture. Categorize each by the
  five failure modes. Look for patterns: which architecture fails how?

- [ ] **Plot DB size vs accuracy** for vector-DB Reflexion.

- [ ] **Write report.** Structure:
  1. Intro and taxonomy (1.5 p)
  2. Architectures under study (2 p)
  3. Benchmark and methodology (1 p)
  4. Results (2 p)
  5. Qualitative analysis (1.5 p)
  6. Discussion and limitations (1 p)

## Stretch / nice-to-have

- [ ] Run one or two τ-bench tasks as a "what happens on harder problems?"
  appendix. Frame as illustrative, not statistically meaningful.
- [ ] Test with a second model (e.g., `qwen35-397b`) on a small subset to
  show architectural ordering is robust across models.
- [ ] Combine REBACT + Reflexion-vector: in-trajectory and cross-trajectory
  reflection together. Probably out of scope, but interesting.

## What NOT to do

- Don't extend to τ-bench full scale (cost + complexity).
- Don't switch models mid-project — fixed model is essential for clean
  comparison.
- Don't implement CoT or ToT on BFCL (they can't act; not meaningful).
- Don't try to fix REBACT's irreversibility limitation with elaborate
  state-rollback mechanisms; just document it as a limitation.