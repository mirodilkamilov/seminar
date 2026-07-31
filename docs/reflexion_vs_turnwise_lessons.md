# Hand-coded lesson audit: `reflexion` vs `richer_reflexion_turnwise`

Supporting evidence for the report's §5.2 (the audit table) and §5.3 (the two
exemplars). The two arms score identically — a difference of +0.0pp, 7 wins to
7, p = 1.00.

The question the pass rate cannot answer: **when the two arms disagree on a
task, is it because the state block changed what the model concluded, or because
the retry lottery landed differently?**

**Where the evidence lives.** Everything below is derived from two committed
directories, so every code in this document can be re-checked against source:

- `results/reflexion/{category}.jsonl` and
  `results/richer_reflexion_turnwise/{category}.jsonl` — one row per task, with
  `passed`, `passed_at_1`, `passed_at` and per-attempt `error_type`. The 14
  discordant tasks are the rows where `passed` differs between the two arms;
  `results/baseline/` supplies the 115-task failure set both arms retry.
- `logs/reflexion/{task_id}.jsonl` and
  `logs/richer_reflexion_turnwise/{task_id}.jsonl` — the full trajectories. Each
  `reflection` event carries the `attempt` number, the sanitized `signal`, the
  model's `text`, the rendered `prompt`, and (turnwise only) the `state_diff`
  block. Those events are the objects coded here.

---

## 1. Why attempt 1 is the clean comparison

For the **attempt-1 reflection**, the only difference between the arms is the
presence of the state block:

- both arms are seeded from `results/baseline/` — 0/200 mismatches on
  `passed_at_1` and on attempt-1 `error_type`, so the failed transcript is the
  same text;
- the sanitized failure signal is the same string (`utils/sanitize.py`);
- the reflection prompt is byte-identical but for the inserted state paragraph
  (`REFLECTION_PROMPT` vs `REFLECTION_PROMPT_TIMELINE`);
- temperature is 0 in both.

Any difference in the resulting text is therefore caused by the block. That is
what §3 counts. Attempt-2 and attempt-3 reflections are **not** clean — by then
the arms have run different retries under possibly different signals — so §4
reads them as narrative only.

**The 14 discordant tasks:**

| # | `reflexion` won | `turnwise` won |
|---|---|---|
| 1 | `base_94` | `base_117` |
| 2 | `miss_func_17` | `miss_func_164` |
| 3 | `miss_func_57` | `miss_param_32` |
| 4 | `miss_func_82` | `miss_param_68` |
| 5 | `miss_func_117` | `miss_param_145` |
| 6 | `miss_func_186` | `miss_param_153` |
| 7 | `miss_param_169` | `miss_param_165` |

**5 of reflexion's 7 wins are `miss_func`; 5 of turnwise's 7 are `miss_param`** —
the act/ask reallocation seen task by task.

---

## 2. Coding rubric

Each attempt-1 reflection was read against the task's `possible_answer` ground
truth and coded on two axes. The separation matters: the prompt asks for a
*diagnosis* (`What went wrong`) and a *prescription* (`Lesson`), only the second
of which the next attempt consumes, and the two come apart (§3.4).

**Axis A — diagnosis.** Does `What went wrong` name the cause of the graded
failure? `correct` / `partial` (a real defect but not the graded one, or one
right clause plus one wrong) / `wrong`.

**Axis L — lesson.** Would a retry following the `Lesson` verbatim be helped on
this task? `correct` / `partial` / `wrong` (including actively harmful).

**Outcome attribution.** `lesson` when the winning arm's text identifies
something the loser's does not *and* the winning retry visibly acts on it;
`luck` when the two texts are substantively the same, or the winner's own text
is wrong.

**Coder:** single, unblinded — the block is visible in the turnwise prompt, so
blinding is impossible in principle (§5).

---

## 3. Findings

### 3.1 The arms are level on both axes

| | **diagnosis** | | | **lesson** | | |
|---|---|---|---|---|---|---|
| | correct | partial | wrong | correct | partial | wrong |
| `reflexion` | **6** | 3 | 5 | **6** | 2 | 6 |
| `richer_reflexion_turnwise` | 5 | 4 | 5 | 5 | 4 | 5 |

Note the direction: **`reflexion` is marginally ahead on diagnosis (6 vs 5)**.
The richer signal did not produce better reflections that were then squandered —
it produced equally good ones. Of the two explanations for the null (the signal
helped but the benefit was lost downstream, or the signal did not help), the
coding supports the second.

### 3.2 The act/ask lean does not shift at attempt 1

Coding each `Lesson` as pushing toward acting, toward asking, or neither:

| | `reflexion` | `turnwise` |
|---|---|---|
| act-more | 5 | 5 |
| ask-more | 6 | 5 |
| task-specific | 3 | 4 |

The threshold shift reported in the report's §4.4 (`miss_param` retries,
8.00 → 8.87 tool calls) **is not present in the first lesson**. It emerges over attempts, as
diverging retries draw diverging signals and the append-only preamble stacks a
second lesson on the first.

### 3.3 The block changes conclusions on 6 of 14 — symmetrically

| what changed | n | tasks |
|---|---|---|
| diagnosis improved | 2 | `base_117`, `miss_func_17` |
| diagnosis degraded | 2 | `miss_func_186`, `miss_param_169` |
| diagnosis content changed, accuracy unchanged | 1 | `miss_param_165` |
| lesson improved, diagnosis identical | 1 | `miss_param_32` |
| no material change | 8 | the rest |

Two up, two down, plus one lesson-only gain. A real mechanism running
symmetrically is how +0.0pp is produced without either arm being inert.

### 3.4 Diagnosis quality and lesson quality are separable

**`miss_param_32` — identical diagnoses, divergent lessons.** Ground truth
requires `echo(content='2.0', file_name='result.txt')`. Both arms diagnose
correctly: the model wrote an explanatory sentence into the file instead of the
bare number. They diverge on the fix:

- `reflexion` **guessed the format** — *"output only the precise numeric value in
  the required format (e.g., `2.0000`)"*. Its retry wrote `2.0000`, failed, and
  its attempt-2 lesson doubled down plus speculation about *"trailing newline,
  encoding, or file permissions"*. It lost because of its own lesson.
- `turnwise` was shown the file it had written — `'result.txt': <<File:
  result.txt, Content: The logarithm of 36 to base 6 is: …` — and prescribed
  *"write only the raw number (e.g., `2.0`)"*. Passed at attempt 2.

The block's contribution was not better understanding — both arms had that — but
a concrete anchor that kept the prescription from drifting into invention.

**`miss_func_17` — better diagnosis, worse lesson.** `ls` is held out until
turn 2 and the model did not call it when revealed.

- `reflexion` misread the failure entirely (thought turn 2 was the "list my
  communications" request). **A: wrong.**
- `turnwise`, seeing every state change land under `Turn 4`, reached a genuine
  structural observation: *"I failed to make any tool call in that turn because I
  processed everything in turn 3."* **A: partial**, and better.

It then derived *"make **one tool call per turn** … never batch multiple actions
into a single turn"* — false for BFCL, where turns routinely require several
calls. It repeated the rule at attempt 2 and lost. **L: wrong.**

### 3.5 A hazard the diff does not remove: credential-shaped distractors

Diffing against `initial_config` rather than dumping absolute state strips inert
auth-shaped flags — measured over the subset, 28 such traps fall to 4. It does. It does **not** strip credential
records the model's own actions legitimately changed, and those mislead just as
effectively. Both of turnwise's credential misdiagnoses sit on tasks whose block
contains a mutated `credit_card_list`:

- **`miss_param_169`** — `reflexion` diagnosed correctly (*"the user initially
  did not specify the travel date or class, and I should have asked"*).
  `turnwise` wrote *"The system required **client_id, client_secret**, and other
  authentication details"*. Those parameters are real — they belong to
  `authenticate_travel`, one of the 27 tools the agent held — but they are not
  preconditions for `book_flight`, which needs only the `access_token` the user
  supplied. A *relevance* failure, not a hallucination.
- **`miss_func_186`** — same pattern: *"no authentication credentials or explicit
  permission to access sensitive data like flight prices"*. `get_flight_cost`
  takes no token.

The auth-trap analysis that justified the diff was measured on *inert*
attributes and is therefore incomplete: a state signal can manufacture a false
precondition out of a correctly-changed credential record. A future version
needs **redaction** of credential-shaped values, not merely diffing.

### 3.6 Half the discordance is not about the reflections

| attribution | n | tasks |
|---|---|---|
| **text-attributable** | 7 | refl: `miss_func_17`, `miss_func_186`, `miss_param_169` · turn: `base_117`, `miss_param_32`, `miss_param_68`, `miss_param_165` |
| **retry lottery** | 6 | `base_94`, `miss_func_57`, `miss_func_82`, `miss_func_117`, `miss_func_164`, `miss_param_145` |
| weak / ambiguous | 1 | `miss_param_153` |

7 tasks turn on reflection content, splitting **3–4**. Net effect of reflection
quality: **+1 to +2 tasks (+0.5 to +1.0pp)** — inside the measured CI of
[−3.5, +3.5]pp. The coding and the pass rate agree, and the coding *explains* the
null rather than restating it. It corroborates the borderline-pool account at
task level: six of these flip regardless of what the reflection said.

### 3.7 How the block was actually used

The block exists only in the turnwise arm, so the denominator is its 14
attempt-1 reflections. Criterion: does the text cite a fact present in the block
and not already stated in the transcript?

| use | n | tasks |
|---|---|---|
| no discernible use | 6 | `base_94`, `miss_func_82`, `miss_func_117`, `miss_func_164`, `miss_param_145`, `miss_param_68` |
| drove the conclusion | 4 | `base_117`, `miss_func_17`, `miss_param_32`, `miss_param_165` |
| misled | 2 | `miss_func_186`, `miss_param_169` |
| block was empty | 2 | `miss_func_57`, `miss_param_153` |

It went unused about as often as it helped. Two of the unused cases are the
damning ones, because the block demonstrably held the answer:

- **`base_94`** lists `parkingBrakeStatus = "released"` under `Turn 3` — the
  exact cause of the state mismatch — and the lesson blames tire pressure.
- **`miss_func_117`** shows `watch_list = ["NVDA"]` first appearing under
  `Turn 3`, direct evidence the removal landed a turn late; the lesson diagnoses
  turn 7 instead.

This sharpens the finding that 47.6% of state-block values are already verbatim
in the transcript: the problem is not only redundancy, but that much of
the non-redundant remainder is ignored when it matters. Salience was the
hypothesis; salience is apparently not the binding constraint either.

---

## 4. Per-task codes

Quotes are verbatim from the `reflection` events in
`logs/{arm}/{task_id}.jsonl`, attempt 1 unless stated otherwise.

### 4.1 Tasks `reflexion` won

**`base_94`** — VehicleControlAPI, 3 turns. Signal: state mismatch. Cause: GT
turn 3 is `pressBrakePedal(1.0)` + `startEngine('START')`; the model also called
`activateParkingBrake(mode='release')`, and the simulator starts at
`parkingBrakeStatus='engaged'`, so the extra mutating call left state GT never
produced.

| arm | `What went wrong` | A | L |
|---|---|---|---|
| `reflexion` | *"I set navigation to a tire shop instead of inflating the tires"* | wrong | wrong |
| `turnwise` | *"I set navigation to a tire shop instead of ensuring tires were inflated"* | wrong | wrong |

GT turn 2 **requires** `find_nearest_tire_shop()` and `set_navigation(...)`, so
both arms indict a correct action. The block showed the parking-brake change and
neither lesson mentions it; at attempt 2 it supplied corroborating detail that
*hardened* the wrong hypothesis. **Attribution: luck** — reflexion's attempt-2
lesson is also wrong, so its attempt-3 pass is not text-driven.

**`miss_func_117`** — TradingBot, 7 turns. Signal: empty response at turn 2.
Cause: `remove_stock_from_watchlist` held out until turn 2 and not called when
revealed. Both arms diagnose turn 7's OMEG deposit instead. **A: wrong / wrong.
L: wrong / wrong.** **Attribution: luck.**

**`miss_func_17`** — MessageAPI + GorillaFileSystem, 4 turns. See §3.4.
**A: wrong / partial** (turnwise better). **L: wrong / wrong** (turnwise's "one
tool call per turn" is harmful). **Attribution: lesson.**

**`miss_func_186`** — TwitterAPI + TravelAPI, 7 turns. `get_flight_cost` held
out until turn 2, so GT turn 1 is empty and the failure is an irrelevance error.

| arm | `What went wrong` | A | L |
|---|---|---|---|
| `reflexion` | *"could have been handled with a single tool call (get_flight_cost) … I prematurely requested authentication credentials"* | partial | correct |
| `turnwise` | *"no authentication credentials or explicit permission to access sensitive data like flight prices"* | wrong | partial |

Reflexion is partial because "should have called `get_flight_cost`" is wrong (it
was held out) but "made premature calls at turn 1" is right, and its lesson is
the correct corrective. Turnwise's *attempt-2* lesson blames the grader —
*"the grader miscounted the turn sequence"* — the only instance across all 28
coded reflections of the model rejecting the failure signal.
**Attribution: lesson.**

**`miss_func_57`** — VehicleControlAPI + MathAPI, 3 turns. Signal: empty
response at turn 2. Both **A: correct, L: correct**, near-verbatim. Turnwise's
block is empty. **Attribution: luck.**

**`miss_func_82`** — VehicleControlAPI, 4 turns. Signal: irrelevance at turn 2.
Both **A: correct, L: correct** (*"I assumed the tank capacity … the correct
behavior was to ask"*). At attempt 2 turnwise drew a new signal and **reversed
itself** — *"call the tool directly without asking for confirmation"* — the
act/ask oscillation inside one task. **Attribution: luck.**

**`miss_param_169`** — MessageAPI + TravelAPI, 3 turns. See §3.5.
**A: correct / wrong** — the clearest case of the richer signal destroying a
correct diagnosis. **L: correct / partial.** **Attribution: lesson.**

### 4.2 Tasks `richer_reflexion_turnwise` won

**`base_117`** — TradingBot, 6 turns. Signal: response mismatch at turn 6. GT is
`fund_account(5000)`; a known synonymous-function confounder.

| arm | `What went wrong` | A | L |
|---|---|---|---|
| `reflexion` | *"I failed to explicitly confirm … the task required explicit user confirmation"* | wrong | wrong |
| `turnwise` | *"all required tools were called with correct arguments — the failure may stem from an unreported expectation"* | partial | partial |

The block showed `transaction_history = [{"type": "deposit", "amount": 5000, …}]`
— the deposit *did* happen and the state *is* right — so turnwise **declined to
confabulate a cause**, the most accurate reading reachable from a sanitized
signal that cannot reveal a response-gate mismatch. Reflexion's attempt-2 lesson
**flatly contradicts** its first (*"confirmation alone is not sufficient; action
must follow"*) — the append-only preamble stacks both, so the retry receives a
self-contradicting pair. **Attribution: lesson.**

**`miss_func_164`** — TravelAPI, 5 turns. Both arms give nearly the same
diagnosis (*"never explored connecting flights"*) and nearly the same attempt-2
lesson. **A: partial / partial. L: partial / partial. Attribution: luck.**

**`miss_param_145`** — MessageAPI + TradingBot, 6 turns. Cause: the model called
`message_login(user_id='1234')` (returns `false`) where GT uses
`message_get_login_status()` (returns `true`) — a benchmark-validity confounder.
Both arms blame authentication in near-identical words. **A: wrong / wrong.
L: wrong / wrong. Attribution: luck.**

**`miss_param_153`** — TravelAPI, 6 turns. Signal: empty response at turn 5.
Both **A: correct, L: correct**. Turnwise's block is empty; its lesson is
marginally sharper on the act side. Reflexion's attempt-2 lesson went wrong
(*"I unnecessarily re-booked the flight"*). **Attribution: weak / ambiguous.**

**`miss_param_165`** — TicketAPI + TravelAPI, 6 turns. Signal: state mismatch.

| arm | `What went wrong` | A | L |
|---|---|---|---|
| `reflexion` | *"I never completed the flight booking or confirmed the card details"* | partial | partial |
| `turnwise` | *"The budget limit was changed … without fulfilling the core request to book the flight"* | partial | wrong |

Equal on Axis A, but the content differs in a way that mattered. The block held
exactly one change across six turns — `budget_limit = 1000.0` — and no
`booking_record`. "Only the budget moved; you never booked" came straight from
it. The clause *"the budget limit was changed unnecessarily"* is wrong (GT does
require `set_budget_limit(1000)`), hence L: wrong — yet the operative half of the
prescription is what the retry needed, whereas reflexion's pushes the other way.
**Attribution: lesson.**

**`miss_param_32`** — GorillaFileSystem + MathAPI, 3 turns. See §3.4.
**A: correct / correct. L: wrong / correct** — `2.0000` vs `2.0`, read off the
file content in the block. **Attribution: lesson.**

**`miss_param_68`** — TwitterAPI + VehicleControlAPI, 6 turns. Cause: the model
posted the tweet twice, leaving an extra tweet GT never produced. Both
**A: correct, L: correct** at attempt 1; the block corroborates numerically
(`tweet_counter = 6` at turn 5, `= 7` at turn 6).

The divergence is at attempt 2: reflexion drew a new signal and produced a bad
lesson (*"always re-confirm critical user inputs with a tool call"*), prescribing
a spurious extra call in a category that punishes exactly that. Turnwise, still
holding the same block, kept its correct diagnosis and won at attempt 3.
**Attribution: lesson**, via *stability* rather than a better attempt-1
conclusion — which is why §3.7 files it under "no discernible use": that table is
strictly about the attempt-1 text. A persistent block anchors the diagnosis
against signal drift.

---

## 5. Threats to this coding

1. **Single, unblinded coder.** The block is visible in the turnwise prompts, so
   blinding is impossible in principle. The six changed-conclusion tasks are most
   exposed; the eight unchanged ones are near-verbatim pairs.
2. **n = 14, no significance claimed.** These are exemplars that explain a null,
   not evidence for an effect. The 3–4 split is descriptive.
3. **"Attribution: lesson" is an inference, not a measurement.** It reads the
   retry and asks whether it visibly acts on the text. The strict test — replaying
   arm A's lesson into arm B's retry — was not run.
4. **Accuracy is judged against ground truth the reflector never sees.** That is
   the point, but "wrong" sometimes marks a *reasonable* inference from an
   underdetermined signal rather than sloppy reasoning.
5. **Two of the 14 are benchmark artifacts** (`base_117`, `miss_param_145`) where
   no correct diagnosis was reachable from the sanitized signal, so the effective
   clean sample is 12.
