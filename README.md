# Reasoning–Action Loop Architectures on BFCL v3

Does self-reflection actually help an LLM agent? This repository evaluates
Reflexion and ReAct on the **BFCL v3 multi-turn** tool-calling benchmark, with a
control that is usually missing: a **sham reflection** that has the same shape as
a real one but says nothing about the failure.

> Agentic AI seminar, University of Passau (WS 2025/2026).
> Presentation 22 May 2026 · report July 2026.

**Research question.** Does the information content of the reflection signal
change **how many** failures an agent recovers, or only **which ones**?

**Answer: only which ones.** Six conditions of increasing signal quality all
land within three tasks of each other, because every one of them draws from the
same small pool of recoverable failures.

| Condition       | What it adds to the retry                  |    pass@3 |
|-----------------|--------------------------------------------|----------:|
| Baseline        | nothing — native function calling          |  85 / 200 |
| ReAct           | a `Thought:` before each action            |  86 / 200 |
| Sham-lite       | retry + sanitised signal + a bland lesson  | 102 / 200 |
| Sham            | + generic corrective advice                | 103 / 200 |
| Reflexion       | + a lesson written from the actual failure | 100 / 200 |
| Reflexion+State | + the agent's own per-turn state timeline  | 100 / 200 |

Real reflection loses to the sham (−1.5 pp, 95% CI [−5.5, +2.5]). A much richer
signal changes nothing (+0.0 pp). Only 28 of the 115 failed tasks are ever
recovered by *any* retry, and every condition recovers 15–18 of those same 28.

## 📄 The report

**[`report/report.pdf`](report/report.pdf)** is the write-up: method, results,
the hand-coded reflection audit, and limitations. Read that first — everything
below is just how to run the code.

Supporting documents:

- [`docs/reflexion_vs_turnwise_lessons.md`](docs/reflexion_vs_turnwise_lessons.md)
  — the full hand-coded audit of all 14 discordant reflection pairs.
- [`baseline_vs_baseline_difference.csv`](baseline_vs_baseline_difference.csv) —
  hand-labelled analysis of the 15 tasks that flip between two identical runs.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install openai python-dotenv httpx
echo "FIM_API_KEY=..." > .env          # gitignored
```

BFCL is used from source, not pip — the repo expects `gorilla/` to be a clone of
`github.com/ShishirPatil/gorilla` at tag `v1.3`. All LLM calls go through
`utils.retry.call_with_retry`, since the endpoint throws occasional 5xx.

Model: `qwen3-next-80b-a3b-instruct`, served over the university FIM endpoint.

## Running

```bash
python run_benchmark.py --make-subset                # freeze task_subset.json and exit
python run_benchmark.py --arch baseline              # full frozen subset
python run_benchmark.py --arch react --category base # one architecture, one category
python run_benchmark.py --arch baseline --sample 5   # quick smoke, no frozen subset
python run_benchmark.py --arch baseline --tag run2   # noise re-run → separate output dir
```

The four retry arms seed attempt 1 from the recorded `baseline` run, so they
need a completed baseline on this machine and cost only the retries:

```bash
python run_benchmark.py --arch reflexion
python run_benchmark.py --arch blind_retry
python run_benchmark.py --arch blind_retry_lite
python run_benchmark.py --arch richer_reflexion_turnwise
```

Re-run individual tasks — the fresh rows are merged into the existing category
file rather than truncating it:

```bash
python run_benchmark.py --task-id multi_turn_base_90 multi_turn_miss_param_1
```

Each run writes a trajectory log to `logs/{arch}/{task_id}.jsonl`, a per-task
results row to `results/{arch}/{category}.jsonl`, and regenerates
`results/results.csv` and `results/summary.csv`.

## Layout

```
architectures/    one file per condition, all subclass a shared FC loop
utils/            retry, schema, executor, sanitize, state_dump,
                  conversation, logging, config
run_benchmark.py  runs a condition over the frozen subset and grades it
task_subset.json  the frozen 200-task subset, shared by every condition
results/          per-task grading + cost, plus the summary CSVs
logs/             per-task trajectory JSONL
docs/             design record and the reflection audit
report/           report.tex, references.bib, report.pdf
gorilla/          cloned BFCL repo at tag v1.3 (third-party, do not edit)
```

## Notes on reproducing

- **The subset is frozen** to `task_subset.json` (50 tasks × 4 categories,
  stratified by which APIs a task involves) and shared by every condition, so
  all comparisons are paired per task.
- **`temperature=0` is not reproducible here.** The endpoint serves a
  mixture-of-experts model, and two identical baseline runs disagreed on 15 of
  200 tasks. Expect re-runs to differ; that noise floor is why the sham control
  exists.
- **Grading** is `multi_turn_checker` AND `multi_turn_irrelevance_checker`. The
  second is what scores the abstention categories, and omitting it inflates
  `miss_param` and `miss_func`.

## AI assistance disclosure

I used Claude (Anthropic) throughout this project: to set up the benchmark
evaluation, to clarify and iterate on my thinking, to discuss the results, and
to review my work. I am solely responsible for any mistakes that remain.
