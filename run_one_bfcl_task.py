"""
Minimal BFCL v3 multi-turn pipeline — Function Calling Baseline.

Runs ONE task end-to-end with Qwen3-Next via native function calling.
Writes a full trajectory log to logs/baseline/{task_id}.jsonl.

Goal: confirm pipeline works end-to-end and produces a PASS/FAIL.

Phase 0 findings (confirmed from source, no runtime needed)
-----------------------------------------------------------
* State persistence: execute_multi_turn_func_call stores simulator instances
  in globals() keyed as "{model}_{task_id}_{class}_instance".  Instances are
  reused across calls in the same process, so state IS preserved turn-to-turn.
* Grading isolation: multi_turn_checker passes is_evaL_run=True, which appends
  "_eval" to the key → completely separate instance from the live-run one.
  No state contamination between execution and grading.
* is_evaL_run spelling: capital-L confirmed in multi_turn_utils.py:31.
  Our call-site (is_evaL_run=False) is correct.
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# --- Make the BFCL repo importable -----------------------------------------
BFCL_REPO = Path(__file__).parent / "gorilla" / "berkeley-function-call-leaderboard"
sys.path.insert(0, str(BFCL_REPO))

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker

# --- Shared utils (Phase 0) -------------------------------------------------
from utils.retry import call_with_retry
from utils.schema import load_tools_for_classes
from utils.logging import TrajectoryLogger

# --- Config -----------------------------------------------------------------
load_dotenv()
client = OpenAI(
    api_key=os.environ["FIM_API_KEY"],
    base_url="https://llms.innkube.fim.uni-passau.de/v1",
)
MODEL = "qwen3-next-80b-a3b-instruct"
ARCHITECTURE = "baseline"

TARGET_TASK_ID = "multi_turn_base_1"
TASK_FILE = BFCL_REPO / "bfcl_eval/data/BFCL_v3_multi_turn_base.json"
ANSWER_FILE = BFCL_REPO / "bfcl_eval/data/possible_answer/BFCL_v3_multi_turn_base.json"
DOCS_DIR = BFCL_REPO / "bfcl_eval/data/multi_turn_func_doc"

# --- Helpers ---------------------------------------------------------------

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def tool_call_to_python_string(name: str, json_args: str) -> str:
    """
    Convert a model's tool call into a Python call string the BFCL executor
    can `eval`. Example:
        name='mv', json_args='{"source":"log.txt","destination":"archive"}'
        -> "mv(source='log.txt', destination='archive')"
    """
    args = json.loads(json_args)
    parts = []
    for k, v in args.items():
        # repr() handles strings, bools, ints, lists, dicts, None correctly
        parts.append(f"{k}={repr(v)}")
    return f"{name}({', '.join(parts)})"


def execute_call_locally(call_string: str, initial_config: dict,
                         involved_classes: list, test_entry_id: str) -> str:
    """
    Run one call string against the *live* simulator so we can feed the
    result back to the model as a tool observation.

    Uses BFCL's own executor so state is consistent with grading.

    State persistence note: execute_multi_turn_func_call stores the simulator
    instance in globals() on first call and reuses it on subsequent calls,
    so state accumulates correctly across multiple calls to this function
    within the same task.
    """
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
    )
    results, _ = execute_multi_turn_func_call(
        func_call_list=[call_string],
        initial_config=initial_config,
        involved_classes=involved_classes,
        model_name=MODEL,
        test_entry_id=test_entry_id,
        long_context=False,
        is_evaL_run=False,  # capital-L is correct — confirmed in multi_turn_utils.py:31
    )
    return results[0] if results else "[no result]"


# --- The main task loop ----------------------------------------------------

def run_task(task: dict, tlog: TrajectoryLogger) -> list[list[list[str]]]:
    """
    Run one BFCL multi-turn task with the FC baseline.

    Returns the per-turn, per-step call strings in the shape
    ``list[list[list[str]]]`` that ``multi_turn_checker`` expects.
    """
    print(f"\n{'='*70}")
    print(f"Running task: {task['id']}")
    print(f"Involved classes: {task['involved_classes']}")
    print(f"Total user turns: {len(task['question'])}")
    print(f"{'='*70}\n")

    tools = load_tools_for_classes(task["involved_classes"], DOCS_DIR)
    print(f"Loaded {len(tools)} tool schemas\n")

    tlog.task_start(task)

    # Build the initial message history with a minimal system prompt.
    messages = [
        {
            "role": "system",
            "content": (
                "You are an agent that completes user tasks by calling the "
                "provided tools. Call tools whenever they are needed. After "
                "you have all the information to satisfy the user's request, "
                "reply in natural language to summarise what you did."
            ),
        }
    ]

    # all_turns_calls[turn_idx][step_idx] = list of call strings made at that step
    all_turns_calls: list[list[list[str]]] = []

    for turn_idx, turn_messages in enumerate(task["question"]):
        print(f"\n--- Turn {turn_idx + 1} ---")
        # Each "turn" in BFCL is a list of messages; usually just one user msg.
        for m in turn_messages:
            messages.append(m)
            print(f"[user] {m['content']}")
        tlog.user_turn(turn_idx, turn_messages)

        # Inner loop: keep calling the model until it stops requesting tools.
        this_turn_calls: list[list[str]] = []
        MAX_STEPS_PER_TURN = 10

        for step in range(MAX_STEPS_PER_TURN):
            tlog.llm_request(turn_idx, step, MODEL, messages, tools)
            t0 = time.monotonic()
            response = call_with_retry(
                client,
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0,
                max_tokens=1024,
            )
            latency = time.monotonic() - t0
            tlog.llm_response(turn_idx, step, response, latency)

            msg = response.choices[0].message

            # The model can return BOTH content and tool_calls. Append both.
            assistant_msg = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            # No tool calls? Model is done with this turn.
            if not msg.tool_calls:
                print(f"[assistant] {msg.content}")
                break

            # Otherwise, execute each tool call and feed results back.
            step_call_strings = []
            for call_idx, tc in enumerate(msg.tool_calls):
                call_str = tool_call_to_python_string(
                    tc.function.name, tc.function.arguments
                )
                args_dict = json.loads(tc.function.arguments)
                print(f"[tool_call] {call_str}")
                tlog.tool_call(turn_idx, step, call_idx, tc.function.name, args_dict, call_str)
                step_call_strings.append(call_str)

                result = execute_call_locally(
                    call_str,
                    task["initial_config"],
                    task["involved_classes"],
                    task["id"],
                )
                print(f"[tool_result] {result}")
                tlog.tool_result(turn_idx, step, call_idx, result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

            this_turn_calls.append(step_call_strings)
        else:
            print(f"⚠ Reached MAX_STEPS_PER_TURN={MAX_STEPS_PER_TURN} without resolution")

        tlog.turn_end(turn_idx, n_steps=len(this_turn_calls))
        all_turns_calls.append(this_turn_calls)

    return all_turns_calls


# --- Entry point ----------------------------------------------------------

def main():
    tasks = load_jsonl(TASK_FILE)
    task = next(t for t in tasks if t["id"] == TARGET_TASK_ID)

    answers = load_jsonl(ANSWER_FILE)
    answer = next(a for a in answers if a["id"] == TARGET_TASK_ID)

    with TrajectoryLogger(ARCHITECTURE, TARGET_TASK_ID) as tlog:
        # Run the model on the task
        model_calls = run_task(task, tlog)

        # Show what the model produced vs what the ground truth expects
        print(f"\n{'='*70}")
        print("MODEL CALLS vs GROUND TRUTH (per turn)")
        print(f"{'='*70}")
        for i, (model_turn, gt_turn) in enumerate(zip(model_calls, answer["ground_truth"])):
            print(f"\nTurn {i + 1}:")
            print(f"  model: {model_turn}")
            print(f"  gt:    {gt_turn}")

        # Grade with BFCL's checker
        print(f"\n{'='*70}")
        print("GRADING WITH BFCL CHECKER")
        print(f"{'='*70}")
        result = multi_turn_checker(
            multi_turn_model_result_list_decoded=model_calls,
            multi_turn_ground_truth_list=answer["ground_truth"],
            test_entry=task,
            test_category="multi_turn_base",
            model_name=MODEL,
        )
        print(json.dumps(result, indent=2, default=str)[:2000])

        passed = bool(result.get("valid"))
        tlog.task_end(
            passed=passed,
            error_message=None if passed else result.get("error_message"),
        )

    print(f"\n{'='*70}")
    if passed:
        print("✓ PASS")
    else:
        print(f"✗ FAIL: {result.get('error_message', 'unknown')}")
    print(f"{'='*70}")
    print(f"  Trajectory log: logs/{ARCHITECTURE}/{TARGET_TASK_ID}.jsonl")


if __name__ == "__main__":
    main()
