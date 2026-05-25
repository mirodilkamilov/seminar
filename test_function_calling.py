"""Does the FIM endpoint support native function calling for Qwen3-Next?
We define one fake tool and ask a question that should trigger it."""
import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI, APIError

load_dotenv()

client = OpenAI(
    api_key=os.environ["FIM_API_KEY"],
    base_url="https://llms.innkube.fim.uni-passau.de/v1",
)

MODEL = "qwen3-next-80b-a3b-instruct"


def call_with_retry(max_retries=3, **kwargs):
    """Tiny retry wrapper: 3 tries, exponential backoff (1s, 2s, 4s)."""
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except APIError as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [retry] {type(e).__name__}: {e}. Waiting {wait}s...")
            time.sleep(wait)


# Define a single dummy tool, in OpenAI's tool-calling format.
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["city"],
            },
        },
    }
]

messages = [
    {"role": "user", "content": "What's the weather in Munich right now? Use celsius."}
]

response = call_with_retry(
    model=MODEL,
    messages=messages,
    tools=tools,
    tool_choice="auto",
    temperature=0,
    max_tokens=200,
)

msg = response.choices[0].message

print("=" * 60)
print("Finish reason:", response.choices[0].finish_reason)
print("Content (text):", repr(msg.content))
print("Tool calls:", msg.tool_calls)
print("=" * 60)

if msg.tool_calls:
    print("\n✓ Function calling WORKS.")
    for tc in msg.tool_calls:
        print(f"  Tool: {tc.function.name}")
        print(f"  Args: {tc.function.arguments}")
        # Verify the args are valid JSON
        try:
            parsed = json.loads(tc.function.arguments)
            print(f"  Parsed args: {parsed}")
        except json.JSONDecodeError as e:
            print(f"  ⚠ Args are not valid JSON: {e}")
else:
    print("\n✗ No tool call was made. Model responded with text only.")
    print("  This means either:")
    print("   (a) The endpoint doesn't pass tools through to the model, or")
    print("   (b) Qwen3-Next decided not to use the tool for this query.")

