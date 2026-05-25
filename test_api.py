"""Smoke test: can we reach the FIM endpoint and get a response?"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["FIM_API_KEY"],
    base_url="https://llms.innkube.fim.uni-passau.de/v1",
)

# Try Qwen3-Next first; if it's been replaced, we'll switch.
MODEL = "qwen3-next-80b-a3b-instruct"

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": "Reply with exactly one word: hello"}
    ],
    max_tokens=20,
    temperature=0,
)

print("Model:", MODEL)
print("Response:", response.choices[0].message.content)
print("Tokens used:", response.usage.total_tokens)

