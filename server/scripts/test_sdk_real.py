"""Test Agent SDK with actual LongMemEval prompt."""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from engram.benchmark.longmemeval.dataset import load_dataset
from engram.embeddings.provider import GeminiProvider

os.environ.pop("CLAUDECODE", None)
os.environ.pop("CLAUDE_CODE_ENTRYPOINT", None)

load_dotenv(Path.home() / ".engram" / ".env", override=False)
load_dotenv()

# Init embedding provider (same as baseline)
try:
    provider = GeminiProvider()
    print("Embedding provider: OK")
except Exception as e:
    print(f"Embedding provider failed: {e}")
    provider = None

# Load first question
dataset = load_dataset("data/longmemeval/longmemeval_oracle.json", variant="auto")
inst = dataset.instances[0]

# Build prompt exactly like baseline
session_text = ""
for session in inst.sessions:
    header = "\n--- Session"
    if session.date:
        header += f" ({session.date})"
    header += " ---\n"
    session_text += header + session.text + "\n"

prompt = (
    f"Here are the user's past conversations:\n"
    f"{session_text}\n"
    f"Question (asked on {inst.question_date}): "
    f"{inst.question}"
)

print(f"Question: {inst.question}")
print(f"Prompt length: {len(prompt)} chars")


async def test():
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    opts = ClaudeAgentOptions(
        model="claude-sonnet-4-6",
        max_turns=1,
        permission_mode="acceptEdits",
        system_prompt="Answer based ONLY on the provided conversations. Be concise.",
        debug_stderr=sys.stderr,
    )
    try:
        async for msg in query(prompt=prompt, options=opts):
            if isinstance(msg, AssistantMessage):
                for b in msg.content:
                    if isinstance(b, TextBlock):
                        print("GOT:", b.text[:300])
            elif isinstance(msg, ResultMessage):
                print("DONE:", msg.subtype, f"cost=${getattr(msg, 'total_cost_usd', '?')}")
    except Exception as e:
        print(f"ERROR: {e}")


asyncio.run(test())
