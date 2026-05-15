"""Test Agent SDK with a long prompt (simulating baseline)."""
import os
os.environ.pop("CLAUDECODE", None)
os.environ.pop("CLAUDE_CODE_ENTRYPOINT", None)

import asyncio
import sys
from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage, AssistantMessage, TextBlock

# Simulate a baseline prompt with session content
fake_sessions = "User: I went to the store today.\nAssistant: What did you buy?\n" * 100
prompt = f"Here are the user's past conversations:\n{fake_sessions}\nQuestion: What did the user do today?"

print(f"Prompt length: {len(prompt)} chars")

async def test():
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
                        print("GOT:", b.text[:200])
            elif isinstance(msg, ResultMessage):
                print("DONE:", msg.subtype)
    except Exception as e:
        print(f"ERROR: {e}")

asyncio.run(test())
