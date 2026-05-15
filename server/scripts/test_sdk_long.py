"""Test Agent SDK with a long prompt (simulating baseline)."""

import asyncio
import os
import sys

os.environ.pop("CLAUDECODE", None)
os.environ.pop("CLAUDE_CODE_ENTRYPOINT", None)

# Simulate a baseline prompt with session content
fake_sessions = "User: I went to the store today.\nAssistant: What did you buy?\n" * 100
prompt = (
    f"Here are the user's past conversations:\n{fake_sessions}\n"
    "Question: What did the user do today?"
)

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
                        print("GOT:", b.text[:200])
            elif isinstance(msg, ResultMessage):
                print("DONE:", msg.subtype)
    except Exception as e:
        print(f"ERROR: {e}")


asyncio.run(test())
