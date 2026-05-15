"""Minimal Agent SDK test."""

import asyncio
import os

os.environ.pop("CLAUDECODE", None)
os.environ.pop("CLAUDE_CODE_ENTRYPOINT", None)


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
    )
    async for msg in query(prompt="Say PONG", options=opts):
        if isinstance(msg, AssistantMessage):
            for b in msg.content:
                if isinstance(b, TextBlock):
                    print("GOT:", b.text)
        elif isinstance(msg, ResultMessage):
            print("DONE:", msg.subtype)


asyncio.run(test())
