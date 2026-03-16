"""HyDE -- Hypothetical Document Embedding for improved retrieval.

Generates a hypothetical answer passage before embedding, bridging the
question-answer embedding asymmetry.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

HYDE_CACHE_SIZE = 256

_HYDE_PROMPT = """\
You are helping a memory retrieval system. Given this question about a \
user's past conversations, write a short paragraph (2-3 sentences) that \
would be found in the conversation transcript containing the answer. \
Write it as if you are the actual conversation text, in first person \
from the user's perspective. Do not answer the question - write what \
the conversation would look like.

Question: {query}

Conversation passage that would contain the answer:"""


async def generate_hypothetical_document(
    query: str,
    *,
    model: str = "claude-haiku-4-5-20251001",
    cache: OrderedDict[str, str] | None = None,
) -> str | None:
    """Generate a hypothetical answer passage for the given query.

    Returns None if generation fails (caller should fall back to original query).
    """
    # Check cache
    if cache is not None and query in cache:
        cache.move_to_end(query)
        return cache[query]

    try:
        import anthropic

        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
        )

        hypothesis = response.content[0].text.strip()

        # Cache result
        if cache is not None:
            cache[query] = hypothesis
            if len(cache) > HYDE_CACHE_SIZE:
                cache.popitem(last=False)

        return hypothesis

    except Exception as e:
        logger.warning("HyDE generation failed (non-fatal): %s", e)
        return None
