"""Shared public-surface helpers for episode adjudication work items."""

from __future__ import annotations

import inspect
from typing import Any


async def load_episode_adjudication_requests(
    manager: Any,
    *,
    episode_id: str,
    group_id: str,
) -> list[dict]:
    """Return episode adjudication work items when supported by the manager."""
    getter = getattr(manager, "get_episode_adjudications", None)
    if getter is None:
        return []
    result = getter(episode_id, group_id)
    if inspect.isawaitable(result):
        result = await result
    return result if isinstance(result, list) else []
