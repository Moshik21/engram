"""Temporal intention scanner for proactive deadline notifications."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from engram.notifications.models import MemoryNotification
from engram.notifications.store import NotificationStore

if TYPE_CHECKING:
    from engram.config import ActivationConfig

logger = logging.getLogger(__name__)

_FIRED_CAP = 500


class TemporalIntentionScanner:
    """Checks for approaching intention deadlines and creates notifications."""

    def __init__(self, store: NotificationStore, cfg: ActivationConfig) -> None:
        self._store = store
        self._cfg = cfg
        self._fired_ids: set[str] = set()

    async def scan(self, group_id: str, graph_store: Any) -> int:
        """Scan intentions for approaching deadlines. Returns count of notifications created."""
        if not self._cfg.notification_temporal_enabled:
            return 0

        try:
            intentions = await graph_store.list_intentions(group_id, enabled_only=True)
        except Exception:
            logger.debug("Failed to list intentions for temporal scan", exc_info=True)
            return 0

        from engram.models.prospective import IntentionMeta

        now = time.time()
        horizon = self._cfg.notification_temporal_horizon_seconds
        count = 0

        for entity in intentions:
            attrs = entity.attributes or {}
            try:
                meta = IntentionMeta(**attrs)
            except Exception:
                continue

            if not meta.expires_at:
                continue

            # expires_at is ISO 8601 string — parse to timestamp
            try:
                from datetime import datetime, timezone

                dt = datetime.fromisoformat(meta.expires_at)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                expires_ts = dt.timestamp()
            except (ValueError, TypeError):
                continue

            entity_id = entity.id
            if entity_id in self._fired_ids:
                continue

            remaining = expires_ts - now
            if remaining <= 0 or remaining > horizon:
                continue

            # Approaching deadline — create notification
            hours_left = remaining / 3600
            if hours_left >= 1:
                time_str = f"{hours_left:.1f} hours"
            else:
                time_str = f"{remaining / 60:.0f} minutes"

            self._store.add(
                MemoryNotification(
                    group_id=group_id,
                    notification_type="temporal_intention",
                    priority="high",
                    title=f"Intention deadline in {time_str}",
                    body=f"Trigger: {meta.trigger_text}\nAction: {meta.action_text}",
                    entity_ids=[entity_id],
                    metadata={"expires_at": meta.expires_at, "remaining_seconds": remaining},
                    created_at=now,
                )
            )
            self._fired_ids.add(entity_id)
            count += 1

            # Bound the set
            if len(self._fired_ids) > _FIRED_CAP:
                # Remove oldest half
                excess = list(self._fired_ids)[: _FIRED_CAP // 2]
                self._fired_ids -= set(excess)

        return count
