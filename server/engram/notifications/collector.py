"""EventBus hook that translates consolidation events into notifications."""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from engram.notifications.models import MemoryNotification, notification_to_dict
from engram.notifications.store import NotificationStore

if TYPE_CHECKING:
    from engram.config import ActivationConfig

logger = logging.getLogger(__name__)

# Max idempotency digests retained
_DEDUP_CAP = 1000


class NotificationCollector:
    """Watches EventBus for consolidation phase completions and creates notifications."""

    def __init__(
        self,
        store: NotificationStore,
        cfg: ActivationConfig,
        consolidation_store: Any = None,
    ) -> None:
        self._store = store
        self._cfg = cfg
        self._consolidation_store = consolidation_store
        self._seen: OrderedDict[str, None] = OrderedDict()

    def _dedup_key(self, ntype: str, cycle_id: str, entity_ids: list[str]) -> str:
        raw = f"{ntype}:{cycle_id}:{','.join(sorted(entity_ids))}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _is_duplicate(self, key: str) -> bool:
        if key in self._seen:
            return True
        self._seen[key] = None
        while len(self._seen) > _DEDUP_CAP:
            self._seen.popitem(last=False)
        return False

    async def on_event(
        self,
        group_id: str,
        event_type: str,
        payload: dict,
        event: dict,
    ) -> None:
        """EventBus on_publish_hook callback."""
        if not event_type.startswith("consolidation.phase."):
            return
        if not event_type.endswith(".completed"):
            return

        items_affected = payload.get("items_affected", 0)
        if items_affected == 0:
            return

        cycle_id = payload.get("cycle_id", "")
        phase = payload.get("phase", "")

        try:
            if phase == "dream" and self._cfg.notification_dream_enabled:
                await self._handle_dream(group_id, cycle_id)
            elif phase == "schema" and self._cfg.notification_schema_enabled:
                await self._handle_schema(group_id, cycle_id)
            elif phase == "mature" and self._cfg.notification_maturation_enabled:
                await self._handle_maturation(group_id, cycle_id)
            elif phase == "merge" and self._cfg.notification_merge_enabled:
                await self._handle_merge(group_id, cycle_id)
        except Exception:
            logger.debug("Notification collector error for %s", phase, exc_info=True)

    def _publish(self, notification: MemoryNotification) -> None:
        """Add to store and publish to EventBus."""
        self._store.add(notification)
        try:
            from engram.events.bus import get_event_bus

            bus = get_event_bus()
            bus.publish(
                notification.group_id,
                "notification.created",
                notification_to_dict(notification),
                _origin="notification_collector",
            )
        except Exception:
            pass

    async def _handle_dream(self, group_id: str, cycle_id: str) -> None:
        cs = self._consolidation_store
        if cs is None:
            return
        records = await cs.get_dream_association_records(cycle_id, group_id)
        if not records:
            return

        entity_ids = []
        lines: list[str] = []
        for r in records[:5]:
            entity_ids.extend([r.source_entity_id, r.target_entity_id])
            lines.append(
                f"{r.source_entity_name} ({r.source_domain}) <-> "
                f"{r.target_entity_name} ({r.target_domain}), "
                f"surprise={r.surprise_score:.2f}"
            )

        key = self._dedup_key("dream_association", cycle_id, entity_ids)
        if self._is_duplicate(key):
            return

        self._publish(
            MemoryNotification(
                group_id=group_id,
                notification_type="dream_association",
                priority="normal",
                title=f"{len(records)} cross-domain connection(s) discovered",
                body="\n".join(lines),
                entity_ids=entity_ids,
                metadata={"record_count": len(records)},
                source_cycle_id=cycle_id,
                created_at=time.time(),
            )
        )

    async def _handle_schema(self, group_id: str, cycle_id: str) -> None:
        cs = self._consolidation_store
        if cs is None:
            return
        records = await cs.get_schema_records(cycle_id, group_id)
        if not records:
            return

        entity_ids = [r.schema_entity_id for r in records]
        lines: list[str] = []
        for r in records[:5]:
            lines.append(
                f"{r.schema_name}: {r.instance_count} instances, "
                f"{r.predicate_count} predicates ({r.action})"
            )

        key = self._dedup_key("schema_discovery", cycle_id, entity_ids)
        if self._is_duplicate(key):
            return

        self._publish(
            MemoryNotification(
                group_id=group_id,
                notification_type="schema_discovery",
                priority="normal",
                title=f"{len(records)} structural pattern(s) detected",
                body="\n".join(lines),
                entity_ids=entity_ids,
                metadata={"record_count": len(records)},
                source_cycle_id=cycle_id,
                created_at=time.time(),
            )
        )

    async def _handle_maturation(self, group_id: str, cycle_id: str) -> None:
        cs = self._consolidation_store
        if cs is None:
            return
        records = await cs.get_maturation_records(cycle_id, group_id)
        if not records:
            return

        entity_ids = [r.entity_id for r in records]
        lines: list[str] = []
        for r in records[:5]:
            lines.append(f"{r.entity_name}: {r.old_tier} -> {r.new_tier}")
        if len(records) > 5:
            lines.append(f"...and {len(records) - 5} more")

        key = self._dedup_key("entity_maturation", cycle_id, entity_ids)
        if self._is_duplicate(key):
            return

        self._publish(
            MemoryNotification(
                group_id=group_id,
                notification_type="entity_maturation",
                priority="normal",
                title=f"{len(records)} entity(ies) graduated memory tier",
                body="\n".join(lines),
                entity_ids=entity_ids,
                metadata={"record_count": len(records)},
                source_cycle_id=cycle_id,
                created_at=time.time(),
            )
        )

    async def _handle_merge(self, group_id: str, cycle_id: str) -> None:
        cs = self._consolidation_store
        if cs is None:
            return
        records = await cs.get_merge_records(cycle_id, group_id)
        if not records:
            return

        entity_ids: list[str] = []
        lines: list[str] = []
        has_identity_core = False
        for r in records[:5]:
            entity_ids.extend([r.keep_id, r.remove_id])
            lines.append(f"'{r.remove_name}' merged into '{r.keep_name}'")
            if r.decision_reason and "identity_core" in r.decision_reason:
                has_identity_core = True
        if len(records) > 5:
            lines.append(f"...and {len(records) - 5} more")

        key = self._dedup_key("entity_merge", cycle_id, entity_ids)
        if self._is_duplicate(key):
            return

        self._publish(
            MemoryNotification(
                group_id=group_id,
                notification_type="entity_merge",
                priority="high" if has_identity_core else "normal",
                title=f"{len(records)} duplicate entity(ies) unified",
                body="\n".join(lines),
                entity_ids=entity_ids,
                metadata={"record_count": len(records), "identity_core": has_identity_core},
                source_cycle_id=cycle_id,
                created_at=time.time(),
            )
        )
