"""HelixDB implementation of GraphStore protocol."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from engram.config import HelixDBConfig
from engram.entity_dedup_policy import NameRegime, analyze_name, entity_identifier_facets
from engram.models.entity import Entity
from engram.models.episode import Attachment, Episode, EpisodeProjectionState, EpisodeStatus
from engram.models.episode_cue import EpisodeCue
from engram.models.relationship import Relationship
from engram.storage.protocols import ENTITY_UPDATABLE_FIELDS, EPISODE_UPDATABLE_FIELDS
from engram.utils.dates import utc_now, utc_now_iso
from engram.utils.text_guards import is_meta_summary

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from engram.models.prospective import Intention


def _parse_dt(value: str | None) -> datetime | None:
    """Parse an ISO 8601 datetime string or return None."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _dedup_summaries(existing: str, incoming: str, max_len: int = 500) -> str:
    """Merge two summaries, removing duplicate sentences via token-set Jaccard."""

    def _sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip() and len(s.strip()) > 5]

    def _tokens(sent: str) -> set[str]:
        return {w.lower() for w in re.findall(r"\b\w{3,}\b", sent)}

    existing_sents = _sentences(existing)
    incoming_sents = _sentences(incoming)

    if not incoming_sents:
        return existing.strip()

    existing_token_sets = [_tokens(s) for s in existing_sents]

    novel = []
    for sent in incoming_sents:
        sent_tokens = _tokens(sent)
        if not sent_tokens:
            continue
        is_dup = False
        for etokens in existing_token_sets:
            if not etokens:
                continue
            intersection = sent_tokens & etokens
            union = sent_tokens | etokens
            jaccard = len(intersection) / len(union) if union else 0
            if jaccard >= 0.6:
                is_dup = True
                break
        if not is_dup:
            novel.append(sent)
            existing_token_sets.append(sent_tokens)

    if not novel:
        result = existing.strip()
    else:
        novel_text = ". ".join(novel)
        result = f"{existing.strip()} {novel_text}".strip()

    if len(result) > max_len:
        result = result[: max_len - 3] + "..."
    return result


def _safe_get(d: dict, key: str, default: Any = None) -> Any:
    """Safely get a value from a dict returned by Helix."""
    v = d.get(key, default)
    return v if v is not None else default


def _evidence_dict_to_storage(d: dict) -> dict:
    """Convert a Helix Evidence node dict to the GraphStore evidence shape."""
    return {
        "evidence_id": d.get("evidence_id", ""),
        "episode_id": d.get("episode_id", ""),
        "group_id": d.get("group_id", "default"),
        "fact_class": d.get("fact_class", ""),
        "confidence": d.get("confidence", 0.0) or 0.0,
        "source_type": d.get("source_type", ""),
        "extractor_name": d.get("extractor_name", ""),
        "payload": json.loads(d.get("payload_json", "{}") or "{}"),
        "source_span": d.get("source_span"),
        "corroborating_signals": json.loads(d.get("signals_json", "[]") or "[]"),
        "ambiguity_tags": json.loads(d.get("ambiguity_tags_json", "[]") or "[]"),
        "ambiguity_score": d.get("ambiguity_score", 0.0) or 0.0,
        "adjudication_request_id": d.get("adjudication_request_id"),
        "status": d.get("status", "pending"),
        "commit_reason": d.get("commit_reason"),
        "committed_id": d.get("committed_id"),
        "deferred_cycles": d.get("deferred_cycles", 0) or 0,
        "created_at": d.get("created_at"),
        "resolved_at": d.get("resolved_at"),
    }


def _adjudication_dict_to_storage(d: dict) -> dict:
    """Convert a Helix AdjudicationRequest node dict to storage shape."""
    return {
        "request_id": d.get("request_id", ""),
        "episode_id": d.get("episode_id", ""),
        "group_id": d.get("group_id", "default"),
        "status": d.get("status", "pending"),
        "ambiguity_tags": json.loads(d.get("ambiguity_tags_json", "[]") or "[]"),
        "evidence_ids": json.loads(d.get("evidence_ids_json", "[]") or "[]"),
        "selected_text": d.get("selected_text", "") or "",
        "request_reason": d.get("request_reason", "") or "",
        "resolution_source": d.get("resolution_source"),
        "resolution_payload": json.loads(
            d.get("resolution_payload_json", "null") or "null",
        ),
        "attempt_count": d.get("attempt_count", 0) or 0,
        "created_at": d.get("created_at"),
        "resolved_at": d.get("resolved_at"),
    }


class HelixGraphStore:
    """Graph store backed by HelixDB — a combined graph + vector database.

    All Helix client calls are synchronous (HTTP POST via urllib). We wrap
    every call with ``asyncio.to_thread()`` to keep the event loop responsive.
    """

    def __init__(self, config: HelixDBConfig, encryptor=None, client=None) -> None:
        self._config = config
        self._encryptor = encryptor
        self._client: Any | None = None
        self._helix_client = client  # Shared HelixClient (async httpx)
        # entity_id (our UUID) -> Helix internal node ID
        self._entity_id_cache: dict[str, Any] = {}
        # episode_id (our UUID) -> Helix internal node ID
        self._episode_id_cache: dict[str, Any] = {}
        # rel_id (our UUID) -> Helix internal edge ID
        self._rel_id_cache: dict[str, Any] = {}
        # evidence_id -> Helix internal node ID
        self._evidence_id_cache: dict[str, Any] = {}
        # adjudication request_id -> Helix internal node ID
        self._adjudication_id_cache: dict[str, Any] = {}
        # intention_id -> Helix internal node ID
        self._intention_id_cache: dict[str, Any] = {}
        # cue episode_id -> Helix internal node ID
        self._cue_id_cache: dict[str, Any] = {}
        # schema_member key -> Helix internal node ID
        self._schema_member_id_cache: dict[str, Any] = {}

    async def _query(self, endpoint: str, payload: dict) -> list[dict]:
        """Execute a Helix query.

        Uses the shared async ``HelixClient`` (httpx with connection pooling)
        when available, falling back to the synchronous ``helix-py`` SDK
        via ``asyncio.to_thread()``.
        """
        # Fast path: shared async client with persistent connections
        if self._helix_client is not None:
            return await self._helix_client.query(endpoint, payload)

        # Legacy fallback: synchronous helix-py SDK
        client = self._client
        if client is None:
            raise RuntimeError("HelixDB graph store not initialized")
        try:
            result = await asyncio.to_thread(client.query, endpoint, payload)
            if result is None:
                return []
            from engram.storage.helix import unwrap_helix_results

            return unwrap_helix_results(result)
        except Exception as exc:
            exc_name = type(exc).__name__
            if "NoValue" in exc_name or "NotFound" in exc_name:
                return []
            raise

    # ------------------------------------------------------------------
    # Internal helpers: extract Helix internal ID from response
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_count(result: list[dict]) -> int:
        """Extract a count value from a Helix query result."""
        if not result:
            return 0
        item = result[0]
        if isinstance(item, dict):
            return item.get("count", 0)
        return 0

    @staticmethod
    def _extract_helix_id(item: dict) -> str | None:
        """Extract the Helix-assigned internal ID from a response dict.

        Helix v2 returns UUIDs (strings), v1 returned ints. We accept both.
        Response may be wrapped in a ``"node"`` or ``"edge"`` envelope.
        """
        # Unwrap envelope if present
        inner = item
        if "node" in item and isinstance(item["node"], dict):
            inner = item["node"]
        elif "edge" in item and isinstance(item["edge"], dict):
            inner = item["edge"]

        for key in ("id", "_id", "node_id", "edge_id"):
            if key in inner and inner[key] is not None:
                return inner[key]
        return None

    def _cache_entity(self, helix_id: Any, entity_id: str) -> None:
        if helix_id is not None:
            self._entity_id_cache[entity_id] = helix_id

    def _cache_episode(self, helix_id: Any, episode_id: str) -> None:
        if helix_id is not None:
            self._episode_id_cache[episode_id] = helix_id

    def _cache_rel(self, helix_id: Any, rel_id: str) -> None:
        if helix_id is not None:
            self._rel_id_cache[rel_id] = helix_id

    # ------------------------------------------------------------------
    # Helix ID resolution: given our UUID, find the Helix internal ID
    # ------------------------------------------------------------------

    async def _resolve_entity_helix_id(self, entity_id: str, group_id: str) -> int | None:
        """Resolve an entity UUID to a Helix internal ID via cache or query."""
        if entity_id in self._entity_id_cache:
            return self._entity_id_cache[entity_id]
        # Scan by group and filter
        results = await self._query(
            "find_entities_exact_name",
            {"name_exact": "__LOOKUP__", "gid": group_id},
        )
        # That won't find it. Instead, search all entities in group.
        results = await self._query("find_entities_by_group", {"gid": group_id})
        for item in results:
            eid = item.get("entity_id", "")
            hid = self._extract_helix_id(item)
            if hid is not None:
                self._entity_id_cache[eid] = hid
            if eid == entity_id:
                return hid
        return None

    async def _resolve_episode_helix_id(self, episode_id: str, group_id: str) -> int | None:
        """Resolve an episode UUID to a Helix internal ID via cache or query."""
        if episode_id in self._episode_id_cache:
            return self._episode_id_cache[episode_id]
        results = await self._query("find_episodes_by_group", {"gid": group_id})
        for item in results:
            eid = item.get("episode_id", "")
            hid = self._extract_helix_id(item)
            if hid is not None:
                self._episode_id_cache[eid] = hid
            if eid == episode_id:
                return hid
        return None

    async def _resolve_rel_helix_id(
        self, rel_id: str, source_helix_id: int
    ) -> int | None:
        """Resolve a relationship UUID to a Helix internal edge ID."""
        if rel_id in self._rel_id_cache:
            return self._rel_id_cache[rel_id]
        # Search outgoing edges from source
        results = await self._query("get_outgoing_edges", {"id": source_helix_id})
        for item in results:
            rid = item.get("rel_id", "")
            hid = self._extract_helix_id(item)
            if hid is not None:
                self._rel_id_cache[rid] = hid
            if rid == rel_id:
                return hid
        return None

    # ------------------------------------------------------------------
    # Encryption helpers
    # ------------------------------------------------------------------

    def _encrypt(self, group_id: str, plaintext: str | None) -> str | None:
        if not plaintext or not self._encryptor:
            return plaintext
        return cast(str | None, self._encryptor.encrypt(group_id, plaintext))

    def _decrypt(self, group_id: str, data: str | None) -> str | None:
        if not data or not self._encryptor:
            return data
        return cast(str | None, self._encryptor.decrypt(group_id, data))

    # ------------------------------------------------------------------
    # Model conversion helpers
    # ------------------------------------------------------------------

    def _dict_to_entity(self, d: dict, group_id: str | None = None) -> Entity:
        """Convert a Helix entity dict to an Entity model."""
        node_group = d.get("group_id", "default")
        decrypt_group = group_id or node_group
        summary = self._decrypt(decrypt_group, d.get("summary"))

        attributes = None
        raw_attrs = d.get("attributes_json")
        if raw_attrs:
            try:
                attributes = json.loads(raw_attrs)
            except (json.JSONDecodeError, TypeError):
                pass

        pii_categories = None
        raw_pii = d.get("pii_categories_json")
        if raw_pii:
            try:
                pii_categories = json.loads(raw_pii)
            except (json.JSONDecodeError, TypeError):
                pass

        # Cache the Helix ID
        hid = self._extract_helix_id(d)
        eid = d.get("entity_id", "")
        self._cache_entity(hid, eid)

        source_episode_ids: list[str] = []
        raw_source_eps = d.get("source_episode_ids")
        if raw_source_eps:
            try:
                source_episode_ids = json.loads(raw_source_eps)
            except (json.JSONDecodeError, TypeError):
                pass

        return Entity(
            id=eid,
            name=d.get("name", ""),
            entity_type=d.get("entity_type", ""),
            summary=summary,
            attributes=attributes,
            group_id=node_group,
            created_at=_parse_dt(d.get("created_at")) or utc_now(),
            updated_at=_parse_dt(d.get("updated_at")) or utc_now(),
            deleted_at=_parse_dt(d.get("deleted_at")),
            access_count=d.get("access_count", 0) or 0,
            last_accessed=_parse_dt(d.get("last_accessed")),
            pii_detected=bool(d.get("pii_detected", False)),
            pii_categories=pii_categories,
            identity_core=bool(d.get("identity_core", False)),
            mat_tier=d.get("mat_tier", "episodic") or "episodic",
            recon_count=d.get("recon_count", 0) or 0,
            lexical_regime=d.get("lexical_regime"),
            canonical_identifier=d.get("canonical_identifier"),
            identifier_label=bool(d.get("identifier_label", False)),
            source_episode_ids=source_episode_ids,
            evidence_count=d.get("evidence_count", 0) or 0,
            evidence_span_start=_parse_dt(d.get("evidence_span_start")),
            evidence_span_end=_parse_dt(d.get("evidence_span_end")),
        )

    def _dict_to_episode(self, d: dict, group_id: str | None = None) -> Episode:
        """Convert a Helix episode dict to an Episode model."""
        node_group = d.get("group_id", "default")
        decrypt_group = group_id or node_group
        content = self._decrypt(decrypt_group, d.get("content", "")) or ""

        hid = self._extract_helix_id(d)
        eid = d.get("episode_id", "")
        self._cache_episode(hid, eid)

        raw_status = d.get("status", "pending")
        status = (
            raw_status
            if isinstance(raw_status, EpisodeStatus)
            else EpisodeStatus(str(raw_status))
        )
        raw_proj = d.get("projection_state", "queued") or "queued"
        projection_state = (
            raw_proj
            if isinstance(raw_proj, EpisodeProjectionState)
            else EpisodeProjectionState(str(raw_proj))
        )

        return Episode(
            id=eid,
            content=content,
            source=d.get("source"),
            status=status,
            group_id=node_group,
            session_id=d.get("session_id"),
            conversation_date=_parse_dt(d.get("conversation_date")),
            created_at=_parse_dt(d.get("created_at")) or utc_now(),
            updated_at=_parse_dt(d.get("updated_at")),
            error=d.get("error") or None,
            retry_count=d.get("retry_count", 0) or 0,
            processing_duration_ms=d.get("processing_duration_ms"),
            encoding_context=d.get("encoding_context_json"),
            memory_tier=d.get("memory_tier", "episodic") or "episodic",
            consolidation_cycles=d.get("consolidation_cycles", 0) or 0,
            entity_coverage=d.get("entity_coverage", 0.0) or 0.0,
            projection_state=projection_state,
            last_projection_reason=d.get("last_projection_reason"),
            last_projected_at=_parse_dt(d.get("last_projected_at")),
            attachments=[
                Attachment(**a)
                for a in json.loads(d.get("attachments_json", "[]") or "[]")
            ],
        )

    def _dict_to_relationship(self, d: dict) -> Relationship:
        """Convert a Helix edge dict to a Relationship model."""
        hid = self._extract_helix_id(d)
        rid = d.get("rel_id", "")
        self._cache_rel(hid, rid)

        # Helix v2 edges have from_node/to_node (Helix internal IDs).
        # Reverse-map to our entity UUIDs via the ID cache.
        source_id = d.get("source_id", "")
        target_id = d.get("target_id", "")
        if not source_id:
            from_hid = d.get("from_node") or d.get("_from", "")
            for eid, cached_hid in self._entity_id_cache.items():
                if str(cached_hid) == str(from_hid):
                    source_id = eid
                    break
        if not target_id:
            to_hid = d.get("to_node") or d.get("_to", "")
            for eid, cached_hid in self._entity_id_cache.items():
                if str(cached_hid) == str(to_hid):
                    target_id = eid
                    break

        return Relationship(
            id=rid,
            source_id=source_id,
            target_id=target_id,
            predicate=d.get("predicate", ""),
            weight=d.get("weight", 1.0) or 1.0,
            valid_from=_parse_dt(d.get("valid_from")),
            valid_to=_parse_dt(d.get("valid_to")),
            created_at=_parse_dt(d.get("created_at")) or utc_now(),
            polarity=d.get("polarity", "positive") or "positive",
            source_episode=d.get("source_episode_id"),
            group_id=d.get("group_id", "default"),
        )

    def _dict_to_episode_cue(self, d: dict) -> EpisodeCue:
        """Convert a Helix EpisodeCue node dict to an EpisodeCue model."""
        hid = self._extract_helix_id(d)
        ep_id = d.get("episode_id", "")
        if hid is not None:
            self._cue_id_cache[ep_id] = hid

        return EpisodeCue(
            episode_id=ep_id,
            group_id=d.get("group_id", "default"),
            cue_text=d.get("cue_text", ""),
            projection_state=d.get("projection_state", "cued"),
            created_at=_parse_dt(d.get("created_at")) or utc_now(),
            updated_at=_parse_dt(d.get("updated_at")),
            entity_mentions=json.loads(d.get("supporting_spans_json", "[]") or "[]"),
        )

    @staticmethod
    def _dict_to_intention(d: dict) -> Intention:
        """Convert a Helix Intention node dict to an Intention model."""
        from engram.models.prospective import Intention

        ctx: dict = {}
        raw_ctx = d.get("context_json")
        if raw_ctx:
            try:
                ctx = json.loads(raw_ctx)
            except (json.JSONDecodeError, TypeError):
                pass

        return Intention(
            id=d.get("intention_id", ""),
            trigger_text=d.get("trigger_text", ""),
            action_text=d.get("action_text", ""),
            trigger_type=ctx.get("trigger_type", "semantic"),
            entity_name=ctx.get("entity_name"),
            threshold=ctx.get("threshold", 0.7),
            max_fires=d.get("max_fires", 5) or 5,
            fire_count=d.get("fire_count", 0) or 0,
            enabled=bool(d.get("enabled", True)),
            group_id=d.get("group_id", "default"),
            created_at=_parse_dt(d.get("created_at")) or utc_now(),
            updated_at=_parse_dt(d.get("updated_at")) or utc_now(),
            expires_at=_parse_dt(d.get("deleted_at")),
        )

    # ------------------------------------------------------------------
    # Temporal/active edge filtering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_active_edge(d: dict) -> bool:
        """Check if an edge dict represents an active (non-expired) edge."""
        if d.get("is_expired", False):
            return False
        valid_to = d.get("valid_to")
        if valid_to:
            dt = _parse_dt(valid_to)
            if dt and dt <= utc_now():
                return False
        return True

    @staticmethod
    def _is_expired_edge(d: dict) -> bool:
        """Check if an edge dict represents an expired edge."""
        if d.get("is_expired", False):
            return True
        valid_to = d.get("valid_to")
        if valid_to:
            dt = _parse_dt(valid_to)
            if dt and dt <= utc_now():
                return True
        return False

    @staticmethod
    def _edge_active_at(d: dict, at_time: datetime) -> bool:
        """Check if an edge was active at a specific point in time."""
        valid_from = d.get("valid_from")
        if valid_from:
            dt = _parse_dt(valid_from)
            if dt and dt > at_time:
                return False
        valid_to = d.get("valid_to")
        if valid_to:
            dt = _parse_dt(valid_to)
            if dt and dt <= at_time:
                return False
        return True

    # ==================================================================
    # GraphStore protocol implementation
    # ==================================================================

    async def initialize(self) -> None:
        """Connect to HelixDB."""
        # Use shared async client if provided, otherwise create one
        if self._helix_client is None:
            from engram.storage.helix.client import HelixClient

            self._helix_client = HelixClient(self._config)
        # Always ensure client is initialized
        if not self._helix_client.is_connected:
            await self._helix_client.initialize()

        # Also create legacy client for any code that references self._client
        # Skip for native transport — no HTTP server to connect to
        transport = getattr(self._config, "transport", "http")
        if transport != "native":
            try:
                from helix import Client  # type: ignore[import-untyped]

                kwargs: dict[str, Any] = {
                    "port": self._config.port,
                    "verbose": False,
                }
                if self._config.api_endpoint:
                    kwargs["url"] = self._config.api_endpoint
                    kwargs["local"] = False
                    if self._config.api_key:
                        kwargs["api_key"] = self._config.api_key
                else:
                    kwargs["local"] = True
                self._client = await asyncio.to_thread(Client, **kwargs)
            except ImportError:
                pass

        logger.info(
            "HelixDB graph store initialized (host=%s, port=%d)",
            self._config.host,
            self._config.port,
        )

    async def close(self) -> None:
        """No-op — Helix sync client has no close method."""
        self._client = None

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    async def create_entity(self, entity: Entity) -> str:
        now = utc_now_iso()
        summary = self._encrypt(entity.group_id, entity.summary)
        results = await self._query(
            "create_entity",
            {
                "entity_id": entity.id,
                "name": entity.name,
                "group_id": entity.group_id,
                "entity_type": entity.entity_type,
                "summary": summary or "",
                "attributes_json": json.dumps(entity.attributes) if entity.attributes else "{}",
                "created_at": entity.created_at.isoformat() if entity.created_at else now,
                "updated_at": now,
                "is_deleted": False,
                "deleted_at": "",
                "identity_core": entity.identity_core,
                "mat_tier": entity.mat_tier if hasattr(entity, "mat_tier") else "episodic",
                "recon_count": entity.recon_count if hasattr(entity, "recon_count") else 0,
                "lexical_regime": entity.lexical_regime or "",
                "canonical_identifier": entity.canonical_identifier or "",
                "identifier_label": "true" if entity.identifier_label else "",
                "pii_detected": bool(entity.pii_detected),
                "pii_categories_json": (
                    json.dumps(entity.pii_categories) if entity.pii_categories else "[]"
                ),
                "access_count": entity.access_count,
                "last_accessed": (
                    entity.last_accessed.isoformat() if entity.last_accessed else ""
                ),
                "source_episode_ids": json.dumps(entity.source_episode_ids),
                "evidence_count": entity.evidence_count,
                "evidence_span_start": (
                    entity.evidence_span_start.isoformat()
                    if entity.evidence_span_start
                    else ""
                ),
                "evidence_span_end": (
                    entity.evidence_span_end.isoformat()
                    if entity.evidence_span_end
                    else ""
                ),
            },
        )
        if results:
            hid = self._extract_helix_id(results[0])
            self._cache_entity(hid, entity.id)
        return entity.id

    async def get_entity(self, entity_id: str, group_id: str) -> Entity | None:
        helix_id = await self._resolve_entity_helix_id(entity_id, group_id)
        if helix_id is None:
            return None
        results = await self._query("get_entity", {"id": helix_id})
        if not results:
            return None
        d = results[0]
        if d.get("is_deleted", False):
            return None
        if d.get("group_id") != group_id:
            return None
        return self._dict_to_entity(d, group_id)

    async def batch_get_entities(
        self,
        entity_ids: list[str],
        group_id: str,
    ) -> dict[str, Entity]:
        if not entity_ids:
            return {}
        result: dict[str, Entity] = {}
        for eid in entity_ids:
            entity = await self.get_entity(eid, group_id)
            if entity is not None:
                result[entity.id] = entity
        return result

    async def update_entity(self, entity_id: str, updates: dict, group_id: str) -> None:
        if not updates:
            return
        if "name" in updates or "entity_type" in updates:
            current = await self.get_entity(entity_id, group_id)
            if current is not None:
                next_name = updates.get("name", current.name)
                facets = entity_identifier_facets(next_name)
                updates["lexical_regime"] = facets["lexical_regime"]
                updates["canonical_identifier"] = facets["canonical_identifier"]
                updates["identifier_label"] = "true" if facets["identifier_label"] else ""
        updates["updated_at"] = utc_now_iso()
        invalid = set(updates.keys()) - ENTITY_UPDATABLE_FIELDS
        if invalid:
            raise ValueError(f"Disallowed entity update fields: {invalid}")

        # Read-modify-write: fetch current entity, apply updates, write back
        helix_id = await self._resolve_entity_helix_id(entity_id, group_id)
        if helix_id is None:
            return
        results = await self._query("get_entity", {"id": helix_id})
        if not results:
            return
        current = results[0]

        # Apply updates
        for key, value in updates.items():
            if key == "attributes":
                current["attributes_json"] = json.dumps(value) if value else "{}"
            elif key == "pii_categories":
                current["pii_categories_json"] = json.dumps(value) if value else "[]"
            elif key == "source_episode_ids":
                current["source_episode_ids"] = (
                    json.dumps(value) if isinstance(value, list) else (value or "[]")
                )
            elif key == "deleted_at":
                dt_val = value.isoformat() if hasattr(value, "isoformat") else (value or "")
                current["deleted_at"] = dt_val
                current["is_deleted"] = bool(value)
            elif key in ("last_accessed", "evidence_span_start", "evidence_span_end"):
                current[key] = value.isoformat() if hasattr(value, "isoformat") else (value or "")
            else:
                current[key] = value

        await self._query(
            "update_entity_full",
            {
                "id": helix_id,
                "name": current.get("name", ""),
                "entity_type": current.get("entity_type", ""),
                "summary": current.get("summary", ""),
                "attributes_json": current.get("attributes_json", "{}"),
                "updated_at": current.get("updated_at", utc_now_iso()),
                "is_deleted": bool(current.get("is_deleted", False)),
                "deleted_at": current.get("deleted_at") or "",
                "identity_core": bool(current.get("identity_core", False)),
                "mat_tier": current.get("mat_tier") or "episodic",
                "recon_count": current.get("recon_count", 0),
                "lexical_regime": current.get("lexical_regime") or "",
                "canonical_identifier": current.get("canonical_identifier") or "",
                "identifier_label": current.get("identifier_label") or "",
                "pii_detected": bool(current.get("pii_detected", False)),
                "pii_categories_json": current.get("pii_categories_json", "[]"),
                "access_count": current.get("access_count", 0),
                "last_accessed": current.get("last_accessed", ""),
                "source_episode_ids": current.get("source_episode_ids", "[]"),
                "evidence_count": current.get("evidence_count", 0),
                "evidence_span_start": current.get("evidence_span_start", ""),
                "evidence_span_end": current.get("evidence_span_end", ""),
            },
        )

    async def delete_entity(self, entity_id: str, soft: bool = True, *, group_id: str) -> None:
        helix_id = await self._resolve_entity_helix_id(entity_id, group_id)
        if helix_id is None:
            return
        if soft:
            await self._query(
                "soft_delete_entity",
                {"id": helix_id, "deleted_at": utc_now_iso()},
            )
        else:
            # Delete connected edges first
            if self._helix_client is not None:
                out_edges, in_edges = await self._helix_client.query_concurrent([
                    ("get_outgoing_edges", {"id": helix_id}),
                    ("get_incoming_edges", {"id": helix_id}),
                ])
            else:
                out_edges = await self._query("get_outgoing_edges", {"id": helix_id})
                in_edges = await self._query("get_incoming_edges", {"id": helix_id})
            for edge in out_edges + in_edges:
                edge_hid = self._extract_helix_id(edge)
                if edge_hid is not None:
                    try:
                        await self._query("drop_edge", {"id": edge_hid})
                    except Exception:
                        pass
            await self._query("hard_delete_entity", {"id": helix_id})
            self._entity_id_cache.pop(entity_id, None)

    async def delete_group(self, group_id: str) -> None:
        """Delete all data belonging to a group by iterating and hard-deleting."""
        # Delete entities
        try:
            entities = await self._query("find_entities_by_group", {"gid": group_id})
            for ent in entities:
                hid = self._extract_helix_id(ent)
                if hid is not None:
                    try:
                        await self._query("hard_delete_entity", {"id": hid})
                    except Exception:
                        pass
        except Exception:
            logger.debug("delete_group: entities failed for %s", group_id, exc_info=True)

        # Delete episodes
        try:
            episodes = await self._query("find_episodes_by_group", {"gid": group_id})
            for ep in episodes:
                hid = self._extract_helix_id(ep)
                if hid is not None:
                    try:
                        await self._query("hard_delete_episode", {"id": hid})
                    except Exception:
                        pass
        except Exception:
            logger.debug("delete_group: episodes failed for %s", group_id, exc_info=True)

        # Delete intentions
        try:
            intentions = await self._query("find_intentions_by_group", {"gid": group_id})
            for it in intentions:
                hid = self._extract_helix_id(it)
                if hid is not None:
                    try:
                        await self._query("hard_delete_intention", {"id": hid})
                    except Exception:
                        pass
        except Exception:
            logger.debug("delete_group: intentions failed for %s", group_id, exc_info=True)

        # Clear caches
        self._entity_id_cache.clear()
        self._episode_id_cache.clear()
        self._rel_id_cache.clear()

    async def find_entities(
        self,
        name: str | None = None,
        entity_type: str | None = None,
        group_id: str | None = None,
        limit: int = 20,
    ) -> list[Entity]:
        if name and entity_type and group_id:
            results = await self._query(
                "find_entities_by_name_and_type",
                {"name_query": name, "etype": entity_type, "gid": group_id},
            )
        elif name and group_id:
            results = await self._query(
                "find_entities_by_name",
                {"name_query": name, "gid": group_id},
            )
        elif entity_type and group_id:
            results = await self._query(
                "find_entities_by_type",
                {"etype": entity_type, "gid": group_id},
            )
        elif group_id:
            results = await self._query("find_entities_by_group", {"gid": group_id})
        else:
            # No group_id — search across all groups (less efficient)
            # Helix requires a group_id for indexed queries, so fallback
            results = await self._query(
                "find_entities_by_group", {"gid": "default"}
            )

        entities = []
        for d in results:
            if name and name.lower() not in d.get("name", "").lower():
                continue
            if entity_type and d.get("entity_type") != entity_type:
                continue
            entities.append(self._dict_to_entity(d, group_id))
        # Sort by updated_at descending
        entities.sort(key=lambda e: e.updated_at or utc_now(), reverse=True)
        return entities[:limit]

    async def find_entity_candidates(
        self,
        name: str,
        group_id: str,
        limit: int = 30,
    ) -> list[Entity]:
        """Retrieve candidate entities for fuzzy resolution."""
        seen_ids: set[str] = set()
        results: list[Entity] = []
        form = analyze_name(name)

        # Phase 1: Exact name match
        exact_results = await self._query(
            "find_entities_exact_name",
            {"name_exact": name.strip(), "gid": group_id},
        )
        for d in exact_results:
            entity = self._dict_to_entity(d, group_id)
            if entity.id not in seen_ids:
                seen_ids.add(entity.id)
                results.append(entity)

        if len(results) >= limit:
            return results[:limit]

        # Phase 1.5: Canonical identifier match
        if form.canonical_code:
            canon_results = await self._query(
                "find_entities_by_canonical",
                {"canon": form.canonical_code, "gid": group_id},
            )
            for d in canon_results:
                entity = self._dict_to_entity(d, group_id)
                if entity.id not in seen_ids:
                    seen_ids.add(entity.id)
                    results.append(entity)

        if len(results) >= limit:
            return results[:limit]

        # Phase 2: CONTAINS match on full name
        contains_results = await self._query(
            "find_entities_by_name",
            {"name_query": name.strip(), "gid": group_id},
        )
        for d in contains_results:
            entity = self._dict_to_entity(d, group_id)
            if entity.id not in seen_ids:
                seen_ids.add(entity.id)
                results.append(entity)

        if len(results) >= limit:
            return results[:limit]

        # Phase 3: Token fallback — search individual tokens >= 3 chars
        tokens = [t for t in name.strip().split() if len(t) >= 3]
        if form.regime != NameRegime.NATURAL_LANGUAGE:
            tokens = []
        for token in tokens:
            if len(results) >= limit:
                break
            token_results = await self._query(
                "find_entities_by_name",
                {"name_query": token, "gid": group_id},
            )
            for d in token_results:
                entity = self._dict_to_entity(d, group_id)
                if entity.id not in seen_ids:
                    seen_ids.add(entity.id)
                    results.append(entity)

        return results[:limit]

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    async def create_relationship(self, rel: Relationship) -> str:
        source_hid = await self._resolve_entity_helix_id(rel.source_id, rel.group_id)
        target_hid = await self._resolve_entity_helix_id(rel.target_id, rel.group_id)
        if source_hid is None or target_hid is None:
            logger.warning(
                "create_relationship: could not resolve entity Helix IDs for %s -> %s",
                rel.source_id,
                rel.target_id,
            )
            return rel.id

        results = await self._query(
            "create_relationship",
            {
                "rel_id": rel.id,
                "group_id": rel.group_id,
                "predicate": rel.predicate,
                "weight": rel.weight,
                "polarity": rel.polarity,
                "valid_from": rel.valid_from.isoformat() if rel.valid_from else "",
                "valid_to": rel.valid_to.isoformat() if rel.valid_to else "",
                "is_expired": False,
                "created_at": rel.created_at.isoformat() if rel.created_at else utc_now_iso(),
                "source_episode_id": rel.source_episode or "",
                "source_id": source_hid,
                "target_id": target_hid,
            },
        )
        if results:
            hid = self._extract_helix_id(results[0])
            self._cache_rel(hid, rel.id)
        return rel.id

    async def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        predicate: str | None = None,
        active_only: bool = True,
        group_id: str = "default",
    ) -> list[Relationship]:
        helix_id = await self._resolve_entity_helix_id(entity_id, group_id)
        if helix_id is None:
            return []

        edges: list[dict] = []
        if direction == "both" and self._helix_client is not None:
            # Fire outgoing + incoming queries concurrently
            if predicate:
                out_results, in_results = await self._helix_client.query_concurrent([
                    ("get_outgoing_edges_by_predicate", {"id": helix_id, "pred": predicate}),
                    ("get_incoming_edges_by_predicate", {"id": helix_id, "pred": predicate}),
                ])
            else:
                out_results, in_results = await self._helix_client.query_concurrent([
                    ("get_outgoing_edges", {"id": helix_id}),
                    ("get_incoming_edges", {"id": helix_id}),
                ])
            edges.extend(out_results)
            edges.extend(in_results)
        else:
            if direction in ("outgoing", "both"):
                if predicate:
                    edges.extend(
                        await self._query(
                            "get_outgoing_edges_by_predicate",
                            {"id": helix_id, "pred": predicate},
                        )
                    )
                else:
                    edges.extend(await self._query("get_outgoing_edges", {"id": helix_id}))
            if direction in ("incoming", "both"):
                if predicate:
                    edges.extend(
                        await self._query(
                            "get_incoming_edges_by_predicate",
                            {"id": helix_id, "pred": predicate},
                        )
                    )
                else:
                    edges.extend(await self._query("get_incoming_edges", {"id": helix_id}))

        rels: list[Relationship] = []
        seen_ids: set[str] = set()
        for edge in edges:
            if edge.get("group_id") != group_id:
                continue
            if active_only and not self._is_active_edge(edge):
                continue
            rel = self._dict_to_relationship(edge)
            if rel.id not in seen_ids:
                seen_ids.add(rel.id)
                rels.append(rel)
        return rels

    async def invalidate_relationship(
        self, rel_id: str, valid_to: datetime, group_id: str
    ) -> None:
        # We need the Helix edge ID. Try to find it.
        helix_id = self._rel_id_cache.get(rel_id)
        if helix_id is None:
            # Search for it across all entities in the group
            all_entities = await self._query("find_entities_by_group", {"gid": group_id})
            for ent in all_entities:
                ehid = self._extract_helix_id(ent)
                if ehid is None:
                    continue
                for edge in await self._query("get_outgoing_edges", {"id": ehid}):
                    rid = edge.get("rel_id", "")
                    edge_hid = self._extract_helix_id(edge)
                    if edge_hid is not None:
                        self._rel_id_cache[rid] = edge_hid
                    if rid == rel_id:
                        helix_id = edge_hid
                        break
                if helix_id is not None:
                    break

        if helix_id is None:
            logger.warning("invalidate_relationship: could not find Helix ID for %s", rel_id)
            return

        await self._query(
            "invalidate_edge",
            {"id": helix_id, "valid_to": valid_to.isoformat()},
        )

    async def find_conflicting_relationships(
        self,
        source_id: str,
        predicate: str,
        group_id: str,
    ) -> list[Relationship]:
        helix_id = await self._resolve_entity_helix_id(source_id, group_id)
        if helix_id is None:
            return []
        edges = await self._query(
            "get_outgoing_edges_by_predicate",
            {"id": helix_id, "pred": predicate},
        )
        rels = []
        for edge in edges:
            if edge.get("group_id") != group_id:
                continue
            # Only active (no valid_to)
            if edge.get("valid_to") and edge.get("valid_to") != "":
                continue
            rels.append(self._dict_to_relationship(edge))
        return rels

    async def find_existing_relationship(
        self,
        source_id: str,
        target_id: str,
        predicate: str,
        group_id: str,
    ) -> Relationship | None:
        helix_id = await self._resolve_entity_helix_id(source_id, group_id)
        if helix_id is None:
            return None
        edges = await self._query(
            "get_outgoing_edges_by_predicate",
            {"id": helix_id, "pred": predicate},
        )
        for edge in edges:
            if edge.get("group_id") != group_id:
                continue
            if not self._is_active_edge(edge):
                continue
            rel = self._dict_to_relationship(edge)
            if rel.target_id == target_id:
                return rel
        return None

    async def get_relationships_at(
        self,
        entity_id: str,
        at_time: datetime,
        direction: str = "both",
        group_id: str = "default",
    ) -> list[Relationship]:
        helix_id = await self._resolve_entity_helix_id(entity_id, group_id)
        if helix_id is None:
            return []

        edges: list[dict] = []
        if direction == "both" and self._helix_client is not None:
            out_results, in_results = await self._helix_client.query_concurrent([
                ("get_outgoing_edges", {"id": helix_id}),
                ("get_incoming_edges", {"id": helix_id}),
            ])
            edges.extend(out_results)
            edges.extend(in_results)
        else:
            if direction in ("outgoing", "both"):
                edges.extend(await self._query("get_outgoing_edges", {"id": helix_id}))
            if direction in ("incoming", "both"):
                edges.extend(await self._query("get_incoming_edges", {"id": helix_id}))

        rels: list[Relationship] = []
        seen_ids: set[str] = set()
        for edge in edges:
            if edge.get("group_id") != group_id:
                continue
            if not self._edge_active_at(edge, at_time):
                continue
            rel = self._dict_to_relationship(edge)
            if rel.id not in seen_ids:
                seen_ids.add(rel.id)
                rels.append(rel)
        return rels

    async def get_neighbors(
        self,
        entity_id: str,
        hops: int = 1,
        group_id: str | None = None,
        max_results: int = 5000,
    ) -> list[tuple[Entity, Relationship]]:
        """Return entities within N hops via iterative BFS."""
        seen_entities: set[str] = {entity_id}
        seen_edges: set[str] = set()
        frontier_helix_ids: set[int] = set()

        start_hid = await self._resolve_entity_helix_id(
            entity_id, group_id or "default"
        )
        if start_hid is None:
            return []
        frontier_helix_ids.add(start_hid)

        results: list[tuple[Entity, Relationship]] = []

        for _hop in range(hops):
            if not frontier_helix_ids:
                break

            next_frontier: set[int] = set()
            for hid in frontier_helix_ids:
                # Fetch all four queries concurrently when shared client available
                if self._helix_client is not None:
                    out_edges, out_neighbors, in_edges, in_neighbors = (
                        await self._helix_client.query_concurrent([
                            ("get_outgoing_edges", {"id": hid}),
                            ("get_outgoing_neighbors", {"id": hid}),
                            ("get_incoming_edges", {"id": hid}),
                            ("get_incoming_neighbors", {"id": hid}),
                        ])
                    )
                else:
                    out_edges = await self._query("get_outgoing_edges", {"id": hid})
                    out_neighbors = await self._query("get_outgoing_neighbors", {"id": hid})
                    in_edges = await self._query("get_incoming_edges", {"id": hid})
                    in_neighbors = await self._query("get_incoming_neighbors", {"id": hid})

                # Process outgoing
                for edge, neighbor in zip(out_edges, out_neighbors):
                    if not self._is_active_edge(edge):
                        continue
                    if group_id and edge.get("group_id") != group_id:
                        continue
                    if neighbor.get("is_deleted", False):
                        continue
                    if group_id and neighbor.get("group_id") != group_id:
                        continue
                    rel = self._dict_to_relationship(edge)
                    entity = self._dict_to_entity(neighbor, group_id)
                    if rel.id not in seen_edges:
                        seen_edges.add(rel.id)
                        results.append((entity, rel))
                    if entity.id not in seen_entities:
                        seen_entities.add(entity.id)
                        nhid = self._extract_helix_id(neighbor)
                        if nhid is not None:
                            next_frontier.add(nhid)
                    if len(results) >= max_results:
                        break

                if len(results) >= max_results:
                    break

                # Process incoming
                for edge, neighbor in zip(in_edges, in_neighbors):
                    if not self._is_active_edge(edge):
                        continue
                    if group_id and edge.get("group_id") != group_id:
                        continue
                    if neighbor.get("is_deleted", False):
                        continue
                    if group_id and neighbor.get("group_id") != group_id:
                        continue
                    rel = self._dict_to_relationship(edge)
                    entity = self._dict_to_entity(neighbor, group_id)
                    if rel.id not in seen_edges:
                        seen_edges.add(rel.id)
                        results.append((entity, rel))
                    if entity.id not in seen_entities:
                        seen_entities.add(entity.id)
                        nhid = self._extract_helix_id(neighbor)
                        if nhid is not None:
                            next_frontier.add(nhid)
                    if len(results) >= max_results:
                        break

                if len(results) >= max_results:
                    break

            frontier_helix_ids = next_frontier

        return results

    async def get_all_edges(
        self,
        group_id: str,
        entity_ids: set[str] | None = None,
        limit: int = 10000,
    ) -> list[Relationship]:
        all_entities = await self._query("find_entities_by_group", {"gid": group_id})

        rels: list[Relationship] = []
        seen_ids: set[str] = set()
        for ent in all_entities:
            ehid = self._extract_helix_id(ent)
            if ehid is None:
                continue
            eid = ent.get("entity_id", "")
            if entity_ids is not None and eid not in entity_ids:
                continue
            edges = await self._query("get_outgoing_edges", {"id": ehid})
            for edge in edges:
                if edge.get("group_id") != group_id:
                    continue
                if not self._is_active_edge(edge):
                    continue
                rel = self._dict_to_relationship(edge)
                if entity_ids is not None and rel.target_id not in entity_ids:
                    continue
                if rel.id not in seen_ids:
                    seen_ids.add(rel.id)
                    rels.append(rel)
                    if len(rels) >= limit:
                        return rels
        return rels

    async def get_active_neighbors_with_weights(
        self, entity_id: str, group_id: str | None = None
    ) -> list[tuple[str, float, str, str]]:
        helix_id = await self._resolve_entity_helix_id(
            entity_id, group_id or "default"
        )
        if helix_id is None:
            return []

        results: list[tuple[str, float, str, str]] = []
        seen: set[str] = set()

        # Fetch all four queries concurrently when shared client available
        if self._helix_client is not None:
            out_edges, out_neighbors, in_edges, in_neighbors = (
                await self._helix_client.query_concurrent([
                    ("get_outgoing_edges", {"id": helix_id}),
                    ("get_outgoing_neighbors", {"id": helix_id}),
                    ("get_incoming_edges", {"id": helix_id}),
                    ("get_incoming_neighbors", {"id": helix_id}),
                ])
            )
            all_pairs = list(zip(out_edges, out_neighbors)) + list(zip(in_edges, in_neighbors))
        else:
            out_edges = await self._query("get_outgoing_edges", {"id": helix_id})
            out_neighbors = await self._query("get_outgoing_neighbors", {"id": helix_id})
            in_edges = await self._query("get_incoming_edges", {"id": helix_id})
            in_neighbors = await self._query("get_incoming_neighbors", {"id": helix_id})
            all_pairs = list(zip(out_edges, out_neighbors)) + list(zip(in_edges, in_neighbors))

        for edge, nbr in all_pairs:
            if not self._is_active_edge(edge):
                continue
            if group_id and edge.get("group_id") != group_id:
                continue
            if nbr.get("is_deleted", False):
                continue

            polarity = edge.get("polarity", "positive") or "positive"
            if polarity == "negative":
                continue

            nbr_id = nbr.get("entity_id", "")
            if nbr_id == entity_id:
                continue
            if nbr_id in seen:
                continue
            seen.add(nbr_id)

            weight = edge.get("weight", 1.0) or 1.0
            if polarity == "uncertain":
                weight *= 0.5

            results.append((
                nbr_id,
                weight,
                edge.get("predicate", ""),
                nbr.get("entity_type", ""),
            ))

        return results

    # ------------------------------------------------------------------
    # Episodes
    # ------------------------------------------------------------------

    async def create_episode(self, episode: Episode) -> str:
        content = self._encrypt(episode.group_id, episode.content)
        now_iso = utc_now_iso()
        status_val = episode.status.value if hasattr(episode.status, "value") else episode.status
        proj_val = (
            episode.projection_state.value
            if hasattr(episode.projection_state, "value")
            else episode.projection_state
        )

        results = await self._query(
            "create_episode",
            {
                "episode_id": episode.id,
                "group_id": episode.group_id,
                "content": content or "",
                "source": episode.source or "",
                "session_id": episode.session_id or "",
                "status": status_val,
                "created_at": episode.created_at.isoformat() if episode.created_at else now_iso,
                "updated_at": episode.updated_at.isoformat() if episode.updated_at else now_iso,
                "error": episode.error or "",
                "retry_count": episode.retry_count,
                "processing_duration_ms": episode.processing_duration_ms or 0,
                "skipped_meta": False,
                "skipped_triage": False,
                "encoding_context_json": episode.encoding_context or "{}",
                "memory_tier": episode.memory_tier or "episodic",
                "consolidation_cycles": episode.consolidation_cycles,
                "entity_coverage": episode.entity_coverage,
                "projection_state": proj_val,
                "last_projection_reason": episode.last_projection_reason or "",
                "last_projected_at": (
                    episode.last_projected_at.isoformat() if episode.last_projected_at else ""
                ),
                "conversation_date": (
                    episode.conversation_date.isoformat() if episode.conversation_date else ""
                ),
                "attachments_json": (
                    json.dumps([a.model_dump() for a in episode.attachments])
                    if episode.attachments
                    else "[]"
                ),
            },
        )
        if results:
            hid = self._extract_helix_id(results[0])
            self._cache_episode(hid, episode.id)
        return episode.id

    async def update_episode(
        self,
        episode_id: str,
        updates: dict,
        group_id: str = "default",
    ) -> None:
        if not updates:
            return
        updates["updated_at"] = utc_now_iso()
        invalid = set(updates.keys()) - EPISODE_UPDATABLE_FIELDS
        if invalid:
            raise ValueError(f"Disallowed episode update fields: {invalid}")

        helix_id = await self._resolve_episode_helix_id(episode_id, group_id)
        if helix_id is None:
            return
        results = await self._query("get_episode", {"id": helix_id})
        if not results:
            return
        current = results[0]

        # Apply updates
        for key, value in updates.items():
            if key == "status" and hasattr(value, "value"):
                current["status"] = value.value
            elif key == "projection_state" and hasattr(value, "value"):
                current["projection_state"] = value.value
            elif key == "attachments_json":
                current["attachments_json"] = value or "[]"
            elif key == "encoding_context":
                current["encoding_context_json"] = value or "{}"
            elif key == "last_projected_at":
                current[key] = value.isoformat() if hasattr(value, "isoformat") else (value or "")
            elif key == "conversation_date":
                current[key] = value.isoformat() if hasattr(value, "isoformat") else (value or "")
            else:
                current[key] = value

        await self._query(
            "update_episode_full",
            {
                "id": helix_id,
                "status": current.get("status", "pending"),
                "updated_at": current.get("updated_at", utc_now_iso()),
                "error": current.get("error", ""),
                "retry_count": current.get("retry_count", 0),
                "processing_duration_ms": current.get("processing_duration_ms", 0) or 0,
                "content": current.get("content", ""),
                "skipped_meta": current.get("skipped_meta", False),
                "skipped_triage": current.get("skipped_triage", False),
                "encoding_context_json": current.get("encoding_context_json", "{}"),
                "memory_tier": current.get("memory_tier", "episodic"),
                "consolidation_cycles": current.get("consolidation_cycles", 0),
                "entity_coverage": current.get("entity_coverage", 0.0),
                "projection_state": current.get("projection_state", "queued"),
                "last_projection_reason": current.get("last_projection_reason", ""),
                "last_projected_at": current.get("last_projected_at", ""),
                "conversation_date": current.get("conversation_date", ""),
                "attachments_json": current.get("attachments_json", "[]"),
            },
        )

    async def get_episodes(
        self,
        group_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Episode]:
        if group_id:
            results = await self._query("find_episodes_by_group", {"gid": group_id})
        else:
            results = await self._query("find_episodes_by_group", {"gid": "default"})
        episodes = [self._dict_to_episode(d, group_id) for d in results]
        episodes.sort(key=lambda e: e.created_at or utc_now(), reverse=True)
        return episodes[offset : offset + limit]

    async def get_episode_by_id(self, episode_id: str, group_id: str) -> Episode | None:
        helix_id = await self._resolve_episode_helix_id(episode_id, group_id)
        if helix_id is None:
            return None
        results = await self._query("get_episode", {"id": helix_id})
        if not results:
            return None
        d = results[0]
        if d.get("group_id") != group_id:
            return None
        return self._dict_to_episode(d, group_id)

    async def get_episode_entities(self, episode_id: str) -> list[str]:
        # Need the Helix ID for the episode
        # Try cache first, then scan default group
        helix_id = self._episode_id_cache.get(episode_id)
        if helix_id is None:
            # Try to find it
            for gid in ("default",):
                helix_id = await self._resolve_episode_helix_id(episode_id, gid)
                if helix_id is not None:
                    break
        if helix_id is None:
            return []
        results = await self._query("get_episode_entities", {"id": helix_id})
        return [d.get("entity_id", "") for d in results if d.get("entity_id")]

    async def get_episodes_for_entity(
        self,
        entity_id: str,
        group_id: str = "default",
        limit: int = 20,
    ) -> list[str]:
        helix_id = await self._resolve_entity_helix_id(entity_id, group_id)
        if helix_id is None:
            return []
        results = await self._query("get_episodes_for_entity", {"id": helix_id})
        episodes = []
        for d in results:
            if d.get("group_id") == group_id:
                episodes.append(d)
        # Sort by created_at desc
        episodes.sort(key=lambda d: d.get("created_at", ""), reverse=True)
        return [d.get("episode_id", "") for d in episodes[:limit] if d.get("episode_id")]

    async def get_adjacent_episodes(
        self,
        episode_id: str,
        group_id: str,
        limit: int = 3,
    ) -> list[Episode]:
        """Get temporally adjacent episodes from the same session."""
        ep = await self.get_episode_by_id(episode_id, group_id)
        if not ep or not ep.session_id:
            return []
        results = await self._query(
            "find_episodes_by_session", {"sid": ep.session_id}
        )
        episodes = []
        for d in results:
            if d.get("episode_id") == episode_id:
                continue
            if d.get("group_id") != group_id:
                continue
            episodes.append(self._dict_to_episode(d, group_id))
        # Sort by temporal proximity
        ref_time = ep.created_at or utc_now()
        episodes.sort(key=lambda e: abs((e.created_at - ref_time).total_seconds()))
        return episodes[:limit]

    async def link_episode_entity(self, episode_id: str, entity_id: str) -> None:
        ep_hid = self._episode_id_cache.get(episode_id)
        ent_hid = self._entity_id_cache.get(entity_id)

        if ep_hid is None:
            for gid in ("default",):
                ep_hid = await self._resolve_episode_helix_id(episode_id, gid)
                if ep_hid is not None:
                    break
        if ent_hid is None:
            for gid in ("default",):
                ent_hid = await self._resolve_entity_helix_id(entity_id, gid)
                if ent_hid is not None:
                    break

        if ep_hid is None or ent_hid is None:
            logger.warning(
                "link_episode_entity: could not resolve Helix IDs for ep=%s, ent=%s",
                episode_id,
                entity_id,
            )
            return
        await self._query(
            "link_episode_entity",
            {"episode_id": ep_hid, "entity_id": ent_hid},
        )

    # ------------------------------------------------------------------
    # Episode Cues
    # ------------------------------------------------------------------

    async def upsert_episode_cue(self, cue: EpisodeCue) -> None:
        # Check if cue already exists
        existing = await self._query(
            "find_cue_by_episode",
            {"ep_id": cue.episode_id, "gid": cue.group_id},
        )
        proj_val = (
            cue.projection_state.value
            if hasattr(cue.projection_state, "value")
            else cue.projection_state
        )
        now = utc_now_iso()

        if existing:
            hid = self._extract_helix_id(existing[0])
            if hid is not None:
                self._cue_id_cache[cue.episode_id] = hid
                await self._query(
                    "update_cue",
                    {
                        "id": hid,
                        "cue_text": cue.cue_text,
                        "supporting_spans_json": json.dumps(cue.entity_mentions),
                        "projection_state": proj_val,
                        "updated_at": cue.updated_at.isoformat() if cue.updated_at else now,
                    },
                )
                return

        results = await self._query(
            "create_episode_cue",
            {
                "episode_id": cue.episode_id,
                "group_id": cue.group_id,
                "cue_text": cue.cue_text,
                "supporting_spans_json": json.dumps(cue.entity_mentions),
                "projection_state": proj_val,
                "created_at": cue.created_at.isoformat() if cue.created_at else now,
                "updated_at": cue.updated_at.isoformat() if cue.updated_at else now,
            },
        )
        if results:
            hid = self._extract_helix_id(results[0])
            if hid is not None:
                self._cue_id_cache[cue.episode_id] = hid

    async def get_episode_cue(self, episode_id: str, group_id: str) -> EpisodeCue | None:
        results = await self._query(
            "find_cue_by_episode",
            {"ep_id": episode_id, "gid": group_id},
        )
        if not results:
            return None
        d = results[0]
        if not d.get("cue_text"):
            return None
        return self._dict_to_episode_cue(d)

    async def update_episode_cue(
        self,
        episode_id: str,
        updates: dict,
        group_id: str = "default",
    ) -> None:
        if not updates:
            return
        existing = await self._query(
            "find_cue_by_episode",
            {"ep_id": episode_id, "gid": group_id},
        )
        if not existing:
            return
        hid = self._extract_helix_id(existing[0])
        if hid is None:
            return
        current = existing[0]

        # Apply updates
        cue_text = updates.get("cue_text", current.get("cue_text", ""))
        spans_json = current.get("supporting_spans_json", "[]")
        if "entity_mentions" in updates:
            spans_json = json.dumps(updates["entity_mentions"])
        proj_state = current.get("projection_state", "cued")
        if "projection_state" in updates:
            v = updates["projection_state"]
            proj_state = v.value if hasattr(v, "value") else v
        now = utc_now_iso()

        await self._query(
            "update_cue",
            {
                "id": hid,
                "cue_text": cue_text,
                "supporting_spans_json": spans_json,
                "projection_state": proj_state,
                "updated_at": now,
            },
        )

    # ------------------------------------------------------------------
    # Stats & Analytics
    # ------------------------------------------------------------------

    async def get_stats(self, group_id: str | None = None) -> dict:
        gid = group_id or "default"

        # Fetch entities and episodes concurrently
        if self._helix_client is not None:
            entities, episodes = await self._helix_client.query_concurrent([
                ("find_entities_by_group", {"gid": gid}),
                ("find_episodes_by_group", {"gid": gid}),
            ])
        else:
            entities = await self._query("find_entities_by_group", {"gid": gid})
            episodes = await self._query("find_episodes_by_group", {"gid": gid})

        entity_count = len(entities)

        # Relationship count is expensive (per-entity edge scan); approximate from first few
        # Parallelize the edge count queries for the sample entities
        sample_ents = entities[:10]
        sample_hids = [self._extract_helix_id(ent) for ent in sample_ents]
        sample_hids = [h for h in sample_hids if h is not None]

        relationship_count = 0
        if sample_hids and self._helix_client is not None:
            edge_results = await self._helix_client.query_concurrent([
                ("get_outgoing_edges", {"id": hid}) for hid in sample_hids
            ])
            for edges in edge_results:
                relationship_count += len(edges)
        else:
            for hid in sample_hids:
                edges = await self._query("get_outgoing_edges", {"id": hid})
                relationship_count += len(edges)
        if entity_count > 10 and sample_hids:
            relationship_count = int(relationship_count * entity_count / 10)

        episode_count = 0

        projection_counts: dict[str, Any] = defaultdict(int)
        total_duration = 0.0
        duration_count = 0
        for ep in episodes:
            ps = ep.get("projection_state", "queued")
            projection_counts[ps] += 1
            dur = ep.get("processing_duration_ms")
            if ps == "projected" and dur:
                total_duration += dur
                duration_count += 1

        attempted = (
            projection_counts.get("projected", 0)
            + projection_counts.get("failed", 0)
            + projection_counts.get("dead_letter", 0)
        )

        cue_metrics = {
            "cue_count": 0,
            "episodes_without_cues": episode_count,
            "cue_coverage": 0.0,
            "cue_hit_count": 0,
            "cue_hit_episode_count": 0,
            "cue_hit_episode_rate": 0.0,
            "cue_surfaced_count": 0,
            "cue_selected_count": 0,
            "cue_used_count": 0,
            "cue_near_miss_count": 0,
            "avg_policy_score": 0.0,
            "avg_projection_attempts": 0.0,
            "projected_cue_count": 0,
            "cue_to_projection_conversion_rate": 0.0,
        }
        projection_metrics = {
            "state_counts": dict(projection_counts),
            "attempted_episode_count": attempted,
            "total_attempts": 0,
            "failure_count": projection_counts.get("failed", 0),
            "dead_letter_count": projection_counts.get("dead_letter", 0),
            "failure_rate": (
                round(
                    (projection_counts.get("failed", 0) + projection_counts.get("dead_letter", 0))
                    / attempted,
                    4,
                )
                if attempted
                else 0.0
            ),
            "avg_processing_duration_ms": (
                round(total_duration / duration_count, 2) if duration_count else 0.0
            ),
            "avg_time_to_projection_ms": 0.0,
            "yield": {
                "linked_entity_count": 0,
                "relationship_count": 0,
                "avg_linked_entities_per_projected_episode": 0.0,
                "avg_relationships_per_projected_episode": 0.0,
            },
        }

        return {
            "entities": entity_count,
            "relationships": relationship_count,
            "episodes": episode_count,
            "cue_metrics": cue_metrics,
            "projection_metrics": projection_metrics,
        }

    async def get_episodes_paginated(
        self,
        group_id: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
        source: str | None = None,
        status: str | None = None,
    ) -> tuple[list[Episode], str | None]:
        gid = group_id or "default"

        if source:
            all_eps = await self._query("find_episodes_by_source", {"gid": gid, "src": source})
        elif status:
            all_eps = await self._query("find_episodes_by_status", {"gid": gid, "st": status})
        else:
            all_eps = await self._query("find_episodes_by_group", {"gid": gid})

        episodes = [self._dict_to_episode(d, group_id) for d in all_eps]
        # Apply additional filters
        if source and status:
            episodes = [e for e in episodes if e.source == source and e.status.value == status]
        elif status and not source:
            # Already filtered by query
            pass
        elif source and not status:
            pass

        episodes.sort(key=lambda e: e.created_at or utc_now(), reverse=True)

        # Apply cursor
        if cursor:
            episodes = [e for e in episodes if e.created_at.isoformat() < cursor]

        result_eps = episodes[: limit]
        next_cursor = None
        if len(episodes) > limit:
            next_cursor = result_eps[-1].created_at.isoformat() if result_eps else None

        return result_eps, next_cursor

    async def get_top_connected(
        self, group_id: str | None = None, limit: int = 10
    ) -> list[dict]:
        gid = group_id or "default"
        all_entities = await self._query("find_entities_by_group", {"gid": gid})

        # Collect valid Helix IDs
        valid_ents: list[tuple[dict, Any]] = []
        for ent in all_entities:
            ehid = self._extract_helix_id(ent)
            if ehid is not None:
                valid_ents.append((ent, ehid))

        entity_edge_counts: list[dict] = []
        if valid_ents and self._helix_client is not None:
            # Fire all outgoing + incoming queries concurrently
            queries = []
            for _ent, ehid in valid_ents:
                queries.append(("get_outgoing_edges", {"id": ehid}))
                queries.append(("get_incoming_edges", {"id": ehid}))
            all_results = await self._helix_client.query_concurrent(queries)
            for i, (ent, _ehid) in enumerate(valid_ents):
                out_edges = all_results[i * 2]
                in_edges = all_results[i * 2 + 1]
                active_count = 0
                for e in out_edges + in_edges:
                    if self._is_active_edge(e) and e.get("group_id") == gid:
                        active_count += 1
                entity_edge_counts.append({
                    "id": ent.get("entity_id", ""),
                    "name": ent.get("name", ""),
                    "entityType": ent.get("entity_type", ""),
                    "edgeCount": active_count,
                })
        else:
            for ent, ehid in valid_ents:
                out_edges = await self._query("get_outgoing_edges", {"id": ehid})
                in_edges = await self._query("get_incoming_edges", {"id": ehid})
                active_count = 0
                for e in out_edges + in_edges:
                    if self._is_active_edge(e) and e.get("group_id") == gid:
                        active_count += 1
                entity_edge_counts.append({
                    "id": ent.get("entity_id", ""),
                    "name": ent.get("name", ""),
                    "entityType": ent.get("entity_type", ""),
                    "edgeCount": active_count,
                })

        entity_edge_counts.sort(key=lambda x: x["edgeCount"], reverse=True)
        return entity_edge_counts[:limit]

    async def get_growth_timeline(
        self, group_id: str | None = None, days: int = 30
    ) -> list[dict]:
        gid = group_id or "default"
        since = (utc_now() - timedelta(days=days)).isoformat()

        if self._helix_client is not None:
            all_eps, all_ents = await self._helix_client.query_concurrent([
                ("find_episodes_by_group", {"gid": gid}),
                ("find_entities_by_group", {"gid": gid}),
            ])
        else:
            all_eps = await self._query("find_episodes_by_group", {"gid": gid})
            all_ents = await self._query("find_entities_by_group", {"gid": gid})

        ep_map: dict[str, Any] = {}
        for d in all_eps:
            ca = d.get("created_at", "")
            if ca >= since:
                day = ca[:10]
                ep_map[day] = ep_map.get(day, 0) + 1

        ent_map: dict[str, Any] = {}
        for d in all_ents:
            ca = d.get("created_at", "")
            if ca >= since:
                day = ca[:10]
                ent_map[day] = ent_map.get(day, 0) + 1

        all_days = sorted(set(list(ep_map.keys()) + list(ent_map.keys())), reverse=True)
        return [
            {
                "date": day,
                "episodes": ep_map.get(day, 0),
                "entities": ent_map.get(day, 0),
            }
            for day in all_days[:days]
        ]

    async def get_entity_type_counts(self, group_id: str | None = None) -> dict[str, Any]:
        gid = group_id or "default"
        all_ents = await self._query("find_entities_by_group", {"gid": gid})
        counts: dict[str, Any] = {}
        for d in all_ents:
            et = d.get("entity_type", "")
            counts[et] = counts.get(et, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Consolidation methods
    # ------------------------------------------------------------------

    async def get_co_occurring_entity_pairs(
        self,
        group_id: str,
        since: datetime | None = None,
        min_co_occurrence: int = 3,
        limit: int = 100,
    ) -> list[tuple[str, str, int]]:
        """Find entity pairs that co-occur in episodes but lack a direct relationship."""
        all_eps = await self._query("find_episodes_by_group", {"gid": group_id})

        # For each episode, get linked entities
        episode_entities: dict[str, list[str]] = {}
        for ep in all_eps:
            ep_id = ep.get("episode_id", "")
            if not ep_id:
                continue
            if since:
                ca = _parse_dt(ep.get("created_at"))
                if ca and ca < since:
                    continue
            ep_hid = self._extract_helix_id(ep)
            if ep_hid is None:
                continue
            ents = await self._query("get_episode_entities", {"id": ep_hid})
            entity_ids = [e.get("entity_id", "") for e in ents if e.get("entity_id")]
            if len(entity_ids) >= 2:
                episode_entities[ep_id] = entity_ids

        # Count co-occurrences
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        for ep_id, eids in episode_entities.items():
            eids_sorted = sorted(eids)
            for i in range(len(eids_sorted)):
                for j in range(i + 1, len(eids_sorted)):
                    pair = (eids_sorted[i], eids_sorted[j])
                    pair_counts[pair] += 1

        # Filter by min_co_occurrence and exclude pairs with existing relationships
        results: list[tuple[str, str, int]] = []
        for (a, b), count in pair_counts.items():
            if count < min_co_occurrence:
                continue
            # Check for existing relationship
            existing = await self.find_existing_relationship(a, b, "", group_id)
            if existing is None:
                existing = await self.find_existing_relationship(b, a, "", group_id)
            if existing is None:
                results.append((a, b, count))

        results.sort(key=lambda x: -x[2])
        return results[:limit]

    async def get_entity_episode_counts(
        self,
        group_id: str,
        entity_ids: list[str],
    ) -> dict[str, Any]:
        if not entity_ids:
            return {}
        result: dict[str, Any] = {}
        for eid in entity_ids:
            count = await self.get_entity_episode_count(eid, group_id)
            result[eid] = count
        return result

    async def find_structural_merge_candidates(
        self,
        group_id: str,
        min_shared_neighbors: int = 3,
        limit: int = 200,
    ) -> list[tuple[str, str, int]]:
        """Find entity pairs sharing many active neighbors."""
        all_entities = await self._query("find_entities_by_group", {"gid": group_id})
        active_ids = {d.get("entity_id", "") for d in all_entities}

        # Collect valid entities
        valid_ents: list[tuple[str, Any]] = []
        for ent in all_entities:
            ehid = self._extract_helix_id(ent)
            eid = ent.get("entity_id", "")
            if ehid is not None and eid:
                valid_ents.append((eid, ehid))

        # Build neighbor sets — fetch all edges concurrently
        neighbors: dict[str, set[str]] = defaultdict(set)
        if valid_ents and self._helix_client is not None:
            queries = []
            for _eid, ehid in valid_ents:
                queries.append(("get_outgoing_edges", {"id": ehid}))
                queries.append(("get_incoming_edges", {"id": ehid}))
            all_results = await self._helix_client.query_concurrent(queries)
            for i, (eid, _ehid) in enumerate(valid_ents):
                out_edges = all_results[i * 2]
                in_edges = all_results[i * 2 + 1]
                for edge in out_edges + in_edges:
                    if not self._is_active_edge(edge):
                        continue
                    if edge.get("group_id") != group_id:
                        continue
                    rel = self._dict_to_relationship(edge)
                    other = rel.target_id if rel.source_id == eid else rel.source_id
                    if other in active_ids and other != eid:
                        neighbors[eid].add(other)
        else:
            for eid, ehid in valid_ents:
                out_edges = await self._query("get_outgoing_edges", {"id": ehid})
                in_edges = await self._query("get_incoming_edges", {"id": ehid})
                for edge in out_edges + in_edges:
                    if not self._is_active_edge(edge):
                        continue
                    if edge.get("group_id") != group_id:
                        continue
                    rel = self._dict_to_relationship(edge)
                    other = rel.target_id if rel.source_id == eid else rel.source_id
                    if other in active_ids and other != eid:
                        neighbors[eid].add(other)

        # Build inverted index
        inv_index: dict[str, set[str]] = defaultdict(set)
        for eid, nbrs in neighbors.items():
            for nbr in nbrs:
                inv_index[nbr].add(eid)

        # Find pairs sharing >= min_shared_neighbors
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        for nbr_id, connected in inv_index.items():
            connected_list = sorted(connected)
            for i in range(len(connected_list)):
                for j in range(i + 1, len(connected_list)):
                    pair = (connected_list[i], connected_list[j])
                    pair_counts[pair] += 1

        results = [
            (a, b, count)
            for (a, b), count in pair_counts.items()
            if count >= min_shared_neighbors
        ]
        results.sort(key=lambda x: -x[2])
        return results[:limit]

    async def get_episode_cooccurrence_count(
        self,
        entity_id_a: str,
        entity_id_b: str,
        group_id: str,
    ) -> int:
        """Count episodes where both entities appear together."""
        eps_a_task = self.get_episodes_for_entity(entity_id_a, group_id, limit=10000)
        eps_b_task = self.get_episodes_for_entity(entity_id_b, group_id, limit=10000)
        eps_a_list, eps_b_list = await asyncio.gather(eps_a_task, eps_b_task)
        return len(set(eps_a_list) & set(eps_b_list))

    async def get_dead_entities(
        self,
        group_id: str,
        min_age_days: int = 30,
        limit: int = 100,
        max_access_count: int = 0,
    ) -> list[Entity]:
        cutoff = (utc_now() - timedelta(days=min_age_days)).isoformat()
        all_entities = await self._query("find_entities_by_group", {"gid": group_id})

        # Pre-filter candidates before querying edges
        candidates: list[tuple[dict, Any]] = []
        for ent in all_entities:
            if ent.get("identity_core", False):
                continue
            if (ent.get("access_count", 0) or 0) > max_access_count:
                continue
            if (ent.get("created_at", "") or "") >= cutoff:
                continue
            ehid = self._extract_helix_id(ent)
            if ehid is None:
                continue
            candidates.append((ent, ehid))

        dead: list[Entity] = []
        if candidates and self._helix_client is not None:
            # Fire all outgoing + incoming queries concurrently
            queries = []
            for _ent, ehid in candidates:
                queries.append(("get_outgoing_edges", {"id": ehid}))
                queries.append(("get_incoming_edges", {"id": ehid}))
            all_results = await self._helix_client.query_concurrent(queries)
            for i, (ent, _ehid) in enumerate(candidates):
                out_edges = all_results[i * 2]
                in_edges = all_results[i * 2 + 1]
                has_active = False
                for edge in out_edges + in_edges:
                    if self._is_active_edge(edge) and edge.get("group_id") == group_id:
                        has_active = True
                        break
                if not has_active:
                    dead.append(self._dict_to_entity(ent, group_id))
                    if len(dead) >= limit:
                        break
        else:
            for ent, ehid in candidates:
                out_edges = await self._query("get_outgoing_edges", {"id": ehid})
                in_edges = await self._query("get_incoming_edges", {"id": ehid})
                has_active = False
                for edge in out_edges + in_edges:
                    if self._is_active_edge(edge) and edge.get("group_id") == group_id:
                        has_active = True
                        break
                if not has_active:
                    dead.append(self._dict_to_entity(ent, group_id))
                    if len(dead) >= limit:
                        break

        dead.sort(key=lambda e: (e.access_count, e.created_at.isoformat() if e.created_at else ""))
        return dead[:limit]

    async def merge_entities(
        self,
        keep_id: str,
        remove_id: str,
        group_id: str,
    ) -> int:
        """Merge remove_id into keep_id."""
        keep_hid = await self._resolve_entity_helix_id(keep_id, group_id)
        remove_hid = await self._resolve_entity_helix_id(remove_id, group_id)
        if keep_hid is None or remove_hid is None:
            return 0

        transferred = 0

        # 1. Fetch outgoing + incoming edges from remove_id concurrently
        if self._helix_client is not None:
            out_edges, in_edges = await self._helix_client.query_concurrent([
                ("get_outgoing_edges", {"id": remove_hid}),
                ("get_incoming_edges", {"id": remove_hid}),
            ])
        else:
            out_edges = await self._query("get_outgoing_edges", {"id": remove_hid})
            in_edges = await self._query("get_incoming_edges", {"id": remove_hid})

        # Re-point outgoing edges from remove_id
        for edge in out_edges:
            if edge.get("group_id") != group_id:
                continue
            rel = self._dict_to_relationship(edge)
            if rel.target_id == keep_id:
                # Would create self-loop, just delete
                edge_hid = self._extract_helix_id(edge)
                if edge_hid is not None:
                    try:
                        await self._query("drop_edge", {"id": edge_hid})
                    except Exception:
                        pass
                continue

            # Check if keeper already has same edge
            existing = await self.find_existing_relationship(
                keep_id, rel.target_id, rel.predicate, group_id
            )
            if existing:
                if rel.weight > existing.weight:
                    await self.update_relationship_weight(
                        keep_id, rel.target_id,
                        rel.weight - existing.weight,
                        group_id=group_id,
                        predicate=rel.predicate,
                    )
            else:
                target_hid = await self._resolve_entity_helix_id(rel.target_id, group_id)
                if target_hid is not None:
                    new_id = f"rel_{uuid.uuid4().hex[:12]}"
                    await self._query(
                        "create_relationship",
                        {
                            "rel_id": new_id,
                            "group_id": group_id,
                            "predicate": rel.predicate,
                            "weight": rel.weight,
                            "polarity": rel.polarity,
                            "valid_from": rel.valid_from.isoformat() if rel.valid_from else "",
                            "valid_to": rel.valid_to.isoformat() if rel.valid_to else "",
                            "is_expired": False,
                            "created_at": (
                                rel.created_at.isoformat() if rel.created_at else utc_now_iso()
                            ),
                            "source_episode_id": rel.source_episode or "",
                            "source_id": keep_hid,
                            "target_id": target_hid,
                        },
                    )

            # Delete old edge
            edge_hid = self._extract_helix_id(edge)
            if edge_hid is not None:
                try:
                    await self._query("drop_edge", {"id": edge_hid})
                except Exception:
                    pass
            transferred += 1

        # 2. Re-point incoming edges to remove_id (already fetched above)
        for edge in in_edges:
            if edge.get("group_id") != group_id:
                continue
            rel = self._dict_to_relationship(edge)
            if rel.source_id == keep_id:
                edge_hid = self._extract_helix_id(edge)
                if edge_hid is not None:
                    try:
                        await self._query("drop_edge", {"id": edge_hid})
                    except Exception:
                        pass
                continue

            existing = await self.find_existing_relationship(
                rel.source_id, keep_id, rel.predicate, group_id
            )
            if existing:
                if rel.weight > existing.weight:
                    await self.update_relationship_weight(
                        rel.source_id, keep_id,
                        rel.weight - existing.weight,
                        group_id=group_id,
                        predicate=rel.predicate,
                    )
            else:
                source_hid = await self._resolve_entity_helix_id(rel.source_id, group_id)
                if source_hid is not None:
                    new_id = f"rel_{uuid.uuid4().hex[:12]}"
                    await self._query(
                        "create_relationship",
                        {
                            "rel_id": new_id,
                            "group_id": group_id,
                            "predicate": rel.predicate,
                            "weight": rel.weight,
                            "polarity": rel.polarity,
                            "valid_from": rel.valid_from.isoformat() if rel.valid_from else "",
                            "valid_to": rel.valid_to.isoformat() if rel.valid_to else "",
                            "is_expired": False,
                            "created_at": (
                                rel.created_at.isoformat() if rel.created_at else utc_now_iso()
                            ),
                            "source_episode_id": rel.source_episode or "",
                            "source_id": source_hid,
                            "target_id": keep_hid,
                        },
                    )

            edge_hid = self._extract_helix_id(edge)
            if edge_hid is not None:
                try:
                    await self._query("drop_edge", {"id": edge_hid})
                except Exception:
                    pass
            transferred += 1

        # 3. Re-point episode links
        remove_episodes = await self._query("get_episodes_for_entity", {"id": remove_hid})
        for ep_dict in remove_episodes:
            ep_hid = self._extract_helix_id(ep_dict)
            if ep_hid is not None:
                try:
                    await self._query(
                        "link_episode_entity",
                        {"episode_id": ep_hid, "entity_id": keep_hid},
                    )
                except Exception:
                    pass

        # 4. Merge summaries — fetch both entities concurrently
        if self._helix_client is not None:
            keep_results, remove_results = await self._helix_client.query_concurrent([
                ("get_entity", {"id": keep_hid}),
                ("get_entity", {"id": remove_hid}),
            ])
        else:
            keep_results = await self._query("get_entity", {"id": keep_hid})
            remove_results = await self._query("get_entity", {"id": remove_hid})
        if keep_results and remove_results:
            keep_data = keep_results[0]
            remove_data = remove_results[0]
            keep_summary = keep_data.get("summary", "") or ""
            remove_summary = remove_data.get("summary", "") or ""
            keep_count = keep_data.get("access_count", 0) or 0
            remove_count = remove_data.get("access_count", 0) or 0

            if remove_summary and remove_summary not in keep_summary:
                if is_meta_summary(remove_summary):
                    logger.warning(
                        "Rejected meta-contaminated summary during merge into %s: %s",
                        keep_id,
                        remove_summary[:80],
                    )
                else:
                    keep_summary = _dedup_summaries(keep_summary, remove_summary)

            # Update keeper
            keep_data["summary"] = keep_summary
            keep_data["access_count"] = keep_count + remove_count
            keep_data["updated_at"] = utc_now_iso()
            await self._query(
                "update_entity_full",
                {
                    "id": keep_hid,
                    "name": keep_data.get("name", ""),
                    "entity_type": keep_data.get("entity_type", ""),
                    "summary": keep_summary,
                    "attributes_json": keep_data.get("attributes_json", "{}"),
                    "updated_at": utc_now_iso(),
                    "is_deleted": keep_data.get("is_deleted", False),
                    "deleted_at": keep_data.get("deleted_at", ""),
                    "identity_core": keep_data.get("identity_core", False),
                    "mat_tier": keep_data.get("mat_tier", "episodic"),
                    "recon_count": keep_data.get("recon_count", 0),
                    "lexical_regime": keep_data.get("lexical_regime", ""),
                    "canonical_identifier": keep_data.get("canonical_identifier", ""),
                    "identifier_label": keep_data.get("identifier_label", ""),
                    "pii_detected": keep_data.get("pii_detected", False),
                    "pii_categories_json": keep_data.get("pii_categories_json", "[]"),
                    "access_count": keep_count + remove_count,
                    "last_accessed": keep_data.get("last_accessed", ""),
                    "source_episode_ids": keep_data.get("source_episode_ids", "[]"),
                    "evidence_count": keep_data.get("evidence_count", 0),
                    "evidence_span_start": keep_data.get("evidence_span_start", ""),
                    "evidence_span_end": keep_data.get("evidence_span_end", ""),
                },
            )

        # 5. Soft-delete the loser
        await self._query(
            "soft_delete_entity",
            {"id": remove_hid, "deleted_at": utc_now_iso()},
        )

        return transferred

    async def get_relationships_by_predicate(
        self,
        group_id: str,
        predicate: str,
        active_only: bool = True,
        limit: int = 10000,
    ) -> list[Relationship]:
        all_entities = await self._query("find_entities_by_group", {"gid": group_id})

        rels: list[Relationship] = []
        seen_ids: set[str] = set()
        for ent in all_entities:
            ehid = self._extract_helix_id(ent)
            if ehid is None:
                continue
            edges = await self._query(
                "get_outgoing_edges_by_predicate",
                {"id": ehid, "pred": predicate},
            )
            for edge in edges:
                if edge.get("group_id") != group_id:
                    continue
                if active_only and not self._is_active_edge(edge):
                    continue
                rel = self._dict_to_relationship(edge)
                if rel.id not in seen_ids:
                    seen_ids.add(rel.id)
                    rels.append(rel)
                    if len(rels) >= limit:
                        return rels
        return rels

    async def update_relationship_weight(
        self,
        source_id: str,
        target_id: str,
        weight_delta: float,
        max_weight: float = 3.0,
        group_id: str = "default",
        predicate: str | None = None,
    ) -> float | None:
        """Increment edge weight via read-modify-write, capped at max_weight."""
        source_hid = await self._resolve_entity_helix_id(source_id, group_id)
        if source_hid is None:
            return None

        # Fetch outgoing + incoming concurrently
        if self._helix_client is not None:
            edges, in_edges = await self._helix_client.query_concurrent([
                ("get_outgoing_edges", {"id": source_hid}),
                ("get_incoming_edges", {"id": source_hid}),
            ])
        else:
            edges = await self._query("get_outgoing_edges", {"id": source_hid})
            in_edges = await self._query("get_incoming_edges", {"id": source_hid})

        for edge in edges + in_edges:
            if edge.get("group_id") != group_id:
                continue
            if not self._is_active_edge(edge):
                continue
            rel = self._dict_to_relationship(edge)
            matches = (
                (rel.source_id == source_id and rel.target_id == target_id)
                or (rel.source_id == target_id and rel.target_id == source_id)
            )
            if not matches:
                continue
            if predicate is not None and rel.predicate != predicate:
                continue

            new_weight = min(max_weight, rel.weight + weight_delta)
            edge_hid = self._extract_helix_id(edge)
            if edge_hid is None:
                continue
            await self._query(
                "update_edge",
                {
                    "id": edge_hid,
                    "weight": new_weight,
                    "is_expired": edge.get("is_expired", False),
                    "valid_to": edge.get("valid_to", ""),
                },
            )
            return new_weight

        return None

    async def get_identity_core_entities(self, group_id: str) -> list[Entity]:
        results = await self._query("find_identity_core_entities", {"gid": group_id})
        return [self._dict_to_entity(d, group_id) for d in results]

    async def path_exists_within_hops(
        self,
        source_id: str,
        target_id: str,
        max_hops: int,
        group_id: str,
    ) -> bool:
        """Check if a path exists between two entities within N hops.

        Fast path: tries the HelixDB ``shortest_path_bfs`` query which
        performs BFS server-side.  Falls back to Python-side BFS when the
        query is unavailable or returns an error.
        """
        # --- Python-side BFS (handles max_hops correctly) ---
        visited: set[str] = {source_id}
        frontier: set[str] = {source_id}

        for _hop in range(max_hops):
            if not frontier:
                break
            next_frontier: set[str] = set()
            for eid in frontier:
                hid = await self._resolve_entity_helix_id(eid, group_id)
                if hid is None:
                    continue
                if self._helix_client is not None:
                    out_edges, in_edges = await self._helix_client.query_concurrent([
                        ("get_outgoing_edges", {"id": hid}),
                        ("get_incoming_edges", {"id": hid}),
                    ])
                else:
                    out_edges = await self._query("get_outgoing_edges", {"id": hid})
                    in_edges = await self._query("get_incoming_edges", {"id": hid})
                for edge in out_edges + in_edges:
                    if not self._is_active_edge(edge):
                        continue
                    if edge.get("group_id") != group_id:
                        continue
                    rel = self._dict_to_relationship(edge)
                    other = rel.target_id if rel.source_id == eid else rel.source_id
                    if other == target_id:
                        return True
                    if other not in visited:
                        visited.add(other)
                        next_frontier.add(other)
            frontier = next_frontier

        return False

    async def get_expired_relationships(
        self,
        group_id: str,
        predicate: str | None = None,
        limit: int = 100,
    ) -> list[Relationship]:
        all_entities = await self._query("find_entities_by_group", {"gid": group_id})

        rels: list[Relationship] = []
        seen_ids: set[str] = set()
        for ent in all_entities:
            ehid = self._extract_helix_id(ent)
            if ehid is None:
                continue
            edges = await self._query("get_outgoing_edges", {"id": ehid})
            for edge in edges:
                if edge.get("group_id") != group_id:
                    continue
                if not self._is_expired_edge(edge):
                    continue
                rel = self._dict_to_relationship(edge)
                if predicate and rel.predicate != predicate:
                    continue
                if rel.id not in seen_ids:
                    seen_ids.add(rel.id)
                    rels.append(rel)
                    if len(rels) >= limit:
                        return rels
        return rels

    async def sample_edges(
        self,
        group_id: str,
        limit: int = 500,
        exclude_ids: set[str] | None = None,
    ) -> list[Relationship]:
        """Return a random sample of active relationships."""
        all_edges = await self.get_all_edges(group_id, limit=limit * 3)
        if exclude_ids:
            all_edges = [r for r in all_edges if r.id not in exclude_ids]
        if len(all_edges) <= limit:
            return all_edges
        return random.sample(all_edges, limit)

    # ------------------------------------------------------------------
    # Maturation queries
    # ------------------------------------------------------------------

    async def get_entity_episode_count(self, entity_id: str, group_id: str) -> int:
        eps = await self.get_episodes_for_entity(entity_id, group_id, limit=100000)
        return len(eps)

    async def get_entity_temporal_span(
        self,
        entity_id: str,
        group_id: str,
    ) -> tuple[str | None, str | None]:
        ep_ids = await self.get_episodes_for_entity(entity_id, group_id, limit=100000)
        if not ep_ids:
            return (None, None)
        timestamps: list[str] = []
        for ep_id in ep_ids:
            ep = await self.get_episode_by_id(ep_id, group_id)
            if ep and ep.created_at:
                timestamps.append(ep.created_at.isoformat())
        if not timestamps:
            return (None, None)
        return (min(timestamps), max(timestamps))

    async def get_entity_relationship_types(
        self,
        entity_id: str,
        group_id: str,
    ) -> list[str]:
        rels = await self.get_relationships(
            entity_id, direction="both", active_only=True, group_id=group_id
        )
        return list({r.predicate for r in rels})

    # ------------------------------------------------------------------
    # Schema Formation
    # ------------------------------------------------------------------

    async def get_schema_members(
        self,
        schema_entity_id: str,
        group_id: str,
    ) -> list[dict]:
        results = await self._query(
            "find_schema_members",
            {"schema_id": schema_entity_id, "gid": group_id},
        )
        return [
            {
                "role_label": d.get("role_label", ""),
                "member_type": "",
                "member_predicate": "",
                "member_entity_id": d.get("member_entity_id", ""),
            }
            for d in results
        ]

    async def save_schema_members(
        self,
        schema_entity_id: str,
        members: list[dict],
        group_id: str,
    ) -> None:
        schema_hid = await self._resolve_entity_helix_id(schema_entity_id, group_id)
        for m in members:
            results = await self._query(
                "create_schema_member",
                {
                    "schema_entity_id": schema_entity_id,
                    "group_id": group_id,
                    "role_label": m.get("role_label", ""),
                    "member_entity_id": m.get("member_entity_id", ""),
                },
            )
            if results and schema_hid is not None:
                member_hid = self._extract_helix_id(results[0])
                if member_hid is not None:
                    try:
                        await self._query(
                            "link_schema_member",
                            {"entity_id": schema_hid, "member_id": member_hid},
                        )
                    except Exception:
                        pass

    async def find_entities_by_type(
        self,
        entity_type: str,
        group_id: str,
        limit: int = 100,
    ) -> list[Entity]:
        results = await self._query(
            "find_entities_by_type",
            {"etype": entity_type, "gid": group_id},
        )
        entities = [self._dict_to_entity(d, group_id) for d in results]
        return entities[:limit]

    # ------------------------------------------------------------------
    # Prospective memory (Intentions)
    # ------------------------------------------------------------------

    async def create_intention(self, intention: object) -> str:
        i = cast("Intention", intention)
        results = await self._query(
            "create_intention",
            {
                "intention_id": i.id,
                "group_id": i.group_id,
                "trigger_text": i.trigger_text,
                "action_text": i.action_text,
                "entity_names_json": "[]",
                "enabled": i.enabled,
                "fire_count": i.fire_count,
                "max_fires": i.max_fires,
                "created_at": i.created_at.isoformat(),
                "updated_at": i.updated_at.isoformat(),
                "deleted_at": i.expires_at.isoformat() if i.expires_at else "",
                "is_deleted": False,
                "context_json": json.dumps({
                    "threshold": i.threshold,
                    "trigger_type": i.trigger_type,
                    "entity_name": i.entity_name,
                }),
            },
        )
        if results:
            hid = self._extract_helix_id(results[0])
            if hid is not None:
                self._intention_id_cache[i.id] = hid
        return i.id

    async def get_intention(self, id: str, group_id: str) -> object | None:
        # Try cache
        hid = self._intention_id_cache.get(id)
        if hid is None:
            all_intentions = await self._query(
                "find_intentions_by_group", {"gid": group_id}
            )
            for d in all_intentions:
                iid = d.get("intention_id", "")
                ihid = self._extract_helix_id(d)
                if ihid is not None:
                    self._intention_id_cache[iid] = ihid
                if iid == id:
                    return self._dict_to_intention(d)
            return None
        results = await self._query("get_intention", {"id": hid})
        if not results:
            return None
        d = results[0]
        # Validate group_id
        if d.get("group_id") != group_id:
            return None
        return self._dict_to_intention(d)

    async def list_intentions(
        self,
        group_id: str,
        enabled_only: bool = True,
    ) -> list:
        if enabled_only:
            results = await self._query("find_enabled_intentions", {"gid": group_id})
        else:
            results = await self._query("find_intentions_by_group", {"gid": group_id})
        intentions = []
        now = utc_now()
        for d in results:
            hid = self._extract_helix_id(d)
            iid = d.get("intention_id", "")
            if hid is not None:
                self._intention_id_cache[iid] = hid
            intention = self._dict_to_intention(d)
            # Filter out expired intentions when listing enabled
            if enabled_only and intention.expires_at and intention.expires_at <= now:
                continue
            intentions.append(intention)
        return intentions

    async def update_intention(
        self,
        id: str,
        updates: dict,
        group_id: str,
    ) -> None:
        hid = self._intention_id_cache.get(id)
        if hid is None:
            # Resolve
            await self.get_intention(id, group_id)
            hid = self._intention_id_cache.get(id)
        if hid is None:
            return

        # Read-modify-write
        results = await self._query("get_intention", {"id": hid})
        if not results:
            return
        current = results[0]

        allowed = {"trigger_text", "action_text", "threshold", "max_fires", "enabled", "expires_at"}
        # Parse existing context_json to preserve threshold/trigger_type
        ctx: dict = {}
        raw_ctx = current.get("context_json")
        if raw_ctx:
            try:
                ctx = json.loads(raw_ctx)
            except (json.JSONDecodeError, TypeError):
                pass

        for key, val in updates.items():
            if key not in allowed:
                continue
            if key == "expires_at":
                dt = val.isoformat() if hasattr(val, "isoformat") else (val or "")
                current["deleted_at"] = dt
            elif key == "threshold":
                ctx["threshold"] = val
            else:
                current[key] = val

        await self._query(
            "update_intention_full",
            {
                "id": hid,
                "trigger_text": current.get("trigger_text", ""),
                "action_text": current.get("action_text", ""),
                "entity_names_json": current.get("entity_names_json", "[]"),
                "enabled": bool(current.get("enabled", True)),
                "fire_count": current.get("fire_count", 0),
                "max_fires": current.get("max_fires", 5),
                "updated_at": utc_now_iso(),
                "deleted_at": current.get("deleted_at", ""),
                "is_deleted": bool(current.get("is_deleted", False)),
                "context_json": json.dumps(ctx),
            },
        )

    async def delete_intention(
        self,
        id: str,
        group_id: str,
        soft: bool = True,
    ) -> None:
        hid = self._intention_id_cache.get(id)
        if hid is None:
            await self.get_intention(id, group_id)
            hid = self._intention_id_cache.get(id)
        if hid is None:
            return
        if soft:
            # Soft delete: disable + mark deleted
            await self.update_intention(id, {"enabled": False}, group_id)
            await self._query(
                "soft_delete_intention",
                {"id": hid, "deleted_at": utc_now_iso()},
            )
        else:
            # Hard delete
            try:
                await self._query("hard_delete_intention", {"id": hid})
            except Exception:
                # Fallback to soft delete
                await self._query(
                    "soft_delete_intention",
                    {"id": hid, "deleted_at": utc_now_iso()},
                )
            self._intention_id_cache.pop(id, None)

    async def increment_intention_fire_count(
        self,
        id: str,
        group_id: str,
    ) -> None:
        hid = self._intention_id_cache.get(id)
        if hid is None:
            await self.get_intention(id, group_id)
            hid = self._intention_id_cache.get(id)
        if hid is None:
            return

        results = await self._query("get_intention", {"id": hid})
        if not results:
            return
        current = results[0]
        new_count = (current.get("fire_count", 0) or 0) + 1

        await self._query(
            "update_intention_full",
            {
                "id": hid,
                "trigger_text": current.get("trigger_text", ""),
                "action_text": current.get("action_text", ""),
                "entity_names_json": current.get("entity_names_json", "[]"),
                "enabled": current.get("enabled", True),
                "fire_count": new_count,
                "max_fires": current.get("max_fires", 5),
                "updated_at": utc_now_iso(),
                "deleted_at": current.get("deleted_at", ""),
                "is_deleted": current.get("is_deleted", False),
                "context_json": current.get("context_json", "{}"),
            },
        )

    # ------------------------------------------------------------------
    # Evidence storage
    # ------------------------------------------------------------------

    async def store_evidence(
        self,
        evidence: list[dict],
        group_id: str = "default",
        *,
        default_status: str = "pending",
    ) -> None:
        if not evidence:
            return
        for ev in evidence:
            status = ev.get("status", default_status)
            resolved_at = ev.get("resolved_at")
            if resolved_at is None and status in {
                "committed", "rejected", "expired", "superseded",
            }:
                resolved_at = utc_now_iso()
            results = await self._query(
                "create_evidence",
                {
                    "evidence_id": ev["evidence_id"],
                    "episode_id": ev["episode_id"],
                    "group_id": group_id,
                    "status": status,
                    "fact_class": ev.get("fact_class") or "",
                    "confidence": ev.get("confidence", 0.0) or 0.0,
                    "source_type": ev.get("source_type") or "",
                    "extractor_name": ev.get("extractor_name") or "",
                    "payload_json": json.dumps(ev.get("payload") or {}),
                    "source_span": ev.get("source_span") or "",
                    "signals_json": json.dumps(ev.get("corroborating_signals") or []),
                    "ambiguity_tags_json": json.dumps(ev.get("ambiguity_tags") or []),
                    "ambiguity_score": ev.get("ambiguity_score") or 0.0,
                    "adjudication_request_id": ev.get("adjudication_request_id") or "",
                    "commit_reason": ev.get("commit_reason") or "",
                    "committed_id": ev.get("committed_id") or "",
                    "deferred_cycles": ev.get("deferred_cycles") or 0,
                    "created_at": ev.get("created_at") or utc_now_iso(),
                    "resolved_at": resolved_at or "",
                },
            )
            if results:
                hid = self._extract_helix_id(results[0])
                if hid is not None:
                    self._evidence_id_cache[ev["evidence_id"]] = hid

    async def get_pending_evidence(
        self,
        group_id: str = "default",
        limit: int = 100,
    ) -> list[dict]:
        results = await self._query("find_pending_evidence", {"gid": group_id})
        evidence = [_evidence_dict_to_storage(d) for d in results]
        evidence.sort(key=lambda e: e.get("confidence", 0.0), reverse=True)
        return evidence[:limit]

    async def get_episode_evidence(
        self,
        episode_id: str,
        group_id: str = "default",
    ) -> list[dict]:
        results = await self._query(
            "find_evidence_by_episode",
            {"ep_id": episode_id, "gid": group_id},
        )
        evidence = [_evidence_dict_to_storage(d) for d in results]
        evidence.sort(key=lambda e: e.get("confidence", 0.0), reverse=True)
        return evidence

    async def update_evidence_status(
        self,
        evidence_id: str,
        status: str,
        updates: dict | None = None,
        group_id: str = "default",
    ) -> None:
        updates = updates or {}
        hid = self._evidence_id_cache.get(evidence_id)
        if hid is None:
            # Search
            pending = await self._query("find_pending_evidence", {"gid": group_id})
            for d in pending:
                eid = d.get("evidence_id", "")
                ehid = self._extract_helix_id(d)
                if ehid is not None:
                    self._evidence_id_cache[eid] = ehid
                if eid == evidence_id:
                    hid = ehid
                    break

        if hid is None:
            return

        resolved_at = ""
        if status in {"committed", "rejected", "expired", "superseded"}:
            resolved_at = utc_now_iso()

        await self._query(
            "update_evidence",
            {
                "id": hid,
                "status": status,
                "resolved_at": resolved_at,
                "commit_reason": updates.get("commit_reason", ""),
                "committed_id": updates.get("committed_id", ""),
            },
        )

    async def get_entity_count(self, group_id: str = "default") -> int:
        """Count entities in a group.

        Fast path: ``find_entity_ids_by_group`` returns only entity IDs
        (lighter payload via projection).  Falls back to the full
        ``find_entities_by_group`` query when the projected version is
        unavailable.
        """
        try:
            results = await self._query(
                "find_entity_ids_by_group", {"gid": group_id}
            )
            return len(results)
        except Exception:
            logger.debug(
                "find_entity_ids_by_group unavailable, falling back to full query"
            )
        results = await self._query("find_entities_by_group", {"gid": group_id})
        return len(results)

    # ------------------------------------------------------------------
    # Adjudication requests
    # ------------------------------------------------------------------

    async def store_adjudication_requests(
        self,
        requests: list[dict],
        group_id: str = "default",
    ) -> None:
        if not requests:
            return
        for req in requests:
            results = await self._query(
                "create_adjudication",
                {
                    "request_id": req["request_id"],
                    "episode_id": req["episode_id"],
                    "group_id": group_id,
                    "status": req.get("status") or "pending",
                    "ambiguity_tags_json": json.dumps(req.get("ambiguity_tags") or []),
                    "evidence_ids_json": json.dumps(req.get("evidence_ids") or []),
                    "selected_text": req.get("selected_text") or "",
                    "request_reason": req.get("request_reason") or "",
                    "resolution_source": req.get("resolution_source") or "",
                    "resolution_payload_json": (
                        json.dumps(req.get("resolution_payload"))
                        if req.get("resolution_payload") is not None
                        else ""
                    ),
                    "attempt_count": req.get("attempt_count") or 0,
                    "created_at": req.get("created_at") or utc_now_iso(),
                    "resolved_at": req.get("resolved_at") or "",
                },
            )
            if results:
                hid = self._extract_helix_id(results[0])
                if hid is not None:
                    self._adjudication_id_cache[req["request_id"]] = hid

    async def get_episode_adjudications(
        self,
        episode_id: str,
        group_id: str = "default",
    ) -> list[dict]:
        results = await self._query(
            "find_adjudications_by_episode",
            {"ep_id": episode_id, "gid": group_id},
        )
        return [_adjudication_dict_to_storage(d) for d in results]

    async def get_adjudication_request(
        self,
        request_id: str,
        group_id: str = "default",
    ) -> dict | None:
        hid = self._adjudication_id_cache.get(request_id)
        if hid is not None:
            results = await self._query("get_adjudication", {"id": hid})
            if results:
                return _adjudication_dict_to_storage(results[0])

        # Search
        pending = await self._query("find_pending_adjudications", {"gid": group_id})
        for d in pending:
            rid = d.get("request_id", "")
            rhid = self._extract_helix_id(d)
            if rhid is not None:
                self._adjudication_id_cache[rid] = rhid
            if rid == request_id:
                return _adjudication_dict_to_storage(d)
        return None

    async def get_pending_adjudication_requests(
        self,
        group_id: str = "default",
        limit: int = 100,
    ) -> list[dict]:
        results = await self._query("find_pending_adjudications", {"gid": group_id})
        adjudications = [_adjudication_dict_to_storage(d) for d in results]
        for d, raw in zip(adjudications, results):
            hid = self._extract_helix_id(raw)
            if hid is not None:
                self._adjudication_id_cache[d["request_id"]] = hid
        return adjudications[:limit]

    async def update_adjudication_request(
        self,
        request_id: str,
        updates: dict,
        group_id: str = "default",
    ) -> None:
        if not updates:
            return
        hid = self._adjudication_id_cache.get(request_id)
        if hid is None:
            await self.get_adjudication_request(request_id, group_id)
            hid = self._adjudication_id_cache.get(request_id)
        if hid is None:
            return

        status = updates.get("status", "pending")
        resolved_at = updates.get("resolved_at", "")
        if not resolved_at and status in {"materialized", "rejected", "expired"}:
            resolved_at = utc_now_iso()

        await self._query(
            "update_adjudication",
            {
                "id": hid,
                "status": status,
                "resolution_source": updates.get("resolution_source", ""),
                "resolution_payload_json": (
                    json.dumps(updates.get("resolution_payload"))
                    if updates.get("resolution_payload") is not None
                    else ""
                ),
                "attempt_count": updates.get("attempt_count", 0),
                "resolved_at": resolved_at,
            },
        )
