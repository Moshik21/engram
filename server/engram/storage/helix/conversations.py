"""HelixDB-backed conversation persistence for the Knowledge tab."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from engram.config import HelixDBConfig

logger = logging.getLogger(__name__)


class ConversationNotFoundError(LookupError):
    """Raised when a conversation is missing or not owned by the caller."""

    def __init__(self, conversation_id: str) -> None:
        super().__init__(f"Conversation '{conversation_id}' not found")
        self.conversation_id = conversation_id


def _safe_get(d: dict, key: str, default: Any = None) -> Any:
    """Safely get a value from a dict returned by Helix."""
    v = d.get(key, default)
    return v if v is not None else default


class HelixConversationStore:
    """Stores chat conversations and their entity associations in HelixDB.

    All Helix client calls are synchronous (HTTP POST via urllib). We wrap
    every call with ``asyncio.to_thread()`` to keep the event loop responsive.
    """

    def __init__(self, config: HelixDBConfig, client=None) -> None:
        self._config = config
        self._client: Any | None = None
        self._helix_client = client  # Shared HelixClient (async httpx)
        # conversation_id (our UUID) -> Helix internal node ID
        self._conv_id_cache: dict[str, Any] = {}
        # message_id (our UUID) -> Helix internal node ID
        self._msg_id_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Internal query helper
    # ------------------------------------------------------------------

    async def _query(self, endpoint: str, payload: dict) -> list[dict]:
        """Execute a Helix query.

        Fast path: shared async HelixClient (httpx connection pool).
        Legacy fallback: synchronous helix-py SDK via thread pool.
        """
        # Fast path: shared async client
        if self._helix_client is not None:
            return await self._helix_client.query(endpoint, payload)

        # Legacy fallback: synchronous helix-py SDK
        client = self._client
        if client is None:
            raise RuntimeError("HelixConversationStore not initialized")
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
    def _extract_helix_id(item: dict):
        """Extract the Helix-assigned internal ID from a response dict."""
        for key in ("id", "_id", "node_id", "edge_id"):
            if key in item and item[key] is not None:
                return item[key]
        return None

    def _cache_conv(self, helix_id, conversation_id: str) -> None:
        if helix_id is not None:
            self._conv_id_cache[conversation_id] = helix_id

    def _cache_msg(self, helix_id: int | None, message_id: str) -> None:
        if helix_id is not None:
            self._msg_id_cache[message_id] = helix_id

    # ------------------------------------------------------------------
    # Helix ID resolution
    # ------------------------------------------------------------------

    async def _resolve_conv_helix_id(
        self, conversation_id: str, group_id: str
    ) -> int | None:
        """Resolve a conversation UUID to a Helix internal ID via cache or query."""
        if conversation_id in self._conv_id_cache:
            return self._conv_id_cache[conversation_id]
        # Scan conversations for the group and populate cache
        results = await self._query(
            "find_conversations_by_group", {"gid": group_id}
        )
        for item in results:
            cid = _safe_get(item, "conversation_id", "")
            hid = self._extract_helix_id(item)
            if hid is not None:
                self._conv_id_cache[cid] = hid
            if cid == conversation_id:
                return hid
        return None

    async def _resolve_msg_helix_id(
        self, message_id: str, conversation_id: str
    ) -> int | None:
        """Resolve a message UUID to a Helix internal ID via cache or query."""
        if message_id in self._msg_id_cache:
            return self._msg_id_cache[message_id]
        results = await self._query(
            "find_messages_by_conversation", {"conv_id": conversation_id}
        )
        for item in results:
            mid = _safe_get(item, "message_id", "")
            hid = self._extract_helix_id(item)
            if hid is not None:
                self._msg_id_cache[mid] = hid
            if mid == message_id:
                return hid
        return None

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dict_to_conversation(d: dict) -> dict:
        """Convert a Helix conversation dict to the API response shape."""
        return {
            "id": _safe_get(d, "conversation_id", ""),
            "title": _safe_get(d, "title"),
            "sessionDate": _safe_get(d, "session_date", ""),
            "createdAt": _safe_get(d, "created_at", ""),
            "updatedAt": _safe_get(d, "updated_at", ""),
        }

    @staticmethod
    def _dict_to_message(d: dict) -> dict:
        """Convert a Helix message dict to the API response shape."""
        return {
            "id": _safe_get(d, "message_id", ""),
            "role": _safe_get(d, "role", ""),
            "content": _safe_get(d, "content", ""),
            "partsJson": _safe_get(d, "parts_json"),
            "createdAt": _safe_get(d, "created_at", ""),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Connect to HelixDB."""
        if self._helix_client is None:
            from engram.storage.helix.client import HelixClient

            self._helix_client = HelixClient(self._config)
        if not self._helix_client.is_connected:
            await self._helix_client.initialize()

        transport = getattr(self._config, "transport", "http")
        if transport == "native":
            logger.info("HelixDB conversation store initialized (native transport)")
            return

        from helix import Client  # type: ignore[import-untyped]

        kwargs: dict[str, Any] = {
            "port": self._config.port,
            "verbose": self._config.verbose,
        }
        if self._config.api_endpoint:
            kwargs["url"] = self._config.api_endpoint
            kwargs["local"] = False
            if self._config.api_key:
                kwargs["api_key"] = self._config.api_key
        else:
            kwargs["local"] = True

        self._client = await asyncio.to_thread(Client, **kwargs)

        logger.info(
            "HelixDB conversation store initialized (host=%s, port=%d)",
            self._config.host,
            self._config.port,
        )

    async def close(self) -> None:
        """No-op -- Helix sync client has no close method."""
        self._client = None

    # ------------------------------------------------------------------
    # Internal guards
    # ------------------------------------------------------------------

    async def _require_conversation(
        self, conversation_id: str, group_id: str
    ) -> int:
        """Ensure a conversation exists and return its Helix internal ID.

        Raises ConversationNotFoundError if not found or not owned by group.
        """
        helix_id = await self._resolve_conv_helix_id(conversation_id, group_id)
        if helix_id is None:
            raise ConversationNotFoundError(conversation_id)
        # Verify it exists and belongs to the group
        results = await self._query("get_conversation", {"id": helix_id})
        if not results:
            raise ConversationNotFoundError(conversation_id)
        d = results[0]
        if _safe_get(d, "group_id", "") != group_id:
            raise ConversationNotFoundError(conversation_id)
        return helix_id

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    async def create_conversation(
        self,
        group_id: str,
        session_date: str | None = None,
        title: str | None = None,
    ) -> str:
        conv_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        if not session_date:
            session_date = now[:10]

        results = await self._query(
            "create_conversation",
            {
                "conversation_id": conv_id,
                "group_id": group_id,
                "title": title or "",
                "session_date": session_date,
                "created_at": now,
                "updated_at": now,
            },
        )
        if results:
            hid = self._extract_helix_id(results[0])
            self._cache_conv(hid, conv_id)
        return conv_id

    async def get_conversation(self, conversation_id: str, group_id: str) -> dict:
        helix_id = await self._resolve_conv_helix_id(conversation_id, group_id)
        if helix_id is None:
            raise ConversationNotFoundError(conversation_id)

        results = await self._query("get_conversation", {"id": helix_id})
        if not results:
            raise ConversationNotFoundError(conversation_id)

        d = results[0]
        if _safe_get(d, "group_id", "") != group_id:
            raise ConversationNotFoundError(conversation_id)

        return self._dict_to_conversation(d)

    async def list_conversations(
        self,
        group_id: str,
        limit: int = 50,
    ) -> list[dict]:
        results = await self._query(
            "find_conversations_by_group", {"gid": group_id}
        )

        conversations: list[dict] = []
        for item in results[:limit]:
            cid = _safe_get(item, "conversation_id", "")
            hid = self._extract_helix_id(item)
            self._cache_conv(hid, cid)

            # Fetch linked entity IDs for this conversation
            entity_ids: list[str] = []
            if hid is not None:
                try:
                    ent_results = await self._query(
                        "find_conversation_entities", {"conv_id": hid}
                    )
                    entity_ids = [
                        _safe_get(e, "entity_id", "")
                        for e in ent_results
                        if _safe_get(e, "entity_id", "")
                    ]
                except Exception:
                    logger.debug(
                        "Failed to fetch entities for conversation %s",
                        cid,
                        exc_info=True,
                    )

            conv = self._dict_to_conversation(item)
            conv["entityIds"] = entity_ids
            conversations.append(conv)

        return conversations

    async def get_messages(
        self, conversation_id: str, group_id: str
    ) -> list[dict]:
        await self._require_conversation(conversation_id, group_id)

        results = await self._query(
            "find_messages_by_conversation", {"conv_id": conversation_id}
        )

        messages: list[dict] = []
        for item in results:
            mid = _safe_get(item, "message_id", "")
            hid = self._extract_helix_id(item)
            self._cache_msg(hid, mid)
            messages.append(self._dict_to_message(item))

        return messages

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        parts_json: str | None = None,
        *,
        group_id: str,
    ) -> str:
        conv_helix_id = await self._require_conversation(conversation_id, group_id)

        msg_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        results = await self._query(
            "create_conversation_message",
            {
                "message_id": msg_id,
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "parts_json": parts_json or "",
                "created_at": now,
            },
        )
        if results:
            hid = self._extract_helix_id(results[0])
            self._cache_msg(hid, msg_id)

        # Update conversation's updated_at via read-modify-write
        # (Helix update overwrites all fields, so we must preserve title)
        conv_results = await self._query("get_conversation", {"id": conv_helix_id})
        existing_title = ""
        if conv_results:
            existing_title = _safe_get(conv_results[0], "title", "")
        await self._query(
            "update_conversation",
            {
                "id": conv_helix_id,
                "title": existing_title or "",
                "updated_at": now,
            },
        )

        return msg_id

    async def add_messages_bulk(
        self,
        conversation_id: str,
        messages: list[dict],
        group_id: str,
    ) -> list[str]:
        conv_helix_id = await self._require_conversation(conversation_id, group_id)

        now = datetime.now(timezone.utc).isoformat()
        ids: list[str] = []

        for msg in messages:
            msg_id = str(uuid.uuid4())
            ids.append(msg_id)
            results = await self._query(
                "create_conversation_message",
                {
                    "message_id": msg_id,
                    "conversation_id": conversation_id,
                    "role": msg["role"],
                    "content": msg["content"],
                    "parts_json": msg.get("partsJson", "") or "",
                    "created_at": now,
                },
            )
            if results:
                hid = self._extract_helix_id(results[0])
                self._cache_msg(hid, msg_id)

        # Update conversation's updated_at timestamp
        conv_results = await self._query("get_conversation", {"id": conv_helix_id})
        existing_title = ""
        if conv_results:
            existing_title = _safe_get(conv_results[0], "title", "")

        await self._query(
            "update_conversation",
            {
                "id": conv_helix_id,
                "title": existing_title or "",
                "updated_at": now,
            },
        )

        return ids

    async def tag_entity(
        self, conversation_id: str, entity_id: str, group_id: str
    ) -> None:
        conv_helix_id = await self._require_conversation(conversation_id, group_id)

        # Resolve entity UUID to Helix internal ID for the edge link
        entity_helix_id = await self._resolve_entity_helix_id_for_tag(
            entity_id, group_id
        )
        if entity_helix_id is None:
            logger.warning(
                "Cannot tag entity %s on conversation %s: entity not found",
                entity_id,
                conversation_id,
            )
            return

        try:
            await self._query(
                "link_conversation_entity",
                {"conv_id": conv_helix_id, "entity_id": entity_helix_id},
            )
        except Exception:
            logger.debug(
                "link_conversation_entity failed for conv=%s entity=%s",
                conversation_id,
                entity_id,
                exc_info=True,
            )

    async def _resolve_entity_helix_id_for_tag(
        self, entity_id: str, group_id: str
    ) -> int | None:
        """Resolve an entity UUID to a Helix internal ID by scanning the group.

        This is a lightweight version of the graph store's resolver,
        scoped to conversation tagging.
        """
        try:
            results = await self._query(
                "find_entities_by_group", {"gid": group_id}
            )
        except Exception:
            return None
        for item in results:
            eid = _safe_get(item, "entity_id", "")
            if eid == entity_id:
                return self._extract_helix_id(item)
        return None

    async def update_conversation(
        self,
        conversation_id: str,
        group_id: str,
        title: str | None = None,
    ) -> bool:
        helix_id = await self._resolve_conv_helix_id(conversation_id, group_id)
        if helix_id is None:
            return False

        # Verify ownership
        results = await self._query("get_conversation", {"id": helix_id})
        if not results:
            return False
        d = results[0]
        if _safe_get(d, "group_id", "") != group_id:
            return False

        now = datetime.now(timezone.utc).isoformat()

        if title is not None:
            await self._query(
                "update_conversation",
                {
                    "id": helix_id,
                    "title": title,
                    "updated_at": now,
                },
            )
        return True

    async def delete_conversation(
        self, conversation_id: str, group_id: str
    ) -> bool:
        helix_id = await self._resolve_conv_helix_id(conversation_id, group_id)
        if helix_id is None:
            return False

        # Verify ownership
        results = await self._query("get_conversation", {"id": helix_id})
        if not results:
            return False
        d = results[0]
        if _safe_get(d, "group_id", "") != group_id:
            return False

        # Delete associated messages first
        try:
            await self._query(
                "delete_conversation_messages",
                {"conv_id": conversation_id},
            )
        except Exception:
            logger.debug(
                "delete_conversation_messages failed for %s",
                conversation_id,
                exc_info=True,
            )

        # Delete the conversation node
        try:
            await self._query("delete_conversation", {"id": helix_id})
        except Exception:
            logger.debug(
                "delete_conversation failed for %s",
                conversation_id,
                exc_info=True,
            )
            return False

        # Clean up conversation cache entry.
        # Message cache entries for this conversation become stale but
        # are harmless — they will simply miss on next resolve attempt.
        self._conv_id_cache.pop(conversation_id, None)

        return True
