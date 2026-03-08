"""SQLite-backed conversation persistence for the Knowledge tab."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import aiosqlite


class ConversationNotFoundError(LookupError):
    """Raised when a conversation is missing or not owned by the caller."""

    def __init__(self, conversation_id: str) -> None:
        super().__init__(f"Conversation '{conversation_id}' not found")
        self.conversation_id = conversation_id


class SQLiteConversationStore:
    """Stores chat conversations and their entity associations."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("ConversationStore not initialized")
        return self._db

    async def initialize(self, db: aiosqlite.Connection | None = None) -> None:
        if db:
            self._db = db
        elif not self._db:
            self._db = await aiosqlite.connect(self._db_path)
            self._db.row_factory = aiosqlite.Row

        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          TEXT PRIMARY KEY,
                group_id    TEXT NOT NULL DEFAULT 'default',
                title       TEXT,
                session_date TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_group_date
                ON conversations(group_id, session_date)
        """)
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id              TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role            TEXT NOT NULL CHECK(role IN ('user','assistant')),
                content         TEXT NOT NULL,
                parts_json      TEXT,
                created_at      TEXT NOT NULL
            )
        """)
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_messages_conv
                ON conversation_messages(conversation_id)
        """)
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS conversation_entities (
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                entity_id       TEXT NOT NULL REFERENCES entities(id),
                PRIMARY KEY (conversation_id, entity_id)
            )
        """)
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_entities_entity
                ON conversation_entities(entity_id)
        """)
        await self.db.commit()

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
        await self.db.execute(
            """INSERT INTO conversations (id, group_id, title, session_date, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (conv_id, group_id, title, session_date, now, now),
        )
        await self.db.commit()
        return conv_id

    async def _get_conversation_row(
        self,
        conversation_id: str,
        group_id: str,
    ) -> aiosqlite.Row | None:
        cursor = await self.db.execute(
            """SELECT id, group_id, title, session_date, created_at, updated_at
               FROM conversations
               WHERE id = ? AND group_id = ?""",
            (conversation_id, group_id),
        )
        return await cursor.fetchone()

    async def get_conversation(self, conversation_id: str, group_id: str) -> dict:
        row = await self._get_conversation_row(conversation_id, group_id)
        if row is None:
            raise ConversationNotFoundError(conversation_id)
        return {
            "id": row["id"],
            "title": row["title"],
            "sessionDate": row["session_date"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }

    async def _require_conversation(self, conversation_id: str, group_id: str) -> None:
        if await self._get_conversation_row(conversation_id, group_id) is None:
            raise ConversationNotFoundError(conversation_id)

    async def list_conversations(
        self, group_id: str, limit: int = 50,
    ) -> list[dict]:
        cursor = await self.db.execute(
            """SELECT c.id, c.title, c.session_date, c.created_at, c.updated_at
               FROM conversations c
               WHERE c.group_id = ?
               ORDER BY c.updated_at DESC
               LIMIT ?""",
            (group_id, limit),
        )
        rows = await cursor.fetchall()

        results = []
        for row in rows:
            conv_id = row["id"]
            # Fetch entity IDs for this conversation
            ent_cursor = await self.db.execute(
                "SELECT entity_id FROM conversation_entities WHERE conversation_id = ?",
                (conv_id,),
            )
            ent_rows = await ent_cursor.fetchall()
            entity_ids = [r["entity_id"] for r in ent_rows]

            results.append({
                "id": conv_id,
                "title": row["title"],
                "sessionDate": row["session_date"],
                "createdAt": row["created_at"],
                "updatedAt": row["updated_at"],
                "entityIds": entity_ids,
            })

        return results

    async def get_messages(self, conversation_id: str, group_id: str) -> list[dict]:
        await self._require_conversation(conversation_id, group_id)
        cursor = await self.db.execute(
            """SELECT id, role, content, parts_json, created_at
               FROM conversation_messages
               WHERE conversation_id = ?
               ORDER BY created_at ASC""",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "partsJson": row["parts_json"],
                "createdAt": row["created_at"],
            }
            for row in rows
        ]

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        parts_json: str | None = None,
        *,
        group_id: str,
    ) -> str:
        await self._require_conversation(conversation_id, group_id)
        msg_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            "INSERT INTO conversation_messages"
            " (id, conversation_id, role, content, parts_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (msg_id, conversation_id, role, content, parts_json, now),
        )
        # Update conversation's updated_at
        await self.db.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ? AND group_id = ?",
            (now, conversation_id, group_id),
        )
        await self.db.commit()
        return msg_id

    async def add_messages_bulk(
        self,
        conversation_id: str,
        messages: list[dict],
        group_id: str,
    ) -> list[str]:
        await self._require_conversation(conversation_id, group_id)
        now = datetime.now(timezone.utc).isoformat()
        ids = []
        for msg in messages:
            msg_id = str(uuid.uuid4())
            ids.append(msg_id)
            await self.db.execute(
                "INSERT INTO conversation_messages"
                " (id, conversation_id, role, content, parts_json, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (msg_id, conversation_id, msg["role"],
                 msg["content"], msg.get("partsJson"), now),
            )
        await self.db.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ? AND group_id = ?",
            (now, conversation_id, group_id),
        )
        await self.db.commit()
        return ids

    async def tag_entity(self, conversation_id: str, entity_id: str, group_id: str) -> None:
        await self._require_conversation(conversation_id, group_id)
        await self.db.execute(
            "INSERT OR IGNORE INTO conversation_entities"
            " (conversation_id, entity_id) VALUES (?, ?)",
            (conversation_id, entity_id),
        )
        await self.db.commit()

    async def update_conversation(
        self, conversation_id: str, group_id: str, title: str | None = None,
    ) -> bool:
        row = await self._get_conversation_row(conversation_id, group_id)
        if row is None:
            return False
        now = datetime.now(timezone.utc).isoformat()
        if title is not None:
            cursor = await self.db.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND group_id = ?",
                (title, now, conversation_id, group_id),
            )
            await self.db.commit()
            return cursor.rowcount > 0
        await self.db.commit()
        return True

    async def delete_conversation(self, conversation_id: str, group_id: str) -> bool:
        cursor = await self.db.execute(
            "DELETE FROM conversations WHERE id = ? AND group_id = ?",
            (conversation_id, group_id),
        )
        await self.db.commit()
        return cursor.rowcount > 0
