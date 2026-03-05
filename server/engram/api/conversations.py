"""Conversation persistence API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from engram.api.deps import get_conversation_store
from engram.security.middleware import get_tenant

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


class CreateConversationBody(BaseModel):
    session_date: str | None = None
    title: str | None = None


class BulkMessagesBody(BaseModel):
    messages: list[dict]


class UpdateConversationBody(BaseModel):
    title: str | None = None


@router.get("/")
async def list_conversations(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
) -> JSONResponse:
    tenant = get_tenant(request)
    store = get_conversation_store()
    conversations = await store.list_conversations(tenant.group_id, limit=limit)
    return JSONResponse(content={"conversations": conversations})


@router.post("/")
async def create_conversation(request: Request, body: CreateConversationBody) -> JSONResponse:
    tenant = get_tenant(request)
    store = get_conversation_store()
    conv_id = await store.create_conversation(
        group_id=tenant.group_id,
        session_date=body.session_date,
        title=body.title,
    )
    return JSONResponse(content={"id": conv_id})


@router.get("/{conversation_id}/messages")
async def get_messages(request: Request, conversation_id: str) -> JSONResponse:
    store = get_conversation_store()
    messages = await store.get_messages(conversation_id)
    return JSONResponse(content={"messages": messages})


@router.post("/{conversation_id}/messages")
async def append_messages(
    request: Request, conversation_id: str, body: BulkMessagesBody,
) -> JSONResponse:
    store = get_conversation_store()
    ids = await store.add_messages_bulk(conversation_id, body.messages)
    return JSONResponse(content={"ids": ids})


@router.patch("/{conversation_id}")
async def update_conversation(
    request: Request, conversation_id: str, body: UpdateConversationBody,
) -> JSONResponse:
    tenant = get_tenant(request)
    store = get_conversation_store()
    await store.update_conversation(conversation_id, tenant.group_id, title=body.title)
    return JSONResponse(content={"status": "updated"})


@router.delete("/{conversation_id}")
async def delete_conversation(request: Request, conversation_id: str) -> JSONResponse:
    tenant = get_tenant(request)
    store = get_conversation_store()
    deleted = await store.delete_conversation(conversation_id, tenant.group_id)
    if not deleted:
        return JSONResponse(status_code=404, content={"detail": "Not found"})
    return JSONResponse(content={"status": "deleted"})
