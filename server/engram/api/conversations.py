"""Conversation persistence API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from engram.api.deps import get_conversation_store
from engram.retrieval.conversation_persistence import (
    append_group_conversation_messages,
    create_group_conversation,
    delete_group_conversation,
    get_group_conversation_messages,
    list_group_conversations,
    update_group_conversation_title,
)
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
    conversations = await list_group_conversations(
        store,
        group_id=tenant.group_id,
        limit=limit,
    )
    return JSONResponse(content={"conversations": conversations})


@router.post("/")
async def create_conversation(request: Request, body: CreateConversationBody) -> JSONResponse:
    tenant = get_tenant(request)
    store = get_conversation_store()
    conv_id = await create_group_conversation(
        store,
        group_id=tenant.group_id,
        session_date=body.session_date,
        title=body.title,
    )
    return JSONResponse(content={"id": conv_id})


@router.get("/{conversation_id}/messages")
async def get_messages(request: Request, conversation_id: str) -> JSONResponse:
    tenant = get_tenant(request)
    store = get_conversation_store()
    messages = await get_group_conversation_messages(
        store,
        conversation_id=conversation_id,
        group_id=tenant.group_id,
    )
    if messages is None:
        return JSONResponse(status_code=404, content={"detail": "Not found"})
    return JSONResponse(content={"messages": messages})


@router.post("/{conversation_id}/messages")
async def append_messages(
    request: Request,
    conversation_id: str,
    body: BulkMessagesBody,
) -> JSONResponse:
    tenant = get_tenant(request)
    store = get_conversation_store()
    ids = await append_group_conversation_messages(
        store,
        conversation_id=conversation_id,
        messages=body.messages,
        group_id=tenant.group_id,
    )
    if ids is None:
        return JSONResponse(status_code=404, content={"detail": "Not found"})
    return JSONResponse(content={"ids": ids})


@router.patch("/{conversation_id}")
async def update_conversation(
    request: Request,
    conversation_id: str,
    body: UpdateConversationBody,
) -> JSONResponse:
    tenant = get_tenant(request)
    store = get_conversation_store()
    updated = await update_group_conversation_title(
        store,
        conversation_id=conversation_id,
        group_id=tenant.group_id,
        title=body.title,
    )
    if not updated:
        return JSONResponse(status_code=404, content={"detail": "Not found"})
    return JSONResponse(content={"status": "updated"})


@router.delete("/{conversation_id}")
async def delete_conversation(request: Request, conversation_id: str) -> JSONResponse:
    tenant = get_tenant(request)
    store = get_conversation_store()
    deleted = await delete_group_conversation(
        store,
        conversation_id=conversation_id,
        group_id=tenant.group_id,
    )
    if not deleted:
        return JSONResponse(status_code=404, content={"detail": "Not found"})
    return JSONResponse(content={"status": "deleted"})
