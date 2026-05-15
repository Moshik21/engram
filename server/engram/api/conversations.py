"""Conversation persistence API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from engram.api.deps import get_conversation_store
from engram.retrieval.conversation_persistence import (
    build_api_conversation_append_messages_surface,
    build_api_conversation_create_surface,
    build_api_conversation_delete_surface,
    build_api_conversation_list_surface,
    build_api_conversation_messages_surface,
    build_api_conversation_update_surface,
    conversation_not_found_payload,
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
    payload = await build_api_conversation_list_surface(
        store,
        group_id=tenant.group_id,
        limit=limit,
    )
    return JSONResponse(content=payload)


@router.post("/")
async def create_conversation(request: Request, body: CreateConversationBody) -> JSONResponse:
    tenant = get_tenant(request)
    store = get_conversation_store()
    payload = await build_api_conversation_create_surface(
        store,
        group_id=tenant.group_id,
        session_date=body.session_date,
        title=body.title,
    )
    return JSONResponse(content=payload)


@router.get("/{conversation_id}/messages")
async def get_messages(request: Request, conversation_id: str) -> JSONResponse:
    tenant = get_tenant(request)
    store = get_conversation_store()
    payload = await build_api_conversation_messages_surface(
        store,
        conversation_id=conversation_id,
        group_id=tenant.group_id,
    )
    if payload is None:
        return JSONResponse(status_code=404, content=conversation_not_found_payload())
    return JSONResponse(content=payload)


@router.post("/{conversation_id}/messages")
async def append_messages(
    request: Request,
    conversation_id: str,
    body: BulkMessagesBody,
) -> JSONResponse:
    tenant = get_tenant(request)
    store = get_conversation_store()
    payload = await build_api_conversation_append_messages_surface(
        store,
        conversation_id=conversation_id,
        messages=body.messages,
        group_id=tenant.group_id,
    )
    if payload is None:
        return JSONResponse(status_code=404, content=conversation_not_found_payload())
    return JSONResponse(content=payload)


@router.patch("/{conversation_id}")
async def update_conversation(
    request: Request,
    conversation_id: str,
    body: UpdateConversationBody,
) -> JSONResponse:
    tenant = get_tenant(request)
    store = get_conversation_store()
    payload = await build_api_conversation_update_surface(
        store,
        conversation_id=conversation_id,
        group_id=tenant.group_id,
        title=body.title,
    )
    if payload is None:
        return JSONResponse(status_code=404, content=conversation_not_found_payload())
    return JSONResponse(content=payload)


@router.delete("/{conversation_id}")
async def delete_conversation(request: Request, conversation_id: str) -> JSONResponse:
    tenant = get_tenant(request)
    store = get_conversation_store()
    payload = await build_api_conversation_delete_surface(
        store,
        conversation_id=conversation_id,
        group_id=tenant.group_id,
    )
    if payload is None:
        return JSONResponse(status_code=404, content=conversation_not_found_payload())
    return JSONResponse(content=payload)
