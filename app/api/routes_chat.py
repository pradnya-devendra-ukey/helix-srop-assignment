"""
POST /v1/chat/{session_id} — send a user message, get assistant reply.

E1 — Idempotency:
  If the request includes an `Idempotency-Key` header, we check whether we've
  already processed a request with that key for this session. If yes, we return
  the stored response immediately without re-running the pipeline.
  This makes retried requests safe (network timeouts, client retries, etc.).
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import IdempotencyRecord
from app.db.session import get_db
from app.srop import pipeline

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    content: str


class ChatResponse(BaseModel):
    reply: str
    routed_to: str
    trace_id: str


@router.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(
    session_id: str,
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> ChatResponse:
    """
    Run one turn of the SROP pipeline.

    Error cases:
    - Session not found → 404 SESSION_NOT_FOUND
    - LLM timeout → 504 UPSTREAM_TIMEOUT
    """
    # ── E1: Idempotency check ─────────────────────────────────────────────────
    if idempotency_key:
        idem_hash = hashlib.sha256(
            f"{session_id}:{idempotency_key}".encode()
        ).hexdigest()
        result = await db.execute(
            select(IdempotencyRecord).where(IdempotencyRecord.idem_hash == idem_hash)
        )
        existing = result.scalar_one_or_none()
        if existing:
            return ChatResponse(
                reply=existing.reply,
                routed_to=existing.routed_to,
                trace_id=existing.trace_id,
            )

    # ── Run pipeline ──────────────────────────────────────────────────────────
    pipeline_result = await pipeline.run(session_id, body.content, db)

    # ── E1: Store idempotency record ──────────────────────────────────────────
    if idempotency_key:
        db.add(IdempotencyRecord(
            idem_hash=idem_hash,
            session_id=session_id,
            reply=pipeline_result.content,
            routed_to=pipeline_result.routed_to,
            trace_id=pipeline_result.trace_id,
            created_at=datetime.now(timezone.utc),
        ))
        await db.commit()

    return ChatResponse(
        reply=pipeline_result.content,
        routed_to=pipeline_result.routed_to,
        trace_id=pipeline_result.trace_id,
    )
