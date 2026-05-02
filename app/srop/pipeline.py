"""
SROP pipeline entrypoint — called by the chat route.

Pattern C session persistence:
  - Load SessionState JSON from sessions.state column
  - Inject it into the root agent's system instruction
  - Run one ADK turn (InMemoryRunner + InMemorySessionService — stateless runner,
    state lives in our DB not in ADK)
  - Extract routing decision and tool calls from the ADK event stream
  - Persist AgentTrace + Messages + updated SessionState to DB
  - Return PipelineResult to the route handler

State survives process restarts because it lives in SQLite, not in ADK's
InMemorySessionService.
"""
from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import structlog
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.orchestrator import build_root_agent
from app.agents.tools.search_docs import extract_chunk_ids_from_result
from app.api.errors import SessionNotFoundError, UpstreamTimeoutError
from app.db.models import AgentTrace, Message, Session as DbSession
from app.settings import settings
from app.srop.state import SessionState

log = structlog.get_logger(__name__)

APP_NAME = "helix_srop"


@dataclass
class PipelineResult:
    content: str
    routed_to: str
    trace_id: str


async def run(session_id: str, user_message: str, db: AsyncSession) -> PipelineResult:
    """Run one SROP pipeline turn. Raises SessionNotFoundError or UpstreamTimeoutError."""
    trace_id = str(uuid.uuid4())
    start_time = time.monotonic()

    structlog.contextvars.bind_contextvars(session_id=session_id, trace_id=trace_id)

    # ── 1. Load session + state ───────────────────────────────────────────────
    result = await db.execute(select(DbSession).where(DbSession.session_id == session_id))
    db_session = result.scalar_one_or_none()
    if db_session is None:
        raise SessionNotFoundError(f"Session '{session_id}' not found.")

    state = SessionState.from_db_dict(db_session.state)
    log.info("pipeline_started", user_id=state.user_id, turn=state.turn_count)

    # ── 2. Build root agent with injected context ─────────────────────────────
    root_agent = build_root_agent(
        user_id=state.user_id,
        plan_tier=state.plan_tier,
        turn_count=state.turn_count,
        last_agent=state.last_agent,
    )

    # ── 3. Run ADK (single-turn, stateless runner) ────────────────────────────
    session_svc = InMemorySessionService()
    adk_session = await session_svc.create_session(
        app_name=APP_NAME,
        user_id=state.user_id,
    )
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_svc)

    tool_calls: list[dict] = []
    retrieved_chunk_ids: list[str] = []
    routed_to = "smalltalk"
    final_text = ""

    async def _collect_events() -> None:
        nonlocal routed_to, final_text

        pending_tool_call: dict | None = None

        events = runner.run_async(
            user_id=state.user_id,
            session_id=adk_session.id,
            new_message={"role": "user", "parts": [{"text": user_message}]},
        )

        async for event in events:
            # ── Tool call started ─────────────────────────────────────────
            content = getattr(event, "content", None)
            if content and getattr(content, "parts", None):
                for part in content.parts:
                    fn_call = getattr(part, "function_call", None)
                    if fn_call:
                        pending_tool_call = {
                            "tool_name": fn_call.name,
                            "args": dict(fn_call.args) if fn_call.args else {},
                            "result": None,
                        }
                        tool_calls.append(pending_tool_call)

                    fn_resp = getattr(part, "function_response", None)
                    if fn_resp:
                        resp_val = fn_resp.response if fn_resp.response is not None else {}
                        # Fill result into the last pending tool call
                        if tool_calls:
                            tool_calls[-1]["result"] = resp_val
                        # Extract chunk IDs if this was a search_docs call
                        resp_text = str(resp_val)
                        retrieved_chunk_ids.extend(extract_chunk_ids_from_result(resp_text))

            # ── Final response ────────────────────────────────────────────
            if event.is_final_response():
                author = getattr(event, "author", "") or ""
                if "knowledge" in author.lower():
                    routed_to = "knowledge"
                elif "account" in author.lower():
                    routed_to = "account"
                else:
                    routed_to = "smalltalk"

                if content and getattr(content, "parts", None):
                    final_text = "".join(
                        getattr(p, "text", "") or "" for p in content.parts
                    )

    try:
        await asyncio.wait_for(_collect_events(), timeout=settings.llm_timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise UpstreamTimeoutError(
            f"LLM did not respond within {settings.llm_timeout_seconds}s"
        ) from exc

    latency_ms = int((time.monotonic() - start_time) * 1000)
    log.info("pipeline_done", routed_to=routed_to, latency_ms=latency_ms)

    # ── 4. Persist trace ──────────────────────────────────────────────────────
    trace = AgentTrace(
        trace_id=trace_id,
        session_id=session_id,
        routed_to=routed_to,
        tool_calls=tool_calls,
        retrieved_chunk_ids=list(dict.fromkeys(retrieved_chunk_ids)),  # deduplicate, keep order
        latency_ms=latency_ms,
        created_at=datetime.now(timezone.utc),
    )
    db.add(trace)

    # ── 5. Persist messages ───────────────────────────────────────────────────
    db.add(Message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role="user",
        content=user_message,
        trace_id=trace_id,
        created_at=datetime.now(timezone.utc),
    ))
    db.add(Message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role="assistant",
        content=final_text,
        trace_id=trace_id,
        created_at=datetime.now(timezone.utc),
    ))

    # ── 6. Update session state ───────────────────────────────────────────────
    state.last_agent = routed_to  # type: ignore[assignment]
    state.turn_count += 1
    db_session.state = state.to_db_dict()
    db_session.updated_at = datetime.now(timezone.utc)

    await db.commit()

    return PipelineResult(content=final_text, routed_to=routed_to, trace_id=trace_id)
