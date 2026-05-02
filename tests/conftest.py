"""
Test fixtures.

Key fixtures:
- `client`: async test client with in-memory SQLite DB
- `mock_adk`: patches pipeline.run at the ADK boundary so tests don't hit real LLM
- `seeded_session`: pre-created session in the DB for convenience
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base
from app.db.session import get_db
from app.main import app
from app.srop.pipeline import PipelineResult

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db() -> None:  # type: ignore[misc]
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield  # type: ignore[misc]
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db() -> AsyncSession:  # type: ignore[misc]
    async with TestSessionLocal() as session:
        yield session  # type: ignore[misc]


@pytest_asyncio.fixture
async def client(db: AsyncSession):  # type: ignore[misc]
    """Async test client with DB overridden to in-memory SQLite."""

    async def _override_get_db():  # type: ignore[return]
        yield db

    app.dependency_overrides[get_db] = _override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def mock_adk(monkeypatch: pytest.MonkeyPatch):
    """
    Patch pipeline.run at the ADK boundary — no real LLM calls in tests.

    The mock inspects the user message and returns a canned PipelineResult:
    - Messages with 'deploy' or 'rotate' → routed_to='knowledge'
    - Messages with 'build' or 'account' or 'plan' → routed_to='account'
    - Everything else → routed_to='smalltalk'

    The reply for account/plan queries includes the plan_tier so state-injection
    tests can assert it appears in the response.
    """
    import uuid

    async def mock_run(session_id: str, user_message: str, db: AsyncSession) -> PipelineResult:
        from app.db.models import AgentTrace, Message, Session as DbSession
        from sqlalchemy import select
        from datetime import datetime, timezone

        # Determine routing
        msg_lower = user_message.lower()
        if any(kw in msg_lower for kw in ("deploy", "rotate", "how", "what", "doc")):
            routed_to = "knowledge"
            reply = "According to [chunk_abc1234567890123], you can rotate a deploy key via the dashboard."
        elif any(kw in msg_lower for kw in ("build", "account", "status", "plan", "tier")):
            # Load session state to reflect plan_tier in reply
            result = await db.execute(select(DbSession).where(DbSession.session_id == session_id))
            db_session = result.scalar_one_or_none()
            plan = db_session.state.get("plan_tier", "free") if db_session else "free"
            routed_to = "account"
            reply = f"Your plan tier is {plan}. You have 3 recent builds."
        else:
            routed_to = "smalltalk"
            reply = "Hello! How can I help you with Helix today?"

        trace_id = str(uuid.uuid4())

        # Write trace + messages so trace endpoint works
        trace = AgentTrace(
            trace_id=trace_id,
            session_id=session_id,
            routed_to=routed_to,
            tool_calls=[{"tool_name": "search_docs", "args": {"query": user_message}, "result": reply}]
            if routed_to == "knowledge" else [],
            retrieved_chunk_ids=["chunk_abc1234567890123"] if routed_to == "knowledge" else [],
            latency_ms=42,
            created_at=datetime.now(timezone.utc),
        )
        db.add(trace)
        db.add(Message(
            message_id=str(uuid.uuid4()), session_id=session_id, role="user",
            content=user_message, trace_id=trace_id, created_at=datetime.now(timezone.utc),
        ))
        db.add(Message(
            message_id=str(uuid.uuid4()), session_id=session_id, role="assistant",
            content=reply, trace_id=trace_id, created_at=datetime.now(timezone.utc),
        ))

        # Update session state turn_count + last_agent
        result2 = await db.execute(select(DbSession).where(DbSession.session_id == session_id))
        db_sess = result2.scalar_one_or_none()
        if db_sess:
            state = dict(db_sess.state)
            state["turn_count"] = state.get("turn_count", 0) + 1
            state["last_agent"] = routed_to
            db_sess.state = state
        await db.commit()

        return PipelineResult(content=reply, routed_to=routed_to, trace_id=trace_id)

    monkeypatch.setattr("app.srop.pipeline.run", mock_run)
    monkeypatch.setattr("app.api.routes_chat.pipeline.run", mock_run)
