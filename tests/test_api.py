"""
Integration tests — exercise the full SROP pipeline.
LLM mocked at the ADK boundary (pipeline.run is patched, not the HTTP layer).
"""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_create_session(client):
    resp = await client.post("/v1/sessions", json={"user_id": "u_test_001"})
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert data["user_id"] == "u_test_001"


@pytest.mark.asyncio
async def test_healthz(client):
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_session_not_found_returns_404(client):
    resp = await client.post("/v1/chat/nonexistent-session-id", json={"content": "hello"})
    assert resp.status_code == 404
    body = resp.json()
    assert body["title"] == "SESSION_NOT_FOUND"


@pytest.mark.asyncio
async def test_knowledge_query_routes_correctly(client, mock_adk):
    """
    Core integration test — two turns, knowledge routing, state persistence.
    Turn 1: knowledge query → routed_to='knowledge', trace has chunk IDs.
    Turn 2: plan_tier query → reply includes 'pro' (from state persisted after turn 1).
    """
    # Create session with plan_tier=pro
    sess = await client.post("/v1/sessions", json={"user_id": "u_test_002", "plan_tier": "pro"})
    assert sess.status_code == 200
    session_id = sess.json()["session_id"]

    # Turn 1 — knowledge query
    r1 = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "How do I rotate a deploy key?"},
    )
    assert r1.status_code == 200
    assert r1.json()["routed_to"] == "knowledge"
    trace_id = r1.json()["trace_id"]

    # Trace must have chunk IDs
    trace = await client.get(f"/v1/traces/{trace_id}")
    assert trace.status_code == 200
    trace_data = trace.json()
    assert len(trace_data["retrieved_chunk_ids"]) > 0

    # Turn 2 — plan tier follow-up (state injection test)
    r2 = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "What is my plan tier?"},
    )
    assert r2.status_code == 200
    # Agent should know plan_tier='pro' from persisted state
    assert "pro" in r2.json()["reply"].lower()


@pytest.mark.asyncio
async def test_trace_not_found_returns_404(client):
    resp = await client.get("/v1/traces/nonexistent-trace-id")
    assert resp.status_code == 404
    assert resp.json()["title"] == "TRACE_NOT_FOUND"


@pytest.mark.asyncio
async def test_idempotency_key_deduplicates(client, mock_adk):
    """E1 — same Idempotency-Key header returns same response without re-running pipeline."""
    sess = await client.post("/v1/sessions", json={"user_id": "u_idem_001"})
    session_id = sess.json()["session_id"]

    headers = {"Idempotency-Key": "test-idem-key-001"}
    r1 = await client.post(
        f"/v1/chat/{session_id}", json={"content": "How do I rotate a deploy key?"}, headers=headers
    )
    assert r1.status_code == 200

    r2 = await client.post(
        f"/v1/chat/{session_id}", json={"content": "How do I rotate a deploy key?"}, headers=headers
    )
    assert r2.status_code == 200
    # Same trace_id means it was served from cache, not re-run
    assert r1.json()["trace_id"] == r2.json()["trace_id"]


@pytest.mark.asyncio
async def test_guardrail_refusal(client, mock_adk):
    """E5 — out-of-scope queries return a refusal (smalltalk routing)."""
    sess = await client.post("/v1/sessions", json={"user_id": "u_guard_001"})
    session_id = sess.json()["session_id"]

    r = await client.post(f"/v1/chat/{session_id}", json={"content": "Write me a poem about the sea"})
    assert r.status_code == 200
    # The mock routes this to smalltalk; in prod the guardrail prompt handles it
    assert r.json()["routed_to"] == "smalltalk"
