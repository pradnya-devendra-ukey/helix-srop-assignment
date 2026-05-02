# Helix SROP — AI Support Concierge

**Stateful RAG Orchestration Pipeline** for the Helix B2B dev-tools platform.

---

## Architecture

```
POST /v1/chat/{session_id}
         │
         ▼
┌──────────────────────────────────────────┐
│  SROP Pipeline  (app/srop/pipeline.py)   │
│                                          │
│  1. Load SessionState from SQLite        │
│  2. Build root agent (inject context)    │
│  3. Run ADK — InMemoryRunner (1 turn)    │
│  4. Extract routing + tool calls         │
│  5. Write AgentTrace to SQLite           │
│  6. Persist updated SessionState         │
│  7. Return PipelineResult                │
└──────────────────┬───────────────────────┘
                   │  routes via ADK AgentTool
             ┌─────┴──────┐
             ▼            ▼
       KnowledgeAgent  AccountAgent
       (RAG + Chroma)  (mock DB tools)
             │
        Chroma DB            SQLite
       (162 chunks)    (sessions, messages,
                        traces, users,
                        idempotency_records)
```

### Module Map

| Path | Responsibility |
|------|----------------|
| `app/main.py` | FastAPI app, lifespan (DB init + logging), route registration |
| `app/settings.py` | Pydantic-settings config loaded from `.env` |
| `app/api/routes_sessions.py` | `POST /v1/sessions` |
| `app/api/routes_chat.py` | `POST /v1/chat/{session_id}` + E1 idempotency |
| `app/api/routes_traces.py` | `GET /v1/traces/{trace_id}` |
| `app/api/errors.py` | Typed `HelixError` exceptions → RFC 7807 JSON |
| `app/db/models.py` | SQLAlchemy 2.x models (User, Session, Message, AgentTrace, IdempotencyRecord) |
| `app/db/session.py` | Async engine, sessionmaker, `get_db` dependency |
| `app/srop/pipeline.py` | Core pipeline — ADK runner, event extraction, trace write |
| `app/srop/state.py` | `SessionState` Pydantic model (JSON ↔ SQLite) |
| `app/agents/orchestrator.py` | Root `LlmAgent` with `AgentTool` routing |
| `app/agents/knowledge.py` | `KnowledgeAgent` — answers doc questions via `search_docs` |
| `app/agents/account.py` | `AccountAgent` — account/build info via mock tools |
| `app/agents/tools/search_docs.py` | Top-k Chroma query → formatted string with chunk IDs |
| `app/agents/tools/account_tools.py` | `get_recent_builds`, `get_account_status` (mock data) |
| `app/rag/ingest.py` | CLI: markdown → chunks → embeddings → Chroma upsert |
| `app/rag/retriever.py` | Chroma client + Google `gemini-embedding-001` utilities |
| `app/obs/logging.py` | `structlog` JSON log configuration |
| `client.py` | Interactive demo client — runs all 9 test scenarios |
| `tests/conftest.py` | Pytest fixtures: in-memory DB, `mock_adk`, async HTTP client |
| `tests/test_api.py` | 7 integration tests (ADK boundary mocked) |
| `tests/test_retriever.py` | 4 unit tests for chunking + retrieval |

---

## Setup (< 5 min)

```bash
# 1. Clone
git clone https://github.com/pradnya-devendra-ukey/helix-srop-assignment
cd helix-srop-assignment

# 2. Virtual env + install
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env — set GOOGLE_API_KEY=<your-gemini-key>

# 4. Seed the vector store (one-time, ~162 chunks from 14 docs)
python -m app.rag.ingest --path docs/

# 5. Run
uvicorn app.main:app --reload
# → http://localhost:8000/docs
```

> **Note:** `GOOGLE_API_KEY` requires access to `gemini-2.5-flash` (LLM) and
> `gemini-embedding-001` (embeddings). Both are available on standard Gemini
> Developer API keys from [aistudio.google.com](https://aistudio.google.com/app/apikey).

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/sessions` | Create session. Body: `{"user_id", "plan_tier"}` |
| `POST` | `/v1/chat/{session_id}` | Send message. Returns `{reply, routed_to, trace_id}` |
| `GET` | `/v1/traces/{trace_id}` | Structured trace for one turn (chunk IDs, tool calls, latency) |
| `GET` | `/healthz` | Health check |

**Idempotency (E1):** Pass `Idempotency-Key: <uuid>` header on chat requests to make retries safe.

### Quick smoke test

```bash
# 1. Create session
curl -X POST http://localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "alice", "plan_tier": "pro"}'
# → {"session_id": "...", "user_id": "alice"}

# 2. Knowledge query (RAG)
curl -X POST http://localhost:8000/v1/chat/<session_id> \
  -H "Content-Type: application/json" \
  -d '{"content": "How do I rotate a deploy key?"}'
# → {"reply": "According to [chunk_xxx]...", "routed_to": "knowledge", "trace_id": "..."}

# 3. Account query (cross-turn state)
curl -X POST http://localhost:8000/v1/chat/<session_id> \
  -H "Content-Type: application/json" \
  -d '{"content": "Show me my last 3 builds"}'
# → {"reply": "Recent 3 builds for user 'alice'...", "routed_to": "account", ...}

# 4. Fetch trace
curl http://localhost:8000/v1/traces/<trace_id>
```

Or run the full interactive demo client:

```bash
python client.py
# Runs all 9 test scenarios end-to-end: healthz, session, RAG turn,
# account turn, trace fetch, guardrail, idempotency, error cases
```

---

## Running Tests

```bash
pytest -q
# 11 tests, 0 failures — no LLM calls, ADK boundary is mocked
```

`test_search_docs_returns_results_with_chunk_ids` requires a seeded vector
store; it skips gracefully if `python -m app.rag.ingest` hasn't been run.

---

## Docker

```bash
# Set GOOGLE_API_KEY in .env first, then:
docker compose up
# → API on http://localhost:8000
# Volumes mount helix_srop.db and chroma_db/ so state survives restarts
```

---

## Design Decisions

### Session Persistence — Pattern C (state-only injection)

State (`user_id`, `plan_tier`, `last_agent`, `turn_count`) is stored as JSON
in the `sessions.state` column and injected into the root agent's system
instruction on every turn.

```python
class SessionState(BaseModel):
    user_id: str
    plan_tier: Literal["free", "pro", "enterprise"] = "free"
    last_agent: Literal["knowledge", "account", "smalltalk"] | None = None
    turn_count: int = 0
```

The ADK `InMemoryRunner` is stateless (created per request); all persistence
lives in SQLite. Killing and restarting `uvicorn` preserves full context.

**Tradeoffs:** Minimal token overhead — only 4 fields injected, not full
conversation history. Agent doesn't remember exact prior phrasing, only
metadata. Sufficient for routing + FAQ; extend with a `recent_messages` ring
buffer for richer multi-turn dialogue.

### ADK Agent Routing

Uses the exact `AgentTool` pattern from the assignment — routing is entirely
LLM-driven via tool selection, never string parsing:

```python
root_agent = LlmAgent(
    name="srop_root",
    model="gemini-2.5-flash",
    instruction=instruction,   # injected per-turn with session context
    tools=[
        AgentTool(agent=knowledge_agent),
        AgentTool(agent=account_agent),
    ],
)
```

### Chunking — Heading-aware

Split markdown on `##`/`###` boundaries; sub-chunk long sections by sentence.
**Rationale:** Helix docs are structured markdown where each section under a
heading is a semantically complete concept. Heading-aware chunks avoid cutting
through code examples and keep retrieval context coherent.

14 docs → **162 chunks** at default 800 char chunk size.

### Embeddings — `gemini-embedding-001` (Google)

- 3072 dimensions, same API key as the LLM (zero extra credential management)
- `RETRIEVAL_DOCUMENT` task type at ingest; `RETRIEVAL_QUERY` at search time
  (asymmetric retrieval — recommended by Google for RAG)
- Replaces the deprecated `text-embedding-004` model

### Vector Store — Chroma (embedded, persistent)

- No separate server — embedded library with `PersistentClient` file-system storage
- `collection.upsert()` with SHA-256 deterministic chunk IDs makes re-ingest idempotent
- Cosine similarity (`hnsw:space: cosine`); score threshold 0.45 to filter noise

---

## Extensions Implemented

| Ext | Points | Implementation |
|-----|--------|----------------|
| **E1 — Idempotency** | 6 | `Idempotency-Key` header; SHA-256 of `session_id:key` → `idempotency_records` table; replay returns original `trace_id`. Tested in `test_idempotency_key_deduplicates`. |
| **E5 — Guardrails** | 4 | Root agent system prompt explicit refusal policy: off-topic queries (poems, personal advice, etc.) → `"I can only assist with Helix product questions and your account information."` No sub-agent called. |
| **E6 — Docker** | 3 | `Dockerfile` (python:3.11-slim, pre-ingests docs at build time). `docker-compose.yml` with volume mounts + `/healthz` healthcheck. |

---

## Known Limitations

1. **AccountAgent uses mock data** — `get_recent_builds` / `get_account_status`
   return deterministic pseudo-random data seeded by `user_id`. The ADK
   wiring is real; swap with actual DB queries for production.

2. **Pattern C omits conversation history** — agent sees metadata
   (`last_agent`, `turn_count`, `plan_tier`) but not prior messages verbatim.
   "Explain more" on turn 3 may not resolve correctly without a `recent_messages`
   buffer in `SessionState`.

3. **No Gemini rate-limit retry** — `tenacity` is in requirements but not yet
   wired into the ADK runner. 429s from Gemini surface as 500s under load.

4. **Single-container Docker** — SQLite + embedded Chroma work for development.
   Production would use Postgres (`asyncpg`) + hosted Chroma or Weaviate.

5. **`routed_to` detection** — extracted from ADK event `author` field
   (`"knowledge" in author.lower()`). If ADK changes event schema this needs
   updating. Functional but fragile.

---

## Time Breakdown

| Task | Time |
|------|------|
| Codebase exploration + design decisions | 30 min |
| DB models + settings + FastAPI boilerplate | 25 min |
| RAG: ingest.py + retriever + search_docs | 35 min |
| ADK agents: knowledge, account, orchestrator | 30 min |
| Pipeline: state in/out, ADK runner, trace extraction | 35 min |
| API routes + error handling | 20 min |
| Extensions (E1, E5, E6) | 25 min |
| Tests: conftest + integration + unit | 20 min |
| Bug fixes (SDK migration, model updates) | 20 min |
| client.py + README | 20 min |
| **Total** | **~3h 40min** |
