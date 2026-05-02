# Helix SROP — AI Support Concierge

**Stateful RAG Orchestration Pipeline** for the Helix B2B dev-tools platform.

## Architecture

```
POST /v1/chat/{session_id}
         │
         ▼
┌─────────────────────────────┐
│  SROP Pipeline              │
│  1. Load SessionState (DB)  │
│  2. Inject context → agent  │
│  3. Run ADK orchestrator    │
│  4. Extract events/trace    │
│  5. Persist state + trace   │
└─────────────┬───────────────┘
              │ routes via ADK AgentTool
        ┌─────┴──────┐
        ▼            ▼
  KnowledgeAgent  AccountAgent
  (RAG + Chroma)  (mock DB tools)
        │
   Chroma DB       SQLite
   (doc chunks)  (sessions, traces)
```

## Setup (< 5 min)

```bash
# 1. Clone and enter
git clone <repo> && cd helix-srop-assignment

# 2. Create venv + install
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env — add your GOOGLE_API_KEY

# 4. Seed the vector store
python -m app.rag.ingest --path docs/

# 5. Run
uvicorn app.main:app --reload
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/sessions` | Create session. Body: `{user_id, plan_tier}` |
| POST | `/v1/chat/{session_id}` | Send message. Returns `{reply, routed_to, trace_id}` |
| GET | `/v1/traces/{trace_id}` | Fetch structured trace for one turn |
| GET | `/healthz` | Health check |

**Idempotency (E1):** Pass `Idempotency-Key: <uuid>` header on chat requests.

## Design Decisions

### Session Persistence — Pattern C (state-only injection)
State (`user_id`, `plan_tier`, `last_agent`, `turn_count`) is stored as JSON
in the `sessions.state` column and injected into the root agent's system
instruction on every turn. The ADK runner is stateless (`InMemoryRunner`
created per request); persistence lives entirely in SQLite.

**Tradeoffs:** Simplest implementation, smallest context window usage, no ADK
session service to maintain. Downside: agent doesn't see raw conversation
history — only the metadata. For this use-case (routing + FAQ), that's fine.

### Chunking — Heading-aware
Split markdown on `##`/`###` boundaries; sub-chunk long sections by sentence.
Rationale: the Helix docs corpus is well-structured markdown where each section
under a heading is a self-contained concept. This keeps semantic coherence and
avoids cutting through code examples mid-block.

### Embedding — `text-embedding-004` (Google)
Same provider as the LLM (single API key), 768-dimensional, free tier adequate
for development.

### Vector Store — Chroma (persistent)
No server required — embedded library with file-system persistence. Upsert
by chunk ID makes re-ingest idempotent. Simple Python API.

## Extensions Implemented
| Ext | Description |
|-----|------------|
| E1 — Idempotency | `Idempotency-Key` header deduplicates requests |
| E5 — Guardrails | Root agent instruction refuses out-of-scope queries |
| E6 — Docker | `Dockerfile` + `docker-compose.yml` |

## Running Tests

```bash
pytest -q
```

Tests mock the ADK boundary (`pipeline.run`) so no LLM calls are made.
`test_search_docs_returns_results_with_chunk_ids` requires the vector store
to be seeded first; it skips gracefully otherwise.

## Docker

```bash
docker compose up
```

Volumes mount `helix_srop.db` and `chroma_db/` so state survives restarts.

## Known Limitations
- AccountAgent uses mock data (the wiring is real; swap for actual DB queries)
- Pattern C doesn't pass full conversation history — the agent doesn't remember
  exact prior messages, only metadata
- No rate-limit handling on Gemini API (tenacity retry is configured but not
  hooked into ADK runner yet)

## Time Spent
| Task | Time |
|------|------|
| Codebase exploration + design decisions | 30 min |
| RAG (ingest + retriever + search_docs) | 35 min |
| ADK agents (knowledge, account, orchestrator) | 30 min |
| Pipeline + state management | 35 min |
| API routes + error handling | 20 min |
| Tests | 20 min |
| Extensions (E1, E5, E6) | 25 min |
| README | 15 min |
| **Total** | **~3h 30min** |
