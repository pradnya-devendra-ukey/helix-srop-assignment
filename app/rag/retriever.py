"""
Chroma vector-store client singleton + Google embedding utilities.

Shared by ingest.py (write path) and search_docs.py (read path).
Uses a module-level singleton so the Chroma client is created once per process.

Uses google.genai (new SDK). Model: gemini-embedding-001 (replaces the
deprecated text-embedding-004 which is no longer available via the API).
"""
from __future__ import annotations

import logging

import chromadb
from google import genai
from google.genai import types

from app.settings import settings

log = logging.getLogger(__name__)

COLLECTION_NAME = "helix_docs"
EMBED_MODEL = "gemini-embedding-001"  # 3072-dim; replaces text-embedding-004

# ── Lazy singletons ────────────────────────────────────────────────────────────
_chroma_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None
_genai_client: genai.Client | None = None


def _get_genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=settings.google_api_key)
    return _genai_client


def get_collection() -> chromadb.Collection:
    """Return (and lazily initialise) the Chroma collection."""
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            "chroma_collection_ready",
            extra={"path": settings.chroma_persist_dir, "name": COLLECTION_NAME},
        )
    return _collection


# ── Embedding helpers ──────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of documents (ingest time). Returns list of vectors."""
    client = _get_genai_client()
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return [e.values for e in result.embeddings]


def embed_query(query: str) -> list[float]:
    """Embed a single query string (retrieval time)."""
    client = _get_genai_client()
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[query],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values
