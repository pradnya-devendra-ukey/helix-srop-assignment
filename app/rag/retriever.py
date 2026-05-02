"""
Chroma vector-store client singleton + Google embedding utilities.

Shared by ingest.py (write path) and search_docs.py (read path).
Uses a module-level singleton so the Chroma client is created once per process.
"""
from __future__ import annotations

import logging

import chromadb
import google.generativeai as genai

from app.settings import settings

log = logging.getLogger(__name__)

COLLECTION_NAME = "helix_docs"
EMBED_MODEL = "models/text-embedding-004"

# ── Lazy singletons ────────────────────────────────────────────────────────────
_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def _configure_genai() -> None:
    genai.configure(api_key=settings.google_api_key)


def get_collection() -> chromadb.Collection:
    """Return (and lazily initialise) the Chroma collection."""
    global _client, _collection
    if _collection is None:
        _configure_genai()
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("chroma_collection_ready", path=settings.chroma_persist_dir, name=COLLECTION_NAME)
    return _collection


# ── Embedding helpers ──────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of documents (ingest time). Returns list of vectors."""
    _configure_genai()
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=texts,
        task_type="retrieval_document",
    )
    embeddings = result["embedding"]
    # API returns list[float] for single input, list[list[float]] for multiple
    if texts and isinstance(embeddings[0], float):
        return [embeddings]  # type: ignore[list-item]
    return embeddings  # type: ignore[return-value]


def embed_query(query: str) -> list[float]:
    """Embed a single query string (retrieval time)."""
    _configure_genai()
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=query,
        task_type="retrieval_query",
    )
    return result["embedding"]  # type: ignore[return-value]
