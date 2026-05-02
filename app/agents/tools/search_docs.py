"""
search_docs tool — used by KnowledgeAgent.

Queries the Chroma vector store for the top-k most relevant documentation
chunks. Returns a formatted string with chunk IDs so the agent can cite
sources, and also a machine-readable list for trace extraction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.retriever import embed_query, get_collection

SCORE_THRESHOLD = 0.45  # discard low-quality results below this


@dataclass
class DocChunk:
    chunk_id: str
    score: float
    content: str
    metadata: dict


def _last_search_chunk_ids() -> list[str]:
    """Module-level storage so the pipeline can grab chunk IDs after tool call."""
    return _LAST_CHUNK_IDS


_LAST_CHUNK_IDS: list[str] = []


async def search_docs(query: str, k: int = 5, product_area: str | None = None) -> str:
    """
    Search Helix product documentation for the top-k most relevant chunks.

    Args:
        query: natural-language question from the user
        k: number of results to return (default 5)
        product_area: optional metadata filter (e.g. "security", "ci-cd")

    Returns:
        Formatted string with chunk IDs and content for the agent to cite.
        Always reference chunk IDs as [chunk_xxx] in your answer.
    """
    global _LAST_CHUNK_IDS

    collection = get_collection()
    query_vec = embed_query(query)

    where: dict | None = {"product_area": product_area} if product_area else None

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(k, collection.count() or k),
        where=where,
        include=["documents", "distances", "metadatas"],
    )

    chunks: list[DocChunk] = []
    if results["ids"] and results["ids"][0]:
        for chunk_id, distance, doc, meta in zip(
            results["ids"][0],
            results["distances"][0],
            results["documents"][0],
            results["metadatas"][0],
        ):
            score = round(1.0 - distance, 4)
            if score >= SCORE_THRESHOLD:
                chunks.append(DocChunk(chunk_id=chunk_id, score=score, content=doc, metadata=meta))

    chunks.sort(key=lambda c: c.score, reverse=True)
    _LAST_CHUNK_IDS = [c.chunk_id for c in chunks]

    if not chunks:
        return "No relevant documentation found for this query."

    parts: list[str] = []
    for c in chunks:
        source = c.metadata.get("source", "unknown")
        parts.append(
            f"[{c.chunk_id}] (score: {c.score:.2f}, source: {source})\n{c.content}"
        )
    return "\n\n---\n\n".join(parts)


def extract_chunk_ids_from_result(result_text: str) -> list[str]:
    """Parse chunk IDs out of a search_docs result string (for pipeline trace)."""
    return re.findall(r"chunk_[a-f0-9]{16}", result_text)
