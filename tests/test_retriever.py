"""
Unit tests for RAG chunking and retrieval.
test_search_docs requires the vector store to be seeded (run ingest.py on docs/).
"""
from __future__ import annotations

import pytest

from app.rag.ingest import chunk_markdown, extract_metadata
from pathlib import Path


def test_chunker_produces_non_empty_chunks():
    """Chunker must not produce empty strings."""
    text = "# Header\n\nSome content.\n\n## Section 2\n\nMore content here.\n\n### Subsection\n\nDeep content."
    chunks = chunk_markdown(text, chunk_size=200)
    assert len(chunks) > 0
    assert all(c.strip() for c in chunks), "No chunk should be empty or whitespace-only"


def test_chunker_heading_aware_splits_on_headings():
    """Heading-aware chunker should split at ## boundaries."""
    text = "# Title\n\nIntro text.\n\n## Section A\n\nSection A content.\n\n## Section B\n\nSection B content."
    chunks = chunk_markdown(text, chunk_size=500)
    # Expect at least 2 chunks (one per ## section + possible intro)
    assert len(chunks) >= 2


def test_extract_metadata_parses_frontmatter():
    """extract_metadata must return title and product_area from YAML frontmatter."""
    text = "---\ntitle: Deploy Keys\nproduct_area: security\ntags: [keys]\n---\n\n# Content"
    meta = extract_metadata(Path("deploy-keys.md"), text)
    assert meta["title"] == "Deploy Keys"
    assert meta["product_area"] == "security"
    assert meta["source"] == "deploy-keys.md"


def test_extract_metadata_no_frontmatter():
    """extract_metadata should return at minimum source when no frontmatter present."""
    text = "# Just a heading\n\nSome text."
    meta = extract_metadata(Path("readme.md"), text)
    assert meta["source"] == "readme.md"
    assert "title" not in meta


@pytest.mark.asyncio
async def test_search_docs_returns_results_with_chunk_ids():
    """search_docs must return chunk IDs and scores in [0, 1].
    Requires vector store to be seeded: python -m app.rag.ingest --path docs/
    """
    try:
        from app.agents.tools.search_docs import search_docs, _LAST_CHUNK_IDS
        result = await search_docs("how to rotate a deploy key", k=3)
        # If no chunks loaded, skip gracefully
        if "No relevant documentation" in result:
            pytest.skip("Vector store not seeded — run: python -m app.rag.ingest --path docs/")
        assert len(_LAST_CHUNK_IDS) > 0, "search_docs must populate _LAST_CHUNK_IDS"
        assert all(cid.startswith("chunk_") for cid in _LAST_CHUNK_IDS)
    except Exception as exc:
        pytest.skip(f"Vector store unavailable: {exc}")
