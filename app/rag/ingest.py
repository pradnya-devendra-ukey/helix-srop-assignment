"""
RAG ingest CLI.

Usage:
    python -m app.rag.ingest --path docs/
    python -m app.rag.ingest --path docs/ --chunk-size 800 --chunk-overlap 0

Strategy: heading-aware chunking (split on ## / ###).
Rationale: the docs/ corpus is structured markdown where each section under a
heading is a self-contained concept. Splitting at heading boundaries keeps
semantic coherence and avoids cutting through code examples. Long sections
are sub-chunked by sentence to stay within the embedding window.

Stable chunk IDs: sha256(file_stem::chunk_index)[:16] — deterministic so
re-ingesting does not duplicate entries (Chroma upsert is idempotent on id).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

from app.rag.retriever import embed_texts, get_collection

# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_sentences(text: str, max_chars: int = 800) -> list[str]:
    """Split text on sentence boundaries, accumulating up to max_chars."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(sentence)
        current_len += len(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_markdown(text: str, chunk_size: int = 800, overlap: int = 0) -> list[str]:  # noqa: ARG001
    """
    Heading-aware chunking: split on ## or ### boundaries.
    Sections longer than chunk_size are further split by sentence.
    Returns non-empty strings only.
    """
    sections = re.split(r"\n(?=#{2,3} )", text)
    chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            chunks.extend(chunk_sentences(section, max_chars=chunk_size))
    return [c for c in chunks if c.strip()]


# ── Metadata ───────────────────────────────────────────────────────────────────

def extract_metadata(file_path: Path, text: str) -> dict[str, Any]:
    """
    Parse YAML frontmatter (--- ... ---) from a markdown file.
    Always adds 'source' (filename) to the returned dict.
    """
    meta: dict[str, Any] = {"source": file_path.name}
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if match:
        try:
            parsed = yaml.safe_load(match.group(1)) or {}
            meta.update(parsed)
        except yaml.YAMLError:
            pass
    return meta


# ── Stable IDs ─────────────────────────────────────────────────────────────────

def make_chunk_id(file_path: Path, chunk_index: int) -> str:
    raw = f"{file_path.stem}::{chunk_index}"
    return "chunk_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Ingest ─────────────────────────────────────────────────────────────────────

BATCH_SIZE = 20


async def ingest_directory(docs_path: Path, chunk_size: int, chunk_overlap: int) -> None:
    """Walk docs_path, chunk + embed every .md file, upsert into Chroma."""
    collection = get_collection()
    md_files = sorted(docs_path.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files in {docs_path}")

    total_chunks = 0
    for file_path in md_files:
        text = file_path.read_text(encoding="utf-8")
        metadata = extract_metadata(file_path, text)
        chunks = chunk_markdown(text, chunk_size, chunk_overlap)
        if not chunks:
            continue

        chunk_ids = [make_chunk_id(file_path, i) for i in range(len(chunks))]
        metadatas = [metadata] * len(chunks)

        # Embed in batches
        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch_texts = chunks[batch_start : batch_start + BATCH_SIZE]
            batch_ids = chunk_ids[batch_start : batch_start + BATCH_SIZE]
            batch_meta = metadatas[batch_start : batch_start + BATCH_SIZE]

            embeddings = embed_texts(batch_texts)
            collection.upsert(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_meta,
            )

        print(f"  {file_path.name}: {len(chunks)} chunks ingested")
        total_chunks += len(chunks)

    print(f"Ingest complete. Total chunks: {total_chunks}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into the vector store")
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(ingest_directory(args.path, args.chunk_size, args.chunk_overlap))


if __name__ == "__main__":
    main()
