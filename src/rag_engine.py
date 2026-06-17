"""
rag_engine.py — Retrieval-Augmented Generation Engine
======================================================

Manages the ChromaDB vector store:
  • Reads markdown files from the knowledge base
  • Chunks text into ~250-word paragraph-aware segments
  • Embeds with sentence-transformers and stores in ChromaDB
  • Provides semantic search for grounding LLM responses
"""

import re
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    KNOWLEDGE_BASE_PATH,
)


# ── Module-level singleton ──────────────────────────────────────────
_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None


def _get_embedding_function() -> embedding_functions.SentenceTransformerEmbeddingFunction:
    """Return a reusable sentence-transformer embedding function."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )


def _get_client() -> chromadb.PersistentClient:
    """Lazy-init the ChromaDB persistent client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    return _client


# ── Text Chunking ───────────────────────────────────────────────────

def _infer_topic(filepath: Path) -> str:
    """Infer the document topic from the filename."""
    name = filepath.stem.lower()
    topic_map = {
        "diseases": "disease",
        "natural_farming": "farming",
        "multilevel_cropping": "cropping",
        "subsidies": "subsidy",
    }
    for key, topic in topic_map.items():
        if key in name:
            return topic
    return "general"


def _split_into_sections(text: str) -> list[str]:
    """
    Split markdown text into sections using ## headings as boundaries.
    Each section includes its heading and all content until the next heading.
    """
    # Split on markdown ## headings (keep the heading with the section)
    sections = re.split(r'\n(?=## )', text)
    # Clean up: remove leading/trailing whitespace, drop empty sections
    return [s.strip() for s in sections if s.strip()]


def _chunk_section(section: str, max_words: int = 300, overlap_words: int = 30) -> list[str]:
    """
    Break a section into chunks of roughly max_words, splitting on paragraph
    boundaries. Adds a small word overlap between consecutive chunks for
    context continuity.
    """
    paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current_chunk_parts: list[str] = []
    current_word_count = 0

    for para in paragraphs:
        para_words = len(para.split())

        # If a single paragraph exceeds max_words, keep it as its own chunk
        if para_words > max_words:
            # Flush current accumulator first
            if current_chunk_parts:
                chunks.append("\n\n".join(current_chunk_parts))
                current_chunk_parts = []
                current_word_count = 0
            chunks.append(para)
            continue

        # Would adding this paragraph exceed the limit?
        if current_word_count + para_words > max_words and current_chunk_parts:
            chunks.append("\n\n".join(current_chunk_parts))
            # Overlap: carry forward the last paragraph fragment
            if overlap_words > 0 and current_chunk_parts:
                last = current_chunk_parts[-1]
                overlap = " ".join(last.split()[-overlap_words:])
                current_chunk_parts = [f"...{overlap}"]
                current_word_count = overlap_words
            else:
                current_chunk_parts = []
                current_word_count = 0

        current_chunk_parts.append(para)
        current_word_count += para_words

    # Flush remaining
    if current_chunk_parts:
        chunks.append("\n\n".join(current_chunk_parts))

    return chunks


def _chunk_document(filepath: Path) -> list[dict]:
    """
    Read a markdown file and return a list of chunk dicts:
        { "id": str, "text": str, "metadata": { "source_file", "topic", "section_heading" } }
    """
    text = filepath.read_text(encoding="utf-8")
    topic = _infer_topic(filepath)
    sections = _split_into_sections(text)

    all_chunks: list[dict] = []
    chunk_index = 0

    for section in sections:
        # Extract section heading (first line if it starts with ##)
        lines = section.split("\n", 1)
        heading = lines[0].strip("# ").strip() if lines[0].startswith("##") else ""

        sub_chunks = _chunk_section(section)
        for sub in sub_chunks:
            chunk_id = f"{filepath.stem}_{chunk_index:03d}"
            all_chunks.append({
                "id": chunk_id,
                "text": sub,
                "metadata": {
                    "source_file": filepath.name,
                    "topic": topic,
                    "section_heading": heading,
                },
            })
            chunk_index += 1

    return all_chunks


# ── Public API ──────────────────────────────────────────────────────

def build_knowledge_base(force_rebuild: bool = False) -> int:
    """
    Read all .md files from the knowledge base directory, chunk them,
    embed them, and store in ChromaDB.

    Args:
        force_rebuild: If True, deletes the existing collection and rebuilds.

    Returns:
        Number of chunks stored.
    """
    client = _get_client()
    ef = _get_embedding_function()

    # Delete existing collection if force rebuild
    if force_rebuild:
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
        except ValueError:
            pass  # Collection didn't exist

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=ef,
    )

    # Skip if collection already has data and not forcing rebuild
    if collection.count() > 0 and not force_rebuild:
        print(f"[RAG] Collection '{CHROMA_COLLECTION_NAME}' already has "
              f"{collection.count()} chunks. Skipping build. "
              f"Use force_rebuild=True to rebuild.")
        return collection.count()

    # Gather all markdown files
    md_files = sorted(KNOWLEDGE_BASE_PATH.glob("*.md"))
    if not md_files:
        print(f"[RAG] No .md files found in {KNOWLEDGE_BASE_PATH}")
        return 0

    all_chunks: list[dict] = []
    for md_file in md_files:
        chunks = _chunk_document(md_file)
        all_chunks.extend(chunks)
        print(f"[RAG] {md_file.name}: {len(chunks)} chunks")

    if not all_chunks:
        return 0

    # Batch add to ChromaDB
    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        metadatas=[c["metadata"] for c in all_chunks],
    )

    print(f"[RAG] Total: {len(all_chunks)} chunks stored in '{CHROMA_COLLECTION_NAME}'")
    return len(all_chunks)


def query_knowledge_base(
    query: str,
    n_results: int = 3,
    topic_filter: Optional[str] = None,
) -> list[str]:
    """
    Perform semantic search on the knowledge base.

    Args:
        query: The search query text.
        n_results: Number of results to return.
        topic_filter: Optional — filter by topic ('disease', 'farming', 'cropping', 'subsidy').

    Returns:
        List of relevant text chunks, ordered by similarity.
    """
    collection = get_or_build_collection()

    where_filter = None
    if topic_filter:
        where_filter = {"topic": topic_filter}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter,
    )

    # results["documents"] is a list of lists (one per query)
    if results and results["documents"]:
        return results["documents"][0]
    return []


def get_or_build_collection() -> chromadb.Collection:
    """
    Get the ChromaDB collection, building it if empty.
    This is the main entry point for other modules.
    """
    global _collection

    if _collection is not None:
        return _collection

    client = _get_client()
    ef = _get_embedding_function()

    _collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=ef,
    )

    # Auto-build if empty
    if _collection.count() == 0:
        print("[RAG] Collection is empty. Building knowledge base...")
        build_knowledge_base()
        # Re-fetch collection after build
        _collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=ef,
        )

    return _collection
