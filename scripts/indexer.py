"""
PatentPilot AI — FAISS Indexer
================================
Reads the JSON patent dataset, chunks text, generates embeddings
with sentence-transformers, and builds a FAISS index.

Optimized for low-resource systems:
- Batch processing (configurable batch size)
- Streaming progress output
- Memory-efficient numpy operations

Usage:
    python scripts/indexer.py [--batch-size 64] [--limit 500]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    CHUNK_MAX_TOKENS,
    CHUNK_MIN_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_METADATA_PATH,
    DATA_DIR,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    PATENTS_JSON,
)


def estimate_tokens(text: str) -> int:
    """Rough token estimation (~0.75 words per token for English)."""
    return int(len(text.split()) * 1.33)


def chunk_text(text: str, min_tokens: int, max_tokens: int, overlap: int) -> List[str]:
    """
    Split text into chunks of approximately min_tokens to max_tokens.
    Uses sentence-aware splitting where possible.
    
    Args:
        text: Input text to chunk
        min_tokens: Minimum chunk size in estimated tokens
        max_tokens: Maximum chunk size in estimated tokens
        overlap: Number of overlapping tokens between chunks
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    # If text is short enough, return as single chunk
    total_tokens = estimate_tokens(text)
    if total_tokens <= max_tokens:
        if total_tokens >= min_tokens // 2:  # Allow smaller chunks if content exists
            return [text.strip()]
        return []

    # Split by sentences
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = estimate_tokens(sentence)

        if current_tokens + sent_tokens > max_tokens and current_chunk:
            # Save current chunk
            chunk_text_str = " ".join(current_chunk).strip()
            if estimate_tokens(chunk_text_str) >= min_tokens // 2:
                chunks.append(chunk_text_str)

            # Keep overlap
            overlap_text = " ".join(current_chunk[-2:]) if len(current_chunk) >= 2 else ""
            current_chunk = [overlap_text] if overlap_text else []
            current_tokens = estimate_tokens(overlap_text) if overlap_text else 0

        current_chunk.append(sentence)
        current_tokens += sent_tokens

    # Don't forget the last chunk
    if current_chunk:
        chunk_text_str = " ".join(current_chunk).strip()
        if estimate_tokens(chunk_text_str) >= min_tokens // 4:
            chunks.append(chunk_text_str)

    return chunks


def build_chunks(patents: list, limit: int = None) -> tuple:
    """
    Build text chunks from patent data and create metadata.
    
    Args:
        patents: List of patent records from JSON
        limit: Optional limit on number of patents to process
        
    Returns:
        (texts, metadata) - lists of chunk texts and metadata dicts
    """
    texts = []
    metadata = []
    chunk_id = 0

    patents_to_process = patents[:limit] if limit else patents
    
    for patent_idx, patent in enumerate(patents_to_process):
        doc_number = patent.get("doc_number", "")
        title = patent.get("title", "")
        abstract = patent.get("abstract", "")
        claims = patent.get("claims", [])

        # Combine abstract as a chunk
        if abstract:
            abstract_with_title = f"{title}. {abstract}" if title else abstract
            abstract_chunks = chunk_text(
                abstract_with_title,
                CHUNK_MIN_TOKENS // 2,  # Allow shorter abstract chunks
                CHUNK_MAX_TOKENS,
                CHUNK_OVERLAP_TOKENS,
            )
            for chunk in abstract_chunks:
                texts.append(chunk)
                metadata.append({
                    "chunk_id": chunk_id,
                    "patent_index": patent_idx,
                    "doc_number": doc_number,
                    "title": title,
                    "chunk_type": "abstract",
                    "text": chunk,
                })
                chunk_id += 1

        # Combine claims into chunks
        if claims:
            claims_text = " ".join(c.get("text", "") for c in claims)
            claims_full = f"{title}. Claims: {claims_text}" if title else claims_text
            claims_chunks = chunk_text(
                claims_full,
                CHUNK_MIN_TOKENS,
                CHUNK_MAX_TOKENS,
                CHUNK_OVERLAP_TOKENS,
            )
            for chunk in claims_chunks:
                texts.append(chunk)
                metadata.append({
                    "chunk_id": chunk_id,
                    "patent_index": patent_idx,
                    "doc_number": doc_number,
                    "title": title,
                    "chunk_type": "claims",
                    "text": chunk,
                })
                chunk_id += 1

    return texts, metadata


def build_index(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int,
) -> faiss.Index:
    """
    Generate embeddings in batches and build a FAISS index.
    Uses IndexFlatIP with normalized embeddings for cosine similarity.
    
    Args:
        texts: List of text chunks to embed
        model: SentenceTransformer model
        batch_size: Number of texts to embed at once
        
    Returns:
        FAISS index
    """
    # Create FAISS index (Inner Product = cosine similarity with normalized vectors)
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)

    total = len(texts)
    start_time = time.time()

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        
        # Generate embeddings with normalization
        embeddings = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        # Add to index
        index.add(embeddings)

        # Progress
        done = min(i + batch_size, total)
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(
            f"  Embedded {done}/{total} chunks "
            f"({done/total*100:.1f}%) "
            f"[{rate:.0f} chunks/sec, ETA: {eta:.0f}s]",
            end="\r",
        )

    print()  # newline after progress
    return index


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index for PatentPilot AI"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=EMBEDDING_BATCH_SIZE,
        help=f"Embedding batch size (default: {EMBEDDING_BATCH_SIZE})"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of patents to index (default: all)"
    )
    args = parser.parse_args()

    # ── Load patents ──────────────────────────────────────────────
    print(f"Loading patents from {PATENTS_JSON}...")
    if not PATENTS_JSON.exists():
        print(f"ERROR: {PATENTS_JSON} not found.")
        print("Run xml_to_json.py first: python scripts/xml_to_json.py")
        sys.exit(1)

    with open(PATENTS_JSON, "r", encoding="utf-8") as f:
        patents = json.load(f)
    print(f"Loaded {len(patents)} patents")

    # ── Build chunks ──────────────────────────────────────────────
    print(f"\nChunking patent text (target: {CHUNK_MIN_TOKENS}-{CHUNK_MAX_TOKENS} tokens)...")
    texts, metadata = build_chunks(patents, limit=args.limit)
    print(f"Created {len(texts)} text chunks from {args.limit or len(patents)} patents")

    if not texts:
        print("ERROR: No text chunks created. Check the patent data.")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────
    print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME}")
    print("(This may take a moment on first run — model will be downloaded)")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Model loaded successfully!")

    # ── Build index ───────────────────────────────────────────────
    print(f"\nBuilding FAISS index (batch_size={args.batch_size})...")
    start_time = time.time()
    index = build_index(texts, model, args.batch_size)
    elapsed = time.time() - start_time
    print(f"Index built: {index.ntotal} vectors in {elapsed:.1f}s")

    # ── Save outputs ──────────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving FAISS index to {FAISS_INDEX_PATH}...")
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    index_size = FAISS_INDEX_PATH.stat().st_size
    print(f"Index saved ({index_size / 1e6:.1f} MB)")

    print(f"Saving chunk metadata to {CHUNK_METADATA_PATH}...")
    with open(CHUNK_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=1, ensure_ascii=False)
    meta_size = CHUNK_METADATA_PATH.stat().st_size
    print(f"Metadata saved ({meta_size / 1e6:.1f} MB)")

    # ── Verification ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("INDEXING COMPLETE")
    print(f"{'='*60}")
    print(f"  Patents processed:  {args.limit or len(patents)}")
    print(f"  Chunks created:     {len(texts)}")
    print(f"  FAISS vectors:      {index.ntotal}")
    print(f"  Index dimensions:   {EMBEDDING_DIMENSION}")
    print(f"  Time elapsed:       {elapsed:.1f}s")
    print(f"{'='*60}")

    # Quick test search
    print("\n[TEST] Quick test search: 'machine learning neural network'")
    test_query = "machine learning neural network"
    q_emb = model.encode([test_query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_emb, 5)
    print(f"Top 5 results:")
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and idx < len(metadata):
            m = metadata[idx]
            print(f"  [{score:.4f}] {m['doc_number']}: {m['title'][:60]}")


if __name__ == "__main__":
    main()
