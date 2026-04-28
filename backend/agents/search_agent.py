"""
PatentPilot AI — Search Agent
==============================
Agent 1 of 6 in the multi-agent pipeline.

Responsibility: Given a user's invention description, embed it and
perform FAISS nearest-neighbor search to find semantically similar
patent chunks in the vector database.

Input:  Invention description (text)
Output: SearchResult with ranked PatentMatch list
"""

import json
import logging
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import (
    CHUNK_METADATA_PATH,
    FAISS_INDEX_PATH,
    FAISS_TOP_K,
    SEARCH_SIMILARITY_THRESHOLD,
    PATENTS_JSON,
)
from backend.models import PatentMatch, SearchResult

logger = logging.getLogger(__name__)


class SearchAgent:
    """
    Performs semantic prior art search using FAISS vector similarity.
    
    Loads the pre-built FAISS index and chunk metadata at initialization,
    then encodes user queries with the same sentence-transformer model
    to find the most similar patent text chunks.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        index: faiss.Index,
        chunk_metadata: list,
        patents_data: list,
    ):
        self.model = model
        self.index = index
        self.chunk_metadata = chunk_metadata
        self.patents_data = patents_data
        logger.info(
            f"SearchAgent initialized with {index.ntotal} vectors, "
            f"{len(patents_data)} patents"
        )

    def search(self, query: str, top_k: int = None) -> SearchResult:
        """
        Perform semantic search for the given invention description.
        
        Args:
            query: The user's invention idea description
            top_k: Number of results to return (default from config)
            
        Returns:
            SearchResult with ranked patent matches
        """
        top_k = top_k or FAISS_TOP_K

        # Encode the query
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        # Search FAISS index
        scores, indices = self.index.search(query_embedding, top_k)

        # Build results, deduplicating by patent doc_number
        seen_docs = set()
        matches: List[PatentMatch] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_metadata):
                continue

            similarity = float(score)
            if similarity < SEARCH_SIMILARITY_THRESHOLD:
                continue

            chunk = self.chunk_metadata[idx]
            doc_number = chunk["doc_number"]

            # Find the full patent record
            patent_data = None
            for p in self.patents_data:
                if p["doc_number"] == doc_number:
                    patent_data = p
                    break

            if doc_number in seen_docs:
                continue
            seen_docs.add(doc_number)

            # Build claims text
            claims_text = ""
            if patent_data and "claims" in patent_data:
                claims_text = " | ".join(
                    c["text"] for c in patent_data["claims"][:5]
                )

            match = PatentMatch(
                doc_number=doc_number,
                title=chunk.get("title", ""),
                abstract=patent_data.get("abstract", "") if patent_data else "",
                claims_text=claims_text,
                similarity_score=similarity,
                chunk_text=chunk.get("text", ""),
            )
            matches.append(match)

        logger.info(
            f"Search complete: {len(matches)} unique patents found "
            f"(threshold={SEARCH_SIMILARITY_THRESHOLD})"
        )

        return SearchResult(
            query=query,
            matches=matches,
            total_searched=self.index.ntotal,
        )
