"""
PatentPilot AI — FastAPI Backend
==================================
Main API server that loads models, FAISS index, and exposes
the analysis pipeline via REST endpoints.

Endpoints:
  POST /api/analyze    — Full multi-agent analysis pipeline
  GET  /api/search     — Standalone semantic search
  GET  /api/health     — Health check
  GET  /api/stats      — Dataset statistics
"""

import json
import logging
import asyncio
import os
from contextlib import asynccontextmanager

# FIX: Force OpenBLAS/MKL to use 1 thread to prevent massive memory spikes
# This fixes "OpenBLAS error: Memory allocation still failed" on 6GB RAM systems
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from backend.config import (
    CHUNK_METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    PATENTS_JSON,
    API_HOST,
    API_PORT,
)
from backend.models import AnalysisRequest, AnalysisResponse
from backend.agents.search_agent import SearchAgent
from backend.agents.pipeline import AnalysisPipeline
from backend.report_generator import ReportGenerator
from backend.llm_client import ollama_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────
model = None
index = None
chunk_metadata = None
patents_data = None
search_agent = None
pipeline = None
report_generator = ReportGenerator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and data on startup."""
    global model, index, chunk_metadata, patents_data, search_agent, pipeline

    logger.info("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info(f"Model loaded: {EMBEDDING_MODEL_NAME}")

    logger.info("Loading FAISS index...")
    if FAISS_INDEX_PATH.exists():
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        logger.info(f"FAISS index loaded: {index.ntotal} vectors")
    else:
        logger.warning(f"FAISS index not found at {FAISS_INDEX_PATH}")
        logger.warning("Run: python scripts/indexer.py")

    logger.info("Loading chunk metadata...")
    if CHUNK_METADATA_PATH.exists():
        with open(CHUNK_METADATA_PATH, "r", encoding="utf-8") as f:
            chunk_metadata = json.load(f)
        logger.info(f"Chunk metadata loaded: {len(chunk_metadata)} chunks")
    else:
        chunk_metadata = []
        logger.warning(f"Chunk metadata not found at {CHUNK_METADATA_PATH}")

    logger.info("Loading patents data...")
    if PATENTS_JSON.exists():
        with open(PATENTS_JSON, "r", encoding="utf-8") as f:
            patents_data = json.load(f)
        logger.info(f"Patents data loaded: {len(patents_data)} patents")
    else:
        patents_data = []
        logger.warning(f"Patents data not found at {PATENTS_JSON}")

    if index and chunk_metadata:
        search_agent = SearchAgent(model, index, chunk_metadata, patents_data)
        pipeline = AnalysisPipeline(search_agent)
        logger.info("Analysis pipeline initialized")
    else:
        logger.warning("Pipeline not initialized — missing index or metadata")

    # Check Ollama
    ollama_available = await ollama_client.is_available()
    if ollama_available:
        logger.info(f"Ollama available with model: {ollama_client.model}")
    else:
        logger.warning(
            "Ollama not available — agents will use fallback heuristics. "
            "Start Ollama with: ollama serve, then: ollama pull mistral"
        )

    yield

    logger.info("Shutting down...")


# ── FastAPI App ───────────────────────────────────────────────────────

app = FastAPI(
    title="PatentPilot AI",
    description="Agentic AI system for patent analysis",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "index_loaded": index is not None,
        "index_vectors": index.ntotal if index else 0,
        "patents_loaded": len(patents_data) if patents_data else 0,
        "ollama_available": await ollama_client.is_available(),
    }


@app.get("/api/stats")
async def stats():
    """Dataset statistics."""
    return {
        "total_patents": len(patents_data) if patents_data else 0,
        "total_chunks": len(chunk_metadata) if chunk_metadata else 0,
        "index_vectors": index.ntotal if index else 0,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "ollama_model": ollama_client.model,
        "ollama_available": await ollama_client.is_available(),
    }


@app.get("/api/search")
async def search(query: str, top_k: int = 10):
    """Standalone semantic search endpoint."""
    if not search_agent:
        raise HTTPException(
            status_code=503,
            detail="Search agent not initialized. Run the indexer first.",
        )

    result = search_agent.search(query, top_k=top_k)
    return {
        "query": result.query,
        "total_searched": result.total_searched,
        "matches": [
            {
                "doc_number": m.doc_number,
                "title": m.title,
                "abstract": m.abstract[:300],
                "similarity_score": round(m.similarity_score, 4),
            }
            for m in result.matches
        ],
    }


@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    """
    Full multi-agent patent analysis pipeline.
    
    Takes an invention description and runs it through all 6 agents:
    Search → Filter → Analysis → Claim Mapping → Legal Reasoning → Citation
    """
    if not pipeline:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Run the indexer first.",
        )

    logger.info(f"Starting analysis: '{request.invention_description[:100]}...'")

    response = await pipeline.run(
        request.invention_description,
        top_k=request.top_k,
    )

    # Generate the strict JSON report format
    json_report = report_generator.generate_json_report(response)

    return {
        **json_report,
        "metadata": {
            "pipeline_steps_completed": response.pipeline_steps_completed,
            "total_time_seconds": response.total_time_seconds,
            "error": response.error,
        },
    }

@app.post("/api/analyze_raw", response_model=AnalysisResponse)
async def analyze_raw(request: AnalysisRequest):
    """Returns the raw AnalysisResponse Pydantic object for the Streamlit frontend."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return await pipeline.run(
        request.invention_description,
        top_k=request.top_k,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=API_HOST, port=API_PORT, reload=True)
