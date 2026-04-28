"""
PatentPilot AI — Backend Configuration
=======================================
Centralized configuration for all backend components.
All paths, model names, and tuning parameters in one place.
"""

import os
from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
PATENTS_JSON = DATA_DIR / "patents.json"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
CHUNK_METADATA_PATH = DATA_DIR / "chunk_metadata.json"

# ── Embedding Model ───────────────────────────────────────────────────

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 output dimension
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# ── FAISS Settings ────────────────────────────────────────────────────

FAISS_TOP_K = int(os.getenv("FAISS_TOP_K", "5"))

# ── Ollama / LLM Settings ────────────────────────────────────────────
# Using llama3.2:3b — best quality model for 6GB RAM systems.
# llama3.1:8b (4.7GB) and mistral:7b (4.5GB) are too large.

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")   # ~986 MB — best for 6GB RAM
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "240"))   # increased timeout for virtual memory
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

# ── Text Chunking ────────────────────────────────────────────────────

CHUNK_MIN_TOKENS = 300
CHUNK_MAX_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

# ── Agent Thresholds ─────────────────────────────────────────────────

SEARCH_SIMILARITY_THRESHOLD = float(
    os.getenv("SEARCH_SIMILARITY_THRESHOLD", "0.4")
)
FILTER_RELEVANCE_THRESHOLD = float(
    os.getenv("FILTER_RELEVANCE_THRESHOLD", "0.5")
)
MAX_FILTERED_PATENTS = int(os.getenv("MAX_FILTERED_PATENTS", "2"))

# ── API Settings ─────────────────────────────────────────────────────

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
