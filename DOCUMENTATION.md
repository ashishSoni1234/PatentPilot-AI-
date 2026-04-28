# PatentPilot AI: System Documentation

## Overview
PatentPilot AI is an advanced, multi-agent AI system designed to perform professional-grade patent analysis. It utilizes a parallelized agentic workflow to analyze an invention idea against a dataset of patents, extracting similarity scores, generating claim mappings, and providing a legal freedom-to-operate (FTO) assessment in seconds.

## Architecture & Technology Stack
- **Frontend**: Streamlit (Responsive SPA architecture using `st.session_state`)
- **Backend**: FastAPI (Async orchestration)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embedding Model**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **LLM Engine**: Groq API (`llama-3.1-8b-instant`) for supersonic token generation

## The 6-Agent Pipeline

The core innovation of PatentPilot AI is its Agentic Pipeline. Instead of relying on a single prompt, the system routes the analysis through 6 specialized AI agents running in parallel using `asyncio.gather` for maximum speed.

### 1. Search Agent
- **Purpose**: Retrieves the top relevant patents from the local FAISS index.
- **Mechanism**: Converts the user's invention idea into dense vector embeddings and compares it against 7,000+ patent chunks.

### 2. Filter Agent
- **Purpose**: Acts as a gatekeeper.
- **Mechanism**: Reviews the search results and filters out irrelevant patents using semantic thresholding to save LLM context window limits.

### 3. Analysis Agent (Novelty)
- **Purpose**: Evaluates the core novelty of the invention.
- **Mechanism**: Instructs the LLM to identify specific structural and functional overlaps between the user's idea and the retrieved prior art.

### 4. Claim Mapping Agent
- **Purpose**: Performs a detailed feature-by-feature legal mapping.
- **Mechanism**: Maps the elements of the user's invention directly to the patent's claims, tagging each element as `IDENTICAL`, `SIMILAR`, or `DIFFERENT`.

### 5. Legal Reasoning Agent
- **Purpose**: Assesses Infringement Risk and FTO.
- **Mechanism**: Consumes the output of the Claim Mapping agent to determine the legal risk level (LOW, MEDIUM, HIGH) and generates a professional Freedom-to-Operate opinion.

### 6. Citation Agent
- **Purpose**: Compiles a bibliography of relevant patents.
- **Mechanism**: Formats the matched prior art into a standard citation format for the final report.

## Performance Optimization
- **Supersonic LLM Execution**: By migrating from standard REST models to Groq's Llama 3.1 8B, the system achieves a full 6-agent analysis in under **3 seconds** (previously 30+ seconds).
- **Parallel Processing**: Asynchronous execution ensures that the Novelty, Claim Mapping, and Legal agents run concurrently without blocking the event loop.

## Setup & Deployment
Please refer to the `README.md` for installation and execution instructions.
