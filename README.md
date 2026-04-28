---
title: PatentPilot AI
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

<div align="center">
  <h1>🚀 PatentPilot AI</h1>
  <p><strong>Agentic AI Patent Analysis — Prior Art Search • Novelty Analysis • Infringement Detection • FTO Assessment</strong></p>
</div>

<p align="center">
  PatentPilot AI is an ultra-fast, multi-agent legal-tech application that automates the patent analysis process. Powered by <strong>FAISS</strong>, <strong>FastAPI</strong>, and the <strong>Groq Llama-3.1 API</strong>, it delivers comprehensive legal reports in under 3 seconds.
</p>

---

## 📖 In-Depth Documentation
For a complete breakdown of the system architecture, the 6-agent parallel pipeline, and technical design decisions, please read the [System Documentation](DOCUMENTATION.md).

---

## 📸 System Walkthrough

PatentPilot AI is designed to be intuitive and fast. Here is a step-by-step walkthrough of how the system works:

### 1. Analysis Dashboard & Novelty Assessment
Within seconds, the dashboard displays the top-level results: Novelty Score, Infringement Risk, and Claim Overlap percentage, along with a detailed textual analysis of exactly which features overlap with prior art.

![Analysis Dashboard](assets/step2.png)

### 2. The Agentic Pipeline Start
Once an idea is submitted, the system activates its 6 parallel AI agents. The Search Agent scans thousands of vector embeddings in the FAISS database to find the closest matches.

![Pipeline Processing](assets/step3.png)

### 3. Describe Your Invention
Users simply enter their invention idea into the application, describing its key technical features and how it works. You can also select from pre-loaded examples to test the system.

![Entering Idea](assets/step4.png)

### 4. Infringement Risk & Claim Element Mapping
The Claim Mapping agent performs a deep, lawyer-like comparison, breaking down the user's idea feature-by-feature and mapping it to the prior art's claims, tagging them as `IDENTICAL`, `SIMILAR`, or `DIFFERENT`.

![Claim Mapping](assets/step5.png)

### 5. Similar Patents, Citations & PDF Export
The system provides a final Freedom-to-Operate (FTO) assessment, lists exact citations for the similar patents, and offers a 1-click PDF download of the entire legal report.

![Citations and Download](assets/step1.png)

---

## ✨ Key Features
- **Supersonic Speed**: Generates detailed patent landscape reports in ~2 seconds using Groq API (`llama-3.1-8b-instant`).
- **Parallel Agentic Workflow**: Uses `asyncio.gather` to run 6 specialized AI agents concurrently.
- **Semantic Vector Search**: Searches through thousands of patents using a local FAISS index and `sentence-transformers`.
- **Zero Hallucination UI**: Uses strict JSON schema enforcement to ensure that LLMs always return structured, parseable data.

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+
- A [Groq API Key](https://console.groq.com/)

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/ashishSoni1234/PatentPilot-AI-.git
cd PatentPilot-AI-
pip install -r requirements.txt
```

### 3. Setup Groq API Key
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 4. Run the Application
You can run the entire system (FastAPI Backend + Streamlit Frontend) using the provided batch script:
```cmd
run.bat
```
The UI will be available at `http://localhost:8501`.

---

## 🛠 Tech Stack
- **Frontend**: Streamlit, HTML/CSS
- **Backend**: FastAPI, Python `asyncio`
- **AI Models**: Llama 3.1 8B (via Groq), `all-MiniLM-L6-v2`
- **Database**: FAISS (Facebook AI Similarity Search)

---

<div align="center">
  <i>Built for high-performance Legal-Tech and Patent Analysis.</i>
</div>
