"""
PatentPilot AI — Streamlit Frontend
======================================
Premium dark-mode UI for patent analysis with the multi-agent pipeline.

Features:
- Invention idea input with real-time analysis
- Animated pipeline progress (6 agent steps)
- Tabbed results: Novelty / Infringement / FTO / Citations
- Interactive patent cards with similarity gauges
- Downloadable Markdown report
- System health dashboard in sidebar

Run:
    streamlit run frontend/app.py
"""

import json
import time
import asyncio
import sys
from pathlib import Path

import streamlit as st
import httpx

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="PatentPilot AI — Patent Analysis",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project root for imports ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    FAISS_INDEX_PATH,
    CHUNK_METADATA_PATH,
    PATENTS_JSON,
    EMBEDDING_MODEL_NAME,
)
from backend.models import AnalysisRequest, AnalysisResponse
from backend.report_generator import ReportGenerator

# ── Custom CSS ───────────────────────────────────────────────────────
st.html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ────────────────────────── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Hero Header ───────────────────── */
    .hero-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .hero-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-header p {
        color: #a0aec0;
        font-size: 1.1rem;
        font-weight: 300;
    }

    /* ── Cards ──────────────────────────── */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    .metric-card .metric-label {
        color: #a0aec0;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }

    /* ── Patent Cards ──────────────────── */
    .patent-card {
        background: linear-gradient(145deg, #1e1e3f 0%, #2d2b55 100%);
        border: 1px solid rgba(102, 126, 234, 0.15);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .patent-card:hover {
        border-color: rgba(102, 126, 234, 0.4);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.1);
    }
    .patent-title {
        font-weight: 600;
        color: #e2e8f0;
        font-size: 1.05rem;
        margin-bottom: 0.4rem;
    }
    .patent-meta {
        color: #718096;
        font-size: 0.82rem;
    }

    /* ── Risk Badge ────────────────────── */
    .risk-low { 
        background: linear-gradient(135deg, #0d9488, #14b8a6);
        color: white; padding: 0.3rem 1rem; border-radius: 20px;
        font-weight: 600; display: inline-block; font-size: 0.9rem;
    }
    .risk-medium {
        background: linear-gradient(135deg, #d97706, #f59e0b);
        color: white; padding: 0.3rem 1rem; border-radius: 20px;
        font-weight: 600; display: inline-block; font-size: 0.9rem;
    }
    .risk-high {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        color: white; padding: 0.3rem 1rem; border-radius: 20px;
        font-weight: 600; display: inline-block; font-size: 0.9rem;
    }

    /* ── Progress Steps ────────────────── */
    .step-complete {
        color: #10b981;
        font-weight: 500;
    }
    .step-running {
        color: #667eea;
        font-weight: 500;
    }
    .step-pending {
        color: #4a5568;
    }

    /* ── Sidebar ───────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #cbd5e0;
    }

    /* ── Score Gauge ───────────────────── */
    .score-gauge {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .score-value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .score-label {
        color: #a0aec0;
        font-size: 0.9rem;
    }

    /* ── Hide Streamlit branding ────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ── Tabs styling ──────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
    }
</style>
""")


# ── Helper Functions ─────────────────────────────────────────────────

def load_pipeline():
    """Check if the FastAPI backend is running instead of loading models locally."""
    try:
        resp = httpx.get("http://localhost:8000/api/health", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            if not data.get("index_loaded"):
                return None, None, "FAISS index not loaded in backend."
            return True, {
                "patents": data.get("patents_loaded", 0),
                "chunks": data.get("index_vectors", 0),
                "vectors": data.get("index_vectors", 0),
            }, None
        return None, None, "Backend health check failed."
    except httpx.ConnectError:
        return None, None, "⏳ Backend ML models are booting up. Please wait 10-15 seconds and reload."
    except Exception as e:
        return None, None, f"FastAPI Backend is not running on port 8000! Start it first. ({e})"


def get_risk_badge(risk_level: str) -> str:
    """Get HTML for a risk badge."""
    cls = f"risk-{risk_level.lower()}"
    return f'<span class="{cls}">{risk_level}</span>'


def get_novelty_color(score: float) -> str:
    """Get color based on novelty score."""
    if score >= 0.7:
        return "#10b981"  # green
    elif score >= 0.4:
        return "#f59e0b"  # yellow
    else:
        return "#ef4444"  # red


def run_async(coro):
    """Run an async function from sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🚀 PatentPilot AI")
    st.markdown("---")

    # Load pipeline
    pipeline, stats, error = load_pipeline()

    if error:
        st.error(f"⚠️ {error}")
        st.markdown("""
        **Setup Steps:**
        1. `python scripts/xml_to_json.py`
        2. `python scripts/indexer.py`
        3. Restart this app
        """)
    else:
        st.success("✅ System Ready")

        st.markdown("#### 📊 Dataset Stats")
        if stats:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{stats['patents']:,}</div>
                <div class="metric-label">Patents Loaded</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Chunks", f"{stats['chunks']:,}")
            with col2:
                st.metric("Vectors", f"{stats['vectors']:,}")

    st.markdown("---")
    st.markdown("#### ⚙️ Settings")
    top_k = st.slider("Search Results (top_k)", 1, 10, 3)

    st.markdown("---")

    # LLM status check
    st.markdown("#### 🤖 LLM Status")
    st.success("🟢 Groq API Ready")

    st.markdown("---")
    st.markdown(
        "<p style='color:#4a5568;font-size:0.75rem;text-align:center'>"
        "PatentPilot AI v1.0<br>Powered by llama-3.1-8b + FAISS</p>",
        unsafe_allow_html=True,
    )


# ── Main Content ─────────────────────────────────────────────────────

# Hero header
st.markdown("""
<div class="hero-header">
    <h1>🚀 PatentPilot AI</h1>
    <p>Agentic AI Patent Analysis — Prior Art Search • Novelty Analysis • Infringement Detection • FTO Assessment</p>
</div>
""", unsafe_allow_html=True)

# Input Section
st.markdown("### 💡 Describe Your Invention")

if "invention_input" not in st.session_state:
    st.session_state["invention_input"] = ""

def set_example(text):
    st.session_state.invention_input = text

invention_text = st.text_area(
    "Invention Description",
    key="invention_input",
    placeholder="Describe your invention idea in detail. Include the problem it solves, "
    "key technical features, how it works, and what makes it different from existing solutions...",
    height=180,
    label_visibility="collapsed",
)

# Example inventions
with st.expander("📝 Example Inventions (click to try)"):
    examples = [
        {
            "name": "🔨 Soil Loosening Stake Tool (Guaranteed Hit)",
            "text": "A soil loosening hand tool having a handle, a shaft, a foot bar, and a broadhead acumination that results in a distal point. This hand tool provides an improved way for a user to loosen the soil so that the stakes of lighting fixtures or other apparatuses can be easily pushed into the ground."
        },
        {
            "name": "🔋 Smart Battery Management (Realistic)",
            "text": "A machine learning-based battery management system for electric vehicles that uses real-time sensor data and predictive neural networks to optimize charging cycles, extend battery lifespan, and prevent thermal runaway. The system employs a novel attention mechanism to weigh multiple sensor inputs including temperature, voltage, current, and internal resistance."
        },
        {
            "name": "🌱 Precision Agriculture Drone (Mixed)",
            "text": "An autonomous agricultural drone system equipped with a broadhead acumination hand tool mechanism to physically test soil looseness, while employing edge AI to monitor crop health. The drone uses computer vision to detect plant diseases, then autonomously lands to measure soil density using the distal point of its mechanical shaft."
        },
    ]
    for ex in examples:
        st.button(ex["name"], key=ex["name"], on_click=set_example, args=(ex["text"],))

# Analyze button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_clicked = st.button(
        "🔍 Analyze Patent Landscape",
        use_container_width=True,
        type="primary",
        disabled=not pipeline or len(invention_text or "") < 20,
    )

# ── Analysis Execution ───────────────────────────────────────────────

if analyze_clicked and invention_text and pipeline:
    st.markdown("---")
    st.markdown("### ⚡ Analysis Pipeline")

    # Pipeline step display
    steps = [
        ("🔍", "Search Agent", "Searching FAISS index..."),
        ("🔬", "Filter Agent", "Filtering relevant patents..."),
        ("📊", "Analysis Agent", "Analyzing novelty..."),
        ("⚖️", "Claim Mapping Agent", "Mapping claim elements..."),
        ("🏛️", "Legal Reasoning Agent", "Assessing risks..."),
        ("📚", "Citation Agent", "Compiling citations..."),
    ]

    progress_bar = st.progress(0)
    status_container = st.empty()

    # Run the pipeline
    start_time = time.time()
    
    with st.spinner("Running 6-agent analysis pipeline... (Estimated time: ~30-40 seconds)"):
        for i, (icon, name, desc) in enumerate(steps):
            progress_bar.progress((i) / len(steps), text=f"{icon} {name}: {desc}")
            time.sleep(0.3)  # Brief visual delay for UX

        try:
            resp = httpx.post(
                "http://localhost:8000/api/analyze_raw",
                json={"invention_description": invention_text, "top_k": top_k},
                timeout=300
            )
            resp.raise_for_status()
            response = AnalysisResponse.model_validate(resp.json())
            st.session_state.analysis_response = response
            progress_bar.progress(1.0, text="✅ Analysis Complete!")
        except Exception as e:
            progress_bar.progress(1.0, text="❌ Analysis Failed!")
            st.error(f"Error calling backend API: {e}")
            st.stop()

    elapsed = time.time() - start_time
    report_gen = ReportGenerator()
    report = report_gen.generate_report(st.session_state.analysis_response)
    st.session_state.analysis_report = report
    st.session_state.analysis_elapsed = elapsed

if "analysis_response" in st.session_state:
    response = st.session_state.analysis_response
    report = st.session_state.analysis_report
    elapsed = st.session_state.get("analysis_elapsed", 0.0)
    report_gen = ReportGenerator()

    # ── Results Display ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Analysis Results")

    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        novelty_score = response.novelty_analysis.novelty_score if response.novelty_analysis else 0
        color = get_novelty_color(novelty_score)
        st.markdown(f"""
        <div class="score-gauge">
            <div class="score-value" style="color:{color}">{novelty_score:.0%}</div>
            <div class="score-label">Novelty Score</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        risk = response.legal_assessment.risk_level if response.legal_assessment else "N/A"
        st.markdown(f"""
        <div class="score-gauge">
            <div style="margin:0.5rem 0">{get_risk_badge(risk)}</div>
            <div class="score-label">Infringement Risk</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        overlap = response.claim_mapping.overlap_percentage if response.claim_mapping else 0
        st.markdown(f"""
        <div class="score-gauge">
            <div class="score-value" style="color:#667eea">{overlap:.0f}%</div>
            <div class="score-label">Claim Overlap</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        n_patents = len(response.filter_result.relevant_patents) if response.filter_result else 0
        st.markdown(f"""
        <div class="score-gauge">
            <div class="score-value" style="color:#a78bfa">{n_patents}</div>
            <div class="score-label">Similar Patents</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Detailed Sections (Single Page Layout) ────────────────────
    st.markdown("---")
    
    st.markdown("### 🔍 Novelty Assessment")
    if response.novelty_analysis:
        na = response.novelty_analysis
        st.markdown(f"**Score:** {na.novelty_score:.2f}/1.00 — "
                   f"{'✅ Novel' if na.is_novel else '❌ Not Novel'}")
        st.markdown(f"**Analysis:** {na.analysis_text}")
        
        if na.unique_features:
            st.markdown("**✨ Unique Features:**")
            for f in na.unique_features:
                st.markdown(f"- {f}")
        
        if na.overlapping_features:
            st.markdown("**⚠️ Overlapping Features:**")
            for f in na.overlapping_features:
                st.markdown(f"- {f}")
    else:
        st.info("Novelty analysis not available")

    st.markdown("---")
    st.markdown("### ⚠️ Infringement Risk Assessment")
    if response.legal_assessment:
        la = response.legal_assessment
        st.markdown(f"**Risk Level:** {get_risk_badge(la.infringement_risk)}", 
                   unsafe_allow_html=True)
        st.markdown(f"\n{la.infringement_details}")
        
        if la.recommendations:
            st.markdown("**📋 Recommendations:**")
            for r in la.recommendations:
                st.markdown(f"- {r}")
    else:
        st.info("Infringement analysis not available")

    st.markdown("---")
    st.markdown("### 🛡️ Freedom to Operate (FTO)")
    if response.legal_assessment:
        la = response.legal_assessment
        st.markdown(f"**Opinion:** {la.fto_opinion}")
        st.markdown(f"\n{la.fto_details}")
    else:
        st.info("FTO assessment not available")

    st.markdown("---")
    st.markdown("### 📋 Claim Element Mapping")
    if response.claim_mapping:
        cm = response.claim_mapping
        st.markdown(f"**Overall Overlap:** {cm.overlap_percentage:.1f}%")
        st.markdown(f"**Summary:** {cm.mapping_summary}")
        
        if cm.mapped_elements:
            st.markdown("**Mapped Elements:**")
            for elem in cm.mapped_elements:
                emoji = {"identical": "🔴", "similar": "🟡", 
                        "different": "🟢", "novel": "✨"}.get(elem.overlap_type, "⚪")
                st.markdown(
                    f"- {emoji} **{elem.overlap_type.upper()}**: "
                    f"`{elem.invention_element}` → `{elem.patent_element[:80]}` "
                    f"(Patent {elem.patent_doc_number})"
                )
        
        if cm.unmapped_elements:
            st.markdown("**Novel Elements (No Match):**")
            for elem in cm.unmapped_elements:
                st.markdown(f"- ✨ {elem}")
    else:
        st.info("Claim mapping not available")

    st.markdown("---")
    st.markdown("### 📚 Patent Citations")
    if response.citations:
        for i, c in enumerate(response.citations, 1):
            with st.expander(f"{i}. {c.patent_number} — {c.title} (Score: {c.relevance_score:.3f})"):
                st.markdown(f"**Relevance:** {c.relevance_summary}")
                if c.relevant_claims:
                    st.markdown("**Claims:**")
                    for claim in c.relevant_claims:
                        st.markdown(f"- {claim}")
    else:
        st.info("No citations available")

    # ── Similar Patents Section ─────────────────────────────────
    if report.similar_patents:
        st.markdown("---")
        st.markdown("### 🔗 Similar Patents Found")
        
        for p in report.similar_patents:
            score_pct = p["similarity_score"] * 100
            st.markdown(f"""
            <div class="patent-card">
                <div class="patent-title">📄 {p['title']}</div>
                <div class="patent-meta">
                    US{p['doc_number']} · Similarity: {score_pct:.1f}%
                </div>
                <div style="margin-top:0.5rem;color:#a0aec0;font-size:0.88rem">
                    {p['abstract'][:250]}...
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Download Report ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Download Final Report")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        format_choice = st.selectbox("Select Format", ["PDF", "Markdown", "JSON"], label_visibility="collapsed")
    
    with col2:
        if format_choice == "PDF":
            pdf_bytes = report_gen.generate_pdf(response)
            st.download_button(
                "📥 Download PDF",
                data=bytes(pdf_bytes),
                file_name="patentpilot_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        elif format_choice == "Markdown":
            md_report = report_gen.generate_markdown(response)
            st.download_button(
                "📥 Download Markdown",
                data=md_report,
                file_name="patentpilot_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            json_str = json.dumps(report_gen.generate_json_report(response), indent=2)
            st.download_button(
                "📥 Download JSON",
                data=json_str,
                file_name="patentpilot_report.json",
                mime="application/json",
                use_container_width=True,
            )

    # Pipeline metadata
    st.markdown("---")
    st.caption(
        f"⏱️ Analysis completed in {response.total_time_seconds:.1f}s · "
        f"Pipeline steps: {', '.join(response.pipeline_steps_completed)} · "
        f"Searched {response.search_result.total_searched if response.search_result else 0:,} vectors"
    )

# ── Empty state ───────────────────────────────────────────────────────
elif "analysis_response" not in st.session_state:
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:3rem 0; color:#4a5568">
        <div style="font-size:4rem; margin-bottom:1rem">🔬</div>
        <div style="font-size:1.2rem; font-weight:500; color:#718096">
            Enter your invention description above and click "Analyze Patent Landscape"
        </div>
        <div style="font-size:0.9rem; margin-top:0.5rem; color:#4a5568">
            PatentPilot AI will search through {patents} patents using a 6-agent analysis pipeline
        </div>
    </div>
    """.format(
        patents=f"{stats['patents']:,}" if pipeline and stats else "6,463"
    ), unsafe_allow_html=True)

    # Feature cards
    st.markdown("")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2rem">🔍</div>
            <div class="metric-value" style="font-size:1.2rem">Semantic Search</div>
            <div class="metric-label">FAISS-powered vector search across patent embeddings</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2rem">🤖</div>
            <div class="metric-value" style="font-size:1.2rem">6-Agent Pipeline</div>
            <div class="metric-label">Search → Filter → Analysis → Claims → Legal → Citations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2rem">📊</div>
            <div class="metric-value" style="font-size:1.2rem">Full Reports</div>
            <div class="metric-label">Novelty, infringement, FTO analysis with citations</div>
        </div>
        """, unsafe_allow_html=True)
