"""
PatentPilot AI — Pydantic Models
=================================
Structured data models for all request/response schemas, agent I/O,
and internal data structures. Ensures type safety across the pipeline.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Patent Data Models ────────────────────────────────────────────────

class PatentClaim(BaseModel):
    """A single patent claim."""
    claim_id: str = ""
    claim_number: int = 0
    text: str = ""


class Patent(BaseModel):
    """A full patent record from the dataset."""
    doc_number: str = ""
    kind: str = ""
    pub_date: str = ""
    title: str = ""
    abstract: str = ""
    claims: List[PatentClaim] = Field(default_factory=list)
    inventors: List[str] = Field(default_factory=list)
    cpc_codes: List[str] = Field(default_factory=list)


# ── Chunk & Search Models ────────────────────────────────────────────

class ChunkMetadata(BaseModel):
    """Metadata for a text chunk stored in the FAISS index."""
    chunk_id: int
    patent_index: int
    doc_number: str
    title: str
    chunk_type: str  # "abstract", "claims", "combined"
    text: str


class PatentMatch(BaseModel):
    """A patent match returned by the search agent."""
    doc_number: str
    title: str
    abstract: str = ""
    claims_text: str = ""
    similarity_score: float
    chunk_text: str = ""
    relevance_reasoning: str = ""


# ── Agent Output Models ──────────────────────────────────────────────

class SearchResult(BaseModel):
    """Output of the Search Agent."""
    query: str
    matches: List[PatentMatch] = Field(default_factory=list)
    total_searched: int = 0


class FilterResult(BaseModel):
    """Output of the Filter Agent."""
    relevant_patents: List[PatentMatch] = Field(default_factory=list)
    filtered_out: int = 0
    reasoning: str = ""


class NoveltyAnalysis(BaseModel):
    """Output of the Analysis Agent."""
    novelty_score: float = Field(0.0, ge=0.0, le=1.0)
    is_novel: bool = False
    overlapping_features: List[str] = Field(default_factory=list)
    unique_features: List[str] = Field(default_factory=list)
    analysis_text: str = ""


class ClaimElement(BaseModel):
    """A single element in a claim mapping."""
    invention_element: str
    patent_element: str = ""
    patent_doc_number: str = ""
    overlap_type: str = ""  # "identical", "similar", "different", "novel"


class ClaimMapping(BaseModel):
    """Output of the Claim Mapping Agent."""
    mapped_elements: List[ClaimElement] = Field(default_factory=list)
    unmapped_elements: List[str] = Field(default_factory=list)
    overlap_percentage: float = 0.0
    mapping_summary: str = ""


class LegalAssessment(BaseModel):
    """Output of the Legal Reasoning Agent."""
    infringement_risk: str = ""  # "LOW", "MEDIUM", "HIGH"
    infringement_details: str = ""
    fto_opinion: str = ""
    fto_details: str = ""
    recommendations: List[str] = Field(default_factory=list)
    risk_level: str = ""  # "LOW", "MEDIUM", "HIGH"


class Citation(BaseModel):
    """A patent citation with source details."""
    patent_number: str
    title: str
    relevant_claims: List[str] = Field(default_factory=list)
    relevance_score: float = 0.0
    relevance_summary: str = ""


# ── Pipeline & Report Models ─────────────────────────────────────────

class AnalysisRequest(BaseModel):
    """Input to the full analysis pipeline."""
    invention_description: str = Field(
        ..., min_length=20,
        description="Detailed description of the invention idea"
    )
    top_k: int = Field(5, ge=1, le=50)


class AnalysisResponse(BaseModel):
    """Full output of the analysis pipeline."""
    # Core results (strict output format)
    novelty: str = ""
    infringement: str = ""
    fto: str = ""
    reasoning: str = ""
    citations: List[Citation] = Field(default_factory=list)

    # Detailed sub-results
    search_result: Optional[SearchResult] = None
    filter_result: Optional[FilterResult] = None
    novelty_analysis: Optional[NoveltyAnalysis] = None
    claim_mapping: Optional[ClaimMapping] = None
    legal_assessment: Optional[LegalAssessment] = None

    # Metadata
    idea_summary: str = ""
    pipeline_steps_completed: List[str] = Field(default_factory=list)
    total_time_seconds: float = 0.0
    error: Optional[str] = None


class ReportOutput(BaseModel):
    """Structured report for display."""
    idea_summary: str = ""
    similar_patents: List[dict] = Field(default_factory=list)
    claim_comparison: str = ""
    risk_analysis: str = ""
    final_verdict: str = ""
    sources: List[dict] = Field(default_factory=list)
    raw_json: dict = Field(default_factory=dict)
