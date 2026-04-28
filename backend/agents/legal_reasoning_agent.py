"""
PatentPilot AI — Legal Reasoning Agent
========================================
Agent 5 of 6 in the multi-agent pipeline.

Responsibility: Synthesizes outputs from the Analysis Agent and
Claim Mapping Agent to provide legal assessments of infringement
risk and freedom-to-operate (FTO) opinion.

Input:  NoveltyAnalysis + ClaimMapping + relevant patents
Output: LegalAssessment with risk level, FTO opinion, recommendations
"""

import logging

from backend.llm_client import LEGAL_REASONING_PROMPT, SYSTEM_PROMPT, ollama_client
from backend.models import (
    ClaimMapping,
    FilterResult,
    LegalAssessment,
    NoveltyAnalysis,
)

logger = logging.getLogger(__name__)


class LegalReasoningAgent:
    """
    Provides legal risk assessment based on patent analysis.
    
    Combines novelty analysis and claim mapping to assess
    infringement risk and freedom-to-operate. Uses LLM for
    nuanced reasoning or falls back to rule-based assessment.
    """

    async def assess(
        self,
        invention: str,
        novelty: NoveltyAnalysis,
        claim_mapping: ClaimMapping,
        filter_result: FilterResult,
    ) -> LegalAssessment:
        """
        Perform legal assessment of infringement and FTO.
        
        Args:
            invention: The user's invention description
            novelty: Novelty analysis results
            claim_mapping: Claim mapping results
            filter_result: Filtered relevant patents
            
        Returns:
            LegalAssessment with risk levels and recommendations
        """
        llm_available = await ollama_client.is_available()

        if llm_available:
            return await self._llm_assessment(
                invention, novelty, claim_mapping, filter_result
            )
        else:
            return self._rule_based_assessment(novelty, claim_mapping, filter_result)

    async def _llm_assessment(
        self,
        invention: str,
        novelty: NoveltyAnalysis,
        claim_mapping: ClaimMapping,
        filter_result: FilterResult,
    ) -> LegalAssessment:
        """Use Mistral for legal reasoning."""
        # Compile patent summaries
        patent_summaries = []
        for p in filter_result.relevant_patents[:5]:
            patent_summaries.append(
                f"Patent {p.doc_number}: {p.title}\n"
                f"  Similarity: {p.similarity_score:.2f}"
            )

        prompt = LEGAL_REASONING_PROMPT.format(
            invention=invention,
            novelty=(
                f"Novelty Score: {novelty.novelty_score:.2f}\n"
                f"Is Novel: {novelty.is_novel}\n"
                f"Overlapping Features: {', '.join(novelty.overlapping_features[:5])}\n"
                f"Unique Features: {', '.join(novelty.unique_features[:5])}\n"
                f"Analysis: {novelty.analysis_text[:500]}"
            ),
            claim_mapping=(
                f"Overlap Percentage: {claim_mapping.overlap_percentage:.1f}%\n"
                f"Mapped Elements: {len(claim_mapping.mapped_elements)}\n"
                f"Unmapped Elements: {len(claim_mapping.unmapped_elements)}\n"
                f"Summary: {claim_mapping.mapping_summary}"
            ),
            patents="\n".join(patent_summaries),
        )

        result = await ollama_client.generate_json(prompt, SYSTEM_PROMPT)

        assessment = LegalAssessment(
            infringement_risk=result.get("infringement_risk", "MEDIUM"),
            infringement_details=result.get(
                "infringement_details", "Assessment via LLM."
            ),
            fto_opinion=result.get("fto_opinion", ""),
            fto_details=result.get("fto_details", ""),
            recommendations=result.get("recommendations", []),
            risk_level=result.get("risk_level", result.get("infringement_risk", "MEDIUM")),
        )

        logger.info(
            f"Legal assessment complete: risk={assessment.risk_level}, "
            f"infringement={assessment.infringement_risk}"
        )
        return assessment

    def _rule_based_assessment(
        self,
        novelty: NoveltyAnalysis,
        claim_mapping: ClaimMapping,
        filter_result: FilterResult,
    ) -> LegalAssessment:
        """Fallback rule-based legal assessment."""
        # Determine risk level based on scores
        overlap = claim_mapping.overlap_percentage
        novelty_score = novelty.novelty_score
        num_similar = len(filter_result.relevant_patents)

        if overlap > 70 or (novelty_score < 0.3 and num_similar > 3):
            risk_level = "HIGH"
        elif overlap > 40 or (novelty_score < 0.5 and num_similar > 1):
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        recommendations = []
        if risk_level == "HIGH":
            recommendations = [
                "Consult a patent attorney before proceeding",
                "Consider design-around strategies to avoid existing claims",
                "Perform a comprehensive freedom-to-operate search with professional tools",
                "Document all differences from prior art carefully",
            ]
        elif risk_level == "MEDIUM":
            recommendations = [
                "Review the overlapping claims in detail with a patent professional",
                "Consider narrowing the invention scope to emphasize novel features",
                "Monitor the status of similar patent applications",
            ]
        else:
            recommendations = [
                "The invention appears to have good freedom to operate",
                "Consider filing a provisional patent application to establish priority",
                "Continue monitoring the patent landscape for new filings",
            ]

        fto_map = {
            "HIGH": "Restricted — significant overlap with existing patents detected",
            "MEDIUM": "Conditional — some overlap exists, modifications recommended",
            "LOW": "Favorable — minimal overlap with existing patents",
        }

        return LegalAssessment(
            infringement_risk=risk_level,
            infringement_details=(
                f"Rule-based assessment: {overlap:.1f}% claim overlap, "
                f"novelty score {novelty_score:.2f}, "
                f"{num_similar} similar patents found."
            ),
            fto_opinion=fto_map.get(risk_level, ""),
            fto_details=(
                f"Based on {num_similar} relevant patents with "
                f"average claim overlap of {overlap:.1f}%."
            ),
            recommendations=recommendations,
            risk_level=risk_level,
        )
