"""
PatentPilot AI — Citation Agent
=================================
Agent 6 of 6 in the multi-agent pipeline.

Responsibility: Compiles properly formatted patent citations from
all previous agent outputs. Each citation includes the patent number,
title, relevant claims, and a relevance summary.

Input:  All previous agent outputs
Output: List[Citation] with formatted references
"""

import logging
from typing import List

from backend.models import (
    Citation,
    ClaimMapping,
    FilterResult,
    NoveltyAnalysis,
)

logger = logging.getLogger(__name__)


class CitationAgent:
    """
    Compiles patent citations from the analysis pipeline results.
    
    Creates properly formatted citations with relevance summaries
    by cross-referencing filter results, claim mappings, and
    novelty analysis.
    """

    def compile_citations(
        self,
        filter_result: FilterResult,
        novelty: NoveltyAnalysis,
        claim_mapping: ClaimMapping,
    ) -> List[Citation]:
        """
        Compile citations from all analysis results.
        
        Args:
            filter_result: Filtered relevant patents
            novelty: Novelty analysis results
            claim_mapping: Claim mapping results
            
        Returns:
            List of formatted citations sorted by relevance
        """
        citations: List[Citation] = []

        for patent in filter_result.relevant_patents:
            # Find relevant claims from claim mapping
            relevant_claims = []
            for elem in claim_mapping.mapped_elements:
                if elem.patent_doc_number == patent.doc_number:
                    claim_desc = (
                        f"{elem.overlap_type}: {elem.invention_element} → "
                        f"{elem.patent_element[:100]}"
                    )
                    relevant_claims.append(claim_desc)

            # Generate relevance summary
            is_overlapping = any(
                patent.doc_number in feat
                for feat in novelty.overlapping_features
            )

            relevance_summary = self._build_relevance_summary(
                patent, is_overlapping, relevant_claims
            )

            citation = Citation(
                patent_number=f"US{patent.doc_number}",
                title=patent.title,
                relevant_claims=relevant_claims if relevant_claims else [
                    "No specific claim mapping performed"
                ],
                relevance_score=patent.similarity_score,
                relevance_summary=relevance_summary,
            )
            citations.append(citation)

        # Sort by relevance score (highest first)
        citations.sort(key=lambda c: c.relevance_score, reverse=True)

        logger.info(f"Compiled {len(citations)} citations")
        return citations

    @staticmethod
    def _build_relevance_summary(patent, is_overlapping: bool, claims: list) -> str:
        """Build a human-readable relevance summary for a citation."""
        parts = []

        # Similarity assessment
        score = patent.similarity_score
        if score >= 0.7:
            parts.append("Highly similar to the proposed invention")
        elif score >= 0.5:
            parts.append("Moderately similar to the proposed invention")
        else:
            parts.append("Some similarity to the proposed invention detected")

        # Overlap status
        if is_overlapping:
            parts.append("Features overlapping with prior art identified")

        # Claims
        if claims:
            parts.append(f"{len(claims)} claim element(s) mapped")

        # Abstract snippet
        if patent.abstract:
            parts.append(f"Abstract: {patent.abstract[:150]}...")

        return ". ".join(parts) + "."
