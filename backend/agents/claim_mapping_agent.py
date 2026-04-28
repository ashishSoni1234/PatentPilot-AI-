"""
PatentPilot AI — Claim Mapping Agent
======================================
Agent 4 of 6 in the multi-agent pipeline.

Responsibility: Performs element-by-element comparison between the
user's invention and the claims of the most similar prior art patents.
Identifies exactly which elements overlap, are similar, or are novel.

Input:  Invention description + relevant patent claims
Output: ClaimMapping with element-level overlap details
"""

import logging

from backend.llm_client import CLAIM_MAPPING_PROMPT, SYSTEM_PROMPT, ollama_client
from backend.models import ClaimElement, ClaimMapping, FilterResult

logger = logging.getLogger(__name__)


class ClaimMappingAgent:
    """
    Maps invention elements to prior art patent claims.
    
    Uses LLM to perform element-by-element claim comparison,
    identifying identical, similar, different, and novel elements.
    """

    async def map_claims(
        self, invention: str, filter_result: FilterResult
    ) -> ClaimMapping:
        """
        Map invention elements to patent claims.
        
        Args:
            invention: The user's invention description
            filter_result: Filtered relevant patents with claims
            
        Returns:
            ClaimMapping with element-level details
        """
        patents = filter_result.relevant_patents
        if not patents:
            return ClaimMapping(
                mapped_elements=[],
                unmapped_elements=["All invention elements appear novel"],
                overlap_percentage=0.0,
                mapping_summary="No relevant patents found for claim comparison.",
            )

        llm_available = await ollama_client.is_available()

        if llm_available:
            return await self._llm_mapping(invention, patents)
        else:
            return self._heuristic_mapping(invention, patents)

    async def _llm_mapping(self, invention: str, patents: list) -> ClaimMapping:
        """Use Mistral for detailed claim mapping."""
        # Compile claims from top patents
        claims_text = []
        for p in patents[:5]:
            if p.claims_text:
                claims_text.append(
                    f"Patent {p.doc_number} ({p.title}):\n{p.claims_text[:600]}"
                )

        if not claims_text:
            return ClaimMapping(
                mapped_elements=[],
                unmapped_elements=["No claims data available for comparison"],
                overlap_percentage=0.0,
                mapping_summary="Referenced patents have no extractable claims.",
            )

        prompt = CLAIM_MAPPING_PROMPT.format(
            invention=invention,
            claims="\n\n---\n\n".join(claims_text),
        )

        result = await ollama_client.generate_json(prompt, SYSTEM_PROMPT)

        mapped_elements = []
        for elem in result.get("mapped_elements", []):
            mapped_elements.append(
                ClaimElement(
                    invention_element=elem.get("invention_element", ""),
                    patent_element=elem.get("patent_element", ""),
                    patent_doc_number=elem.get("patent_doc_number", ""),
                    overlap_type=elem.get("overlap_type", "unknown"),
                )
            )

        mapping = ClaimMapping(
            mapped_elements=mapped_elements,
            unmapped_elements=result.get("unmapped_elements", []),
            overlap_percentage=float(result.get("overlap_percentage", 0.0)),
            mapping_summary=result.get("mapping_summary", "Claim mapping completed."),
        )

        logger.info(
            f"Claim mapping complete: {len(mapped_elements)} mapped, "
            f"{len(mapping.unmapped_elements)} unmapped, "
            f"overlap={mapping.overlap_percentage:.1f}%"
        )
        return mapping

    def _heuristic_mapping(self, invention: str, patents: list) -> ClaimMapping:
        """Fallback: basic claim overlap estimation."""
        inv_words = set(invention.lower().split())

        mapped = []
        total_overlap = 0

        for p in patents[:5]:
            if p.claims_text:
                claim_words = set(p.claims_text.lower().split())
                common = inv_words & claim_words
                overlap_ratio = len(common) / max(len(inv_words), 1)
                total_overlap += overlap_ratio

                if common:
                    mapped.append(
                        ClaimElement(
                            invention_element=", ".join(list(common)[:5]),
                            patent_element=p.claims_text[:200],
                            patent_doc_number=p.doc_number,
                            overlap_type="similar" if overlap_ratio > 0.3 else "different",
                        )
                    )

        avg_overlap = (total_overlap / max(len(patents[:5]), 1)) * 100

        return ClaimMapping(
            mapped_elements=mapped,
            unmapped_elements=["Detailed mapping requires Ollama/Mistral"],
            overlap_percentage=avg_overlap,
            mapping_summary=f"Heuristic word-overlap mapping. "
            f"Average overlap: {avg_overlap:.1f}%",
        )
