"""
PatentPilot AI — Analysis Agent
=================================
Agent 3 of 6 in the multi-agent pipeline.

Responsibility: Analyzes the novelty of the user's invention
compared to the filtered prior art patents. Identifies overlapping
and unique features, and assigns a novelty score.

Input:  Invention description + FilterResult
Output: NoveltyAnalysis
"""

import logging

from backend.llm_client import NOVELTY_PROMPT, SYSTEM_PROMPT, ollama_client
from backend.models import FilterResult, NoveltyAnalysis

logger = logging.getLogger(__name__)


class AnalysisAgent:
    """
    Performs novelty analysis by comparing the invention to prior art.
    
    Uses Mistral to identify which features of the proposed invention
    are genuinely novel vs. already present in existing patents.
    Falls back to heuristic analysis when Ollama is unavailable.
    """

    async def analyze(
        self, invention: str, filter_result: FilterResult
    ) -> NoveltyAnalysis:
        """
        Analyze novelty of the invention against filtered prior art.
        
        Args:
            invention: The user's invention description
            filter_result: Filtered relevant patents from FilterAgent
            
        Returns:
            NoveltyAnalysis with score, overlapping/unique features
        """
        patents = filter_result.relevant_patents
        if not patents:
            return NoveltyAnalysis(
                novelty_score=0.95,
                is_novel=True,
                overlapping_features=[],
                unique_features=["No prior art found — invention appears novel"],
                analysis_text="No relevant prior art was found in the patent database. "
                "This suggests the invention may be highly novel, though a broader "
                "search beyond this database is recommended.",
            )

        llm_available = await ollama_client.is_available()

        if llm_available:
            return await self._llm_analysis(invention, filter_result)
        else:
            return self._heuristic_analysis(invention, filter_result)

    async def _llm_analysis(
        self, invention: str, filter_result: FilterResult
    ) -> NoveltyAnalysis:
        """Use Mistral for novelty analysis."""
        patent_summaries = []
        for p in filter_result.relevant_patents[:8]:
            summary = (
                f"Patent {p.doc_number}: {p.title}\n"
                f"Abstract: {p.abstract[:400]}\n"
                f"Key Claims: {p.claims_text[:400]}"
            )
            patent_summaries.append(summary)

        prompt = NOVELTY_PROMPT.format(
            invention=invention,
            patents="\n\n---\n\n".join(patent_summaries),
        )

        result = await ollama_client.generate_json(prompt, SYSTEM_PROMPT)

        novelty_score = float(result.get("novelty_score", 0.5))

        analysis = NoveltyAnalysis(
            novelty_score=novelty_score,
            is_novel=result.get("is_novel", novelty_score > 0.5),
            overlapping_features=result.get("overlapping_features", []),
            unique_features=result.get("unique_features", []),
            analysis_text=result.get(
                "analysis_text",
                result.get("raw_text", "Analysis completed via LLM."),
            ),
        )

        logger.info(
            f"Novelty analysis complete: score={analysis.novelty_score:.2f}, "
            f"novel={analysis.is_novel}"
        )
        return analysis

    def _heuristic_analysis(
        self, invention: str, filter_result: FilterResult
    ) -> NoveltyAnalysis:
        """Fallback heuristic analysis based on similarity scores."""
        patents = filter_result.relevant_patents
        avg_similarity = sum(p.similarity_score for p in patents) / len(patents)

        # Higher similarity to prior art = lower novelty
        novelty_score = max(0.0, min(1.0, 1.0 - avg_similarity))

        overlapping = [
            f"Overlap with {p.doc_number}: {p.title} "
            f"(similarity: {p.similarity_score:.2f})"
            for p in patents[:5]
        ]

        analysis = NoveltyAnalysis(
            novelty_score=novelty_score,
            is_novel=novelty_score > 0.5,
            overlapping_features=overlapping,
            unique_features=[
                "Detailed feature analysis requires Ollama/Mistral. "
                "Based on vector similarity, novelty score is estimated."
            ],
            analysis_text=(
                f"Heuristic analysis based on {len(patents)} similar patents. "
                f"Average similarity: {avg_similarity:.2f}. "
                f"Estimated novelty score: {novelty_score:.2f}. "
                f"For detailed feature-level analysis, ensure Ollama is running "
                f"with the Mistral model."
            ),
        )

        logger.info(f"Heuristic novelty: score={novelty_score:.2f}")
        return analysis
