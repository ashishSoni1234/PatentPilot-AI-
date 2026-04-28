"""
PatentPilot AI — Filter Agent
===============================
Agent 2 of 6 in the multi-agent pipeline.

Responsibility: Takes the raw search results from the Search Agent
and uses LLM-based reasoning to filter out false positives. Only
patents that are genuinely relevant to the invention idea pass through.

Input:  SearchResult (from Search Agent) + invention description
Output: FilterResult with filtered PatentMatch list
"""

import logging
from typing import List

from backend.config import FILTER_RELEVANCE_THRESHOLD, MAX_FILTERED_PATENTS
from backend.llm_client import FILTER_PROMPT, SYSTEM_PROMPT, ollama_client
from backend.models import FilterResult, PatentMatch, SearchResult

logger = logging.getLogger(__name__)


class FilterAgent:
    """
    Filters search results using LLM-based relevance assessment.
    
    For each candidate patent, the agent prompts Mistral to evaluate
    whether the patent is genuinely relevant to the invention idea,
    providing a relevance score and reasoning.
    
    Falls back to similarity-score-based filtering when Ollama is
    unavailable.
    """

    async def filter(
        self, invention: str, search_result: SearchResult
    ) -> FilterResult:
        """
        Filter search results to only genuinely relevant patents.
        
        Args:
            invention: The user's invention description
            search_result: Raw search results from SearchAgent
            
        Returns:
            FilterResult with relevant patents and filtering reasoning
        """
        candidates = search_result.matches
        if not candidates:
            return FilterResult(
                relevant_patents=[],
                filtered_out=0,
                reasoning="No search results to filter.",
            )

        # Check if LLM is available
        llm_available = await ollama_client.is_available()

        if llm_available:
            return await self._llm_filter(invention, candidates)
        else:
            logger.warning("Ollama not available — using score-based filtering")
            return self._score_filter(candidates)

    async def _llm_filter(
        self, invention: str, candidates: List[PatentMatch]
    ) -> FilterResult:
        """Use Mistral to assess relevance of each candidate patent."""
        # Prepare patent summaries for the prompt
        patent_summaries = []
        for i, p in enumerate(candidates[:15]):  # Limit to top 15 for context window
            summary = (
                f"Patent {i+1} (Doc: {p.doc_number}):\n"
                f"  Title: {p.title}\n"
                f"  Abstract: {p.abstract[:300]}...\n"
                f"  Similarity Score: {p.similarity_score:.3f}"
            )
            patent_summaries.append(summary)

        prompt = FILTER_PROMPT.format(
            invention=invention,
            patents="\n\n".join(patent_summaries),
        )

        result = await ollama_client.generate_json(prompt, SYSTEM_PROMPT)

        # Parse LLM response
        assessments = result.get("assessments", [])
        relevant_patents: List[PatentMatch] = []
        filtered_count = 0

        for assessment in assessments:
            doc_num = assessment.get("doc_number", "")
            score = float(assessment.get("relevance_score", 0))
            reason = assessment.get("reason", "")

            # Find matching candidate
            for candidate in candidates:
                if candidate.doc_number == doc_num:
                    if score >= FILTER_RELEVANCE_THRESHOLD:
                        candidate.relevance_reasoning = reason
                        relevant_patents.append(candidate)
                    else:
                        filtered_count += 1
                    break

        # If LLM didn't return assessments for all, include remaining
        # high-score ones as a safety net
        assessed_docs = {a.get("doc_number") for a in assessments}
        for candidate in candidates:
            if (
                candidate.doc_number not in assessed_docs
                and candidate.similarity_score >= 0.5
                and len(relevant_patents) < MAX_FILTERED_PATENTS
            ):
                relevant_patents.append(candidate)

        # Cap results
        relevant_patents = relevant_patents[:MAX_FILTERED_PATENTS]

        logger.info(
            f"Filter complete: {len(relevant_patents)} relevant, "
            f"{filtered_count} filtered out"
        )

        return FilterResult(
            relevant_patents=relevant_patents,
            filtered_out=filtered_count,
            reasoning=f"LLM-assessed {len(assessments)} patents, "
            f"{len(relevant_patents)} passed threshold {FILTER_RELEVANCE_THRESHOLD}",
        )

    def _score_filter(self, candidates: List[PatentMatch]) -> FilterResult:
        """Fallback: filter by similarity score only."""
        relevant = [
            p for p in candidates
            if p.similarity_score >= FILTER_RELEVANCE_THRESHOLD
        ]
        relevant = relevant[:MAX_FILTERED_PATENTS]
        filtered_count = len(candidates) - len(relevant)

        return FilterResult(
            relevant_patents=relevant,
            filtered_out=filtered_count,
            reasoning=f"Score-based filtering (Ollama unavailable). "
            f"Threshold: {FILTER_RELEVANCE_THRESHOLD}",
        )
