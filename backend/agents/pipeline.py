"""
PatentPilot AI — Pipeline Orchestrator
========================================
Chains all 6 agents in sequence to produce a complete patent analysis.

Pipeline Flow:
  User Input → Search Agent → Filter Agent → Analysis Agent →
  Claim Mapping Agent → Legal Reasoning Agent → Citation Agent →
  Structured Report

Each agent takes the structured output of previous agents as input.
If any agent fails, the pipeline continues with partial results.
"""

import logging
import time

from backend.agents.analysis_agent import AnalysisAgent
from backend.agents.citation_agent import CitationAgent
from backend.agents.claim_mapping_agent import ClaimMappingAgent
from backend.agents.filter_agent import FilterAgent
from backend.agents.legal_reasoning_agent import LegalReasoningAgent
from backend.agents.search_agent import SearchAgent
from backend.models import AnalysisResponse

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """
    Orchestrates the 6-agent patent analysis pipeline.
    
    Runs agents sequentially (not parallel) to conserve memory on
    resource-constrained systems. Collects all outputs and compiles
    the final AnalysisResponse.
    """

    def __init__(self, search_agent: SearchAgent):
        self.search_agent = search_agent
        self.filter_agent = FilterAgent()
        self.analysis_agent = AnalysisAgent()
        self.claim_mapping_agent = ClaimMappingAgent()
        self.legal_reasoning_agent = LegalReasoningAgent()
        self.citation_agent = CitationAgent()

    async def run(
        self, invention: str, top_k: int = 20
    ) -> AnalysisResponse:
        """
        Execute the full patent analysis pipeline.
        
        Args:
            invention: User's invention description
            top_k: Number of search results to retrieve
            
        Returns:
            Complete AnalysisResponse with all agent outputs
        """
        start_time = time.time()
        steps_completed = []
        response = AnalysisResponse(idea_summary=invention[:500])

        try:
            # ── Step 1: Search Agent ──────────────────────────────
            logger.info("Pipeline Step 1/6: Search Agent")
            search_result = self.search_agent.search(invention, top_k=top_k)
            response.search_result = search_result
            steps_completed.append("search")
            logger.info(f"  → Found {len(search_result.matches)} matches")

            # ── Step 2: Filter Agent ──────────────────────────────
            logger.info("Pipeline Step 2/6: Filter Agent")
            filter_result = await self.filter_agent.filter(invention, search_result)
            response.filter_result = filter_result
            steps_completed.append("filter")
            logger.info(
                f"  → {len(filter_result.relevant_patents)} relevant patents"
            )

            # ── Step 3 & 4: Analysis & Claim Mapping IN PARALLEL ──
            logger.info("Pipeline Step 3/6 & 4/6: Analysis & Claim Mapping Agents (Parallel)")
            import asyncio
            novelty, claim_mapping = await asyncio.gather(
                self.analysis_agent.analyze(invention, filter_result),
                self.claim_mapping_agent.map_claims(invention, filter_result)
            )
            
            response.novelty_analysis = novelty
            steps_completed.append("analysis")
            logger.info(f"  → Novelty score: {novelty.novelty_score:.2f}")
            
            response.claim_mapping = claim_mapping
            steps_completed.append("claim_mapping")
            logger.info(f"  → {len(claim_mapping.mapped_elements)} elements mapped")

            # ── Step 5: Legal Reasoning Agent ─────────────────────
            logger.info("Pipeline Step 5/6: Legal Reasoning Agent")
            legal = await self.legal_reasoning_agent.assess(
                invention, novelty, claim_mapping, filter_result
            )
            response.legal_assessment = legal
            steps_completed.append("legal_reasoning")
            logger.info(f"  → Risk level: {legal.risk_level}")

            # ── Step 6: Citation Agent ────────────────────────────
            logger.info("Pipeline Step 6/6: Citation Agent")
            citations = self.citation_agent.compile_citations(
                filter_result, novelty, claim_mapping
            )
            response.citations = citations
            steps_completed.append("citations")
            logger.info(f"  → {len(citations)} citations compiled")

            # ── Compile Final Output ──────────────────────────────
            response.novelty = (
                f"Novelty Score: {novelty.novelty_score:.2f}/1.00 — "
                f"{'Novel' if novelty.is_novel else 'Not Novel'}. "
                f"{novelty.analysis_text}"
            )
            response.infringement = (
                f"Risk Level: {legal.infringement_risk}. "
                f"{legal.infringement_details}"
            )
            response.fto = (
                f"FTO Assessment: {legal.fto_opinion}. "
                f"{legal.fto_details}"
            )
            response.reasoning = (
                f"Analysis based on {len(filter_result.relevant_patents)} "
                f"relevant patents from {search_result.total_searched} indexed chunks. "
                f"Claim overlap: {claim_mapping.overlap_percentage:.1f}%. "
                f"{'; '.join(legal.recommendations[:3])}"
            )

        except Exception as e:
            logger.error(f"Pipeline error at step {len(steps_completed)+1}: {e}")
            response.error = str(e)

        elapsed = time.time() - start_time
        response.pipeline_steps_completed = steps_completed
        response.total_time_seconds = round(elapsed, 2)

        logger.info(
            f"Pipeline complete: {len(steps_completed)}/6 steps in {elapsed:.1f}s"
        )
        return response
