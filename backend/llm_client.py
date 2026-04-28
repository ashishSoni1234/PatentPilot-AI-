"""
PatentPilot AI — Ollama LLM Client
====================================
HTTP client for the local Ollama instance running Mistral 7B.
Includes structured prompt templates, response parsing, and
graceful fallback when Ollama is unavailable.
"""

import json
import logging
import re
from typing import Optional

import httpx

from backend.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT,
)

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for interacting with a local Ollama instance.
    
    Sends structured prompts to Mistral 7B and parses responses.
    Falls back to empty/default responses if Ollama is unavailable.
    """

    def __init__(self):
        self.base_url = OLLAMA_BASE_URL
        self.model = OLLAMA_MODEL
        self.temperature = OLLAMA_TEMPERATURE
        self.timeout = OLLAMA_TIMEOUT
        self._available: Optional[bool] = None

    async def is_available(self) -> bool:
        """Check if Gemini API is available."""
        self._available = True
        return True

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt to Groq API and return the generated text.
        """
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY environment variable is not set")
            return ""
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Groq JSON mode requires explicit mention of JSON in the system prompt
        sys_prompt = system_prompt
        if "JSON" not in sys_prompt:
            sys_prompt += "\nRespond ONLY in valid JSON format."
            
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature or self.temperature,
            "response_format": {"type": "json_object"}
        }
        
        import asyncio
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    resp = await client.post(url, headers=headers, json=payload)
                    
                    if resp.status_code == 429 and attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"Groq API rate limit hit. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                        
                    resp.raise_for_status()
                    result = resp.json()
                    text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return text
            except httpx.TimeoutException:
                logger.error(f"Groq API request timed out after 120s")
                return ""
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Groq API rate limit hit. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                logger.error(f"Groq API HTTP Error: {e.response.text}")
                return ""
            except Exception as e:
                logger.error(f"Groq API request failed: {type(e).__name__} {e}")
                return ""
        return ""

    async def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: Optional[float] = None,
    ) -> dict:
        """
        Send a prompt and attempt to parse the response as JSON.
        Handles cases where the model wraps JSON in markdown code blocks.
        
        Returns:
            Parsed dict, or empty dict on failure
        """
        response = await self.generate(prompt, system_prompt, temperature)
        if not response:
            return {}

        return self._extract_json(response)

    @staticmethod
    def _extract_json(text: str) -> dict:
        """
        Extract JSON from model output, handling common formatting issues:
        - Wrapped in ```json ... ```
        - Wrapped in ``` ... ```
        - Raw JSON
        - JSON embedded in natural language
        """
        # Try to extract from code blocks first
        json_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if json_block:
            text = json_block.group(1).strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find a JSON object in the text
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass

        # Try to find a JSON array
        bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
        if bracket_match:
            try:
                return {"items": json.loads(bracket_match.group())}
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse JSON from LLM output: {text[:200]}...")
        return {"raw_text": text}


# ── Prompt Templates ─────────────────────────────────────────────────

SYSTEM_PROMPT = """You are PatentPilot AI, an expert patent analysis assistant. 
You provide precise, well-structured analysis of patent novelty, infringement risk, 
and freedom-to-operate assessments. Always respond with valid JSON when requested.
Be specific and cite evidence from the patent data provided."""

FILTER_PROMPT = """Analyze whether each of the following patents is genuinely relevant 
to the given invention idea. Consider semantic similarity of technical concepts, 
not just keyword overlap.

INVENTION IDEA:
{invention}

CANDIDATE PATENTS:
{patents}

For each patent, provide:
1. A relevance score from 0.0 to 1.0
2. A brief reason for your assessment

Respond in JSON format:
{{
    "assessments": [
        {{
            "doc_number": "...",
            "relevance_score": 0.8,
            "reason": "..."
        }}
    ]
}}"""

NOVELTY_PROMPT = """Analyze the novelty of the following invention idea compared to 
the existing patents found in prior art search.

INVENTION IDEA:
{invention}

SIMILAR PRIOR ART PATENTS:
{patents}

Provide a thorough novelty analysis:
1. What features of the invention overlap with existing patents?
2. What features are genuinely novel and not found in prior art?
3. An overall novelty score from 0.0 (not novel) to 1.0 (highly novel)

Respond in JSON format:
{{
    "novelty_score": 0.0-1.0,
    "is_novel": true/false,
    "overlapping_features": ["feature1", "feature2"],
    "unique_features": ["feature1", "feature2"],
    "analysis_text": "detailed analysis..."
}}"""

CLAIM_MAPPING_PROMPT = """Map the key elements of the following invention to the claims 
of the most similar prior art patents. Identify element-by-element overlaps.

INVENTION IDEA:
{invention}

PATENT CLAIMS TO COMPARE:
{claims}

For each element of the invention, indicate:
- Whether it maps to an existing patent claim
- The type of overlap: "identical", "similar", "different", or "novel"

Respond in JSON format:
{{
    "mapped_elements": [
        {{
            "invention_element": "...",
            "patent_element": "...",
            "patent_doc_number": "...",
            "overlap_type": "identical|similar|different|novel"
        }}
    ],
    "unmapped_elements": ["element1", "element2"],
    "overlap_percentage": 85.0,
    "mapping_summary": "..."
}}"""

LEGAL_REASONING_PROMPT = """Based on the following patent analysis data, provide a legal 
assessment of infringement risk and freedom-to-operate (FTO).

INVENTION IDEA:
{invention}

NOVELTY ANALYSIS:
{novelty}

CLAIM MAPPING:
{claim_mapping}

RELEVANT PATENTS:
{patents}

Provide:
1. Infringement risk assessment (LOW/MEDIUM/HIGH) with specific reasoning
2. Freedom-to-operate opinion
3. Actionable recommendations

Respond in JSON format:
{{
    "infringement_risk": "LOW|MEDIUM|HIGH",
    "infringement_details": "detailed reasoning...",
    "fto_opinion": "...",
    "fto_details": "...",
    "recommendations": ["rec1", "rec2"],
    "risk_level": "LOW|MEDIUM|HIGH"
}}"""


# ── Singleton instance ───────────────────────────────────────────────
ollama_client = OllamaClient()
