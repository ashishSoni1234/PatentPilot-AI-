"""
PatentPilot AI — Report Generator
====================================
Transforms the AnalysisResponse from the multi-agent pipeline into
structured output formats: JSON report and human-readable Markdown.
"""

import logging
from typing import Dict
from fpdf import FPDF

from backend.models import AnalysisResponse, ReportOutput

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates structured reports from pipeline analysis results.
    
    Outputs:
    1. Strict JSON format (novelty, infringement, fto, reasoning, citations)
    2. ReportOutput with sections for UI display
    3. Markdown report for export
    """

    def generate_json_report(self, response: AnalysisResponse) -> Dict:
        """
        Generate the strict JSON output format as specified.
        
        Returns:
            {
                "novelty": "...",
                "infringement": "...",
                "fto": "...",
                "reasoning": "...",
                "citations": [...]
            }
        """
        return {
            "novelty": response.novelty,
            "infringement": response.infringement,
            "fto": response.fto,
            "reasoning": response.reasoning,
            "citations": [
                {
                    "patent_number": c.patent_number,
                    "title": c.title,
                    "relevant_claims": c.relevant_claims,
                    "relevance_score": round(c.relevance_score, 3),
                    "relevance_summary": c.relevance_summary,
                }
                for c in response.citations
            ],
        }

    def generate_report(self, response: AnalysisResponse) -> ReportOutput:
        """
        Generate a full report for the Streamlit UI.
        
        Sections:
        - Idea Summary
        - Similar Patents
        - Claim Comparison
        - Risk Analysis
        - Final Verdict
        - Sources
        """
        # Similar patents section
        similar_patents = []
        if response.filter_result:
            for p in response.filter_result.relevant_patents:
                similar_patents.append({
                    "doc_number": p.doc_number,
                    "title": p.title,
                    "abstract": p.abstract[:300],
                    "similarity_score": round(p.similarity_score, 3),
                    "relevance_reasoning": p.relevance_reasoning,
                })

        # Claim comparison section
        claim_comparison = ""
        if response.claim_mapping:
            cm = response.claim_mapping
            claim_comparison = f"Overlap: {cm.overlap_percentage:.1f}%\n\n"
            if cm.mapped_elements:
                claim_comparison += "**Mapped Elements:**\n"
                for elem in cm.mapped_elements:
                    claim_comparison += (
                        f"- [{elem.overlap_type.upper()}] "
                        f"{elem.invention_element} → "
                        f"{elem.patent_element[:100]} "
                        f"(Patent {elem.patent_doc_number})\n"
                    )
            if cm.unmapped_elements:
                claim_comparison += "\n**Novel Elements (No Match):**\n"
                for elem in cm.unmapped_elements:
                    claim_comparison += f"- {elem}\n"
            claim_comparison += f"\n{cm.mapping_summary}"

        # Risk analysis section
        risk_analysis = ""
        if response.legal_assessment:
            la = response.legal_assessment
            risk_analysis = (
                f"**Infringement Risk:** {la.infringement_risk}\n"
                f"{la.infringement_details}\n\n"
                f"**Freedom to Operate:** {la.fto_opinion}\n"
                f"{la.fto_details}\n\n"
                f"**Recommendations:**\n"
            )
            for rec in la.recommendations:
                risk_analysis += f"- {rec}\n"

        # Final verdict
        final_verdict = self._generate_verdict(response)

        # Sources
        sources = [
            {
                "patent_number": c.patent_number,
                "title": c.title,
                "relevance_score": round(c.relevance_score, 3),
                "summary": c.relevance_summary,
            }
            for c in response.citations
        ]

        return ReportOutput(
            idea_summary=response.idea_summary,
            similar_patents=similar_patents,
            claim_comparison=claim_comparison,
            risk_analysis=risk_analysis,
            final_verdict=final_verdict,
            sources=sources,
            raw_json=self.generate_json_report(response),
        )

    def generate_markdown(self, response: AnalysisResponse) -> str:
        """Generate a Markdown report for export/download."""
        report = self.generate_report(response)

        md = f"""# PatentPilot AI — Patent Analysis Report

## 📋 Idea Summary
{report.idea_summary}

---

## 🔍 Similar Patents Found ({len(report.similar_patents)})
"""
        for i, p in enumerate(report.similar_patents, 1):
            md += f"""
### {i}. {p['title']}
- **Patent Number:** US{p['doc_number']}
- **Similarity Score:** {p['similarity_score']:.3f}
- **Abstract:** {p['abstract']}
"""

        md += f"""
---

## ⚖️ Claim Comparison
{report.claim_comparison}

---

## ⚠️ Risk Analysis
{report.risk_analysis}

---

## 🏛️ Final Verdict
{report.final_verdict}

---

## 📚 Sources
"""
        for s in report.sources:
            md += f"- **{s['patent_number']}** — {s['title']} (relevance: {s['relevance_score']:.3f})\n"

        md += f"""
---

*Report generated by PatentPilot AI*
*Analysis time: {response.total_time_seconds:.1f}s*
*Pipeline steps completed: {', '.join(response.pipeline_steps_completed)}*
"""
        return md

    @staticmethod
    def _generate_verdict(response: AnalysisResponse) -> str:
        """Generate a final verdict paragraph."""
        parts = []

        if response.novelty_analysis:
            na = response.novelty_analysis
            if na.is_novel:
                parts.append(
                    f"The invention shows **good novelty** with a score of "
                    f"{na.novelty_score:.2f}/1.00."
                )
            else:
                parts.append(
                    f"The invention has **limited novelty** with a score of "
                    f"{na.novelty_score:.2f}/1.00. Significant overlap with "
                    f"existing prior art was detected."
                )

        if response.legal_assessment:
            la = response.legal_assessment
            risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(
                la.risk_level, "⚪"
            )
            parts.append(
                f"Infringement risk is assessed as {risk_emoji} **{la.risk_level}**."
            )
            parts.append(f"Freedom to operate: {la.fto_opinion}")

        if response.claim_mapping:
            cm = response.claim_mapping
            parts.append(
                f"Claim overlap with existing patents: **{cm.overlap_percentage:.1f}%**."
            )

        if not parts:
            return "Analysis incomplete — insufficient data to render a verdict."

        return " ".join(parts)

    def generate_pdf(self, response: AnalysisResponse) -> bytes:
        """Generate a PDF report for export/download."""
        report = self.generate_report(response)
        
        pdf = FPDF()
        pdf.add_page()
        
        # Helper to handle unicode
        def safe_text(txt):
            return txt.encode('latin-1', 'replace').decode('latin-1')

        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 10, safe_text("PatentPilot AI — Patent Analysis Report"), ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, safe_text("Idea Summary"), ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, safe_text(report.idea_summary))
        pdf.ln(5)

        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, safe_text(f"Similar Patents Found ({len(report.similar_patents)})"), ln=True)
        pdf.set_font("Helvetica", "", 11)
        for i, p in enumerate(report.similar_patents, 1):
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 6, safe_text(f"{i}. {p['title']} (US{p['doc_number']})"), ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, safe_text(f"Similarity Score: {p['similarity_score']:.3f}"), ln=True)
            pdf.multi_cell(0, 5, safe_text(f"Abstract: {p['abstract'][:300]}..."))
            pdf.ln(3)

        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, safe_text("Claim Comparison"), ln=True)
        pdf.set_font("Helvetica", "", 11)
        # Strip simple markdown formatting for PDF display
        cc_clean = report.claim_comparison.replace("**", "").replace("`", "")
        pdf.multi_cell(0, 6, safe_text(cc_clean))
        pdf.ln(5)

        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, safe_text("Risk Analysis"), ln=True)
        pdf.set_font("Helvetica", "", 11)
        ra_clean = report.risk_analysis.replace("**", "")
        pdf.multi_cell(0, 6, safe_text(ra_clean))
        pdf.ln(5)

        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, safe_text("Final Verdict"), ln=True)
        pdf.set_font("Helvetica", "", 11)
        fv_clean = report.final_verdict.replace("**", "")
        pdf.multi_cell(0, 6, safe_text(fv_clean))
        pdf.ln(10)

        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 5, safe_text(f"Report generated by PatentPilot AI in {response.total_time_seconds:.1f}s"), ln=True)

        return pdf.output(dest="S")
