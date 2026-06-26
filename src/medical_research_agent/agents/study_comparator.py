"""Compare studies to surface agreement, conflict and the strongest evidence."""

from __future__ import annotations

from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage

from medical_research_agent.agents.base import BaseAgent
from medical_research_agent.llm.factory import get_chat_model
from medical_research_agent.models.comparison import StudyComparison
from medical_research_agent.models.state import ResearchState

_SYSTEM_PROMPT = (
    "You are comparing biomedical studies that have already been retrieved and "
    "appraised. Identify points of agreement, disagreement, and emerging trends "
    "across them. You may only reference the PMIDs given to you below — never "
    "invent a PMID that is not in the provided list."
)


def _build_comparison_matrix(state: ResearchState) -> list[dict[str, str]]:
    """Build the tabular row-per-study matrix deterministically from known fields,
    rather than trusting the LLM to format it — the data is already on hand.
    """
    extracted_by_pmid = {item.pmid: item for item in state.extracted}
    assessment_by_pmid = {item.pmid: item for item in state.assessments}

    matrix: list[dict[str, str]] = []
    for study in state.studies:
        extracted = extracted_by_pmid.get(study.pmid)
        assessment = assessment_by_pmid.get(study.pmid)
        matrix.append(
            {
                "pmid": study.pmid,
                "title": study.title,
                "evidence_level": assessment.evidence_level.label if assessment else "Ungraded",
                "strength": assessment.strength if assessment else "",
                "main_finding": (extracted.main_findings if extracted else "") or "",
            }
        )
    return matrix


class StudyComparatorAgent(BaseAgent):
    """Compare studies to surface agreement, conflict and the strongest evidence."""

    name = "study_comparator"

    async def run(self, state: ResearchState) -> dict[str, object]:
        if not state.studies:
            return {"comparison": StudyComparison()}

        comparison_matrix = _build_comparison_matrix(state)
        context = self._build_context(state)
        model = get_chat_model().with_structured_output(StudyComparison)
        try:
            comparison = cast(
                StudyComparison,
                await model.ainvoke(
                    [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=context)]
                ),
            )
        except Exception as exc:  # noqa: BLE001 - failure shouldn't crash the graph
            return {
                "comparison": StudyComparison(comparison_matrix=comparison_matrix),
                "errors": [f"{self.name}: comparison failed: {exc}"],
            }

        valid_pmids = {study.pmid for study in state.studies}
        comparison.strongest_evidence_pmids = [
            pmid for pmid in comparison.strongest_evidence_pmids if pmid in valid_pmids
        ]
        comparison.conflicting_evidence_pmids = [
            pmid for pmid in comparison.conflicting_evidence_pmids if pmid in valid_pmids
        ]
        comparison.comparison_matrix = comparison_matrix
        return {"comparison": comparison}

    def _build_context(self, state: ResearchState) -> str:
        extracted_by_pmid = {item.pmid: item for item in state.extracted}
        assessment_by_pmid = {item.pmid: item for item in state.assessments}

        lines = ["Studies (cite only these PMIDs):"]
        for study in state.studies:
            extracted = extracted_by_pmid.get(study.pmid)
            assessment = assessment_by_pmid.get(study.pmid)
            lines.append(f"- PMID {study.pmid}: {study.title}")
            if assessment:
                lines.append(
                    f"  Evidence level: {assessment.evidence_level.label}; "
                    f"strength: {assessment.strength or 'unknown'}"
                )
            if extracted and extracted.main_findings:
                lines.append(f"  Main findings: {extracted.main_findings}")
            elif study.abstract:
                lines.append(f"  Abstract: {study.abstract}")
        return "\n".join(lines)
