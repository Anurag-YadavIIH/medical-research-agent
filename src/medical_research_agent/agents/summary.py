"""Synthesise the final evidence report with Vancouver references and caveats.

Citation safety is enforced in code, not just by prompting: references are built
deterministically from retrieved study metadata, and any ``[PMID: ...]`` the model
writes that isn't in the retrieved set is stripped from the narrative before it
ever reaches the report.
"""

from __future__ import annotations

from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from medical_research_agent.agents.base import BaseAgent
from medical_research_agent.citation_utils import strip_fabricated_citations
from medical_research_agent.llm.factory import get_chat_model
from medical_research_agent.models.comparison import StudyComparison
from medical_research_agent.models.report import EvidenceReport, ReferenceEntry
from medical_research_agent.models.state import ResearchState
from medical_research_agent.models.study import Study

_SYSTEM_PROMPT = (
    "You are writing the narrative sections of an evidence-synthesis report for "
    "clinicians. You may cite findings ONLY using the exact notation '[PMID: <id>]' "
    "and ONLY for PMIDs explicitly listed below — never cite, invent, or imply a "
    "study, statistic, or PMID that is not in that list. If few studies were "
    "retrieved or the evidence is otherwise thin, say so explicitly rather than "
    "overstating the conclusion."
)

_THIN_EVIDENCE_THRESHOLD = 3


class _NarrativeSections(BaseModel):
    """LLM-authored prose only — citations, references and disclaimer are not."""

    evidence_summary: str = ""
    clinical_implications: str = ""
    limitations: str = ""
    future_directions: str = ""


def _vancouver_reference(study: Study) -> ReferenceEntry:
    authors = ", ".join(study.authors[:6]) if study.authors else "[No authors listed]"
    if len(study.authors) > 6:
        authors += ", et al"
    year = str(study.publication_year) if study.publication_year else "n.d."
    title = study.title or "[No title available]"
    journal = f" {study.journal}." if study.journal else ""
    vancouver = f"{authors}. {title}.{journal} {year}. PMID: {study.pmid}."
    return ReferenceEntry(pmid=study.pmid, doi=study.doi, vancouver=vancouver, url=study.url)


def _sanitized_comparison(
    comparison: StudyComparison | None, valid_pmids: set[str]
) -> StudyComparison | None:
    """Re-filter PMIDs against state.studies — defense in depth on top of the
    filtering study_comparator already does, since this is what gets serialized
    and persisted as part of the report.
    """
    if comparison is None:
        return None
    return comparison.model_copy(
        update={
            "strongest_evidence_pmids": [
                pmid for pmid in comparison.strongest_evidence_pmids if pmid in valid_pmids
            ],
            "conflicting_evidence_pmids": [
                pmid for pmid in comparison.conflicting_evidence_pmids if pmid in valid_pmids
            ],
        }
    )


def _render_markdown(report: EvidenceReport) -> str:
    lines = [
        f"# {report.question}",
        "",
        "## Evidence Summary",
        report.evidence_summary or "_Insufficient evidence retrieved to summarize._",
        "",
        "## Clinical Implications",
        report.clinical_implications or "_Not assessed._",
        "",
        "## Limitations",
        report.limitations or "_Not assessed._",
        "",
        "## Future Directions",
        report.future_directions or "_Not assessed._",
        "",
        "## References",
    ]
    lines += [f"{i}. {ref.vancouver}" for i, ref in enumerate(report.references, start=1)]
    lines += ["", f"_Disclaimer: {report.DISCLAIMER}_"]
    return "\n".join(lines)


class SummaryAgent(BaseAgent):
    """Synthesise the final evidence report with Vancouver references and caveats."""

    name = "summary"

    async def run(self, state: ResearchState) -> dict[str, object]:
        valid_pmids = {study.pmid for study in state.studies}
        references = [_vancouver_reference(study) for study in state.studies]

        sections, errors = await self._generate_narrative(state, valid_pmids)

        report = EvidenceReport(
            question=state.question,
            studies=state.studies,
            extracted=state.extracted,
            assessments=state.assessments,
            comparison=_sanitized_comparison(state.comparison, valid_pmids),
            references=references,
            evidence_summary=sections.evidence_summary,
            clinical_implications=sections.clinical_implications,
            limitations=sections.limitations,
            future_directions=sections.future_directions,
        )
        report.markdown = _render_markdown(report)

        update: dict[str, object] = {"report": report}
        if errors:
            update["errors"] = errors
        return update

    async def _generate_narrative(
        self, state: ResearchState, valid_pmids: set[str]
    ) -> tuple[_NarrativeSections, list[str]]:
        errors: list[str] = []
        sections = _NarrativeSections()

        if state.studies:
            model = get_chat_model().with_structured_output(_NarrativeSections)
            try:
                sections = cast(
                    _NarrativeSections,
                    await model.ainvoke(
                        [
                            SystemMessage(content=_SYSTEM_PROMPT),
                            HumanMessage(content=self._build_context(state, valid_pmids)),
                        ]
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - failure shouldn't crash the graph
                errors.append(f"{self.name}: narrative generation failed: {exc}")

        cleaned: dict[str, str] = {}
        for field in (
            "evidence_summary",
            "clinical_implications",
            "limitations",
            "future_directions",
        ):
            text, fabricated = strip_fabricated_citations(getattr(sections, field), valid_pmids)
            if fabricated:
                errors.append(
                    f"{self.name}: dropped fabricated PMID citation(s) {fabricated} from {field}"
                )
            cleaned[field] = text
        sections = _NarrativeSections(**cleaned)

        if len(state.studies) < _THIN_EVIDENCE_THRESHOLD:
            note = (
                f"Note: only {len(state.studies)} study(ies) were retrieved for this "
                "question. Treat this summary as preliminary and consult full-text "
                "literature before drawing conclusions."
            )
            sections.evidence_summary = f"{note}\n\n{sections.evidence_summary}".strip()

        return sections, errors

    def _build_context(self, state: ResearchState, valid_pmids: set[str]) -> str:
        extracted_by_pmid = {item.pmid: item for item in state.extracted}
        assessment_by_pmid = {item.pmid: item for item in state.assessments}

        lines = [
            f"Clinical question: {state.question}",
            f"Citable PMIDs (use only these): {sorted(valid_pmids)}",
            "",
            "Studies:",
        ]
        for study in state.studies:
            extracted = extracted_by_pmid.get(study.pmid)
            assessment = assessment_by_pmid.get(study.pmid)
            lines.append(f"- PMID {study.pmid}: {study.title}")
            if assessment:
                lines.append(f"  Evidence level: {assessment.evidence_level.label}")
            if extracted and extracted.main_findings:
                lines.append(f"  Main findings: {extracted.main_findings}")
            elif study.abstract:
                lines.append(f"  Abstract: {study.abstract}")
        if state.comparison:
            lines.append("")
            lines.append("Cross-study comparison:")
            lines.append(f"  Agreements: {state.comparison.agreements}")
            lines.append(f"  Disagreements: {state.comparison.disagreements}")
        return "\n".join(lines)
