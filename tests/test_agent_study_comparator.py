"""Tests for the Study Comparator agent."""

from __future__ import annotations

from medical_research_agent.agents.study_comparator import StudyComparatorAgent
from medical_research_agent.models.comparison import StudyComparison
from medical_research_agent.models.state import ResearchState
from medical_research_agent.models.study import Study


async def test_filters_out_pmids_not_in_retrieved_studies(fake_chat_model) -> None:
    # The LLM hallucinates a PMID that was never retrieved; the agent must drop it.
    fake_comparison = StudyComparison(
        agreements=["Both studies report symptom improvement."],
        strongest_evidence_pmids=["90000001", "99999999"],
        conflicting_evidence_pmids=["99999999"],
    )
    fake_chat_model("medical_research_agent.agents.study_comparator", fake_comparison)

    agent = StudyComparatorAgent()
    state = ResearchState(
        question="Q",
        studies=[
            Study(pmid="90000001", title="A", abstract="First."),
            Study(pmid="90000002", title="B", abstract="Second."),
        ],
    )

    delta = await agent.run(state)

    comparison = delta["comparison"]
    assert comparison.strongest_evidence_pmids == ["90000001"]
    assert comparison.conflicting_evidence_pmids == []


async def test_no_studies_returns_empty_comparison_without_llm_call(fake_chat_model) -> None:
    fake_chat_model("medical_research_agent.agents.study_comparator")

    agent = StudyComparatorAgent()
    state = ResearchState(question="Q")

    delta = await agent.run(state)

    assert delta["comparison"] == StudyComparison()


async def test_llm_failure_still_returns_deterministic_matrix_and_error(fake_chat_model) -> None:
    fake_chat_model("medical_research_agent.agents.study_comparator", RuntimeError("boom"))

    agent = StudyComparatorAgent()
    state = ResearchState(question="Q", studies=[Study(pmid="90000001", title="A")])

    delta = await agent.run(state)

    comparison = delta["comparison"]
    assert comparison.agreements == []
    assert comparison.comparison_matrix == [
        {
            "pmid": "90000001",
            "title": "A",
            "evidence_level": "Ungraded",
            "strength": "",
            "main_finding": "",
        }
    ]
    assert any("boom" in e for e in delta["errors"])


async def test_comparison_matrix_is_built_deterministically_from_state(fake_chat_model) -> None:
    from medical_research_agent.models.evidence import EvidenceAssessment, EvidenceLevel
    from medical_research_agent.models.study import ExtractedStudy

    fake_chat_model("medical_research_agent.agents.study_comparator", StudyComparison())

    agent = StudyComparatorAgent()
    state = ResearchState(
        question="Q",
        studies=[Study(pmid="90000001", title="A")],
        extracted=[ExtractedStudy(pmid="90000001", main_findings="Symptom improvement observed.")],
        assessments=[
            EvidenceAssessment(
                pmid="90000001", evidence_level=EvidenceLevel.LEVEL_II, strength="strong"
            )
        ],
    )

    delta = await agent.run(state)

    assert delta["comparison"].comparison_matrix == [
        {
            "pmid": "90000001",
            "title": "A",
            "evidence_level": "Level II — Randomized controlled trial",
            "strength": "strong",
            "main_finding": "Symptom improvement observed.",
        }
    ]
