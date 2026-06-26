"""Tests for the Summary agent — citation safety is enforced in code, not just prompting."""

from __future__ import annotations

from medical_research_agent.agents.summary import SummaryAgent, _NarrativeSections
from medical_research_agent.models.state import ResearchState
from medical_research_agent.models.study import Study


def _studies(n: int) -> list[Study]:
    return [
        Study(
            pmid=f"9000000{i}",
            title=f"Synthetic Study {i}",
            authors=[f"Author{i} A"],
            journal="Journal of Synthetic Testing",
            publication_year=2023,
        )
        for i in range(1, n + 1)
    ]


async def test_builds_vancouver_references_from_retrieved_metadata(fake_chat_model) -> None:
    fake_chat_model(
        "medical_research_agent.agents.summary",
        _NarrativeSections(evidence_summary="Some synthesized evidence [PMID: 90000001]."),
    )

    agent = SummaryAgent()
    state = ResearchState(question="Q", studies=_studies(3))

    delta = await agent.run(state)

    report = delta["report"]
    assert len(report.references) == 3
    assert report.references[0].pmid == "90000001"
    assert "Author1 A" in report.references[0].vancouver
    assert "Journal of Synthetic Testing" in report.references[0].vancouver
    assert "PMID: 90000001" in report.references[0].vancouver
    assert report.DISCLAIMER in report.markdown


async def test_strips_fabricated_pmid_citation_and_records_error(fake_chat_model) -> None:
    fake_chat_model(
        "medical_research_agent.agents.summary",
        _NarrativeSections(
            evidence_summary=(
                "Treatment A improved outcomes [PMID: 90000001], and so did an "
                "unrelated study [PMID: 12345678] that was never retrieved."
            )
        ),
    )

    agent = SummaryAgent()
    state = ResearchState(question="Q", studies=_studies(3))

    delta = await agent.run(state)

    report = delta["report"]
    assert "[PMID: 90000001]" in report.evidence_summary
    assert "12345678" not in report.evidence_summary
    assert any("12345678" in e for e in delta["errors"])


async def test_states_uncertainty_when_evidence_is_thin(fake_chat_model) -> None:
    fake_chat_model(
        "medical_research_agent.agents.summary",
        _NarrativeSections(evidence_summary="Limited findings observed."),
    )

    agent = SummaryAgent()
    state = ResearchState(question="Q", studies=_studies(1))

    delta = await agent.run(state)

    report = delta["report"]
    assert "only 1 study(ies) were retrieved" in report.evidence_summary
    assert "preliminary" in report.evidence_summary


async def test_no_studies_skips_llm_call_and_preserves_disclaimer(fake_chat_model) -> None:
    fake_chat_model("medical_research_agent.agents.summary")

    agent = SummaryAgent()
    state = ResearchState(question="Q")

    delta = await agent.run(state)

    report = delta["report"]
    assert report.references == []
    assert report.DISCLAIMER in report.markdown
    assert "preliminary" in report.markdown


async def test_llm_failure_still_produces_report_with_error(fake_chat_model) -> None:
    fake_chat_model("medical_research_agent.agents.summary", RuntimeError("llm down"))

    agent = SummaryAgent()
    state = ResearchState(question="Q", studies=_studies(3))

    delta = await agent.run(state)

    report = delta["report"]
    assert len(report.references) == 3
    assert any("llm down" in e for e in delta["errors"])
