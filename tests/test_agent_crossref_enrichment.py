"""Tests for the CrossRef Enrichment agent (service is mocked, not the network)."""

from __future__ import annotations

from medical_research_agent.agents.crossref_enrichment import CrossRefEnrichmentAgent
from medical_research_agent.models.state import ResearchState
from medical_research_agent.models.study import Study


class _FakeCrossRefService:
    closed = False

    def __init__(self, settings: object = None) -> None:
        pass

    async def enrich(self, study: Study) -> Study:
        return study.model_copy(update={"doi": f"10.9999/test.synthetic.{study.pmid}"})

    async def aclose(self) -> None:
        _FakeCrossRefService.closed = True


async def test_enriches_each_study_and_closes_service(monkeypatch) -> None:
    monkeypatch.setattr(
        "medical_research_agent.agents.crossref_enrichment.CrossRefService", _FakeCrossRefService
    )
    agent = CrossRefEnrichmentAgent()
    state = ResearchState(
        question="Q",
        studies=[Study(pmid="90000001", title="A"), Study(pmid="90000002", title="B")],
    )

    delta = await agent.run(state)

    studies = delta["studies"]
    assert [s.doi for s in studies] == [
        "10.9999/test.synthetic.90000001",
        "10.9999/test.synthetic.90000002",
    ]
    assert _FakeCrossRefService.closed is True


async def test_no_studies_returns_empty_list(monkeypatch) -> None:
    monkeypatch.setattr(
        "medical_research_agent.agents.crossref_enrichment.CrossRefService", _FakeCrossRefService
    )
    agent = CrossRefEnrichmentAgent()
    state = ResearchState(question="Q")

    delta = await agent.run(state)

    assert delta["studies"] == []
