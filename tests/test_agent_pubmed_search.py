"""Tests for the PubMed Search agent (service is mocked, not the network)."""

from __future__ import annotations

from medical_research_agent.agents.pubmed_search import PubMedSearchAgent
from medical_research_agent.models.query import QueryUnderstanding, SearchFilters
from medical_research_agent.models.state import ResearchState
from medical_research_agent.models.study import Study


class _FakePubMedService:
    closed = False
    received_query: str | None = None

    def __init__(self, settings: object = None) -> None:
        pass

    async def search(self, query: str, filters: SearchFilters) -> list[Study]:
        _FakePubMedService.received_query = query
        return [Study(pmid="90000001", title="Synthetic fixture study")]

    async def aclose(self) -> None:
        _FakePubMedService.closed = True


async def test_uses_reformulated_search_query_when_available(monkeypatch) -> None:
    monkeypatch.setattr(
        "medical_research_agent.agents.pubmed_search.PubMedService", _FakePubMedService
    )
    agent = PubMedSearchAgent()
    state = ResearchState(
        question="What treats keratoconus?",
        query_understanding=QueryUnderstanding(search_query="keratoconus AND crosslinking"),
    )

    delta = await agent.run(state)

    assert _FakePubMedService.received_query == "keratoconus AND crosslinking"
    assert delta["studies"] == [Study(pmid="90000001", title="Synthetic fixture study")]
    assert _FakePubMedService.closed is True


async def test_falls_back_to_raw_question_without_query_understanding(monkeypatch) -> None:
    monkeypatch.setattr(
        "medical_research_agent.agents.pubmed_search.PubMedService", _FakePubMedService
    )
    agent = PubMedSearchAgent()
    state = ResearchState(question="What treats keratoconus?")

    await agent.run(state)

    assert _FakePubMedService.received_query == "What treats keratoconus?"


async def test_closes_service_even_if_search_raises(monkeypatch) -> None:
    class _FailingService(_FakePubMedService):
        async def search(self, query: str, filters: SearchFilters) -> list[Study]:
            raise RuntimeError("network down")

    monkeypatch.setattr(
        "medical_research_agent.agents.pubmed_search.PubMedService", _FailingService
    )
    agent = PubMedSearchAgent()
    state = ResearchState(question="What treats keratoconus?")

    delta = await agent(state)  # via __call__ so the raised error is caught

    assert any("network down" in e for e in delta["errors"])
    assert _FakePubMedService.closed is True
