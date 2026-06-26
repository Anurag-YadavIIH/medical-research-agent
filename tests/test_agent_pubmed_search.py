"""Tests for the PubMed Search agent (service and cache are mocked, not the network)."""

from __future__ import annotations

import pytest

from medical_research_agent.agents.pubmed_search import PubMedSearchAgent, _cache_key
from medical_research_agent.models.query import QueryUnderstanding, SearchFilters
from medical_research_agent.models.state import ResearchState
from medical_research_agent.models.study import Study


class _FakePubMedService:
    closed = False
    received_query: str | None = None
    call_count = 0

    def __init__(self, settings: object = None) -> None:
        pass

    async def search(self, query: str, filters: SearchFilters) -> list[Study]:
        _FakePubMedService.received_query = query
        _FakePubMedService.call_count += 1
        return [Study(pmid="90000001", title="Synthetic fixture study")]

    async def aclose(self) -> None:
        _FakePubMedService.closed = True


class _FakeCache:
    """No-op cache by default — tests opt into a populated store explicitly."""

    store: dict[str, object] = {}

    def __init__(self, settings: object = None) -> None:
        pass

    async def get(self, key: str) -> object | None:
        return self.store.get(key)

    async def set(self, key: str, value: object, ttl: int | None = None) -> None:
        self.store[key] = value

    async def aclose(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _reset_fake_cache_store() -> None:
    _FakeCache.store = {}


def _patch(monkeypatch, service: type = _FakePubMedService, cache: type = _FakeCache) -> None:
    monkeypatch.setattr("medical_research_agent.agents.pubmed_search.PubMedService", service)
    monkeypatch.setattr("medical_research_agent.agents.pubmed_search.Cache", cache)


async def test_uses_reformulated_search_query_when_available(monkeypatch) -> None:
    _patch(monkeypatch)
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
    _patch(monkeypatch)
    agent = PubMedSearchAgent()
    state = ResearchState(question="What treats keratoconus?")

    await agent.run(state)

    assert _FakePubMedService.received_query == "What treats keratoconus?"


async def test_closes_service_even_if_search_raises(monkeypatch) -> None:
    class _FailingService(_FakePubMedService):
        async def search(self, query: str, filters: SearchFilters) -> list[Study]:
            raise RuntimeError("network down")

    _patch(monkeypatch, service=_FailingService)
    agent = PubMedSearchAgent()
    state = ResearchState(question="What treats keratoconus?")

    delta = await agent(state)  # via __call__ so the raised error is caught

    assert any("network down" in e for e in delta["errors"])
    assert _FakePubMedService.closed is True


async def test_cache_hit_skips_pubmed_call_entirely(monkeypatch) -> None:
    class _UncalledService(_FakePubMedService):
        async def search(self, query: str, filters: SearchFilters) -> list[Study]:
            raise AssertionError("PubMedService.search should not be called on a cache hit")

    filters = SearchFilters()
    key = _cache_key("What treats keratoconus?", filters)
    _FakeCache.store = {key: [{"pmid": "90000001", "title": "Cached study"}]}

    _patch(monkeypatch, service=_UncalledService)
    agent = PubMedSearchAgent()
    state = ResearchState(question="What treats keratoconus?", filters=filters)

    delta = await agent.run(state)

    assert delta["studies"] == [Study(pmid="90000001", title="Cached study")]


async def test_cache_miss_calls_pubmed_and_populates_cache(monkeypatch) -> None:
    _FakeCache.store = {}
    _patch(monkeypatch)
    agent = PubMedSearchAgent()
    filters = SearchFilters()
    state = ResearchState(question="What treats keratoconus?", filters=filters)

    delta = await agent.run(state)

    key = _cache_key("What treats keratoconus?", filters)
    assert key in _FakeCache.store
    assert delta["studies"] == [Study(pmid="90000001", title="Synthetic fixture study")]


async def test_cache_key_is_stable_for_same_question_and_filters() -> None:
    filters = SearchFilters(year_min=2020)
    assert _cache_key("Q", filters) == _cache_key("Q", filters)
    assert _cache_key("Q", filters) != _cache_key("Different question", filters)
