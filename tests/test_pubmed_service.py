"""Tests for the PubMed E-utilities client.

All PMIDs in fixtures are clearly synthetic (90000001/90000002 — not real
PubMed identifiers) so cited evidence can never be mistaken for a real study.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
import respx

from medical_research_agent.config import Settings
from medical_research_agent.models.query import SearchFilters
from medical_research_agent.services.pubmed import _EUTILS_BASE, PubMedService

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def settings() -> Settings:
    return Settings(ncbi_email="tester@example.com", ncbi_tool="mra-tests")


@pytest.fixture
def service(settings: Settings) -> PubMedService:
    return PubMedService(settings=settings)


@pytest.mark.respx(base_url=_EUTILS_BASE)
async def test_esearch_returns_pmids(respx_mock: respx.MockRouter, service: PubMedService) -> None:
    respx_mock.get("/esearch.fcgi").mock(
        return_value=httpx.Response(
            200,
            json={"esearchresult": {"idlist": ["90000001", "90000002"]}},
        )
    )

    pmids = await service.esearch("keratoconus", retmax=10)

    assert pmids == ["90000001", "90000002"]
    await service.aclose()


@pytest.mark.respx(base_url=_EUTILS_BASE)
async def test_efetch_parses_studies(respx_mock: respx.MockRouter, service: PubMedService) -> None:
    xml = (FIXTURES / "pubmed_efetch_sample.xml").read_bytes()
    respx_mock.get("/efetch.fcgi").mock(return_value=httpx.Response(200, content=xml))

    studies = await service.efetch(["90000001", "90000002"])

    assert len(studies) == 2
    first, second = studies

    assert first.pmid == "90000001"
    assert first.title == "A Synthetic Study of Keratoconus Treatment Outcomes"
    assert first.journal == "Journal of Synthetic Testing"
    assert first.publication_year == 2023
    assert first.authors == ["Testauthor J", "Synthetic Research Group"]
    assert "BACKGROUND:" in first.abstract
    assert "METHODS:" in first.abstract
    assert first.publication_types == ["Randomized Controlled Trial", "Journal Article"]

    assert second.pmid == "90000002"
    assert second.publication_year == 2021  # parsed from MedlineDate fallback
    assert second.authors == ["Fixture S"]
    assert (
        second.abstract
        == "Single-section synthetic abstract with no label, used purely for test coverage."
    )

    await service.aclose()


@pytest.mark.respx(base_url=_EUTILS_BASE)
async def test_efetch_empty_pmids_skips_request(
    respx_mock: respx.MockRouter, service: PubMedService
) -> None:
    studies = await service.efetch([])

    assert studies == []
    assert respx_mock.calls.call_count == 0
    await service.aclose()


@pytest.mark.respx(base_url=_EUTILS_BASE)
async def test_search_combines_esearch_and_efetch(
    respx_mock: respx.MockRouter, service: PubMedService
) -> None:
    xml = (FIXTURES / "pubmed_efetch_sample.xml").read_bytes()
    esearch_route = respx_mock.get("/esearch.fcgi").mock(
        return_value=httpx.Response(
            200, json={"esearchresult": {"idlist": ["90000001", "90000002"]}}
        )
    )
    respx_mock.get("/efetch.fcgi").mock(return_value=httpx.Response(200, content=xml))

    studies = await service.search("keratoconus", SearchFilters(max_papers=5))

    assert len(studies) == 2
    sent_term = esearch_route.calls.last.request.url.params["term"]
    assert "keratoconus" in sent_term
    assert "humans" in sent_term.lower()

    await service.aclose()


@pytest.mark.respx(base_url=_EUTILS_BASE)
async def test_search_applies_year_and_type_filters(
    respx_mock: respx.MockRouter, service: PubMedService
) -> None:
    esearch_route = respx_mock.get("/esearch.fcgi").mock(
        return_value=httpx.Response(200, json={"esearchresult": {"idlist": []}})
    )

    filters = SearchFilters(
        year_min=2020,
        year_max=2024,
        article_types=["Randomized Controlled Trial"],
        humans_only=True,
        max_papers=3,
    )
    await service.search("keratoconus", filters)

    sent_term = esearch_route.calls.last.request.url.params["term"]
    assert "2020/01/01" in sent_term
    assert "2024/12/31" in sent_term
    assert "Randomized Controlled Trial" in sent_term

    await service.aclose()


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("NCBI_EMAIL"),
    reason="Live PubMed integration test requires NCBI_EMAIL to be set.",
)
async def test_live_pubmed_search_smoke() -> None:
    """Optional smoke test against the real NCBI API. Skipped in normal CI runs."""
    service = PubMedService()
    try:
        studies = await service.search("aspirin", SearchFilters(max_papers=1))
        assert isinstance(studies, list)
    finally:
        await service.aclose()
