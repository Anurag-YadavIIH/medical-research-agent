"""Tests for the CrossRef enrichment client.

DOIs in fixtures use the reserved/synthetic ``10.9999`` test prefix and an
explicit ``synthetic`` suffix so they can never be mistaken for a real DOI.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from medical_research_agent.config import Settings
from medical_research_agent.models.study import Study
from medical_research_agent.services.crossref import _CROSSREF_BASE, CrossRefService


@pytest.fixture
def settings() -> Settings:
    return Settings(crossref_mailto="tester@example.com")


@pytest.fixture
def service(settings: Settings) -> CrossRefService:
    return CrossRefService(settings=settings)


def _study(**overrides: object) -> Study:
    defaults: dict[str, object] = {
        "pmid": "90000001",
        "title": "A Synthetic Study of Keratoconus Treatment Outcomes",
    }
    defaults.update(overrides)
    return Study(**defaults)  # type: ignore[arg-type]


@pytest.mark.respx(base_url=_CROSSREF_BASE)
async def test_enrich_attaches_doi_on_strong_match(
    respx_mock: respx.MockRouter, service: CrossRefService
) -> None:
    respx_mock.get("/works").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "items": [
                        {
                            "DOI": "10.9999/test.synthetic.0001",
                            "title": ["A Synthetic Study of Keratoconus Treatment Outcomes"],
                            "publisher": "Synthetic Test Publishing",
                            "is-referenced-by-count": 7,
                            "URL": "https://doi.org/10.9999/test.synthetic.0001",
                        }
                    ]
                }
            },
        )
    )

    enriched = await service.enrich(_study())

    assert enriched.doi == "10.9999/test.synthetic.0001"
    assert enriched.citation_count == 7
    assert enriched.publisher == "Synthetic Test Publishing"
    assert enriched.url == "https://doi.org/10.9999/test.synthetic.0001"
    await service.aclose()


@pytest.mark.respx(base_url=_CROSSREF_BASE)
async def test_enrich_ignores_weak_title_match(
    respx_mock: respx.MockRouter, service: CrossRefService
) -> None:
    respx_mock.get("/works").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "items": [
                        {
                            "DOI": "10.9999/test.synthetic.unrelated",
                            "title": ["Completely Unrelated Synthetic Fixture Title"],
                            "publisher": "Synthetic Test Publishing",
                            "is-referenced-by-count": 1,
                            "URL": "https://doi.org/10.9999/test.synthetic.unrelated",
                        }
                    ]
                }
            },
        )
    )

    study = _study()
    enriched = await service.enrich(study)

    assert enriched == study
    assert enriched.doi is None
    await service.aclose()


@pytest.mark.respx(base_url=_CROSSREF_BASE)
async def test_enrich_no_results_returns_unchanged(
    respx_mock: respx.MockRouter, service: CrossRefService
) -> None:
    respx_mock.get("/works").mock(return_value=httpx.Response(200, json={"message": {"items": []}}))

    study = _study()
    enriched = await service.enrich(study)

    assert enriched == study
    await service.aclose()


@pytest.mark.respx(base_url=_CROSSREF_BASE)
async def test_enrich_http_error_returns_unchanged(
    respx_mock: respx.MockRouter, service: CrossRefService
) -> None:
    respx_mock.get("/works").mock(return_value=httpx.Response(503))

    study = _study()
    enriched = await service.enrich(study)

    assert enriched == study
    await service.aclose()


@pytest.mark.respx(base_url=_CROSSREF_BASE)
async def test_enrich_malformed_json_returns_unchanged(
    respx_mock: respx.MockRouter, service: CrossRefService
) -> None:
    respx_mock.get("/works").mock(
        return_value=httpx.Response(
            200, content=b"not json", headers={"Content-Type": "application/json"}
        )
    )

    study = _study()
    enriched = await service.enrich(study)

    assert enriched == study
    await service.aclose()


async def test_enrich_skips_request_for_empty_title(service: CrossRefService) -> None:
    study = _study(title="")

    enriched = await service.enrich(study)

    assert enriched == study
    await service.aclose()
