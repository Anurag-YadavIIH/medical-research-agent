"""Integration tests for POST /research and GET /studies/{query_id}.

The LangGraph pipeline, PubMed/CrossRef and the LLM are all mocked here — only
the FastAPI route + persistence layer (against an in-memory SQLite DB, see
conftest.py) is exercised for real.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from medical_research_agent.api.schemas import DISCLAIMER
from medical_research_agent.config import Settings
from medical_research_agent.models.report import EvidenceReport, ReferenceEntry
from medical_research_agent.models.study import Study

_STUDY = Study(
    pmid="90000001",
    title="Synthetic Study of Keratoconus Treatment",
    authors=["Testauthor J"],
    journal="Journal of Synthetic Testing",
    publication_year=2023,
)
_REPORT = EvidenceReport(
    question="What treats keratoconus?",
    markdown="# What treats keratoconus?\n\nSynthetic report content.",
    evidence_summary="Synthetic evidence summary [PMID: 90000001].",
    studies=[_STUDY],
    references=[
        ReferenceEntry(
            pmid="90000001",
            vancouver="Testauthor J. Synthetic Study of Keratoconus Treatment. "
            "Journal of Synthetic Testing. 2023. PMID: 90000001.",
        )
    ],
)


class _FakeGraph:
    def __init__(self, final_state: dict[str, Any]) -> None:
        self._final_state = final_state

    async def ainvoke(self, state: object, config: object = None) -> dict[str, Any]:
        return self._final_state


def _configured_settings() -> Settings:
    return Settings(default_llm_provider="groq", groq_api_key="fake-key-for-tests")


def _patch_llm_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "medical_research_agent.api.routes.research.get_settings", _configured_settings
    )


def _patch_graph(monkeypatch: pytest.MonkeyPatch, final_state: dict[str, Any]) -> None:
    monkeypatch.setattr(
        "medical_research_agent.api.routes.research.build_research_graph",
        lambda: _FakeGraph(final_state),
    )


def test_research_success_persists_and_returns_expected_shape(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_llm_configured(monkeypatch)
    _patch_graph(
        monkeypatch,
        {"errors": [], "studies": [_STUDY], "report": _REPORT},
    )

    response = client.post("/research", json={"question": "What treats keratoconus?"})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "What treats keratoconus?"
    assert body["report_markdown"] == _REPORT.markdown
    assert body["machine_json"] == _REPORT.to_machine_json()
    assert body["warnings"] == []
    assert body["disclaimer"] == DISCLAIMER
    assert body["query_id"]

    studies_response = client.get(f"/studies/{body['query_id']}")
    assert studies_response.status_code == 200
    persisted = studies_response.json()
    assert len(persisted) == 1
    assert persisted[0]["pmid"] == "90000001"


def test_research_seeded_errors_surface_as_warnings_with_200(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_llm_configured(monkeypatch)
    _patch_graph(
        monkeypatch,
        {
            "errors": ["query_understanding: LLM call failed: boom"],
            "studies": [_STUDY],
            "report": _REPORT,
        },
    )

    response = client.post("/research", json={"question": "What treats keratoconus?"})

    assert response.status_code == 200
    body = response.json()
    assert body["warnings"] == ["query_understanding: LLM call failed: boom"]


def test_research_missing_llm_key_returns_503_with_actionable_message(
    client: TestClient,
) -> None:
    # Deliberately does not patch get_settings/build_research_graph: the repo's
    # real .env ships with blank provider keys, so this exercises the actual
    # pre-flight check without needing the graph to ever be constructed.
    response = client.post("/research", json={"question": "What treats keratoconus?"})

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "API key" in detail
    assert "GROQ_API_KEY" in detail


def test_research_pipeline_crash_returns_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_llm_configured(monkeypatch)

    class _CrashingGraph:
        async def ainvoke(self, state: object, config: object = None) -> dict[str, Any]:
            raise RuntimeError("graph compile exploded")

    monkeypatch.setattr(
        "medical_research_agent.api.routes.research.build_research_graph",
        lambda: _CrashingGraph(),
    )

    response = client.post("/research", json={"question": "What treats keratoconus?"})

    assert response.status_code == 500


def test_get_studies_404_for_unknown_query_id(client: TestClient) -> None:
    response = client.get("/studies/does-not-exist")

    assert response.status_code == 404
