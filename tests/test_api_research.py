"""Integration tests for POST /research and GET /studies/{query_id}.

The LangGraph pipeline, PubMed/CrossRef and the LLM are all mocked here — only
the FastAPI route + persistence layer (against an in-memory SQLite DB, see
conftest.py) is exercised for real.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from structlog.testing import capture_logs

from medical_research_agent.api.schemas import DISCLAIMER
from medical_research_agent.config import Settings
from medical_research_agent.database.repository import ResearchRepository
from medical_research_agent.models.comparison import StudyComparison
from medical_research_agent.models.evidence import EvidenceAssessment, EvidenceLevel
from medical_research_agent.models.report import EvidenceReport, ReferenceEntry
from medical_research_agent.models.study import ExtractedStudy, Study

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
    extracted=[
        ExtractedStudy(pmid="90000001", main_findings="Synthetic symptom improvement observed.")
    ],
    assessments=[
        EvidenceAssessment(
            pmid="90000001", evidence_level=EvidenceLevel.LEVEL_II, strength="moderate"
        )
    ],
    comparison=StudyComparison(
        agreements=["Synthetic agreement."],
        strongest_evidence_pmids=["90000001"],
        comparison_matrix=[
            {
                "pmid": "90000001",
                "title": "Synthetic Study of Keratoconus Treatment",
                "evidence_level": "Level II — Randomized controlled trial",
                "strength": "moderate",
                "main_finding": "Synthetic symptom improvement observed.",
            }
        ],
    ),
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


def test_research_question_over_max_length_returns_422(client: TestClient) -> None:
    response = client.post(
        "/research",
        json={"question": "x" * 501},
    )

    assert response.status_code == 422


def test_research_too_many_keywords_returns_422(client: TestClient) -> None:
    response = client.post(
        "/research",
        json={
            "question": "What treats keratoconus?",
            "filters": {"keywords": [f"kw{i}" for i in range(21)]},
        },
    )

    assert response.status_code == 422


def test_research_overlong_keyword_returns_422(client: TestClient) -> None:
    response = client.post(
        "/research",
        json={
            "question": "What treats keratoconus?",
            "filters": {"keywords": ["x" * 101]},
        },
    )

    assert response.status_code == 422


def test_research_extra_top_level_field_returns_422(client: TestClient) -> None:
    # max_papers at the top level (instead of nested under filters) must be a
    # hard 422, not a silent no-op, so callers learn immediately that their
    # request is malformed.
    response = client.post(
        "/research",
        json={"question": "What treats keratoconus?", "max_papers": 5},
    )

    assert response.status_code == 422


def test_research_missing_llm_key_returns_503_with_actionable_message(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Explicitly force blank keys rather than relying on the ambient .env being
    # unconfigured — a developer's real .env (e.g. for a --live run) may have
    # real keys set, and this test must still exercise the pre-flight check
    # without ever reaching build_research_graph().
    monkeypatch.setattr(
        "medical_research_agent.api.routes.research.get_settings",
        lambda: Settings(default_llm_provider="groq", groq_api_key=None, openai_api_key=None),
    )

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
            raise RuntimeError("graph compile exploded: internal db dsn leaked here")

    monkeypatch.setattr(
        "medical_research_agent.api.routes.research.build_research_graph",
        lambda: _CrashingGraph(),
    )

    with capture_logs() as cap_logs:
        response = client.post("/research", json={"question": "What treats keratoconus?"})

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail == "Research pipeline failed unexpectedly. Please try again or contact support."
    assert "internal db dsn leaked here" not in detail
    assert "RuntimeError" not in detail

    assert any("internal db dsn leaked here" in str(entry.get("error", "")) for entry in cap_logs)
    assert any(entry.get("exc_type") == "RuntimeError" for entry in cap_logs)


def test_get_studies_404_for_unknown_query_id(client: TestClient) -> None:
    response = client.get("/studies/does-not-exist")

    assert response.status_code == 404


def test_research_falls_back_to_minimal_report_when_summary_node_crashed_outright(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No "report" key at all -- simulates the summary node raising before it
    # could build anything (caught by BaseAgent.__call__, not summary.py's own
    # try/except), as opposed to a normal LLM error which still yields a report.
    _patch_llm_configured(monkeypatch)
    _patch_graph(
        monkeypatch,
        {
            "errors": ["summary: unexpected crash"],
            "studies": [_STUDY],
        },
    )

    response = client.post("/research", json={"question": "What treats keratoconus?"})

    assert response.status_code == 200
    body = response.json()
    assert body["warnings"] == ["summary: unexpected crash"]
    assert "Report generation failed" in body["report_markdown"]
    assert DISCLAIMER in body["report_markdown"]
    assert body["machine_json"]["studies"][0]["pmid"] == "90000001"


def test_research_persistence_failure_returns_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_llm_configured(monkeypatch)
    _patch_graph(monkeypatch, {"errors": [], "studies": [_STUDY], "report": _REPORT})

    async def _failing_save_research_run(*args: object, **kwargs: object) -> None:
        raise RuntimeError("database is down: postgresql://mra:secretpw@host/db")

    monkeypatch.setattr(
        "medical_research_agent.database.repository.ResearchRepository.save_research_run",
        _failing_save_research_run,
    )

    with capture_logs() as cap_logs:
        response = client.post("/research", json={"question": "What treats keratoconus?"})

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail == "Failed to persist research results. Please try again."
    assert "secretpw" not in detail
    assert "RuntimeError" not in detail

    assert any("secretpw" in str(entry.get("error", "")) for entry in cap_logs)
    assert any(entry.get("exc_type") == "RuntimeError" for entry in cap_logs)


async def test_extracted_and_comparison_survive_the_persistence_round_trip(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, db_session: AsyncSession
) -> None:
    _patch_llm_configured(monkeypatch)
    _patch_graph(monkeypatch, {"errors": [], "studies": [_STUDY], "report": _REPORT})

    response = client.post("/research", json={"question": "What treats keratoconus?"})
    assert response.status_code == 200
    body = response.json()

    # The exact same dict returned to the client...
    assert body["machine_json"]["extracted"] == [
        {
            "pmid": "90000001",
            "objective": "",
            "sample_size": None,
            "sample_size_description": "",
            "study_design": "",
            "population": "",
            "intervention": "",
            "comparator": "",
            "outcomes": [],
            "main_findings": "Synthetic symptom improvement observed.",
            "statistical_significance": "",
            "limitations": "",
        }
    ]
    assert body["machine_json"]["comparison"]["agreements"] == ["Synthetic agreement."]
    assert body["machine_json"]["comparison"]["strongest_evidence_pmids"] == ["90000001"]
    assert body["machine_json"]["comparison"]["comparison_matrix"] == [
        {
            "pmid": "90000001",
            "title": "Synthetic Study of Keratoconus Treatment",
            "evidence_level": "Level II — Randomized controlled trial",
            "strength": "moderate",
            "main_finding": "Synthetic symptom improvement observed.",
        }
    ]

    # ...must be exactly what's stored, queried back via the repository directly.
    repo = ResearchRepository(db_session)
    summary_record = await repo.get_summary(body["query_id"])
    assert summary_record is not None
    assert summary_record.machine_json == body["machine_json"]


def test_research_rate_limit_returns_429_after_threshold(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_llm_configured(monkeypatch)
    _patch_graph(monkeypatch, {"errors": [], "studies": [_STUDY], "report": _REPORT})

    statuses = [
        client.post("/research", json={"question": "What treats keratoconus?"}).status_code
        for _ in range(11)
    ]

    assert statuses[:10] == [200] * 10
    assert statuses[10] == 429
