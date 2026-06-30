"""Integration tests for /projects endpoints: CRUD, scoped search, PDF
upload, and chat. The LangGraph pipeline, LLM and embeddings model are all
mocked — only the FastAPI routes + persistence layer (in-memory SQLite) are
exercised for real.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from medical_research_agent.models.report import EvidenceReport
from medical_research_agent.models.study import ExtractedStudy, Study
from medical_research_agent.services import pdf_extraction
from tests.test_api_research import _patch_graph, _patch_llm_configured

_STUDY = Study(
    pmid="90000001",
    title="Synthetic Study of Keratoconus Treatment",
    authors=["Testauthor J"],
    journal="Journal of Synthetic Testing",
    publication_year=2023,
    abstract="Synthetic abstract.",
)
_REPORT = EvidenceReport(
    question="What treats keratoconus?",
    markdown="# What treats keratoconus?\n\nSynthetic report content.",
    studies=[_STUDY],
    extracted=[ExtractedStudy(pmid="90000001", main_findings="Synthetic improvement observed.")],
)


def _create_project(client: TestClient, name: str = "Keratoconus review") -> dict[str, Any]:
    response = client.post("/projects", json={"name": name})
    assert response.status_code == 200
    return response.json()


# --- CRUD --------------------------------------------------------------


def test_create_and_get_project(client: TestClient) -> None:
    project = _create_project(client)

    response = client.get(f"/projects/{project['id']}")

    assert response.status_code == 200
    body = response.json()
    assert body["project"]["name"] == "Keratoconus review"
    assert body["history"] == []
    assert body["documents"] == []


def test_create_project_name_too_long_returns_422(client: TestClient) -> None:
    response = client.post("/projects", json={"name": "x" * 201})

    assert response.status_code == 422


def test_create_project_empty_name_returns_422(client: TestClient) -> None:
    response = client.post("/projects", json={"name": ""})

    assert response.status_code == 422


def test_list_projects(client: TestClient) -> None:
    assert client.get("/projects").json() == []

    _create_project(client, "A")
    _create_project(client, "B")

    names = {p["name"] for p in client.get("/projects").json()}
    assert names == {"A", "B"}


def test_get_project_404_for_unknown_id(client: TestClient) -> None:
    response = client.get("/projects/does-not-exist")

    assert response.status_code == 404


def test_delete_project(client: TestClient) -> None:
    project = _create_project(client)

    delete_response = client.delete(f"/projects/{project['id']}")
    assert delete_response.status_code == 204

    assert client.get(f"/projects/{project['id']}").status_code == 404


def test_delete_project_404_for_unknown_id(client: TestClient) -> None:
    response = client.delete("/projects/does-not-exist")

    assert response.status_code == 404


# --- Scoped research -----------------------------------------------------


def test_research_in_project_persists_scoped_and_embeds_papers(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_embeddings_model
) -> None:
    _patch_llm_configured(monkeypatch)
    _patch_graph(monkeypatch, {"errors": [], "studies": [_STUDY], "report": _REPORT})
    fake_embeddings_model("medical_research_agent.services.embeddings")
    project = _create_project(client)

    response = client.post(
        f"/projects/{project['id']}/research", json={"question": "What treats keratoconus?"}
    )

    assert response.status_code == 200
    assert response.json()["warnings"] == []

    detail = client.get(f"/projects/{project['id']}").json()
    assert len(detail["history"]) == 1
    assert detail["history"][0]["question"] == "What treats keratoconus?"
    assert [s["pmid"] for s in detail["history"][0]["studies"]] == ["90000001"]
    assert len(detail["documents"]) == 1
    assert detail["documents"][0]["pmid"] == "90000001"
    assert detail["documents"][0]["source"] == "pubmed"


def test_research_in_project_unknown_project_returns_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_llm_configured(monkeypatch)
    _patch_graph(monkeypatch, {"errors": [], "studies": [_STUDY], "report": _REPORT})

    response = client.post(
        "/projects/does-not-exist/research", json={"question": "What treats keratoconus?"}
    )

    assert response.status_code == 404


def test_research_in_project_does_not_appear_in_normal_search_history(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_embeddings_model
) -> None:
    """Normal (ungrouped) /research must stay completely unaffected by Projects."""
    _patch_llm_configured(monkeypatch)
    _patch_graph(monkeypatch, {"errors": [], "studies": [_STUDY], "report": _REPORT})
    fake_embeddings_model("medical_research_agent.services.embeddings")
    project = _create_project(client)

    normal_response = client.post("/research", json={"question": "What treats keratoconus?"})
    assert normal_response.status_code == 200

    detail = client.get(f"/projects/{project['id']}").json()
    assert detail["history"] == []


def test_research_in_project_embedding_failure_is_non_fatal(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_llm_configured(monkeypatch)
    _patch_graph(monkeypatch, {"errors": [], "studies": [_STUDY], "report": _REPORT})

    async def _failing_embed_texts(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("no embedding key configured")

    monkeypatch.setattr(
        "medical_research_agent.api.routes.projects.embed_texts", _failing_embed_texts
    )
    project = _create_project(client)

    response = client.post(
        f"/projects/{project['id']}/research", json={"question": "What treats keratoconus?"}
    )

    assert response.status_code == 200
    assert any("embedding" in w.lower() for w in response.json()["warnings"])


# --- PDF upload ------------------------------------------------------------


def _patch_pdf_reader(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    _patch_pdf_reader_pages(monkeypatch, [text])


def _patch_pdf_reader_pages(monkeypatch: pytest.MonkeyPatch, pages_text: list[str]) -> None:
    class _FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class _FakeReader:
        pages = [_FakePage(t) for t in pages_text]

    monkeypatch.setattr(pdf_extraction, "PdfReader", lambda _file: _FakeReader())


def test_upload_document_happy_path(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_embeddings_model
) -> None:
    _patch_pdf_reader(monkeypatch, "This uploaded paper studies corneal transplantation.")
    fake_embeddings_model("medical_research_agent.services.embeddings")
    project = _create_project(client)

    response = client.post(
        f"/projects/{project['id']}/documents",
        files={"file": ("paper.pdf", b"%PDF-1.4 fake bytes", "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "upload"
    assert body["pmid"] is None
    assert body["title"] == "paper.pdf"

    detail = client.get(f"/projects/{project['id']}").json()
    assert len(detail["documents"]) == 1
    assert detail["documents"][0]["source"] == "upload"


def test_upload_document_rejects_non_pdf_content_type(client: TestClient) -> None:
    project = _create_project(client)

    response = client.post(
        f"/projects/{project['id']}/documents",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )

    assert response.status_code == 422


def test_upload_document_rejects_oversized_file(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("medical_research_agent.api.routes.projects.MAX_UPLOAD_BYTES", 10)
    project = _create_project(client)

    response = client.post(
        f"/projects/{project['id']}/documents",
        files={"file": ("paper.pdf", b"x" * 100, "application/pdf")},
    )

    assert response.status_code == 413


def test_upload_document_unknown_project_returns_404(client: TestClient) -> None:
    response = client.post(
        "/projects/does-not-exist/documents",
        files={"file": ("paper.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 404


def test_upload_document_unparseable_pdf_returns_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise(_file: Any) -> None:
        raise ValueError("corrupt")

    monkeypatch.setattr(pdf_extraction, "PdfReader", _raise)
    project = _create_project(client)

    response = client.post(
        f"/projects/{project['id']}/documents",
        files={"file": ("paper.pdf", b"not a real pdf", "application/pdf")},
    )

    assert response.status_code == 422


def test_upload_document_rejects_too_many_pages(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercises pdf_extraction's MAX_PAGES bound through the actual HTTP route,
    not just the service-level unit test.
    """
    too_many_pages = ["some text"] * (pdf_extraction.MAX_PAGES + 1)
    _patch_pdf_reader_pages(monkeypatch, too_many_pages)
    project = _create_project(client)

    response = client.post(
        f"/projects/{project['id']}/documents",
        files={"file": ("paper.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 422
    assert "page limit" in response.json()["detail"]


def test_upload_document_rejects_when_exceeding_max_chars(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercises pdf_extraction's MAX_CHARS bound (now a hard reject, not a
    silent truncation) through the actual HTTP route.
    """
    huge_page = "x" * (pdf_extraction.MAX_CHARS + 1000)
    _patch_pdf_reader_pages(monkeypatch, [huge_page])
    project = _create_project(client)

    response = client.post(
        f"/projects/{project['id']}/documents",
        files={"file": ("paper.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 422
    assert "character limit" in response.json()["detail"]


def test_upload_document_rejects_empty_pdf(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PDF with no extractable text (e.g. scanned/image-only), through the
    actual HTTP route.
    """
    _patch_pdf_reader_pages(monkeypatch, ["", "   ", ""])
    project = _create_project(client)

    response = client.post(
        f"/projects/{project['id']}/documents",
        files={"file": ("paper.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 422
    assert "No extractable text" in response.json()["detail"]


def test_upload_document_rejects_overlong_filename(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_pdf_reader(monkeypatch, "Some real text.")
    project = _create_project(client)

    overlong_name = "x" * 256 + ".pdf"
    response = client.post(
        f"/projects/{project['id']}/documents",
        files={"file": (overlong_name, b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 422


def test_upload_document_rejects_when_project_at_document_cap(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_embeddings_model
) -> None:
    monkeypatch.setattr("medical_research_agent.api.routes.projects.MAX_DOCUMENTS_PER_PROJECT", 1)
    fake_embeddings_model("medical_research_agent.services.embeddings")
    _patch_pdf_reader(monkeypatch, "First paper text.")
    project = _create_project(client)

    first = client.post(
        f"/projects/{project['id']}/documents",
        files={"file": ("first.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert first.status_code == 200

    second = client.post(
        f"/projects/{project['id']}/documents",
        files={"file": ("second.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert second.status_code == 422
    assert "maximum" in second.json()["detail"]


def test_research_in_project_skips_embedding_beyond_document_cap(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_embeddings_model
) -> None:
    _patch_llm_configured(monkeypatch)
    monkeypatch.setattr("medical_research_agent.api.routes.projects.MAX_DOCUMENTS_PER_PROJECT", 0)
    fake_embeddings_model("medical_research_agent.services.embeddings")
    _patch_graph(monkeypatch, {"errors": [], "studies": [_STUDY], "report": _REPORT})
    project = _create_project(client)

    response = client.post(
        f"/projects/{project['id']}/research", json={"question": "What treats keratoconus?"}
    )

    assert response.status_code == 200
    assert any("paper cap" in w for w in response.json()["warnings"])
    detail = client.get(f"/projects/{project['id']}").json()
    assert detail["documents"] == []


# --- Chat --------------------------------------------------------------


def test_chat_happy_path_persists_history(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_chat_model, fake_embeddings_model
) -> None:
    _patch_pdf_reader(monkeypatch, "This paper studies corneal crosslinking outcomes.")
    fake_embeddings_model("medical_research_agent.services.embeddings", query_vector=[1.0, 0.0])
    project = _create_project(client)
    upload_response = client.post(
        f"/projects/{project['id']}/documents",
        files={"file": ("paper.pdf", b"%PDF-1.4", "application/pdf")},
    )
    doc_id = upload_response.json()["id"]

    fake_chat_model(
        "medical_research_agent.services.project_chat",
        f"Crosslinking improves outcomes [DOC: {doc_id}].",
    )

    response = client.post(f"/projects/{project['id']}/chat", json={"message": "Does it work?"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == f"Crosslinking improves outcomes [DOC: {doc_id}]."
    assert body["cited_document_ids"] == [doc_id]

    history = client.get(f"/projects/{project['id']}/chat").json()
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[0]["content"] == "Does it work?"


def test_chat_on_empty_project_does_not_call_the_llm(client: TestClient, fake_chat_model) -> None:
    fake_chat_model("medical_research_agent.services.project_chat")  # no results queued
    project = _create_project(client)

    response = client.post(f"/projects/{project['id']}/chat", json={"message": "Anything?"})

    assert response.status_code == 200
    assert "doesn't have any papers yet" in response.json()["reply"]


def test_chat_unknown_project_returns_404(client: TestClient) -> None:
    response = client.post("/projects/does-not-exist/chat", json={"message": "Hello?"})

    assert response.status_code == 404


def test_chat_message_too_long_returns_422(client: TestClient) -> None:
    project = _create_project(client)

    response = client.post(f"/projects/{project['id']}/chat", json={"message": "x" * 1001})

    assert response.status_code == 422


def test_chat_empty_message_returns_422(client: TestClient) -> None:
    project = _create_project(client)

    response = client.post(f"/projects/{project['id']}/chat", json={"message": ""})

    assert response.status_code == 422


def test_get_chat_history_unknown_project_returns_404(client: TestClient) -> None:
    response = client.get("/projects/does-not-exist/chat")

    assert response.status_code == 404


def test_get_chat_history_empty_for_new_project(client: TestClient) -> None:
    project = _create_project(client)

    assert client.get(f"/projects/{project['id']}/chat").json() == []


# --- Rate limiting -------------------------------------------------------


def test_create_project_rate_limit_returns_429_after_threshold(client: TestClient) -> None:
    statuses = [client.post("/projects", json={"name": "P"}).status_code for _ in range(11)]

    assert statuses[:10] == [200] * 10
    assert statuses[10] == 429
