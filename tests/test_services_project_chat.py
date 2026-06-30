"""Tests for the project chat service: retrieval grounding, citation
discipline, and the empty-project short-circuit. LLM/embeddings are faked —
no live API calls.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from medical_research_agent.database.project_repository import ProjectRepository
from medical_research_agent.services.project_chat import (
    _NO_PAPERS_REPLY,
    _NOT_RELEVANT_REPLY,
    answer_project_question,
)


async def test_empty_project_short_circuits_without_calling_the_llm(
    db_session: AsyncSession, fake_chat_model, fake_embeddings_model
) -> None:
    fake_chat_model("medical_research_agent.services.project_chat")  # no results queued
    fake_embeddings_model("medical_research_agent.services.embeddings")

    repo = ProjectRepository(db_session)
    project = await repo.create_project("Empty")

    answer = await answer_project_question(repo, project.id, "What's the evidence?")

    assert answer.reply == _NO_PAPERS_REPLY
    assert answer.cited_document_ids == []


async def test_answer_cites_only_retrieved_documents(
    db_session: AsyncSession, fake_chat_model, fake_embeddings_model
) -> None:
    repo = ProjectRepository(db_session)
    project = await repo.create_project("P")
    doc = await repo.add_pubmed_document(
        project.id, "111", "A study", "Crosslinking halted progression.", [1.0, 0.0]
    )

    fake_embeddings_model("medical_research_agent.services.embeddings", query_vector=[1.0, 0.0])
    fake_chat_model(
        "medical_research_agent.services.project_chat",
        f"Crosslinking is effective [DOC: {doc.id}].",
    )

    answer = await answer_project_question(repo, project.id, "Does crosslinking work?")

    assert answer.reply == f"Crosslinking is effective [DOC: {doc.id}]."
    assert answer.cited_document_ids == [doc.id]


async def test_fabricated_document_citation_is_stripped(
    db_session: AsyncSession, fake_chat_model, fake_embeddings_model
) -> None:
    repo = ProjectRepository(db_session)
    project = await repo.create_project("P")
    await repo.add_pubmed_document(project.id, "111", "A study", "Some findings.", [1.0, 0.0])

    fake_embeddings_model("medical_research_agent.services.embeddings", query_vector=[1.0, 0.0])
    fake_chat_model(
        "medical_research_agent.services.project_chat",
        "This is supported [DOC: nonexistent-id-999].",
    )

    answer = await answer_project_question(repo, project.id, "Question?")

    assert "nonexistent-id-999" not in answer.reply
    assert answer.cited_document_ids == []


async def test_llm_failure_propagates_to_the_caller(
    db_session: AsyncSession, fake_chat_model, fake_embeddings_model
) -> None:
    repo = ProjectRepository(db_session)
    project = await repo.create_project("P")
    await repo.add_pubmed_document(project.id, "111", "A study", "Some findings.", [1.0, 0.0])

    fake_embeddings_model("medical_research_agent.services.embeddings", query_vector=[1.0, 0.0])
    fake_chat_model(
        "medical_research_agent.services.project_chat", RuntimeError("provider exploded")
    )

    with pytest.raises(RuntimeError, match="provider exploded"):
        await answer_project_question(repo, project.id, "Question?")


async def test_irrelevant_question_never_calls_the_llm(
    db_session: AsyncSession, fake_chat_model, fake_embeddings_model
) -> None:
    """Below MIN_RELEVANCE_SCORE, the chat model must not be invoked at all —
    not just "the response happens to decline" but a hard skip, asserted via
    the fake's call count rather than inferring it from the reply text.
    """
    repo = ProjectRepository(db_session)
    project = await repo.create_project("P")
    await repo.add_pubmed_document(
        project.id, "111", "A study", "Crosslinking findings.", [1.0, 0.0]
    )

    # Orthogonal query vector -> cosine similarity 0.0, well below the 0.3 floor.
    fake_embeddings_model("medical_research_agent.services.embeddings", query_vector=[0.0, 1.0])
    fake_model = fake_chat_model("medical_research_agent.services.project_chat")  # no results

    answer = await answer_project_question(repo, project.id, "Unrelated question")

    assert answer.reply == _NOT_RELEVANT_REPLY
    assert answer.cited_document_ids == []
    assert fake_model.call_count == 0


async def test_unretrieved_real_document_cannot_be_cited(
    db_session: AsyncSession, fake_chat_model, fake_embeddings_model
) -> None:
    """Adversarial case: the model tries to cite a document that genuinely
    exists in this project, but wasn't among the top-k chunks actually
    retrieved/shown to it for THIS question. The guarantee is per-turn, not
    "anything in the project is fair game" — citing an un-retrieved id is
    fabrication just as much as citing an id from nowhere.
    """
    repo = ProjectRepository(db_session)
    project = await repo.create_project("P")

    # 9 documents, embeddings spread so the query (orthogonal-ish to the last
    # one) ranks doc_8 last — DEFAULT_TOP_K is 8, so doc_8 is never retrieved.
    docs = []
    for i in range(9):
        # Embeddings biased toward [1, 0] except the last, which points away.
        vec = [1.0, 0.0] if i < 8 else [0.0, 1.0]
        doc = await repo.add_pubmed_document(
            project.id, f"pmid-{i}", f"Study {i}", f"Findings {i}.", vec
        )
        docs.append(doc)
    unretrieved_doc_id = docs[8].id

    fake_embeddings_model("medical_research_agent.services.embeddings", query_vector=[1.0, 0.0])
    # The model hallucinates a citation to the one doc that wasn't retrieved.
    fake_chat_model(
        "medical_research_agent.services.project_chat",
        f"This is well established [DOC: {unretrieved_doc_id}].",
    )

    answer = await answer_project_question(repo, project.id, "Off-topic or unsupported claim?")

    assert unretrieved_doc_id not in answer.reply
    assert unretrieved_doc_id not in answer.cited_document_ids
    assert answer.cited_document_ids == []


async def test_chat_never_retrieves_or_cites_another_projects_documents(
    db_session: AsyncSession, fake_chat_model, fake_embeddings_model
) -> None:
    """Cross-project isolation: project B's content must never be visible to
    project A's chat, even when B's content would embed as a closer match to
    the query than anything actually in A.
    """
    repo = ProjectRepository(db_session)
    project_a = await repo.create_project("A")
    project_b = await repo.create_project("B")

    # B's doc is a near-perfect match for the query vector; A's is a weaker
    # match (0.6 similarity) — still above MIN_RELEVANCE_SCORE (0.3) so this
    # test exercises the LLM-call path, not the relevance short-circuit.
    doc_a = await repo.add_pubmed_document(
        project_a.id, "aaa", "Study A", "Loosely related findings.", [0.6, 0.8]
    )
    doc_b = await repo.add_pubmed_document(
        project_b.id, "bbb", "Study B", "Highly relevant findings.", [1.0, 0.0]
    )

    fake_embeddings_model("medical_research_agent.services.embeddings", query_vector=[1.0, 0.0])
    fake_chat_model(
        "medical_research_agent.services.project_chat",
        f"Based on the excerpts [DOC: {doc_a.id}].",
    )

    answer = await answer_project_question(repo, project_a.id, "Relevant question")

    # Even though doc_b is the better embedding match, it's project B's — the
    # retrieval query itself is scoped by project_id (get_project_chunks), so
    # project A's chat can never see, retrieve, or cite it.
    assert doc_b.id not in answer.reply
    assert doc_b.id not in answer.cited_document_ids
    chunks_visible_to_a = await repo.get_project_chunks(project_a.id)
    assert all(doc_id != doc_b.id for doc_id, _content, _embedding in chunks_visible_to_a)
