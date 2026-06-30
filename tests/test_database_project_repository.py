"""Direct tests of ProjectRepository: CRUD, scoped history, document/chunk
dedup, and chat persistence.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from medical_research_agent.database.project_repository import ProjectRepository
from medical_research_agent.database.repository import ResearchRepository
from medical_research_agent.models.query import SearchFilters
from medical_research_agent.models.report import EvidenceReport
from medical_research_agent.models.study import Study


async def test_create_and_get_project(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)

    project = await repo.create_project("Keratoconus review")

    assert project.id
    fetched = await repo.get_project(project.id)
    assert fetched is not None
    assert fetched.name == "Keratoconus review"


async def test_get_project_returns_none_for_unknown_id(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)

    assert await repo.get_project("does-not-exist") is None


async def test_list_projects_returns_all_created_projects(db_session: AsyncSession) -> None:
    # Not asserting exact order here: SQLite's CURRENT_TIMESTAMP is
    # second-granularity, so two rapid inserts in a test can tie — ordering
    # itself (DESC by created_at) is the production concern, covered by the
    # query shape, not by racing the clock in a unit test.
    repo = ProjectRepository(db_session)
    first = await repo.create_project("First")
    second = await repo.create_project("Second")

    projects = await repo.list_projects()

    assert {p.id for p in projects} == {first.id, second.id}


async def test_delete_project_removes_it(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    project = await repo.create_project("Temp")

    assert await repo.delete_project(project.id) is True
    assert await repo.get_project(project.id) is None


async def test_delete_project_returns_false_for_unknown_id(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)

    assert await repo.delete_project("does-not-exist") is False


async def test_delete_project_cascades_to_documents_and_chat(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    project = await repo.create_project("Temp")
    await repo.add_pubmed_document(project.id, "111", "A study", "abstract text", [0.1, 0.2])
    await repo.save_chat_message(project.id, "user", "hello")

    await repo.delete_project(project.id)

    assert await repo.get_project_chunks(project.id) == []
    assert await repo.get_chat_history(project.id) == []


async def test_list_project_history_only_returns_scoped_queries(db_session: AsyncSession) -> None:
    project_repo = ProjectRepository(db_session)
    research_repo = ResearchRepository(db_session)
    project = await project_repo.create_project("Keratoconus review")

    report = EvidenceReport(
        question="Q1", markdown="# Q1", studies=[Study(pmid="111", title="In project")]
    )
    await research_repo.save_research_run("Q1", SearchFilters(), report, project_id=project.id)
    # A normal, ungrouped search (no project_id) must not show up in the project's history.
    await research_repo.save_research_run(
        "Q2", SearchFilters(), EvidenceReport(question="Q2", markdown="# Q2")
    )

    history = await project_repo.list_project_history(project.id)

    assert len(history) == 1
    assert history[0].question == "Q1"
    assert [s.pmid for s in history[0].studies] == ["111"]
    assert history[0].summary is not None


async def test_add_pubmed_document_and_get_existing_pmids(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    project = await repo.create_project("P")

    assert await repo.get_existing_pmids(project.id) == set()

    await repo.add_pubmed_document(project.id, "111", "A study", "abstract text", [0.1, 0.2])

    assert await repo.get_existing_pmids(project.id) == {"111"}
    chunks = await repo.get_project_chunks(project.id)
    assert len(chunks) == 1
    assert chunks[0][1] == "abstract text"
    assert chunks[0][2] == [0.1, 0.2]


async def test_count_documents(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    project = await repo.create_project("P")

    assert await repo.count_documents(project.id) == 0

    await repo.add_pubmed_document(project.id, "111", "A study", "text a", [0.1])
    await repo.add_pubmed_document(project.id, "222", "B study", "text b", [0.2])

    assert await repo.count_documents(project.id) == 2


async def test_count_documents_is_scoped_per_project(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    project_a = await repo.create_project("A")
    project_b = await repo.create_project("B")
    await repo.add_pubmed_document(project_a.id, "111", "Study A", "text a", [0.1])

    assert await repo.count_documents(project_a.id) == 1
    assert await repo.count_documents(project_b.id) == 0


async def test_add_uploaded_document_creates_one_chunk_per_text(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    project = await repo.create_project("P")

    await repo.add_uploaded_document(
        project.id,
        "uploaded.pdf",
        chunks=["chunk one", "chunk two"],
        embeddings=[[0.1, 0.1], [0.2, 0.2]],
    )

    chunks = await repo.get_project_chunks(project.id)
    assert sorted(c[1] for c in chunks) == ["chunk one", "chunk two"]

    documents = await repo.list_documents(project.id)
    assert len(documents) == 1
    assert documents[0].source == "upload"
    assert documents[0].pmid is None


async def test_get_project_chunks_is_scoped_per_project(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    project_a = await repo.create_project("A")
    project_b = await repo.create_project("B")
    await repo.add_pubmed_document(project_a.id, "111", "Study A", "text a", [0.1])
    await repo.add_pubmed_document(project_b.id, "222", "Study B", "text b", [0.2])

    chunks_a = await repo.get_project_chunks(project_a.id)

    assert len(chunks_a) == 1
    assert chunks_a[0][1] == "text a"


async def test_save_and_get_chat_history_in_order(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    project = await repo.create_project("P")

    await repo.save_chat_message(project.id, "user", "What's the strongest evidence?")
    await repo.save_chat_message(project.id, "assistant", "Level I, per [DOC: 111].")

    history = await repo.get_chat_history(project.id)

    assert [(m.role, m.content) for m in history] == [
        ("user", "What's the strongest evidence?"),
        ("assistant", "Level I, per [DOC: 111]."),
    ]
