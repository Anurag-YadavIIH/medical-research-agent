"""Direct tests of ResearchRepository — the persistence transaction and
rollback behavior in particular, which the API-level tests never force a
failure path for.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from medical_research_agent.database.models import QueryRecord
from medical_research_agent.database.repository import ResearchRepository
from medical_research_agent.models.query import SearchFilters
from medical_research_agent.models.report import EvidenceReport
from medical_research_agent.models.study import Study


def _report(**overrides: object) -> EvidenceReport:
    defaults: dict[str, object] = {
        "question": "What treats keratoconus?",
        "markdown": "# What treats keratoconus?",
        "studies": [Study(pmid="90000001", title="Synthetic Study")],
    }
    defaults.update(overrides)
    return EvidenceReport(**defaults)  # type: ignore[arg-type]


async def test_save_research_run_persists_query_studies_and_summary(
    db_session: AsyncSession,
) -> None:
    repo = ResearchRepository(db_session)
    report = _report()

    query = await repo.save_research_run("What treats keratoconus?", SearchFilters(), report)

    assert query.id
    studies = await repo.get_studies(query.id)
    assert [s.pmid for s in studies] == ["90000001"]

    summary = await repo.get_summary(query.id)
    assert summary is not None
    assert summary.markdown == report.markdown
    assert summary.machine_json == report.to_machine_json()


async def test_save_research_run_rolls_back_on_failure(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = ResearchRepository(db_session)
    report = _report()

    rollback_called = False
    original_rollback = db_session.rollback

    async def _tracking_rollback() -> None:
        nonlocal rollback_called
        rollback_called = True
        await original_rollback()

    async def _failing_commit() -> None:
        raise RuntimeError("simulated DB failure")

    monkeypatch.setattr(db_session, "commit", _failing_commit)
    monkeypatch.setattr(db_session, "rollback", _tracking_rollback)

    with pytest.raises(RuntimeError, match="simulated DB failure"):
        await repo.save_research_run("What treats keratoconus?", SearchFilters(), report)

    assert rollback_called

    # Nothing from the failed transaction should have survived the rollback.
    monkeypatch.undo()
    result = await db_session.execute(select(QueryRecord))
    assert result.scalars().all() == []


async def test_get_query_returns_none_for_unknown_id(db_session: AsyncSession) -> None:
    repo = ResearchRepository(db_session)

    assert await repo.get_query("does-not-exist") is None


async def test_get_studies_returns_empty_list_for_unknown_query_id(
    db_session: AsyncSession,
) -> None:
    repo = ResearchRepository(db_session)

    assert await repo.get_studies("does-not-exist") == []


async def test_get_summary_returns_none_for_unknown_query_id(db_session: AsyncSession) -> None:
    repo = ResearchRepository(db_session)

    assert await repo.get_summary("does-not-exist") is None
