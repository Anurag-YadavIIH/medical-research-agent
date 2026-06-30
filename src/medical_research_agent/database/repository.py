"""Data-access helpers, isolating ORM details from the API/agents."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from medical_research_agent.database.models import QueryRecord, StudyRecord, SummaryRecord
from medical_research_agent.models.query import SearchFilters
from medical_research_agent.models.report import EvidenceReport


class ResearchRepository:
    """Persistence operations for a research run."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save_research_run(
        self,
        question: str,
        filters: SearchFilters,
        report: EvidenceReport,
        project_id: str | None = None,
    ) -> QueryRecord:
        """Persist the query, its studies and the generated summary in one transaction.

        Rolls back the whole run if any part fails, so a partially-written query
        never ends up in the database. ``project_id`` is None for normal,
        ungrouped search (unchanged default) and set when run from inside a
        Project.
        """
        try:
            query = QueryRecord(
                question=question,
                filters=filters.model_dump(mode="json"),
                project_id=project_id,
            )
            self.session.add(query)
            await self.session.flush()  # assigns query.id for the foreign keys below

            for study in report.studies:
                self.session.add(
                    StudyRecord(
                        query_id=query.id,
                        pmid=study.pmid,
                        payload=study.model_dump(mode="json"),
                    )
                )
            self.session.add(
                SummaryRecord(
                    query_id=query.id,
                    markdown=report.markdown,
                    machine_json=report.to_machine_json(),
                )
            )
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        return query

    async def get_query(self, query_id: str) -> QueryRecord | None:
        return await self.session.get(QueryRecord, query_id)

    async def get_studies(self, query_id: str) -> list[StudyRecord]:
        result = await self.session.execute(
            select(StudyRecord).where(StudyRecord.query_id == query_id)
        )
        return list(result.scalars().all())

    async def get_summary(self, query_id: str) -> SummaryRecord | None:
        result = await self.session.execute(
            select(SummaryRecord).where(SummaryRecord.query_id == query_id)
        )
        return result.scalar_one_or_none()
