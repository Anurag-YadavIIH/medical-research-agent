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

    async def create_query(self, question: str, filters: SearchFilters) -> QueryRecord:
        record = QueryRecord(question=question, filters=filters.model_dump())
        self.session.add(record)
        await self.session.flush()
        return record

    async def save_report(self, query_id: str, report: EvidenceReport) -> None:
        for study in report.studies:
            self.session.add(
                StudyRecord(query_id=query_id, pmid=study.pmid, payload=study.model_dump())
            )
        self.session.add(
            SummaryRecord(
                query_id=query_id,
                markdown=report.markdown,
                machine_json=report.to_machine_json(),
            )
        )
        await self.session.commit()

    async def get_studies(self, query_id: str) -> list[StudyRecord]:
        result = await self.session.execute(
            select(StudyRecord).where(StudyRecord.query_id == query_id)
        )
        return list(result.scalars().all())
