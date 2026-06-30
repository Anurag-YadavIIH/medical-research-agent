"""Data-access helpers for Projects: scoped search history, uploaded paper
documents/chunks, and chat history. Mirrors ResearchRepository's style.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from medical_research_agent.database.models import (
    ChatMessageRecord,
    DocumentChunkRecord,
    ProjectDocumentRecord,
    ProjectRecord,
    QueryRecord,
)


class ProjectRepository:
    """Persistence operations for projects, their documents/chunks, and chat."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # --- Projects ------------------------------------------------------

    async def create_project(self, name: str) -> ProjectRecord:
        project = ProjectRecord(name=name)
        self.session.add(project)
        await self.session.commit()
        await self.session.refresh(project)
        return project

    async def list_projects(self) -> list[ProjectRecord]:
        result = await self.session.execute(
            select(ProjectRecord).order_by(ProjectRecord.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_project(self, project_id: str) -> ProjectRecord | None:
        return await self.session.get(ProjectRecord, project_id)

    async def delete_project(self, project_id: str) -> bool:
        project = await self.get_project(project_id)
        if project is None:
            return False
        await self.session.delete(project)
        await self.session.commit()
        return True

    # --- Scoped search history ------------------------------------------

    async def list_project_history(self, project_id: str) -> list[QueryRecord]:
        """Past searches run inside this project, with studies/summary eager-loaded."""
        result = await self.session.execute(
            select(QueryRecord)
            .where(QueryRecord.project_id == project_id)
            .options(selectinload(QueryRecord.studies), selectinload(QueryRecord.summary))
            .order_by(QueryRecord.created_at.desc())
        )
        return list(result.scalars().all())

    # --- Documents / chunks (the RAG corpus) ----------------------------

    async def get_existing_pmids(self, project_id: str) -> set[str]:
        """PMIDs already embedded in this project, so re-searching a paper is a no-op."""
        result = await self.session.execute(
            select(ProjectDocumentRecord.pmid).where(
                ProjectDocumentRecord.project_id == project_id,
                ProjectDocumentRecord.pmid.isnot(None),
            )
        )
        return {pmid for pmid in result.scalars().all() if pmid is not None}

    async def add_pubmed_document(
        self,
        project_id: str,
        pmid: str,
        title: str,
        chunk_content: str,
        embedding: list[float],
    ) -> ProjectDocumentRecord:
        """Add a single-chunk document for a newly-seen PubMed paper.

        Callers must check ``get_existing_pmids`` first — this does not itself
        dedupe, to avoid a redundant query per paper in a batch.
        """
        document = ProjectDocumentRecord(
            project_id=project_id, source="pubmed", pmid=pmid, title=title
        )
        self.session.add(document)
        await self.session.flush()
        self.session.add(
            DocumentChunkRecord(
                document_id=document.id, chunk_index=0, content=chunk_content, embedding=embedding
            )
        )
        await self.session.commit()
        return document

    async def add_uploaded_document(
        self, project_id: str, title: str, chunks: list[str], embeddings: list[list[float]]
    ) -> ProjectDocumentRecord:
        document = ProjectDocumentRecord(
            project_id=project_id, source="upload", pmid=None, title=title
        )
        self.session.add(document)
        await self.session.flush()
        for index, (content, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            self.session.add(
                DocumentChunkRecord(
                    document_id=document.id,
                    chunk_index=index,
                    content=content,
                    embedding=embedding,
                )
            )
        await self.session.commit()
        return document

    async def list_documents(self, project_id: str) -> list[ProjectDocumentRecord]:
        result = await self.session.execute(
            select(ProjectDocumentRecord)
            .where(ProjectDocumentRecord.project_id == project_id)
            .order_by(ProjectDocumentRecord.created_at.desc())
        )
        return list(result.scalars().all())

    async def count_documents(self, project_id: str) -> int:
        result = await self.session.execute(
            select(func.count())
            .select_from(ProjectDocumentRecord)
            .where(ProjectDocumentRecord.project_id == project_id)
        )
        return result.scalar_one()

    async def get_project_chunks(self, project_id: str) -> list[tuple[str, str, list[float]]]:
        """All ``(document_id, content, embedding)`` triples for retrieval ranking."""
        result = await self.session.execute(
            select(
                DocumentChunkRecord.document_id,
                DocumentChunkRecord.content,
                DocumentChunkRecord.embedding,
            )
            .join(
                ProjectDocumentRecord,
                DocumentChunkRecord.document_id == ProjectDocumentRecord.id,
            )
            .where(ProjectDocumentRecord.project_id == project_id)
        )
        return [(doc_id, content, embedding) for doc_id, content, embedding in result.all()]

    # --- Chat ------------------------------------------------------------

    async def save_chat_message(
        self, project_id: str, role: str, content: str
    ) -> ChatMessageRecord:
        message = ChatMessageRecord(project_id=project_id, role=role, content=content)
        self.session.add(message)
        await self.session.commit()
        await self.session.refresh(message)
        return message

    async def get_chat_history(self, project_id: str) -> list[ChatMessageRecord]:
        result = await self.session.execute(
            select(ChatMessageRecord)
            .where(ChatMessageRecord.project_id == project_id)
            .order_by(ChatMessageRecord.created_at.asc())
        )
        return list(result.scalars().all())
