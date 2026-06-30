"""SQLAlchemy 2.0 ORM models for queries, studies and generated summaries."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, ForeignKey, Index, String, Text, func
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Postgres in production, plain JSON under SQLite (used by the test suite — see
# tests/conftest.py for why: SQLite has no JSONB type).
_JSONType = JSONB().with_variant(JSON(), "sqlite")


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def _uuid() -> str:
    return str(uuid.uuid4())


class ProjectRecord(Base):
    """A named container grouping scoped searches, uploaded papers and chat."""

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # cascade="all, delete-orphan" at the ORM level (not just the DB-level
    # ondelete="CASCADE" FKs below) so deleting a project cascades correctly
    # under SQLite too, which doesn't enforce FK-level cascade by default —
    # see tests/conftest.py's SQLite engine, which has no `PRAGMA foreign_keys`.
    queries: Mapped[list[QueryRecord]] = relationship(cascade="all, delete-orphan")
    documents: Mapped[list[ProjectDocumentRecord]] = relationship(cascade="all, delete-orphan")
    chat_messages: Mapped[list[ChatMessageRecord]] = relationship(cascade="all, delete-orphan")


class QueryRecord(Base):
    """A clinician question and its run metadata."""

    __tablename__ = "queries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    # NULL for normal/ungrouped search (unchanged, original behavior); set when the
    # search was run from inside a project (see ProjectRecord).
    project_id: Mapped[str | None] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=True, index=True
    )
    question: Mapped[str] = mapped_column(Text)
    filters: Mapped[dict[str, Any]] = mapped_column(_JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    studies: Mapped[list[StudyRecord]] = relationship(
        back_populates="query", cascade="all, delete-orphan"
    )
    summary: Mapped[SummaryRecord | None] = relationship(
        back_populates="query", cascade="all, delete-orphan", uselist=False
    )


class StudyRecord(Base):
    """A retrieved study's metadata, stored per query."""

    __tablename__ = "studies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    query_id: Mapped[str] = mapped_column(ForeignKey("queries.id", ondelete="CASCADE"))
    pmid: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(_JSONType)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    query: Mapped[QueryRecord] = relationship(back_populates="studies")


class SummaryRecord(Base):
    """The generated evidence report (human + machine forms) for a query."""

    __tablename__ = "summaries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    query_id: Mapped[str] = mapped_column(ForeignKey("queries.id", ondelete="CASCADE"))
    markdown: Mapped[str] = mapped_column(Text, default="")
    machine_json: Mapped[dict[str, Any]] = mapped_column(_JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    query: Mapped[QueryRecord] = relationship(back_populates="summary")


class ProjectDocumentRecord(Base):
    """One paper in a project's corpus — either a PubMed result or an uploaded PDF.

    A PubMed-sourced document is deduped per project via the partial unique
    index below (re-searching the same paper in a project must not re-embed it).
    """

    __tablename__ = "project_documents"
    __table_args__ = (
        Index(
            "ix_project_documents_project_pmid_unique",
            "project_id",
            "pmid",
            unique=True,
            postgresql_where=sql_text("pmid IS NOT NULL"),
            sqlite_where=sql_text("pmid IS NOT NULL"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    source: Mapped[str] = mapped_column(String(16))  # "pubmed" | "upload"
    pmid: Mapped[str | None] = mapped_column(String(32), nullable=True)
    title: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    chunks: Mapped[list[DocumentChunkRecord]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunkRecord(Base):
    """One embeddable unit of a project document's text, with its embedding vector."""

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("project_documents.id", ondelete="CASCADE"))
    chunk_index: Mapped[int]
    content: Mapped[str] = mapped_column(Text)
    # Stored as a plain JSON float array (not a Postgres `vector` column) so this
    # stays portable to SQLite in tests — see services/embeddings.py for why a
    # pure-Python similarity search is the right trade-off at this corpus scale.
    embedding: Mapped[list[float]] = mapped_column(_JSONType)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    document: Mapped[ProjectDocumentRecord] = relationship(back_populates="chunks")


class ChatMessageRecord(Base):
    """One turn of a project's persisted chat history."""

    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
