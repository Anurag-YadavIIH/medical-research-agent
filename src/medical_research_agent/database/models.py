"""SQLAlchemy 2.0 ORM models for queries, studies and generated summaries."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def _uuid() -> str:
    return str(uuid.uuid4())


class QueryRecord(Base):
    """A clinician question and its run metadata."""

    __tablename__ = "queries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    question: Mapped[str] = mapped_column(Text)
    filters: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
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
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    query: Mapped[QueryRecord] = relationship(back_populates="studies")


class SummaryRecord(Base):
    """The generated evidence report (human + machine forms) for a query."""

    __tablename__ = "summaries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    query_id: Mapped[str] = mapped_column(ForeignKey("queries.id", ondelete="CASCADE"))
    markdown: Mapped[str] = mapped_column(Text, default="")
    machine_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    query: Mapped[QueryRecord] = relationship(back_populates="summary")
