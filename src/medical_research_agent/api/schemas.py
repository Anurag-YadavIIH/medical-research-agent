"""Request/response schemas for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from medical_research_agent.models.query import SearchFilters

DISCLAIMER = (
    "For research and educational purposes only. Clinical decisions should rely "
    "on professional judgment and full-text evidence review."
)


class ResearchRequest(BaseModel):
    """Body for POST /research."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=3, max_length=500, examples=["What are recent treatments for keratoconus?"]
    )
    filters: SearchFilters = Field(default_factory=SearchFilters)


class ResearchResponse(BaseModel):
    """Response for POST /research — human report + machine JSON."""

    query_id: str
    question: str
    report_markdown: str
    machine_json: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    version: str
    checks: dict[str, str]


class CreateProjectRequest(BaseModel):
    """Body for POST /projects."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)


class ProjectResponse(BaseModel):
    """A project, as returned by the projects list/detail/create endpoints."""

    id: str
    name: str
    created_at: datetime


class ProjectDocumentResponse(BaseModel):
    """One paper in a project's RAG corpus — PubMed result or uploaded PDF."""

    id: str
    source: str
    pmid: str | None
    title: str
    created_at: datetime


class ProjectHistoryItem(BaseModel):
    """One past scoped search run inside a project."""

    query_id: str
    question: str
    created_at: datetime
    studies: list[dict[str, Any]]
    report_markdown: str


class ProjectDetailResponse(BaseModel):
    """Response for GET /projects/{id} — the project plus its full history/corpus."""

    project: ProjectResponse
    history: list[ProjectHistoryItem]
    documents: list[ProjectDocumentResponse]


class ChatRequest(BaseModel):
    """Body for POST /projects/{id}/chat."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=1000)


class ChatMessageResponse(BaseModel):
    """One turn of a project's chat history."""

    role: str
    content: str
    created_at: datetime


class ChatResponse(BaseModel):
    """Response for POST /projects/{id}/chat."""

    reply: str
    cited_document_ids: list[str] = Field(default_factory=list)
