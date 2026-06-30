"""Domain models for the Projects feature: named research workspaces with a
persistent paper corpus (PubMed + uploaded PDFs) and a grounded chat history.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

DocumentSource = Literal["pubmed", "upload"]
ChatRole = Literal["user", "assistant"]


class Project(BaseModel):
    """A named container grouping scoped searches, uploaded papers and chat."""

    id: str
    name: str
    created_at: datetime


class ProjectDocument(BaseModel):
    """One paper in a project's corpus — either a PubMed result or an uploaded PDF."""

    id: str
    project_id: str
    source: DocumentSource
    pmid: str | None = None
    title: str = ""
    created_at: datetime


class ChatMessage(BaseModel):
    """One turn of a project's persisted chat history."""

    role: ChatRole
    content: str
    created_at: datetime


class ChatAnswer(BaseModel):
    """The assistant's reply to a project chat question, with its grounding."""

    reply: str
    cited_document_ids: list[str] = Field(default_factory=list)
