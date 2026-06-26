"""Request/response schemas for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from medical_research_agent.models.query import SearchFilters

DISCLAIMER = (
    "For research and educational purposes only. Clinical decisions should rely "
    "on professional judgment and full-text evidence review."
)


class ResearchRequest(BaseModel):
    """Body for POST /research."""

    question: str = Field(min_length=3, examples=["What are recent treatments for keratoconus?"])
    filters: SearchFilters = Field(default_factory=SearchFilters)


class ResearchResponse(BaseModel):
    """Response for POST /research — human report + machine JSON."""

    query_id: str
    question: str
    report_markdown: str
    machine_json: dict[str, Any]
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    version: str
    checks: dict[str, str]
