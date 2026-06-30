"""Request/response schemas for the HTTP API."""

from __future__ import annotations

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
