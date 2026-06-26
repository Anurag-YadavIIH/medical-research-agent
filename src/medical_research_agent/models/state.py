"""The typed LangGraph state object threaded through every agent node.

We use a Pydantic ``BaseModel`` as the graph schema (LangGraph supports this) so
state is validated and self-documenting. The pipeline is linear, so each node
writes a distinct field; ``errors`` uses an additive reducer so failures from any
node accumulate rather than overwrite each other.
"""

from __future__ import annotations

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from medical_research_agent.models.comparison import StudyComparison
from medical_research_agent.models.evidence import EvidenceAssessment
from medical_research_agent.models.query import QueryUnderstanding, SearchFilters
from medical_research_agent.models.report import EvidenceReport
from medical_research_agent.models.study import ExtractedStudy, Study


class ResearchState(BaseModel):
    """Mutable state carried from START to END through the agent graph."""

    # --- Inputs ------------------------------------------------------------
    question: str
    filters: SearchFilters = Field(default_factory=SearchFilters)

    # --- Per-stage outputs -------------------------------------------------
    query_understanding: QueryUnderstanding | None = None
    studies: list[Study] = Field(default_factory=list)
    extracted: list[ExtractedStudy] = Field(default_factory=list)
    assessments: list[EvidenceAssessment] = Field(default_factory=list)
    comparison: StudyComparison | None = None
    report: EvidenceReport | None = None

    # --- Diagnostics -------------------------------------------------------
    current_step: str = "start"
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)
