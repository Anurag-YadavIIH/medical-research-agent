"""Models for the user's clinical question and the reformulated search."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SearchFilters(BaseModel):
    """User-controllable constraints applied to the literature search."""

    model_config = ConfigDict(extra="forbid")

    year_min: int | None = Field(default=None, description="Earliest publication year.")
    year_max: int | None = Field(default=None, description="Latest publication year.")
    article_types: list[str] = Field(
        default_factory=list,
        description="PubMed publication types, e.g. 'Randomized Controlled Trial'.",
    )
    humans_only: bool = Field(default=True, description="Restrict to human studies.")
    max_papers: int = Field(default=15, ge=1, le=100, description="Cap on retrieved papers.")


class QueryUnderstanding(BaseModel):
    """Structured PICO-style interpretation of a clinician question.

    Produced by the Query Understanding Agent. Fields mirror the PICO framework
    (Population, Intervention, Comparison, Outcomes) plus the disease focus and a
    reformulated PubMed query string.
    """

    disease: str = Field(default="", description="Condition or disease in focus.")
    intervention: str = Field(default="", description="Treatment / intervention of interest.")
    comparison: str = Field(default="", description="Comparator, if any.")
    population: str = Field(default="", description="Target population (e.g. 'pediatric').")
    outcomes: str = Field(default="", description="Outcomes of interest.")
    search_query: str = Field(default="", description="Reformulated PubMed search string.")
