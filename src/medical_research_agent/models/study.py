"""Models describing a retrieved study and its extracted clinical content."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Study(BaseModel):
    """A single biomedical study as retrieved and enriched from sources.

    Populated by the PubMed Search Agent and augmented by the CrossRef
    Enrichment Agent. Enrichment fields are optional and degrade gracefully when
    a DOI or citation count cannot be resolved.
    """

    pmid: str = Field(description="PubMed identifier.")
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    journal: str = ""
    publication_year: int | None = None
    abstract: str = ""
    publication_types: list[str] = Field(default_factory=list)

    # --- CrossRef enrichment (best-effort) ---------------------------------
    doi: str | None = None
    citation_count: int | None = None
    publisher: str | None = None
    url: str | None = None


class ExtractedStudy(BaseModel):
    """Structured clinical content extracted from a study abstract.

    Produced by the Paper Reader Agent. Carries the source ``pmid`` so it can be
    joined back to the originating :class:`Study`.
    """

    pmid: str
    objective: str = ""
    sample_size: int | None = None
    study_design: str = ""
    population: str = ""
    intervention: str = ""
    comparator: str = ""
    outcomes: list[str] = Field(default_factory=list)
    main_findings: str = ""
    statistical_significance: str = ""
    limitations: str = ""
