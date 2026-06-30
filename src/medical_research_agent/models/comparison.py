"""Cross-study comparison model produced by the Study Comparator Agent."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StudyComparisonNarrative(BaseModel):
    """The LLM-facing subset of a comparison: narrative judgment only.

    ``comparison_matrix`` is deliberately excluded — it's built deterministically
    from already-known fields (see ``study_comparator.py``), and a free-form
    ``dict[str, str]`` field isn't expressible in OpenAI's strict structured-output
    JSON schema mode, so it must never be part of what we ask the model to produce.
    """

    agreements: list[str] = Field(default_factory=list, description="Consistent findings.")
    disagreements: list[str] = Field(default_factory=list, description="Conflicting findings.")
    trends: list[str] = Field(default_factory=list, description="Emerging directional signals.")
    strongest_evidence_pmids: list[str] = Field(
        default_factory=list, description="PMIDs carrying the strongest evidence."
    )
    conflicting_evidence_pmids: list[str] = Field(default_factory=list)


class StudyComparison(StudyComparisonNarrative):
    """Synthesis of agreement / disagreement across the appraised studies."""

    comparison_matrix: list[dict[str, str]] = Field(
        default_factory=list,
        description="Row-per-study matrix of key comparable fields for tabular display.",
    )
