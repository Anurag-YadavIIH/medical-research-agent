"""Cross-study comparison model produced by the Study Comparator Agent."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StudyComparison(BaseModel):
    """Synthesis of agreement / disagreement across the appraised studies."""

    agreements: list[str] = Field(default_factory=list, description="Consistent findings.")
    disagreements: list[str] = Field(default_factory=list, description="Conflicting findings.")
    trends: list[str] = Field(default_factory=list, description="Emerging directional signals.")
    strongest_evidence_pmids: list[str] = Field(
        default_factory=list, description="PMIDs carrying the strongest evidence."
    )
    conflicting_evidence_pmids: list[str] = Field(default_factory=list)
    comparison_matrix: list[dict[str, str]] = Field(
        default_factory=list,
        description="Row-per-study matrix of key comparable fields for tabular display.",
    )
