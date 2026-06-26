"""Final, user-facing report models (human-readable + machine-readable)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from medical_research_agent.models.study import Study


class ReferenceEntry(BaseModel):
    """A single Vancouver-formatted reference with its source identifiers."""

    pmid: str
    doi: str | None = None
    vancouver: str = Field(description="Vancouver-formatted citation string.")
    url: str | None = None


class EvidenceReport(BaseModel):
    """The Summary Agent's final synthesis.

    ``markdown`` is the human-readable report; the remaining fields form the
    machine-readable JSON contract required by the API and the Streamlit UI.
    """

    question: str
    markdown: str = Field(default="", description="Rendered human-readable report.")
    evidence_summary: str = ""
    clinical_implications: str = ""
    limitations: str = ""
    future_directions: str = ""
    studies: list[Study] = Field(default_factory=list)
    references: list[ReferenceEntry] = Field(default_factory=list)

    DISCLAIMER: str = (
        "For research and educational purposes only. Clinical decisions should "
        "rely on professional judgment and full-text evidence review."
    )

    def to_machine_json(self) -> dict[str, object]:
        """Return the compact machine-readable contract."""
        return {
            "question": self.question,
            "studies": [s.model_dump() for s in self.studies],
            "evidence_summary": self.evidence_summary,
            "limitations": self.limitations,
            "references": [r.model_dump() for r in self.references],
            "disclaimer": self.DISCLAIMER,
        }
