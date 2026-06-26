"""Evidence-grading models (Oxford/levels-of-evidence style hierarchy)."""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, Field


class EvidenceLevel(IntEnum):
    """Level-of-evidence hierarchy (lower number = stronger evidence).

    LEVEL_I    Systematic reviews / meta-analyses
    LEVEL_II   Randomized controlled trials
    LEVEL_III  Cohort studies
    LEVEL_IV   Case-control studies
    LEVEL_V    Case reports / expert opinion
    UNGRADED   Could not be determined
    """

    LEVEL_I = 1
    LEVEL_II = 2
    LEVEL_III = 3
    LEVEL_IV = 4
    LEVEL_V = 5
    UNGRADED = 99

    @property
    def label(self) -> str:
        return {
            1: "Level I — Systematic review / meta-analysis",
            2: "Level II — Randomized controlled trial",
            3: "Level III — Cohort study",
            4: "Level IV — Case-control study",
            5: "Level V — Case report / expert opinion",
            99: "Ungraded",
        }[int(self)]


class EvidenceAssessment(BaseModel):
    """Per-study evidence appraisal produced by the Evidence Evaluator Agent."""

    pmid: str
    evidence_level: EvidenceLevel = EvidenceLevel.UNGRADED
    strength: str = Field(default="", description="e.g. strong / moderate / weak.")
    bias_risk: str = Field(default="", description="e.g. low / some concerns / high.")
    confidence_reasoning: str = Field(
        default="", description="Transparent justification for the grade and concerns."
    )
