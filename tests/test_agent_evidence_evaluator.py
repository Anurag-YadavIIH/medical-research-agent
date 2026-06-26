"""Tests for the Evidence Evaluator agent.

The evidence_level must come ONLY from the deterministic publication_types
mapping; the LLM is responsible solely for strength/bias_risk/confidence_reasoning.
"""

from __future__ import annotations

import pytest

from medical_research_agent.agents.evidence_evaluator import (
    EvidenceEvaluatorAgent,
    deterministic_level,
)
from medical_research_agent.models.evidence import EvidenceLevel
from medical_research_agent.models.state import ResearchState
from medical_research_agent.models.study import Study


@pytest.mark.parametrize(
    ("publication_types", "expected"),
    [
        (["Meta-Analysis"], EvidenceLevel.LEVEL_I),
        (["Systematic Review"], EvidenceLevel.LEVEL_I),
        (["Randomized Controlled Trial"], EvidenceLevel.LEVEL_II),
        (["Cohort Studies"], EvidenceLevel.LEVEL_III),
        (["Case-Control Studies"], EvidenceLevel.LEVEL_IV),
        (["Case Reports"], EvidenceLevel.LEVEL_V),
        (["Expert Opinion"], EvidenceLevel.LEVEL_V),
        (["Journal Article"], EvidenceLevel.UNGRADED),
        ([], EvidenceLevel.UNGRADED),
        (["Journal Article", "Meta-Analysis"], EvidenceLevel.LEVEL_I),
    ],
)
def test_deterministic_level_mapping(publication_types: list[str], expected: EvidenceLevel) -> None:
    assert deterministic_level(publication_types) == expected


async def test_llm_cannot_override_deterministic_level(fake_chat_model) -> None:
    class _SneakyJudgment:
        """Mimics an LLM trying to also set evidence_level — must be ignored."""

        strength = "strong"
        bias_risk = "low"
        confidence_reasoning = "Looks rigorous."
        evidence_level = EvidenceLevel.LEVEL_I  # not a real field on _LLMJudgment

    fake_chat_model("medical_research_agent.agents.evidence_evaluator", _SneakyJudgment())

    agent = EvidenceEvaluatorAgent()
    state = ResearchState(
        question="Q",
        studies=[
            Study(
                pmid="90000001",
                title="A",
                abstract="Some abstract text.",
                publication_types=["Case Reports"],
            )
        ],
    )

    delta = await agent.run(state)

    assessment = delta["assessments"][0]
    assert assessment.evidence_level == EvidenceLevel.LEVEL_V
    assert assessment.strength == "strong"


async def test_skips_llm_call_for_studies_without_abstract(fake_chat_model) -> None:
    fake_chat_model("medical_research_agent.agents.evidence_evaluator")

    agent = EvidenceEvaluatorAgent()
    state = ResearchState(
        question="Q",
        studies=[
            Study(
                pmid="90000002",
                title="No abstract",
                abstract="",
                publication_types=["Randomized Controlled Trial"],
            )
        ],
    )

    delta = await agent.run(state)

    assessment = delta["assessments"][0]
    assert assessment.evidence_level == EvidenceLevel.LEVEL_II
    assert assessment.strength == ""


async def test_per_study_failure_does_not_drop_other_assessments(fake_chat_model) -> None:
    class _Judgment:
        strength = "moderate"
        bias_risk = "some concerns"
        confidence_reasoning = "Reasonable design."

    fake_chat_model(
        "medical_research_agent.agents.evidence_evaluator",
        RuntimeError("appraisal failed"),
        _Judgment(),
    )

    agent = EvidenceEvaluatorAgent()
    state = ResearchState(
        question="Q",
        studies=[
            Study(
                pmid="90000001", title="A", abstract="First.", publication_types=["Cohort Studies"]
            ),
            Study(
                pmid="90000002",
                title="B",
                abstract="Second.",
                publication_types=["Randomized Controlled Trial"],
            ),
        ],
    )

    delta = await agent.run(state)

    assessments = delta["assessments"]
    assert len(assessments) == 2
    assert assessments[0].evidence_level == EvidenceLevel.LEVEL_III
    assert assessments[0].strength == ""  # failed -> default judgment
    assert assessments[1].evidence_level == EvidenceLevel.LEVEL_II
    assert assessments[1].strength == "moderate"
    assert any("appraisal failed" in e for e in delta["errors"])
