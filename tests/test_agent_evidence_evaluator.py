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


@pytest.mark.parametrize(
    "publication_types",
    [
        ["Meta-Analysis", "Randomized Controlled Trial"],
        ["Cohort Studies", "Case-Control Studies", "Case Reports"],
    ],
)
def test_multiple_matching_categories_resolve_to_strongest_level(
    publication_types: list[str],
) -> None:
    expected = min(deterministic_level([pt]) for pt in publication_types)
    assert deterministic_level(publication_types) == expected


@pytest.mark.parametrize(
    "publication_types",
    [[], ["Journal Article"], ["Comment"], ["Letter", "Editorial"]],
)
def test_unmapped_publication_types_are_ungraded_not_level_v(
    publication_types: list[str],
) -> None:
    assert deterministic_level(publication_types) == EvidenceLevel.UNGRADED


@pytest.mark.parametrize("publication_types", [["Review"], ["Review", "Journal Article"]])
def test_bare_review_does_not_map_to_level_i(publication_types: list[str]) -> None:
    assert deterministic_level(publication_types) != EvidenceLevel.LEVEL_I
    assert deterministic_level(publication_types) == EvidenceLevel.UNGRADED


async def test_mixed_publication_types_are_noted_in_confidence_reasoning(
    fake_chat_model,
) -> None:
    class _Judgment:
        strength = "moderate"
        bias_risk = "low"
        confidence_reasoning = "Well-designed per the abstract."

    fake_chat_model("medical_research_agent.agents.evidence_evaluator", _Judgment())

    agent = EvidenceEvaluatorAgent()
    state = ResearchState(
        question="Q",
        studies=[
            Study(
                pmid="90000001",
                title="A",
                abstract="Some abstract text.",
                publication_types=["Meta-Analysis", "Randomized Controlled Trial"],
            )
        ],
    )

    delta = await agent.run(state)

    assessment = delta["assessments"][0]
    assert assessment.evidence_level == EvidenceLevel.LEVEL_I
    assert "multiple evidence categories" in assessment.confidence_reasoning
    assert "Well-designed per the abstract." in assessment.confidence_reasoning


async def test_single_matching_category_does_not_add_mixed_note(fake_chat_model) -> None:
    class _Judgment:
        strength = "moderate"
        bias_risk = "low"
        confidence_reasoning = "Well-designed per the abstract."

    fake_chat_model("medical_research_agent.agents.evidence_evaluator", _Judgment())

    agent = EvidenceEvaluatorAgent()
    state = ResearchState(
        question="Q",
        studies=[
            Study(
                pmid="90000001",
                title="A",
                abstract="Some abstract text.",
                # "Journal Article" doesn't map to any category, so only one
                # category actually matches here — not a mixed-typing case.
                publication_types=["Journal Article", "Meta-Analysis"],
            )
        ],
    )

    delta = await agent.run(state)

    assessment = delta["assessments"][0]
    assert assessment.confidence_reasoning == "Well-designed per the abstract."


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
