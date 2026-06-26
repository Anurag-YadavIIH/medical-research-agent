"""Assign a level-of-evidence grade and appraise bias for each study."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from medical_research_agent.agents.base import BaseAgent
from medical_research_agent.llm.factory import get_chat_model
from medical_research_agent.models.evidence import EvidenceAssessment, EvidenceLevel
from medical_research_agent.models.state import ResearchState
from medical_research_agent.models.study import Study

# Checked in order — first match wins, so combinations (e.g. a study tagged both
# "Meta-Analysis" and "Review") resolve to the strongest applicable level.
_LEVEL_KEYWORDS: list[tuple[EvidenceLevel, tuple[str, ...]]] = [
    (EvidenceLevel.LEVEL_I, ("meta-analysis", "systematic review")),
    (EvidenceLevel.LEVEL_II, ("randomized controlled trial",)),
    (EvidenceLevel.LEVEL_III, ("cohort",)),
    (EvidenceLevel.LEVEL_IV, ("case-control",)),
    (EvidenceLevel.LEVEL_V, ("case report", "expert opinion")),
]

_SYSTEM_PROMPT = (
    "You are appraising a biomedical study's strength of evidence and risk of bias "
    "based solely on its abstract. The level-of-evidence grade has already been "
    "determined separately from its publication type — do not state or imply a "
    "different level. Provide only: strength (e.g. strong/moderate/weak), bias_risk "
    "(e.g. low/some concerns/high), and a short, transparent confidence_reasoning."
)


class _LLMJudgment(BaseModel):
    """LLM-derived fields only — evidence_level is never delegated to the model."""

    strength: str = ""
    bias_risk: str = ""
    confidence_reasoning: str = ""


def deterministic_level(publication_types: list[str]) -> EvidenceLevel:
    """Map PubMed publication types to a level-of-evidence grade deterministically."""
    lowered = " | ".join(pt.lower() for pt in publication_types)
    for level, keywords in _LEVEL_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return level
    return EvidenceLevel.UNGRADED


class EvidenceEvaluatorAgent(BaseAgent):
    """Assign a level-of-evidence grade and appraise bias for each study."""

    name = "evidence_evaluator"

    async def run(self, state: ResearchState) -> dict[str, object]:
        model = get_chat_model().with_structured_output(_LLMJudgment)
        assessments: list[EvidenceAssessment] = []
        errors: list[str] = []

        for study in state.studies:
            level = deterministic_level(study.publication_types)
            judgment = await self._judge(model, study, errors)
            assessments.append(
                EvidenceAssessment(
                    pmid=study.pmid,
                    evidence_level=level,
                    strength=judgment.strength,
                    bias_risk=judgment.bias_risk,
                    confidence_reasoning=judgment.confidence_reasoning,
                )
            )

        update: dict[str, object] = {"assessments": assessments}
        if errors:
            update["errors"] = errors
        return update

    async def _judge(
        self, model: Runnable[Any, Any], study: Study, errors: list[str]
    ) -> _LLMJudgment:
        if not study.abstract.strip():
            return _LLMJudgment()
        try:
            return cast(
                _LLMJudgment,
                await model.ainvoke(
                    [
                        SystemMessage(content=_SYSTEM_PROMPT),
                        HumanMessage(content=f"Abstract (PMID {study.pmid}):\n{study.abstract}"),
                    ]
                ),
            )
        except Exception as exc:  # noqa: BLE001 - per-study failure shouldn't drop the rest
            errors.append(f"{self.name}: failed to appraise PMID {study.pmid}: {exc}")
            return _LLMJudgment()
