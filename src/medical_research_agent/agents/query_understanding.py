"""Interpret the clinician question into PICO terms and a PubMed search strategy."""

from __future__ import annotations

from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage

from medical_research_agent.agents.base import BaseAgent
from medical_research_agent.llm.factory import get_chat_model
from medical_research_agent.models.query import QueryUnderstanding
from medical_research_agent.models.state import ResearchState

_SYSTEM_PROMPT = (
    "You are a biomedical research assistant. Given a clinician's question, extract a "
    "PICO-style structured interpretation (population, intervention, comparison, "
    "outcomes, disease focus) and reformulate it as a concise PubMed search query. "
    "Do not attempt to answer the clinical question itself."
)


def _build_question_prompt(question: str, keywords: list[str]) -> str:
    """Fold user-supplied keywords into the question sent to the LLM, if any."""
    if not keywords:
        return question
    keyword_str = ", ".join(keywords)
    return f"{question}\n\nAdditional keywords to incorporate into the search: {keyword_str}"


class QueryUnderstandingAgent(BaseAgent):
    """Interpret the clinician question into PICO terms and a PubMed search strategy."""

    name = "query_understanding"

    async def run(self, state: ResearchState) -> dict[str, object]:
        model = get_chat_model().with_structured_output(QueryUnderstanding)
        prompt = _build_question_prompt(state.question, state.filters.keywords)
        result = cast(
            QueryUnderstanding,
            await model.ainvoke(
                [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]
            ),
        )
        return {"query_understanding": result}
