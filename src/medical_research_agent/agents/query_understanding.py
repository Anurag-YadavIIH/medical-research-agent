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


class QueryUnderstandingAgent(BaseAgent):
    """Interpret the clinician question into PICO terms and a PubMed search strategy."""

    name = "query_understanding"

    async def run(self, state: ResearchState) -> dict[str, object]:
        model = get_chat_model().with_structured_output(QueryUnderstanding)
        result = cast(
            QueryUnderstanding,
            await model.ainvoke(
                [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=state.question)]
            ),
        )
        return {"query_understanding": result}
