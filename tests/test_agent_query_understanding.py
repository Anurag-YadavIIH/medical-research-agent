"""Tests for the Query Understanding agent."""

from __future__ import annotations

from medical_research_agent.agents.query_understanding import QueryUnderstandingAgent
from medical_research_agent.models.query import QueryUnderstanding
from medical_research_agent.models.state import ResearchState


async def test_returns_structured_query_understanding(fake_chat_model) -> None:
    expected = QueryUnderstanding(
        disease="keratoconus",
        intervention="corneal crosslinking",
        search_query="keratoconus AND crosslinking",
    )
    fake_chat_model("medical_research_agent.agents.query_understanding", expected)

    agent = QueryUnderstandingAgent()
    state = ResearchState(question="What are recent treatments for keratoconus?")

    delta = await agent.run(state)

    assert delta == {"query_understanding": expected}


async def test_llm_failure_is_recorded_as_error_not_raised(fake_chat_model) -> None:
    fake_chat_model("medical_research_agent.agents.query_understanding", RuntimeError("boom"))

    agent = QueryUnderstandingAgent()
    state = ResearchState(question="What are recent treatments for keratoconus?")

    delta = await agent(state)  # via __call__, mirroring how LangGraph invokes nodes

    assert delta["current_step"] == "query_understanding"
    assert any("boom" in e for e in delta["errors"])
