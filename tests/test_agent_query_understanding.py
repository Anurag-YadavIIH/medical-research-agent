"""Tests for the Query Understanding agent."""

from __future__ import annotations

from medical_research_agent.agents.query_understanding import (
    QueryUnderstandingAgent,
    _build_question_prompt,
)
from medical_research_agent.models.query import QueryUnderstanding, SearchFilters
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


async def test_returns_structured_query_understanding_with_user_keywords(fake_chat_model) -> None:
    expected = QueryUnderstanding(
        disease="keratoconus",
        intervention="corneal crosslinking",
        search_query="keratoconus AND crosslinking AND pediatric",
    )
    fake_chat_model("medical_research_agent.agents.query_understanding", expected)

    agent = QueryUnderstandingAgent()
    state = ResearchState(
        question="What are recent treatments for keratoconus?",
        filters=SearchFilters(keywords=["pediatric", "crosslinking"]),
    )

    delta = await agent.run(state)

    assert delta == {"query_understanding": expected}


def test_build_question_prompt_is_unchanged_without_keywords() -> None:
    assert _build_question_prompt("What treats keratoconus?", []) == "What treats keratoconus?"


def test_build_question_prompt_appends_keywords() -> None:
    prompt = _build_question_prompt("What treats keratoconus?", ["pediatric", "crosslinking"])

    assert prompt == (
        "What treats keratoconus?\n\n"
        "Additional keywords to incorporate into the search: pediatric, crosslinking"
    )
