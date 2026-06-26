"""Tests for the Paper Reader agent — extraction must never go beyond the abstract."""

from __future__ import annotations

from medical_research_agent.agents.paper_reader import PaperReaderAgent
from medical_research_agent.models.state import ResearchState
from medical_research_agent.models.study import ExtractedStudy, Study


async def test_extracts_per_study_and_forces_correct_pmid(fake_chat_model) -> None:
    # The LLM is given the wrong pmid on purpose to prove the agent overrides it
    # rather than trusting whatever the model echoes back.
    fake_result = ExtractedStudy(pmid="wrong-pmid", objective="Synthetic objective")
    fake_chat_model("medical_research_agent.agents.paper_reader", fake_result)

    agent = PaperReaderAgent()
    state = ResearchState(
        question="Q",
        studies=[Study(pmid="90000001", title="A", abstract="Some synthetic abstract text.")],
    )

    delta = await agent.run(state)

    extracted = delta["extracted"]
    assert len(extracted) == 1
    assert extracted[0].pmid == "90000001"
    assert extracted[0].objective == "Synthetic objective"


async def test_skips_llm_call_for_studies_without_abstract(fake_chat_model) -> None:
    fake_model = fake_chat_model("medical_research_agent.agents.paper_reader")

    agent = PaperReaderAgent()
    state = ResearchState(
        question="Q",
        studies=[Study(pmid="90000002", title="No abstract", abstract="")],
    )

    delta = await agent.run(state)

    extracted = delta["extracted"]
    assert extracted == [ExtractedStudy(pmid="90000002")]
    # FakeStructuredRunnable would raise AssertionError if called with no queued
    # results, so an empty extracted-but-no-error result confirms no LLM call occurred.
    assert fake_model is not None


async def test_per_study_failure_does_not_drop_other_studies(fake_chat_model) -> None:
    fake_chat_model(
        "medical_research_agent.agents.paper_reader",
        RuntimeError("parse failed"),
        ExtractedStudy(pmid="will-be-overridden", objective="Second study objective"),
    )

    agent = PaperReaderAgent()
    state = ResearchState(
        question="Q",
        studies=[
            Study(pmid="90000001", title="A", abstract="First abstract."),
            Study(pmid="90000002", title="B", abstract="Second abstract."),
        ],
    )

    delta = await agent.run(state)

    extracted = delta["extracted"]
    assert len(extracted) == 1
    assert extracted[0].pmid == "90000002"
    assert any("parse failed" in e for e in delta["errors"])
