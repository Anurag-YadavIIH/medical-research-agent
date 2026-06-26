"""End-to-end wiring test: build_research_graph() with every agent mocked.

This proves the graph topology (START -> ... -> END) and state-merging work,
without making any real LLM or HTTP calls.
"""

from __future__ import annotations

from medical_research_agent.graphs.research_graph import build_research_graph
from medical_research_agent.models.comparison import StudyComparison
from medical_research_agent.models.evidence import EvidenceAssessment
from medical_research_agent.models.query import QueryUnderstanding
from medical_research_agent.models.report import EvidenceReport
from medical_research_agent.models.state import ResearchState
from medical_research_agent.models.study import ExtractedStudy, Study

_STUDY = Study(pmid="90000001", title="Synthetic fixture study", abstract="Synthetic abstract.")


async def test_graph_runs_start_to_end_with_mocked_agents(monkeypatch) -> None:
    monkeypatch.setattr(
        "medical_research_agent.agents.query_understanding.QueryUnderstandingAgent.run",
        lambda self, state: _result({"query_understanding": QueryUnderstanding(search_query="q")}),
    )
    monkeypatch.setattr(
        "medical_research_agent.agents.pubmed_search.PubMedSearchAgent.run",
        lambda self, state: _result({"studies": [_STUDY]}),
    )
    monkeypatch.setattr(
        "medical_research_agent.agents.crossref_enrichment.CrossRefEnrichmentAgent.run",
        lambda self, state: _result({"studies": state.studies}),
    )
    monkeypatch.setattr(
        "medical_research_agent.agents.paper_reader.PaperReaderAgent.run",
        lambda self, state: _result({"extracted": [ExtractedStudy(pmid="90000001")]}),
    )
    monkeypatch.setattr(
        "medical_research_agent.agents.evidence_evaluator.EvidenceEvaluatorAgent.run",
        lambda self, state: _result({"assessments": [EvidenceAssessment(pmid="90000001")]}),
    )
    monkeypatch.setattr(
        "medical_research_agent.agents.study_comparator.StudyComparatorAgent.run",
        lambda self, state: _result({"comparison": StudyComparison()}),
    )
    monkeypatch.setattr(
        "medical_research_agent.agents.summary.SummaryAgent.run",
        lambda self, state: _result(
            {"report": EvidenceReport(question=state.question, markdown="# Done")}
        ),
    )

    graph = build_research_graph()
    final_state = await graph.ainvoke(
        ResearchState(question="What treats keratoconus?"),
        config={"configurable": {"thread_id": "test-thread"}},
    )

    assert final_state["current_step"] == "summary"
    assert final_state["errors"] == []
    assert final_state["studies"] == [_STUDY]
    assert final_state["report"].markdown == "# Done"


async def test_graph_accumulates_errors_across_nodes_without_crashing(monkeypatch) -> None:
    async def _failing_run(self: object, state: ResearchState) -> dict[str, object]:
        raise RuntimeError("stage failed")

    monkeypatch.setattr(
        "medical_research_agent.agents.query_understanding.QueryUnderstandingAgent.run",
        _failing_run,
    )
    monkeypatch.setattr(
        "medical_research_agent.agents.pubmed_search.PubMedSearchAgent.run",
        lambda self, state: _result({"studies": []}),
    )
    monkeypatch.setattr(
        "medical_research_agent.agents.crossref_enrichment.CrossRefEnrichmentAgent.run",
        lambda self, state: _result({"studies": []}),
    )
    monkeypatch.setattr(
        "medical_research_agent.agents.paper_reader.PaperReaderAgent.run",
        lambda self, state: _result({"extracted": []}),
    )
    monkeypatch.setattr(
        "medical_research_agent.agents.evidence_evaluator.EvidenceEvaluatorAgent.run",
        lambda self, state: _result({"assessments": []}),
    )
    monkeypatch.setattr(
        "medical_research_agent.agents.study_comparator.StudyComparatorAgent.run",
        lambda self, state: _result({"comparison": StudyComparison()}),
    )
    monkeypatch.setattr(
        "medical_research_agent.agents.summary.SummaryAgent.run",
        lambda self, state: _result(
            {"report": EvidenceReport(question=state.question, markdown="# Done")}
        ),
    )

    graph = build_research_graph()
    final_state = await graph.ainvoke(
        ResearchState(question="What treats keratoconus?"),
        config={"configurable": {"thread_id": "test-thread-2"}},
    )

    assert any("stage failed" in e for e in final_state["errors"])
    assert final_state["current_step"] == "summary"


async def _result(value: dict[str, object]) -> dict[str, object]:
    return value
