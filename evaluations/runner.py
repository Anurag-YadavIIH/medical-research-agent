"""Runs the real LangGraph pipeline against fixture-backed HTTP + a stubbed LLM.

This is deliberately the same `build_research_graph().ainvoke(...)` call the
API route makes — the evaluation harness exercises the actual pipeline, not a
re-implementation of it.
"""

from __future__ import annotations

import uuid

import respx

from evaluations.fixtures import patched_llm, register_http_fixtures
from evaluations.models import GoldCase, RunResult
from medical_research_agent.graphs.research_graph import build_research_graph
from medical_research_agent.models.state import ResearchState


async def run_case(case: GoldCase) -> RunResult:
    """Run the pipeline for one gold case against its recorded fixtures."""
    with respx.mock(assert_all_called=False) as router, patched_llm(case):
        register_http_fixtures(router, case)

        graph = build_research_graph()
        state = ResearchState(question=case.question, filters=case.filters)
        final_state = await graph.ainvoke(
            state, config={"configurable": {"thread_id": str(uuid.uuid4())}}
        )

    report = final_state.get("report")
    machine_json = report.to_machine_json() if report else {}
    studies = machine_json.get("studies") or []

    return RunResult(
        case_id=case.case_id,
        question=case.question,
        retrieved_pmids=[study["pmid"] for study in studies if study.get("pmid")],
        retrieved_dois=[study["doi"] for study in studies if study.get("doi")],
        machine_json=machine_json,
        warnings=list(final_state.get("errors") or []),
    )


async def run_case_live(case: GoldCase) -> RunResult:
    """Run against the real NCBI/CrossRef APIs and a real configured LLM.

    No mocking at all — this is the true end-to-end path, gated behind
    `--live` in the CLI and never run by default or in CI.
    """
    graph = build_research_graph()
    state = ResearchState(question=case.question, filters=case.filters)
    final_state = await graph.ainvoke(
        state, config={"configurable": {"thread_id": str(uuid.uuid4())}}
    )

    report = final_state.get("report")
    machine_json = report.to_machine_json() if report else {}
    studies = machine_json.get("studies") or []

    return RunResult(
        case_id=case.case_id,
        question=case.question,
        retrieved_pmids=[study["pmid"] for study in studies if study.get("pmid")],
        retrieved_dois=[study["doi"] for study in studies if study.get("doi")],
        machine_json=machine_json,
        warnings=list(final_state.get("errors") or []),
    )
