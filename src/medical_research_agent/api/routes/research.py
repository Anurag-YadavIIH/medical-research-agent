"""Research endpoints: run the evidence-synthesis pipeline and persist results."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from medical_research_agent.api.rate_limit import enforce_rate_limit
from medical_research_agent.api.schemas import ResearchRequest, ResearchResponse
from medical_research_agent.config import get_settings
from medical_research_agent.database.repository import ResearchRepository
from medical_research_agent.database.session import get_session
from medical_research_agent.graphs.research_graph import build_research_graph
from medical_research_agent.logging_config import get_logger
from medical_research_agent.models.report import EvidenceReport
from medical_research_agent.models.state import ResearchState
from medical_research_agent.models.study import Study

router = APIRouter()
logger = get_logger(__name__)


def _ensure_llm_configured() -> None:
    """Fail fast with an actionable message rather than a bare 500 mid-pipeline."""
    settings = get_settings()
    if settings.api_key_for(settings.default_llm_provider):
        return
    key_names: dict[str, str] = {
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    key_name = key_names[settings.default_llm_provider]
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=(
            f"No API key configured for the default LLM provider "
            f"'{settings.default_llm_provider}'. Set {key_name} in your environment "
            "or .env file before running research."
        ),
    )


def _fallback_report(question: str, studies: list[Study]) -> EvidenceReport:
    """Used only if the summary node itself crashed outright (not just an LLM error)."""
    report = EvidenceReport(
        question=question,
        studies=studies,
        evidence_summary="Report generation failed before a summary could be produced.",
    )
    report.markdown = (
        f"# {question}\n\n"
        "_Report generation failed before a summary could be produced. "
        "See warnings for details._\n\n"
        f"_Disclaimer: {report.DISCLAIMER}_"
    )
    return report


async def run_research(
    request: ResearchRequest,
    session: AsyncSession,
    *,
    project_id: str | None = None,
) -> tuple[ResearchResponse, EvidenceReport]:
    """Run the evidence-synthesis pipeline and persist the result.

    Shared by ``POST /research`` (project_id=None, normal/ungrouped search,
    unchanged behavior) and ``POST /projects/{id}/research`` (project-scoped) —
    one place doing pipeline-run + persist + error-leakage-safe logging,
    rather than duplicating it per route.

    Agent-level failures (recorded in ``state.errors``) are surfaced as ``warnings``
    with a 200 — a thin or partial result is a normal research outcome. 5xx is
    reserved for genuine infrastructure failures (missing LLM key, DB down, an
    unexpected crash in the pipeline itself).

    Returns the response plus the underlying ``EvidenceReport`` so callers that
    need the retrieved studies (e.g. to embed them into a project's RAG corpus)
    don't have to re-derive them from the response's ``machine_json``.
    """
    _ensure_llm_configured()

    graph = build_research_graph()
    initial_state = ResearchState(question=request.question, filters=request.filters)
    try:
        final_state = await graph.ainvoke(
            initial_state, config={"configurable": {"thread_id": str(uuid.uuid4())}}
        )
    except Exception as exc:
        # The raw exception (DB/LLM internals, stack-trace-shaped text) must never
        # reach the client — log it fully server-side, return a generic message.
        logger.error(
            "research.pipeline_failed",
            error=str(exc),
            exc_type=type(exc).__name__,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Research pipeline failed unexpectedly. Please try again or contact support.",
        ) from exc

    warnings = list(final_state.get("errors") or [])
    report = final_state.get("report") or _fallback_report(
        request.question, final_state.get("studies") or []
    )

    repo = ResearchRepository(session)
    try:
        query_record = await repo.save_research_run(
            request.question, request.filters, report, project_id=project_id
        )
    except Exception as exc:
        logger.error(
            "research.persistence_failed",
            error=str(exc),
            exc_type=type(exc).__name__,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist research results. Please try again.",
        ) from exc

    response = ResearchResponse(
        query_id=query_record.id,
        question=request.question,
        report_markdown=report.markdown,
        machine_json=report.to_machine_json(),
        warnings=warnings,
    )
    return response, report


@router.post(
    "/research",
    response_model=ResearchResponse,
    dependencies=[Depends(enforce_rate_limit)],
)
async def research(
    request: ResearchRequest, session: AsyncSession = Depends(get_session)
) -> ResearchResponse:
    """Run the evidence-synthesis pipeline for a clinical question and persist the result."""
    response, _report = await run_research(request, session)
    return response


@router.get("/studies/{query_id}", response_model=list[dict[str, Any]])
async def get_studies(
    query_id: str, session: AsyncSession = Depends(get_session)
) -> list[dict[str, Any]]:
    """Return persisted studies for a previous query."""
    repo = ResearchRepository(session)
    query_record = await repo.get_query(query_id)
    if query_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No research run found for query_id '{query_id}'.",
        )
    studies = await repo.get_studies(query_id)
    return [study.payload for study in studies]
