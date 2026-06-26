"""Research endpoints. Full orchestration is wired in Phases 3-4."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from medical_research_agent.api.schemas import ResearchRequest, ResearchResponse

router = APIRouter()


@router.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> ResearchResponse:
    """Run the evidence-synthesis pipeline for a clinical question.

    Stubbed in Phase 1: returns 501 until the LangGraph pipeline and persistence
    are connected (Phases 3-4).
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Research pipeline is implemented in Phases 3-4.",
    )


@router.get("/studies/{query_id}", response_model=list[dict[str, Any]])
async def get_studies(query_id: str) -> list[dict[str, Any]]:
    """Return persisted studies for a previous query. (Phase 4.)"""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Study retrieval is implemented in Phase 4.",
    )
