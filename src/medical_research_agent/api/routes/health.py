"""Health and readiness endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from medical_research_agent import __version__
from medical_research_agent.api.schemas import HealthResponse
from medical_research_agent.services.cache import Cache

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness + lightweight dependency checks (non-fatal)."""
    checks: dict[str, str] = {}
    try:
        cache = Cache()
        checks["redis"] = "ok" if await cache.ping() else "unavailable"
        await cache.aclose()
    except Exception:  # noqa: BLE001
        checks["redis"] = "unavailable"
    return HealthResponse(status="ok", version=__version__, checks=checks)
