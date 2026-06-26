"""Async CrossRef client for DOI / citation enrichment.

Failures are non-fatal: enrichment returns the study unchanged when CrossRef has
no match, so a missing DOI never breaks the pipeline.
"""

from __future__ import annotations

import httpx

from medical_research_agent.config import Settings, get_settings
from medical_research_agent.models.study import Study

_CROSSREF_BASE = "https://api.crossref.org"


class CrossRefService:
    """Best-effort enrichment via the CrossRef REST API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        headers = {"User-Agent": f"medical-research-agent (mailto:{self.settings.crossref_mailto})"}
        self._client = httpx.AsyncClient(base_url=_CROSSREF_BASE, timeout=30.0, headers=headers)

    async def enrich(self, study: Study) -> Study:
        """Return a copy of ``study`` enriched with DOI/citations. (Phase 2.)"""
        raise NotImplementedError("CrossRef enrichment lands in Phase 2.")

    async def aclose(self) -> None:
        await self._client.aclose()
