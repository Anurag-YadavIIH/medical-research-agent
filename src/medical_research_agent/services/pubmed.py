"""Async client for NCBI E-utilities (PubMed).

Phase 1 establishes the client, rate-limit-aware configuration and method
contracts. The esearch/efetch HTTP calls and XML parsing are implemented in
Phase 2.
"""

from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from medical_research_agent.config import Settings, get_settings
from medical_research_agent.models.query import SearchFilters
from medical_research_agent.models.study import Study

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedService:
    """Thin async wrapper over NCBI E-utilities."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = httpx.AsyncClient(base_url=_EUTILS_BASE, timeout=30.0)

    def _common_params(self) -> dict[str, str]:
        params = {"tool": self.settings.ncbi_tool, "email": self.settings.ncbi_email}
        if self.settings.ncbi_api_key:
            params["api_key"] = self.settings.ncbi_api_key
        return params

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def esearch(self, query: str, retmax: int) -> list[str]:
        """Return a list of PMIDs for a query. (Implemented in Phase 2.)"""
        raise NotImplementedError("PubMed esearch lands in Phase 2.")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def efetch(self, pmids: list[str]) -> list[Study]:
        """Fetch and parse study records for PMIDs. (Implemented in Phase 2.)"""
        raise NotImplementedError("PubMed efetch lands in Phase 2.")

    async def search(self, query: str, filters: SearchFilters) -> list[Study]:
        """High-level search: build the query with filters, esearch, then efetch."""
        raise NotImplementedError("PubMed search lands in Phase 2.")

    async def aclose(self) -> None:
        await self._client.aclose()
