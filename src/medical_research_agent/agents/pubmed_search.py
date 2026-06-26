"""Search PubMed via NCBI E-utilities and return structured studies."""

from __future__ import annotations

import hashlib

from medical_research_agent.agents.base import BaseAgent
from medical_research_agent.models.query import SearchFilters
from medical_research_agent.models.state import ResearchState
from medical_research_agent.models.study import Study
from medical_research_agent.services.cache import Cache
from medical_research_agent.services.pubmed import PubMedService


def _cache_key(question: str, filters: SearchFilters) -> str:
    digest = hashlib.sha256(f"{question}|{filters.model_dump_json()}".encode()).hexdigest()
    return f"pubmed_search:{digest}"


class PubMedSearchAgent(BaseAgent):
    """Search PubMed via NCBI E-utilities and return structured studies."""

    name = "pubmed_search"

    async def run(self, state: ResearchState) -> dict[str, object]:
        query = (
            state.query_understanding.search_query
            if state.query_understanding and state.query_understanding.search_query
            else state.question
        )

        cache = Cache(settings=self.settings)
        cache_key = _cache_key(state.question, state.filters)
        cached = await cache.get(cache_key)
        if cached is not None:
            await cache.aclose()
            return {"studies": [Study(**item) for item in cached]}

        service = PubMedService(settings=self.settings)
        try:
            studies = await service.search(query, state.filters)
        finally:
            await service.aclose()

        await cache.set(cache_key, [study.model_dump(mode="json") for study in studies])
        await cache.aclose()
        return {"studies": studies}
