"""Search PubMed via NCBI E-utilities and return structured studies."""

from __future__ import annotations

from medical_research_agent.agents.base import BaseAgent
from medical_research_agent.models.state import ResearchState
from medical_research_agent.services.pubmed import PubMedService


class PubMedSearchAgent(BaseAgent):
    """Search PubMed via NCBI E-utilities and return structured studies."""

    name = "pubmed_search"

    async def run(self, state: ResearchState) -> dict[str, object]:
        query = (
            state.query_understanding.search_query
            if state.query_understanding and state.query_understanding.search_query
            else state.question
        )
        service = PubMedService(settings=self.settings)
        try:
            studies = await service.search(query, state.filters)
        finally:
            await service.aclose()
        return {"studies": studies}
