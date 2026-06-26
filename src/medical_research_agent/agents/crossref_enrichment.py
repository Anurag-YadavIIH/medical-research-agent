"""Enrich studies with DOI, citation count, publisher and URL via CrossRef."""

from __future__ import annotations

from medical_research_agent.agents.base import BaseAgent
from medical_research_agent.models.state import ResearchState
from medical_research_agent.services.crossref import CrossRefService


class CrossRefEnrichmentAgent(BaseAgent):
    """Enrich studies with DOI, citation count, publisher and URL via CrossRef."""

    name = "crossref_enrichment"

    async def run(self, state: ResearchState) -> dict[str, object]:
        service = CrossRefService(settings=self.settings)
        try:
            enriched = [await service.enrich(study) for study in state.studies]
        finally:
            await service.aclose()
        return {"studies": enriched}
