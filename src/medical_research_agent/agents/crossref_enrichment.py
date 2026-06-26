"""Enrich studies with DOI, citation count, publisher and URL via CrossRef."""

from __future__ import annotations

from medical_research_agent.agents.base import BaseAgent
from medical_research_agent.models.state import ResearchState


class CrossRefEnrichmentAgent(BaseAgent):
    """Enrich studies with DOI, citation count, publisher and URL via CrossRef."""

    name = "crossref_enrichment"

    async def run(self, state: ResearchState) -> dict[str, object]:
        # TODO (Phase 2/3): implement. Stub passes state through unchanged so the
        # graph is wired and executable end-to-end during scaffolding.
        self.log.warning("agent.stub", step=self.name)
        return {}
