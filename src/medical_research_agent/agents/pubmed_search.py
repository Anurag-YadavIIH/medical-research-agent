"""Search PubMed via NCBI E-utilities and return structured studies."""

from __future__ import annotations

from medical_research_agent.agents.base import BaseAgent
from medical_research_agent.models.state import ResearchState


class PubMedSearchAgent(BaseAgent):
    """Search PubMed via NCBI E-utilities and return structured studies."""

    name = "pubmed_search"

    async def run(self, state: ResearchState) -> dict[str, object]:
        # TODO (Phase 2/3): implement. Stub passes state through unchanged so the
        # graph is wired and executable end-to-end during scaffolding.
        self.log.warning("agent.stub", step=self.name)
        return {}
