"""Interpret the clinician question into PICO terms and a PubMed search strategy."""

from __future__ import annotations

from medical_research_agent.agents.base import BaseAgent
from medical_research_agent.models.state import ResearchState


class QueryUnderstandingAgent(BaseAgent):
    """Interpret the clinician question into PICO terms and a PubMed search strategy."""

    name = "query_understanding"

    async def run(self, state: ResearchState) -> dict[str, object]:
        # TODO (Phase 2/3): implement. Stub passes state through unchanged so the
        # graph is wired and executable end-to-end during scaffolding.
        self.log.warning("agent.stub", step=self.name)
        return {}
