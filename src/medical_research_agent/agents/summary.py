"""Synthesise the final evidence report with Vancouver references and caveats."""

from __future__ import annotations

from medical_research_agent.agents.base import BaseAgent
from medical_research_agent.models.state import ResearchState


class SummaryAgent(BaseAgent):
    """Synthesise the final evidence report with Vancouver references and caveats."""

    name = "summary"

    async def run(self, state: ResearchState) -> dict[str, object]:
        # TODO (Phase 2/3): implement. Stub passes state through unchanged so the
        # graph is wired and executable end-to-end during scaffolding.
        self.log.warning("agent.stub", step=self.name)
        return {}
