"""Extract structured clinical content from each study abstract."""

from __future__ import annotations

from medical_research_agent.agents.base import BaseAgent
from medical_research_agent.models.state import ResearchState


class PaperReaderAgent(BaseAgent):
    """Extract structured clinical content from each study abstract."""

    name = "paper_reader"

    async def run(self, state: ResearchState) -> dict[str, object]:
        # TODO (Phase 2/3): implement. Stub passes state through unchanged so the
        # graph is wired and executable end-to-end during scaffolding.
        self.log.warning("agent.stub", step=self.name)
        return {}
