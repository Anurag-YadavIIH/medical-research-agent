"""Base class shared by every agent node."""

from __future__ import annotations

import abc

from medical_research_agent.config import Settings, get_settings
from medical_research_agent.logging_config import get_logger
from medical_research_agent.models.state import ResearchState


class BaseAgent(abc.ABC):
    """Common scaffolding for agent nodes.

    Agents are callables that take the current :class:`ResearchState` and return
    a *partial* state update (a dict of fields to merge). They never mutate state
    in place — LangGraph applies the returned delta. This keeps nodes pure and
    independently testable.
    """

    #: Stable node name used in the graph and in logs.
    name: str = "base"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.log = get_logger(self.name)

    @abc.abstractmethod
    async def run(self, state: ResearchState) -> dict[str, object]:
        """Execute the agent and return a partial state update."""

    async def __call__(self, state: ResearchState) -> dict[str, object]:
        self.log.info("agent.start", step=self.name)
        try:
            update = await self.run(state)
        except Exception as exc:  # noqa: BLE001 - surfaced as graph-level error
            self.log.error("agent.error", step=self.name, error=str(exc))
            return {"current_step": self.name, "errors": [f"{self.name}: {exc}"]}
        self.log.info("agent.done", step=self.name)
        return {"current_step": self.name, **update}
