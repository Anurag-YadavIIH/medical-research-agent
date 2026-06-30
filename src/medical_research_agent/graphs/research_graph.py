"""Construct the linear evidence-synthesis state graph.

    START
      -> query_understanding
      -> pubmed_search
      -> crossref_enrichment
      -> paper_reader
      -> evidence_evaluator
      -> study_comparator
      -> summary
      -> END

LangGraph is imported lazily inside :func:`build_research_graph` so that simply
importing the FastAPI app (e.g. for the health check) does not require the full
LangChain/LangGraph dependency tree to be installed or initialised.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from medical_research_agent.agents.crossref_enrichment import CrossRefEnrichmentAgent
from medical_research_agent.agents.evidence_evaluator import EvidenceEvaluatorAgent
from medical_research_agent.agents.paper_reader import PaperReaderAgent
from medical_research_agent.agents.pubmed_search import PubMedSearchAgent
from medical_research_agent.agents.query_understanding import QueryUnderstandingAgent
from medical_research_agent.agents.study_comparator import StudyComparatorAgent
from medical_research_agent.agents.summary import SummaryAgent
from medical_research_agent.models.state import ResearchState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langgraph.graph.state import CompiledStateGraph

# Pipeline order: (node name, agent class). Linear by design.
_PIPELINE = [
    QueryUnderstandingAgent,
    PubMedSearchAgent,
    CrossRefEnrichmentAgent,
    PaperReaderAgent,
    EvidenceEvaluatorAgent,
    StudyComparatorAgent,
    SummaryAgent,
]


def build_research_graph(
    checkpointer: Any | None = None,
) -> CompiledStateGraph[ResearchState, Any, Any]:
    """Build and compile the research agent graph.

    Args:
        checkpointer: Optional LangGraph checkpointer for durable, resumable
            runs. Defaults to an in-memory saver in development.
    """
    from langgraph.graph import END, START, StateGraph

    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()

    graph = StateGraph(ResearchState)

    agents = [cls() for cls in _PIPELINE]  # type: ignore[abstract]
    for agent in agents:
        graph.add_node(agent.name, agent)

    graph.add_edge(START, agents[0].name)
    for prev, nxt in zip(agents, agents[1:], strict=False):
        graph.add_edge(prev.name, nxt.name)
    graph.add_edge(agents[-1].name, END)

    return graph.compile(checkpointer=checkpointer)  # type: ignore[return-value]
