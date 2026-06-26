"""Loads gold cases and wires up fixture-backed HTTP + a deterministic LLM stub.

Nothing here makes a real network or LLM call by default — `runner.py` uses
this to run the *real* graph (build_research_graph) against recorded NCBI/
CrossRef responses with every agent's get_chat_model() patched to a canned,
per-case answer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import respx

from evaluations.models import GoldCase
from medical_research_agent.services.crossref import _CROSSREF_BASE
from medical_research_agent.services.pubmed import _EUTILS_BASE

DATASETS_DIR = Path(__file__).parent / "datasets"
FIXTURES_DIR = Path(__file__).parent / "fixtures"

_PMID_IN_MESSAGE = re.compile(r"PMID (\w+)")


def list_gold_case_files() -> list[Path]:
    return sorted(DATASETS_DIR.glob("*.json"))


def load_gold_case(path: Path) -> GoldCase:
    data = json.loads(path.read_text(encoding="utf-8"))
    return GoldCase(
        case_id=data["case_id"],
        question=data["question"],
        filters=data.get("filters", {}),
        min_studies=data.get("min_studies", 0),
        retrieved_pmids=data["retrieved_pmids"],
        gold_extractions=data["gold_extractions"],
        stub_query_understanding=data["stub_query_understanding"],
        stub_extractions=data["stub_extractions"],
        stub_narrative=data["stub_narrative"],
        fixtures_dir=str(FIXTURES_DIR / data["case_id"]),
    )


def load_all_gold_cases() -> list[GoldCase]:
    return [load_gold_case(path) for path in list_gold_case_files()]


def _crossref_side_effect(case: GoldCase) -> Any:
    """Returns the recorded CrossRef response for each study in turn.

    CrossRefEnrichmentAgent.run() awaits ``enrich()`` sequentially, once per
    study, in ``state.studies`` order — which matches efetch's article order,
    which matches ``retrieved_pmids`` order (that's how the fixtures were
    recorded). So the n-th CrossRef request corresponds to the n-th PMID.
    """
    pmids = iter(case.retrieved_pmids)

    def _respond(request: httpx.Request) -> httpx.Response:
        pmid = next(pmids)
        path = Path(case.fixtures_dir) / f"crossref_{pmid}.json"
        return httpx.Response(200, content=path.read_bytes())

    return _respond


def register_http_fixtures(router: respx.MockRouter, case: GoldCase) -> None:
    fixtures_dir = Path(case.fixtures_dir)
    router.get(f"{_EUTILS_BASE}/esearch.fcgi").mock(
        return_value=httpx.Response(200, content=(fixtures_dir / "esearch.json").read_bytes())
    )
    router.get(f"{_EUTILS_BASE}/efetch.fcgi").mock(
        return_value=httpx.Response(200, content=(fixtures_dir / "efetch.xml").read_bytes())
    )
    router.get(f"{_CROSSREF_BASE}/works").mock(side_effect=_crossref_side_effect(case))


class _StubStructuredRunnable:
    def __init__(self, resolver: Any) -> None:
        self._resolver = resolver

    async def ainvoke(self, messages: list[Any]) -> Any:
        return self._resolver(messages)


class _StubChatModel:
    def __init__(self, resolver: Any) -> None:
        self._resolver = resolver

    def with_structured_output(self, schema: type[Any]) -> _StubStructuredRunnable:
        return _StubStructuredRunnable(self._resolver)


def _pmid_from_messages(messages: list[Any]) -> str | None:
    for message in messages:
        match = _PMID_IN_MESSAGE.search(getattr(message, "content", ""))
        if match:
            return match.group(1)
    return None


@contextmanager
def patched_llm(case: GoldCase) -> Iterator[None]:
    """Patch get_chat_model in every agent module to a deterministic stub
    derived from this case's stub_* fields — no real LLM call is made.
    """
    from medical_research_agent.models.evidence import EvidenceAssessment  # noqa: F401
    from medical_research_agent.models.query import QueryUnderstanding
    from medical_research_agent.models.study import ExtractedStudy

    def _query_understanding_resolver(_messages: list[Any]) -> QueryUnderstanding:
        return QueryUnderstanding(**case.stub_query_understanding)

    def _paper_reader_resolver(messages: list[Any]) -> ExtractedStudy:
        pmid = _pmid_from_messages(messages)
        fields = case.stub_extractions.get(pmid or "", {})
        return ExtractedStudy(pmid=pmid or "", **fields)

    class _Judgment:
        strength = "moderate"
        bias_risk = "low"
        confidence_reasoning = "Deterministic evaluation stub judgment."

    def _evidence_evaluator_resolver(_messages: list[Any]) -> Any:
        return _Judgment()

    from medical_research_agent.models.comparison import StudyComparison

    def _study_comparator_resolver(_messages: list[Any]) -> StudyComparison:
        return StudyComparison(
            agreements=["Deterministic evaluation stub: studies broadly agree."],
            strongest_evidence_pmids=list(case.retrieved_pmids[:1]),
        )

    def _summary_resolver(_messages: list[Any]) -> Any:
        return _NarrativeSectionsLike(**case.stub_narrative)

    resolvers = {
        "medical_research_agent.agents.query_understanding": _query_understanding_resolver,
        "medical_research_agent.agents.paper_reader": _paper_reader_resolver,
        "medical_research_agent.agents.evidence_evaluator": _evidence_evaluator_resolver,
        "medical_research_agent.agents.study_comparator": _study_comparator_resolver,
        "medical_research_agent.agents.summary": _summary_resolver,
    }

    with ExitStack() as stack:
        for module_path, resolver in resolvers.items():
            # Bind fake_model as a default arg — otherwise every lambda would
            # share the loop variable and all resolve to the last iteration's
            # stub once actually called.
            fake_model = _StubChatModel(resolver)
            stack.enter_context(
                patch(f"{module_path}.get_chat_model", lambda *a, _m=fake_model, **kw: _m)
            )
        yield


class _NarrativeSectionsLike:
    """Duck-typed stand-in for summary.py's private _NarrativeSections —
    only attribute access is needed, so no import of a private class.
    """

    def __init__(
        self,
        evidence_summary: str = "",
        clinical_implications: str = "",
        limitations: str = "",
        future_directions: str = "",
    ) -> None:
        self.evidence_summary = evidence_summary
        self.clinical_implications = clinical_implications
        self.limitations = limitations
        self.future_directions = future_directions
