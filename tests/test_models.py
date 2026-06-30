"""Tests for domain-model invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from medical_research_agent.models.evidence import EvidenceLevel
from medical_research_agent.models.query import SearchFilters
from medical_research_agent.models.report import EvidenceReport
from medical_research_agent.models.state import ResearchState


def test_search_filters_keywords_rejects_too_many_items() -> None:
    with pytest.raises(ValidationError):
        SearchFilters(keywords=[f"kw{i}" for i in range(21)])


def test_search_filters_keywords_rejects_overlong_item() -> None:
    with pytest.raises(ValidationError):
        SearchFilters(keywords=["x" * 101])


def test_search_filters_keywords_accepts_within_bounds() -> None:
    filters = SearchFilters(keywords=["pediatric", "crosslinking"])
    assert filters.keywords == ["pediatric", "crosslinking"]


def test_evidence_level_ordering() -> None:
    assert EvidenceLevel.LEVEL_I < EvidenceLevel.LEVEL_II
    assert "meta-analysis" in EvidenceLevel.LEVEL_I.label.lower()


def test_report_machine_json_shape() -> None:
    report = EvidenceReport(question="Q", evidence_summary="S", limitations="L")
    payload = report.to_machine_json()
    assert set(payload) >= {"question", "studies", "evidence_summary", "limitations", "references"}


def test_state_errors_accumulate_reducer_default() -> None:
    state = ResearchState(question="Q")
    assert state.errors == []
    assert state.current_step == "start"
