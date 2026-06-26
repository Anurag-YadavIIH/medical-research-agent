"""Shared data shapes for the evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GoldCase:
    """A single labelled evaluation case loaded from datasets/*.json."""

    case_id: str
    question: str
    filters: dict[str, object]
    min_studies: int
    retrieved_pmids: list[str]
    gold_extractions: dict[str, dict[str, object]]
    stub_query_understanding: dict[str, str]
    stub_extractions: dict[str, dict[str, object]]
    stub_narrative: dict[str, str]
    fixtures_dir: str


@dataclass
class RunResult:
    """The output of running the pipeline (real or fixture-backed) for one case."""

    case_id: str
    question: str
    retrieved_pmids: list[str]
    retrieved_dois: list[str]
    machine_json: dict[str, object]
    warnings: list[str] = field(default_factory=list)


@dataclass
class MetricResult:
    """One metric's score plus the explicit list of failures/gaps behind it."""

    name: str
    score: float
    details: dict[str, object] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)


@dataclass
class CaseReport:
    """All metric results for a single case."""

    case_id: str
    question: str
    metrics: list[MetricResult]
