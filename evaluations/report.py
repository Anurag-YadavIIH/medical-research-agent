"""Builds a CaseReport from a RunResult + GoldCase, and renders JSON/Markdown."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluations import metrics
from evaluations.models import CaseReport, GoldCase, RunResult

REPORTS_DIR = Path(__file__).parent / "reports"


def _list_of_dicts(machine_json: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = machine_json.get(key) or []
    return list(value)


def evaluate_case(case: GoldCase, result: RunResult) -> CaseReport:
    retrieved_pmids = set(result.retrieved_pmids)
    retrieved_dois = set(result.retrieved_dois)
    extracted = _list_of_dicts(result.machine_json, "extracted")

    case_metrics = [
        metrics.citation_completeness(result.machine_json, retrieved_pmids, retrieved_dois),
        metrics.extraction_accuracy(
            case.gold_extractions,
            {item["pmid"]: item for item in extracted},
        ),
        metrics.evidence_consistency(
            _list_of_dicts(result.machine_json, "assessments"),
            _list_of_dicts(result.machine_json, "studies"),
        ),
        metrics.hallucination_check(result.machine_json, retrieved_pmids, retrieved_dois),
    ]
    return CaseReport(case_id=case.case_id, question=case.question, metrics=case_metrics)


def render_json(
    case_reports: list[CaseReport], *, generated_at: str | None = None
) -> dict[str, Any]:
    return {
        "generated_at": generated_at or datetime.now(UTC).isoformat(),
        "cases": [
            {
                "case_id": report.case_id,
                "question": report.question,
                "metrics": [
                    {
                        "name": metric.name,
                        "score": metric.score,
                        "details": metric.details,
                        "failures": metric.failures,
                        "gaps": metric.gaps,
                    }
                    for metric in report.metrics
                ],
            }
            for report in case_reports
        ],
    }


def render_markdown(case_reports: list[CaseReport], *, generated_at: str | None = None) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"_Generated: {generated_at or datetime.now(UTC).isoformat()}_",
        "",
    ]

    for report in case_reports:
        lines.append(f"## {report.case_id}")
        lines.append(f"_{report.question}_")
        lines.append("")
        lines.append("| Metric | Score |")
        lines.append("|---|---|")
        for metric in report.metrics:
            lines.append(f"| {metric.name} | {metric.score:.2%} |")
        lines.append("")

        for metric in report.metrics:
            if metric.failures:
                lines.append(f"**{metric.name} — failures:**")
                lines += [f"- {failure}" for failure in metric.failures]
                lines.append("")
            if metric.gaps:
                lines.append(f"**{metric.name} — gaps:**")
                lines += [f"- {gap}" for gap in metric.gaps]
                lines.append("")

    all_failures = sum(len(metric.failures) for report in case_reports for metric in report.metrics)
    lines.append("---")
    lines.append(
        f"**Total failures across all cases and metrics: {all_failures}**"
        if all_failures
        else "**No failures detected.**"
    )
    return "\n".join(lines)


def write_reports(case_reports: list[CaseReport]) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    generated_at = datetime.now(UTC).isoformat()

    json_path = REPORTS_DIR / f"{timestamp}.json"
    md_path = REPORTS_DIR / f"{timestamp}.md"

    json_path.write_text(
        json.dumps(render_json(case_reports, generated_at=generated_at), indent=2), encoding="utf-8"
    )
    md_path.write_text(render_markdown(case_reports, generated_at=generated_at), encoding="utf-8")

    return json_path, md_path
