"""Evaluation harness CLI.

    python -m evaluations.cli              # fixtures, all cases
    python -m evaluations.cli --case keratoconus_crosslinking
    python -m evaluations.cli --live       # real NCBI/CrossRef + a real LLM key

Exit code is 1 only for *critical* failures — a fabricated citation
(citation_completeness or hallucination_check finding an ID outside the
retrieved set) or an evidence_consistency mismatch (a deterministic-mapping
bug). extraction_accuracy misses and untraceable-numeric-claim flags are
real, reported findings but don't fail the run — they're closer to a
continuous quality signal than a hard invariant.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from evaluations.fixtures import load_all_gold_cases
from evaluations.models import CaseReport
from evaluations.report import evaluate_case, write_reports
from evaluations.runner import run_case, run_case_live

_CRITICAL_METRICS = {"citation_completeness", "evidence_consistency"}


def _is_critical_failure(report: CaseReport) -> bool:
    for metric in report.metrics:
        if metric.name in _CRITICAL_METRICS and metric.failures:
            return True
        if metric.name == "hallucination_check" and any(
            f.startswith("ID not in retrieved set") for f in metric.failures
        ):
            return True
    return False


async def _main(case_ids: list[str] | None, live: bool) -> int:
    all_cases = load_all_gold_cases()
    if case_ids:
        cases = [c for c in all_cases if c.case_id in case_ids]
        missing = set(case_ids) - {c.case_id for c in cases}
        if missing:
            print(f"Unknown case id(s): {sorted(missing)}", file=sys.stderr)
            return 1
    else:
        cases = all_cases

    case_reports = []
    for case in cases:
        result = await (run_case_live(case) if live else run_case(case))
        case_reports.append(evaluate_case(case, result))

    json_path, md_path = write_reports(case_reports)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    if any(_is_critical_failure(report) for report in case_reports):
        print("CRITICAL: a citation or evidence-level invariant was violated.", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the evaluation harness.")
    parser.add_argument(
        "--case", action="append", dest="cases", help="Run only this case_id (repeatable)."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run against real NCBI/CrossRef + a real configured LLM, instead of fixtures.",
    )
    args = parser.parse_args()

    exit_code = asyncio.run(_main(args.cases, args.live))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
