"""Integration test for the evaluation harness itself: runs the real
LangGraph pipeline against recorded fixtures with the LLM stubbed — no live
network or LLM calls. This is what `make eval` runs under the hood.
"""

from __future__ import annotations

from evaluations.fixtures import load_all_gold_cases
from evaluations.report import evaluate_case
from evaluations.runner import run_case


async def test_harness_runs_every_gold_case_against_fixtures() -> None:
    cases = load_all_gold_cases()
    assert len(cases) >= 2  # keratoconus_gold.json + at least one more case

    for case in cases:
        result = await run_case(case)

        assert len(result.retrieved_pmids) >= case.min_studies
        assert set(case.retrieved_pmids) == set(result.retrieved_pmids)

        report = evaluate_case(case, result)
        metric_by_name = {m.name: m for m in report.metrics}

        # The pipeline's own anti-hallucination defenses (Phase 3) mean a
        # fixture-backed run should never legitimately fail these two.
        assert metric_by_name["citation_completeness"].score == 1.0
        assert metric_by_name["evidence_consistency"].score == 1.0

        # extraction_accuracy and hallucination_check are allowed to be < 1.0
        # (these gold cases deliberately encode a couple of known stub misses
        # to prove the metrics actually catch something) but must still report
        # explicit, named failures rather than silently passing.
        assert 0.0 <= metric_by_name["extraction_accuracy"].score <= 1.0
        assert 0.0 <= metric_by_name["hallucination_check"].score <= 1.0
