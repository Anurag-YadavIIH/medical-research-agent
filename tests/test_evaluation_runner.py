"""Integration test for the evaluation harness itself: runs the real
LangGraph pipeline against recorded fixtures with the LLM stubbed — no live
network or LLM calls. This is what `make eval` runs under the hood.

Assertions here are deliberately concrete (not just "it didn't crash"):
machine_json shape, citation subset, deterministic evidence levels recomputed
independently, and comparison-PMID filtering are each checked directly
against the real pipeline output, not only via the aggregate metric scores.
"""

from __future__ import annotations

import re

from evaluations.fixtures import load_all_gold_cases
from evaluations.report import evaluate_case
from evaluations.runner import run_case

from medical_research_agent.agents.evidence_evaluator import deterministic_level

_PMID_CITATION = re.compile(r"\[PMID:\s*([\w.-]+)\]")

_EXPECTED_MACHINE_JSON_KEYS = {
    "question",
    "studies",
    "extracted",
    "assessments",
    "comparison",
    "evidence_summary",
    "clinical_implications",
    "limitations",
    "future_directions",
    "references",
    "disclaimer",
}


async def test_harness_runs_every_gold_case_against_fixtures() -> None:
    cases = load_all_gold_cases()
    assert len(cases) >= 2  # keratoconus_gold.json + at least one more case

    seen_hallucination_failure = False

    for case in cases:
        result = await run_case(case)
        machine_json = result.machine_json
        retrieved = set(result.retrieved_pmids)

        # --- machine_json shape: every documented top-level key is present ---
        assert machine_json.keys() >= _EXPECTED_MACHINE_JSON_KEYS
        assert machine_json["disclaimer"]

        # --- retrieved set matches the gold case exactly ---
        assert len(retrieved) >= case.min_studies
        assert retrieved == set(case.retrieved_pmids)

        # --- citation subset: every [PMID: x] in the narrative is retrieved ---
        narrative = "\n".join(
            machine_json[f]
            for f in (
                "evidence_summary",
                "clinical_implications",
                "limitations",
                "future_directions",
            )
        )
        cited = set(_PMID_CITATION.findall(narrative))
        assert cited, "expected at least one citation in this fixture's narrative"
        assert cited <= retrieved

        # --- deterministic evidence levels: recomputed independently per study,
        # not just trusted because the metric says so ---
        studies_by_pmid = {s["pmid"]: s for s in machine_json["studies"]}
        assert machine_json["assessments"], "expected at least one assessment"
        for assessment in machine_json["assessments"]:
            study = studies_by_pmid[assessment["pmid"]]
            expected_level = int(deterministic_level(study["publication_types"]))
            assert assessment["evidence_level"] == expected_level

        # --- comparison-PMID filtering: strongest/conflicting lists never
        # reference a PMID outside the retrieved set ---
        comparison = machine_json["comparison"]
        assert comparison is not None
        assert set(comparison["strongest_evidence_pmids"]) <= retrieved
        assert set(comparison["conflicting_evidence_pmids"]) <= retrieved

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
        assert metric_by_name["extraction_accuracy"].failures
        seen_hallucination_failure |= bool(metric_by_name["hallucination_check"].failures)

    # At least one fixture must exercise the untraceable-numeric-claim path —
    # otherwise this assertion (and the metric's failure path) would never run.
    assert seen_hallucination_failure
