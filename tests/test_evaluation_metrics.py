"""Unit tests for the four evaluation metrics — crafted inputs, no network."""

from __future__ import annotations

from evaluations.metrics import (
    citation_completeness,
    evidence_consistency,
    extraction_accuracy,
    hallucination_check,
)

# --- citation_completeness --------------------------------------------------


def test_citation_completeness_perfect_when_all_citations_are_retrieved() -> None:
    machine_json = {
        "evidence_summary": "Treatment improved outcomes [PMID: 90000001].",
        "references": [{"pmid": "90000001", "doi": None}],
    }
    result = citation_completeness(machine_json, {"90000001"}, set())

    assert result.score == 1.0
    assert result.failures == []


def test_citation_completeness_fabricated_pmid_cannot_score_1_and_is_listed() -> None:
    machine_json = {
        "evidence_summary": (
            "Treatment improved outcomes [PMID: 90000001] and so did an unrelated "
            "study [PMID: 12345678] that was never retrieved."
        ),
        "references": [{"pmid": "90000001", "doi": None}],
    }
    result = citation_completeness(machine_json, {"90000001"}, set())

    assert result.score < 1.0
    assert any("12345678" in f for f in result.failures)


def test_citation_completeness_no_citations_is_vacuously_complete() -> None:
    result = citation_completeness({"evidence_summary": "No citations here."}, set(), set())

    assert result.score == 1.0
    assert result.details["cited_count"] == 0


def test_citation_completeness_fabricated_doi_is_also_a_failure() -> None:
    machine_json = {
        "evidence_summary": "",
        "references": [{"pmid": "90000001", "doi": "10.9999/fabricated.doi"}],
    }
    result = citation_completeness(machine_json, {"90000001"}, {"10.1234/real.doi"})

    assert result.score < 1.0
    assert any("10.9999/fabricated.doi" in f for f in result.failures)


# --- extraction_accuracy ----------------------------------------------------


def test_extraction_accuracy_perfect_match() -> None:
    gold = {
        "1": {"study_design": "RCT", "sample_size": 50, "population": "Adults with condition X"}
    }
    model = {
        "1": {"study_design": "RCT", "sample_size": 50, "population": "Adults with condition X"}
    }

    result = extraction_accuracy(gold, model)

    assert result.score == 1.0
    assert result.failures == []


def test_extraction_accuracy_exact_field_mismatch_is_scored_as_a_miss() -> None:
    gold = {"1": {"study_design": "Randomized Controlled Trial"}}
    model = {"1": {"study_design": "Cohort Study"}}

    result = extraction_accuracy(gold, model)

    assert result.score == 0.0
    assert "field 'study_design'" in result.failures[0]


def test_extraction_accuracy_numeric_field_exact_mismatch() -> None:
    gold = {"1": {"sample_size": 100}}
    model = {"1": {"sample_size": 95}}

    result = extraction_accuracy(gold, model)

    assert result.score == 0.0


def test_extraction_accuracy_fuzzy_field_passes_above_threshold() -> None:
    gold = {"1": {"population": "Adults aged 18 to 65 with type 2 diabetes"}}
    model = {"1": {"population": "adults aged 18-65 with type 2 diabetes"}}

    result = extraction_accuracy(gold, model)

    assert result.score == 1.0


def test_extraction_accuracy_fuzzy_field_fails_below_threshold() -> None:
    gold = {"1": {"main_findings": "Treatment A significantly reduced symptom severity."}}
    model = {"1": {"main_findings": "No relevant data."}}

    result = extraction_accuracy(gold, model)

    assert result.score == 0.0
    assert "similarity=" in result.failures[0]


def test_extraction_accuracy_skips_fields_gold_does_not_label() -> None:
    gold = {"1": {"study_design": "RCT", "comparator": None}}
    model = {"1": {"study_design": "RCT", "comparator": "Something the model invented"}}

    result = extraction_accuracy(gold, model)

    # comparator isn't labelled in gold, so it's not scored at all -> perfect score.
    assert result.score == 1.0
    assert "comparator" not in result.details["per_field"]


def test_extraction_accuracy_missing_model_value_for_labelled_gold_field_is_a_miss() -> None:
    gold = {"1": {"comparator": "Placebo"}}
    model = {"1": {}}

    result = extraction_accuracy(gold, model)

    assert result.score == 0.0


# --- evidence_consistency ---------------------------------------------------


def test_evidence_consistency_matches_deterministic_map() -> None:
    studies = [{"pmid": "1", "publication_types": ["Randomized Controlled Trial"]}]
    assessments = [{"pmid": "1", "evidence_level": 2}]

    result = evidence_consistency(assessments, studies)

    assert result.score == 1.0
    assert result.failures == []
    assert result.gaps == []


def test_evidence_consistency_mismatch_is_a_hard_failure() -> None:
    studies = [{"pmid": "1", "publication_types": ["Randomized Controlled Trial"]}]
    # Level II is what the map says for an RCT; asserting Level I here is a bug.
    assessments = [{"pmid": "1", "evidence_level": 1}]

    result = evidence_consistency(assessments, studies)

    assert result.score == 0.0
    assert "PMID 1" in result.failures[0]


def test_evidence_consistency_ungraded_publication_type_is_a_gap() -> None:
    studies = [{"pmid": "1", "publication_types": ["Letter"]}]
    assessments = [{"pmid": "1", "evidence_level": 99}]  # UNGRADED

    result = evidence_consistency(assessments, studies)

    # Technically self-consistent (UNGRADED == UNGRADED) so the score is 1.0...
    assert result.score == 1.0
    # ...but it must still be visible as a gap, not silently passed over.
    assert len(result.gaps) == 1
    assert "Letter" in result.gaps[0]


# --- hallucination_check ----------------------------------------------------


def test_hallucination_check_clean_output_scores_1() -> None:
    machine_json = {
        "evidence_summary": "n=50 patients showed improvement, p < 0.001 [PMID: 90000001].",
        "references": [{"pmid": "90000001", "doi": None}],
        "extracted": [
            {"pmid": "90000001", "sample_size": 50, "statistical_significance": "p < 0.001"}
        ],
        "comparison": {"strongest_evidence_pmids": ["90000001"], "conflicting_evidence_pmids": []},
    }
    result = hallucination_check(machine_json, {"90000001"}, set())

    assert result.score == 1.0
    assert result.failures == []


def test_hallucination_check_flags_pmid_in_comparison_not_in_retrieved_set() -> None:
    machine_json = {
        "evidence_summary": "",
        "references": [],
        "extracted": [],
        "comparison": {"strongest_evidence_pmids": ["99999999"], "conflicting_evidence_pmids": []},
    }
    result = hallucination_check(machine_json, {"90000001"}, set())

    assert result.score < 1.0
    assert any("99999999" in f for f in result.failures)


def test_hallucination_check_flags_untraceable_sample_size_claim() -> None:
    machine_json = {
        "evidence_summary": "The trial enrolled 250 patients with good outcomes.",
        "references": [],
        "extracted": [{"pmid": "90000001", "sample_size": 50}],
    }
    result = hallucination_check(machine_json, {"90000001"}, set())

    assert result.score < 1.0
    assert any("250 patients" in f for f in result.failures)


def test_hallucination_check_flags_untraceable_pvalue_claim() -> None:
    machine_json = {
        "evidence_summary": "Outcomes were highly significant, p < 0.0001.",
        "references": [],
        "extracted": [{"pmid": "90000001", "statistical_significance": "p < 0.05"}],
    }
    result = hallucination_check(machine_json, {"90000001"}, set())

    assert result.score < 1.0
    assert any("0.0001" in f for f in result.failures)


def test_hallucination_check_no_claims_at_all_scores_1() -> None:
    machine_json = {"evidence_summary": "No numeric claims or citations here.", "references": []}
    result = hallucination_check(machine_json, {"90000001"}, set())

    assert result.score == 1.0
