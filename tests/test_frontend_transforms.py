"""Unit tests for frontend/transforms.py — pure functions, no Streamlit/HTTP."""

from __future__ import annotations

from frontend.transforms import (
    comparison_matrix_rows,
    comparison_narrative,
    evidence_level_label,
    reference_rows,
    study_detail_rows,
)

_MACHINE_JSON = {
    "question": "What treats keratoconus?",
    "studies": [
        {
            "pmid": "90000001",
            "title": "Synthetic Study A",
            "authors": ["Testauthor J"],
            "journal": "Journal of Synthetic Testing",
            "publication_year": 2023,
            "abstract": "Synthetic abstract text.",
            "doi": "10.9999/test.synthetic.0001",
        }
    ],
    "extracted": [
        {
            "pmid": "90000001",
            "objective": "Assess synthetic outcomes.",
            "main_findings": "Synthetic improvement observed.",
            "outcomes": ["symptom score"],
        }
    ],
    "assessments": [
        {
            "pmid": "90000001",
            "evidence_level": 2,
            "strength": "moderate",
            "bias_risk": "low",
            "confidence_reasoning": "Well-designed per the abstract.",
        }
    ],
    "comparison": {
        "agreements": ["Both studies report improvement."],
        "disagreements": [],
        "trends": ["Growing interest in crosslinking."],
        "comparison_matrix": [
            {
                "pmid": "90000001",
                "title": "Synthetic Study A",
                "evidence_level": "Level II — Randomized controlled trial",
                "strength": "moderate",
                "main_finding": "Synthetic improvement observed.",
            }
        ],
    },
    "evidence_summary": "Synthetic evidence summary.",
    "limitations": "Synthetic limitations.",
    "references": [
        {
            "pmid": "90000001",
            "doi": "10.9999/test.synthetic.0001",
            "vancouver": "Testauthor J. Synthetic Study A. Journal of Synthetic Testing. 2023.",
            "url": None,
        }
    ],
    "disclaimer": "For research and educational purposes only.",
}


def test_evidence_level_label_known_and_unknown_values() -> None:
    assert evidence_level_label(1) == "Level I — Systematic review / meta-analysis"
    assert evidence_level_label(99) == "Ungraded"
    assert evidence_level_label(None) == "Ungraded"
    assert evidence_level_label(12345) == "Ungraded"


def test_comparison_matrix_rows_extracts_matrix() -> None:
    rows = comparison_matrix_rows(_MACHINE_JSON)
    assert rows == _MACHINE_JSON["comparison"]["comparison_matrix"]


def test_comparison_matrix_rows_handles_missing_comparison() -> None:
    assert comparison_matrix_rows({}) == []
    assert comparison_matrix_rows({"comparison": None}) == []


def test_comparison_narrative_extracts_bullet_lists() -> None:
    narrative = comparison_narrative(_MACHINE_JSON)
    assert narrative == {
        "agreements": ["Both studies report improvement."],
        "disagreements": [],
        "trends": ["Growing interest in crosslinking."],
    }


def test_comparison_narrative_handles_missing_comparison() -> None:
    assert comparison_narrative({}) == {"agreements": [], "disagreements": [], "trends": []}


def test_study_detail_rows_joins_study_extracted_and_assessment_by_pmid() -> None:
    rows = study_detail_rows(_MACHINE_JSON)
    assert len(rows) == 1
    row = rows[0]
    assert row["pmid"] == "90000001"
    assert row["title"] == "Synthetic Study A"
    assert row["objective"] == "Assess synthetic outcomes."
    assert row["main_findings"] == "Synthetic improvement observed."
    assert row["evidence_level"] == "Level II — Randomized controlled trial"
    assert row["strength"] == "moderate"


def test_study_detail_rows_handles_study_with_no_extraction_or_assessment() -> None:
    machine_json = {
        "studies": [{"pmid": "90000002", "title": "Unextracted Study"}],
        "extracted": [],
        "assessments": [],
    }
    rows = study_detail_rows(machine_json)
    assert len(rows) == 1
    assert rows[0]["pmid"] == "90000002"
    assert rows[0]["objective"] == ""
    assert rows[0]["evidence_level"] == "Ungraded"


def test_reference_rows_builds_pmid_and_doi_links() -> None:
    rows = reference_rows(_MACHINE_JSON)
    assert len(rows) == 1
    assert rows[0]["pmid_url"] == "https://pubmed.ncbi.nlm.nih.gov/90000001/"
    assert rows[0]["doi_url"] == "https://doi.org/10.9999/test.synthetic.0001"


def test_reference_rows_falls_back_to_url_when_no_doi() -> None:
    machine_json = {
        "references": [
            {"pmid": "90000003", "doi": None, "vancouver": "X.", "url": "https://example.org/x"}
        ]
    }
    rows = reference_rows(machine_json)
    assert rows[0]["doi_url"] == "https://example.org/x"


def test_reference_rows_handles_missing_references() -> None:
    assert reference_rows({}) == []
