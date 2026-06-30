"""Pure functions turning a /research response into display-ready data.

No Streamlit, no HTTP — just shaping ``machine_json`` for rendering, so this
module can be unit-tested without the Streamlit runtime or a live backend. The
frontend container doesn't install the backend package (see
docker/frontend.Dockerfile), so this intentionally has zero dependency on
``medical_research_agent`` — including duplicating the evidence-level label
text, which must be kept in sync with ``models/evidence.py`` by hand.
"""

from __future__ import annotations

from typing import Any

_EVIDENCE_LEVEL_LABELS: dict[int, str] = {
    1: "Level I — Systematic review / meta-analysis",
    2: "Level II — Randomized controlled trial",
    3: "Level III — Cohort study",
    4: "Level IV — Case-control study",
    5: "Level V — Case report / expert opinion",
    99: "Ungraded",
}


def evidence_level_label(value: int | None) -> str:
    if value is None:
        return "Ungraded"
    return _EVIDENCE_LEVEL_LABELS.get(value, "Ungraded")


def comparison_matrix_rows(machine_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows for the Study Comparison tab's dataframe."""
    comparison = machine_json.get("comparison")
    if not comparison:
        return []
    return list(comparison.get("comparison_matrix") or [])


def comparison_narrative(machine_json: dict[str, Any]) -> dict[str, list[str]]:
    """Agreements/disagreements/trends for the Study Comparison tab's bullet lists."""
    comparison = machine_json.get("comparison") or {}
    return {
        "agreements": list(comparison.get("agreements") or []),
        "disagreements": list(comparison.get("disagreements") or []),
        "trends": list(comparison.get("trends") or []),
    }


def study_detail_rows(machine_json: dict[str, Any]) -> list[dict[str, Any]]:
    """One merged record per study (Study + ExtractedStudy + EvidenceAssessment,
    joined by pmid) for the Study Details tab's expanders.
    """
    extracted_by_pmid = {item["pmid"]: item for item in machine_json.get("extracted") or []}
    assessment_by_pmid = {item["pmid"]: item for item in machine_json.get("assessments") or []}

    rows: list[dict[str, Any]] = []
    for study in machine_json.get("studies") or []:
        pmid = study.get("pmid", "")
        doi = study.get("doi")
        extracted = extracted_by_pmid.get(pmid, {})
        assessment = assessment_by_pmid.get(pmid, {})
        rows.append(
            {
                "pmid": pmid,
                "title": study.get("title", ""),
                "authors": study.get("authors") or [],
                "journal": study.get("journal", ""),
                "publication_year": study.get("publication_year"),
                "abstract": study.get("abstract", ""),
                "pmid_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                "doi_url": f"https://doi.org/{doi}" if doi else (study.get("url") or ""),
                "objective": extracted.get("objective", ""),
                "population": extracted.get("population", ""),
                "intervention": extracted.get("intervention", ""),
                "comparator": extracted.get("comparator", ""),
                "outcomes": extracted.get("outcomes") or [],
                "main_findings": extracted.get("main_findings", ""),
                "statistical_significance": extracted.get("statistical_significance", ""),
                "limitations": extracted.get("limitations", ""),
                "evidence_level": evidence_level_label(assessment.get("evidence_level")),
                "strength": assessment.get("strength", ""),
                "bias_risk": assessment.get("bias_risk", ""),
                "confidence_reasoning": assessment.get("confidence_reasoning", ""),
            }
        )
    return rows


def evidence_summary_stats(machine_json: dict[str, Any]) -> dict[str, Any]:
    """At-a-glance counts for the Evidence Summary tab, derived purely from
    already-returned ``machine_json`` — no new claims, just aggregation.
    """
    studies = machine_json.get("studies") or []
    assessments = machine_json.get("assessments") or []
    references = machine_json.get("references") or []

    level_counts: dict[str, int] = {}
    levels_present: list[int] = []
    for assessment in assessments:
        level = assessment.get("evidence_level")
        level_counts[evidence_level_label(level)] = (
            level_counts.get(evidence_level_label(level), 0) + 1
        )
        if level is not None and level != 99:
            levels_present.append(level)

    years = sorted({s.get("publication_year") for s in studies if s.get("publication_year")})

    return {
        "total_studies": len(studies),
        "total_references": len(references),
        "level_counts": level_counts,
        # Evidence levels are ranked 1 (strongest) through 5 (weakest); 99 = ungraded.
        "strongest_level_label": (
            evidence_level_label(min(levels_present)) if levels_present else "—"
        ),
        "year_range": f"{years[0]}–{years[-1]}" if years else "—",
    }


def reference_rows(machine_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows for the References tab, with PubMed/DOI links resolved."""
    rows: list[dict[str, Any]] = []
    for ref in machine_json.get("references") or []:
        pmid = ref.get("pmid", "")
        doi = ref.get("doi")
        rows.append(
            {
                "vancouver": ref.get("vancouver", ""),
                "pmid": pmid,
                "pmid_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                "doi": doi,
                "doi_url": f"https://doi.org/{doi}" if doi else (ref.get("url") or ""),
            }
        )
    return rows
