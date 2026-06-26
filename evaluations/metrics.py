"""The four evaluation metrics, as pure functions over a run's machine_json.

Each function takes plain data (no I/O, no LLM, no network) so it can be
unit-tested with crafted inputs, including adversarial ones.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from evaluations.models import MetricResult
from medical_research_agent.agents.evidence_evaluator import deterministic_level
from medical_research_agent.models.evidence import EvidenceLevel

_PMID_CITATION = re.compile(r"\[PMID:\s*([\w.-]+)\]")
_PVALUE = re.compile(r"p\s*[<>=]\s*0?\.\d+")
_SAMPLE_SIZE_MENTION = re.compile(
    r"\b(\d{2,6})\b\s*(?:participants|patients|eyes|subjects|individuals)\b", re.IGNORECASE
)

# extraction_accuracy matching rule: exact (normalized) for structured/short
# fields, fuzzy-normalized for free text. See _FUZZY_THRESHOLD.
_EXACT_FIELDS = ("study_design",)
_NUMERIC_FIELDS = ("sample_size",)
_FUZZY_FIELDS = (
    "population",
    "intervention",
    "comparator",
    "main_findings",
    "statistical_significance",
    "sample_size_description",
)
_FUZZY_THRESHOLD = 0.6

_NARRATIVE_FIELDS = (
    "evidence_summary",
    "clinical_implications",
    "limitations",
    "future_directions",
)


def _normalize(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _narrative_text(machine_json: dict[str, Any]) -> str:
    return "\n".join(str(machine_json.get(field, "")) for field in _NARRATIVE_FIELDS)


def _cited_pmids(machine_json: dict[str, Any]) -> set[str]:
    return set(_PMID_CITATION.findall(_narrative_text(machine_json)))


def citation_completeness(
    machine_json: dict[str, Any], retrieved_pmids: set[str], retrieved_dois: set[str]
) -> MetricResult:
    """Every PMID/DOI cited in the narrative or listed as a reference must exist
    in the run's retrieved studies. Score = (cited and present) / cited. A cited
    ID absent from the retrieved set is a hard failure — by construction this
    makes it impossible to score 1.0 while citing a fabricated ID.
    """
    cited_pmids = _cited_pmids(machine_json) | {
        ref.get("pmid") for ref in machine_json.get("references") or [] if ref.get("pmid")
    }
    cited_dois = {ref.get("doi") for ref in machine_json.get("references") or [] if ref.get("doi")}

    cited = cited_pmids | cited_dois
    if not cited:
        return MetricResult(
            name="citation_completeness",
            score=1.0,
            details={"cited_count": 0, "note": "No citations were made; vacuously complete."},
        )

    present = retrieved_pmids | retrieved_dois
    fabricated = sorted(cited - present)
    matched = len(cited) - len(fabricated)

    failures = [f"Cited ID not in retrieved set: {pmid_or_doi}" for pmid_or_doi in fabricated]
    return MetricResult(
        name="citation_completeness",
        score=matched / len(cited),
        details={"cited_count": len(cited), "matched_count": matched},
        failures=failures,
    )


def extraction_accuracy(
    gold_extractions: dict[str, dict[str, Any]], model_extractions: dict[str, dict[str, Any]]
) -> MetricResult:
    """Compare extracted fields against the labelled gold set, per (pmid, field).

    Matching rule:
      - study_design: exact match, case/whitespace-normalized.
      - sample_size: exact integer match.
      - population, intervention, comparator, main_findings,
        statistical_significance, sample_size_description: fuzzy match via
        difflib.SequenceMatcher ratio on normalized text, threshold >= 0.6.
    A field is only scored when gold provides a non-empty label for it — there's
    nothing to grade against an unlabelled field. If gold has a label and the
    model's value is empty or wrong, that's a scored miss, not a skip.
    """
    failures: list[str] = []
    per_field_matches: dict[str, int] = {}
    per_field_totals: dict[str, int] = {}
    matched = 0
    total = 0

    for pmid, gold in gold_extractions.items():
        model = model_extractions.get(pmid, {})

        for field in _EXACT_FIELDS:
            gold_value = gold.get(field)
            if not gold_value:
                continue
            total += 1
            per_field_totals[field] = per_field_totals.get(field, 0) + 1
            model_value = model.get(field) or ""
            if _normalize(str(gold_value)) == _normalize(str(model_value)):
                matched += 1
                per_field_matches[field] = per_field_matches.get(field, 0) + 1
            else:
                failures.append(
                    f"PMID {pmid} field '{field}': expected {gold_value!r}, got {model_value!r}"
                )

        for field in _NUMERIC_FIELDS:
            gold_value = gold.get(field)
            if gold_value is None:
                continue
            total += 1
            per_field_totals[field] = per_field_totals.get(field, 0) + 1
            model_value = model.get(field)
            if gold_value == model_value:
                matched += 1
                per_field_matches[field] = per_field_matches.get(field, 0) + 1
            else:
                failures.append(
                    f"PMID {pmid} field '{field}': expected {gold_value!r}, got {model_value!r}"
                )

        for field in _FUZZY_FIELDS:
            gold_value = gold.get(field)
            if not gold_value:
                continue
            total += 1
            per_field_totals[field] = per_field_totals.get(field, 0) + 1
            model_value = model.get(field) or ""
            ratio = _fuzzy_ratio(str(gold_value), str(model_value))
            if ratio >= _FUZZY_THRESHOLD:
                matched += 1
                per_field_matches[field] = per_field_matches.get(field, 0) + 1
            else:
                failures.append(
                    f"PMID {pmid} field '{field}': expected {gold_value!r}, got {model_value!r} "
                    f"(similarity={ratio:.2f} < {_FUZZY_THRESHOLD})"
                )

    score = matched / total if total else 1.0
    per_field_scores = {
        field: per_field_matches.get(field, 0) / per_field_totals[field]
        for field in per_field_totals
    }
    return MetricResult(
        name="extraction_accuracy",
        score=score,
        details={"matched": matched, "total": total, "per_field": per_field_scores},
        failures=failures,
    )


def evidence_consistency(
    assessments: list[dict[str, Any]], studies: list[dict[str, Any]]
) -> MetricResult:
    """Assigned evidence_level must agree with the deterministic publication_types
    map (reusing the real deterministic_level() from evidence_evaluator.py — not
    a re-implementation). Since the map is deterministic, this should be ~100%
    on real data; a true mismatch is a hard failure (a pipeline bug). Separately,
    any study whose publication_types the map can't classify (UNGRADED) is
    surfaced as a coverage gap, not silently passed.
    """
    publication_types_by_pmid = {
        study.get("pmid"): study.get("publication_types") or [] for study in studies
    }

    matched = 0
    total = 0
    failures: list[str] = []
    gaps: list[str] = []

    for assessment in assessments:
        pmid = assessment.get("pmid")
        assigned = assessment.get("evidence_level")
        publication_types = publication_types_by_pmid.get(pmid, [])
        expected = deterministic_level(publication_types)

        total += 1
        if assigned == int(expected):
            matched += 1
        else:
            failures.append(
                f"PMID {pmid}: assigned evidence_level={assigned} but publication_types "
                f"{publication_types} deterministically map to {int(expected)} ({expected.label})"
            )

        if expected == EvidenceLevel.UNGRADED:
            gaps.append(
                f"PMID {pmid}: publication_types {publication_types} aren't handled by the "
                "deterministic map (UNGRADED) — consider extending it"
            )

    score = matched / total if total else 1.0
    return MetricResult(
        name="evidence_consistency",
        score=score,
        details={"matched": matched, "total": total},
        failures=failures,
        gaps=gaps,
    )


def hallucination_check(
    machine_json: dict[str, Any], retrieved_pmids: set[str], retrieved_dois: set[str]
) -> MetricResult:
    """Broader anti-hallucination sweep: every PMID/DOI anywhere in the output
    (references, comparison's strongest/conflicting PMIDs, narrative citations)
    must be in the retrieved set, AND every numeric claim that looks like a
    sample size or p-value in the narrative must trace to some extracted
    study's sample_size or statistical_significance text. An untraceable
    numeric claim is flagged, not silently passed.
    """
    failures: list[str] = []

    ids_used: set[str] = _cited_pmids(machine_json)
    for ref in machine_json.get("references") or []:
        if ref.get("pmid"):
            ids_used.add(ref["pmid"])
        if ref.get("doi"):
            ids_used.add(ref["doi"])

    comparison = machine_json.get("comparison") or {}
    ids_used |= set(comparison.get("strongest_evidence_pmids") or [])
    ids_used |= set(comparison.get("conflicting_evidence_pmids") or [])

    present = retrieved_pmids | retrieved_dois
    fabricated_ids = sorted(ids_used - present)
    failures += [f"ID not in retrieved set: {value}" for value in fabricated_ids]

    extracted = machine_json.get("extracted") or []
    sample_sizes = {
        str(item.get("sample_size")) for item in extracted if item.get("sample_size") is not None
    }
    significance_texts = " ".join(
        str(item.get("statistical_significance") or "") for item in extracted
    )

    narrative = _narrative_text(machine_json)
    numeric_claims = list(_SAMPLE_SIZE_MENTION.finditer(narrative))
    untraceable_claims: list[str] = []
    for match in numeric_claims:
        if match.group(1) not in sample_sizes:
            untraceable_claims.append(match.group(0))

    pvalue_claims = _PVALUE.findall(narrative)
    for claim in pvalue_claims:
        if _normalize(claim) not in _normalize(significance_texts):
            untraceable_claims.append(claim)

    failures += [
        f"Untraceable numeric claim in narrative: '{claim}'" for claim in untraceable_claims
    ]

    total_checks = len(ids_used) + len(numeric_claims) + len(pvalue_claims)
    total_failures = len(fabricated_ids) + len(untraceable_claims)
    score = (total_checks - total_failures) / total_checks if total_checks else 1.0

    return MetricResult(
        name="hallucination_check",
        score=score,
        details={
            "ids_checked": len(ids_used),
            "numeric_claims_checked": len(numeric_claims) + len(pvalue_claims),
        },
        failures=failures,
    )
