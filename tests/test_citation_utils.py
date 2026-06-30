"""Tests for the shared anti-fabrication citation utility."""

from __future__ import annotations

from medical_research_agent.citation_utils import (
    strip_fabricated_citations,
    strip_fabricated_doc_citations,
)


def test_keeps_citations_for_valid_pmids() -> None:
    text = "Symptoms improved [PMID: 123] and recurred [PMID: 456]."
    cleaned, fabricated = strip_fabricated_citations(text, {"123", "456"})

    assert cleaned == text
    assert fabricated == []


def test_strips_citations_for_unknown_pmids() -> None:
    text = "Symptoms improved [PMID: 123] and recurred [PMID: 999]."
    cleaned, fabricated = strip_fabricated_citations(text, {"123"})

    assert cleaned == "Symptoms improved [PMID: 123] and recurred ."
    assert fabricated == ["999"]


def test_no_citations_is_a_no_op() -> None:
    cleaned, fabricated = strip_fabricated_citations("No citations here.", {"123"})

    assert cleaned == "No citations here."
    assert fabricated == []


def test_doc_citations_keeps_valid_and_strips_unknown() -> None:
    text = "Per the abstract [DOC: abc-123], outcomes improved [DOC: zzz-999]."
    cleaned, fabricated = strip_fabricated_doc_citations(text, {"abc-123"})

    assert cleaned == "Per the abstract [DOC: abc-123], outcomes improved ."
    assert fabricated == ["zzz-999"]
