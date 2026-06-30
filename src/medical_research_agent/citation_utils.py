"""Shared anti-fabrication citation handling.

Used by both the Summary Agent (narrative report, cites ``[PMID: <id>]``) and
the project chat service (NotebookLM-style Q&A, cites ``[DOC: <id>]`` since
not every source has a PMID — an uploaded PDF doesn't). In both cases the LLM
is told to cite only ids explicitly given in its context, and anything it
writes outside that allowed set is stripped before it ever reaches a user —
citation safety is enforced in code, not just by prompting.
"""

from __future__ import annotations

import re

PMID_CITATION = re.compile(r"\[PMID:\s*([\w.-]+)\]")
DOC_CITATION = re.compile(r"\[DOC:\s*([\w.-]+)\]")


def _strip_citations(
    text: str, valid_ids: set[str], pattern: re.Pattern[str]
) -> tuple[str, list[str]]:
    fabricated: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        cited_id = match.group(1)
        if cited_id in valid_ids:
            return match.group(0)
        fabricated.append(cited_id)
        return ""

    return pattern.sub(_replace, text), fabricated


def strip_fabricated_citations(text: str, valid_pmids: set[str]) -> tuple[str, list[str]]:
    """Remove any ``[PMID: x]`` citation whose id isn't in ``valid_pmids``.

    Returns the cleaned text and the list of fabricated ids that were dropped.
    """
    return _strip_citations(text, valid_pmids, PMID_CITATION)


def strip_fabricated_doc_citations(text: str, valid_doc_ids: set[str]) -> tuple[str, list[str]]:
    """Remove any ``[DOC: x]`` citation whose id isn't in ``valid_doc_ids``.

    Returns the cleaned text and the list of fabricated ids that were dropped.
    """
    return _strip_citations(text, valid_doc_ids, DOC_CITATION)
