"""Tests for PDF text extraction — pypdf itself is mocked out; this tests our
bounding/joining/error-handling logic around it, not pypdf's own parsing.
"""

from __future__ import annotations

from typing import Any

import pytest

from medical_research_agent.services import pdf_extraction


class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakeReader:
    def __init__(self, pages: list[_FakePage]) -> None:
        self.pages = pages


def _patch_reader(monkeypatch: pytest.MonkeyPatch, pages: list[_FakePage]) -> None:
    monkeypatch.setattr(pdf_extraction, "PdfReader", lambda _file: _FakeReader(pages))


def test_extract_text_joins_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_reader(monkeypatch, [_FakePage("Page one."), _FakePage("Page two.")])

    text = pdf_extraction.extract_text(b"fake-pdf-bytes")

    assert text == "Page one.\n\nPage two."


def test_extract_text_skips_blank_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_reader(monkeypatch, [_FakePage(""), _FakePage("Content."), _FakePage("   ")])

    text = pdf_extraction.extract_text(b"fake-pdf-bytes")

    assert text == "Content."


def test_extract_text_raises_on_no_extractable_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_reader(monkeypatch, [_FakePage(""), _FakePage("   ")])

    with pytest.raises(pdf_extraction.PdfExtractionError, match="No extractable text"):
        pdf_extraction.extract_text(b"fake-pdf-bytes")


def test_extract_text_rejects_too_many_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = [_FakePage("x") for _ in range(pdf_extraction.MAX_PAGES + 1)]
    _patch_reader(monkeypatch, pages)

    with pytest.raises(pdf_extraction.PdfExtractionError, match="page limit"):
        pdf_extraction.extract_text(b"fake-pdf-bytes")


def test_extract_text_rejects_when_exceeding_max_chars(monkeypatch: pytest.MonkeyPatch) -> None:
    huge_page = _FakePage("x" * (pdf_extraction.MAX_CHARS + 1000))
    _patch_reader(monkeypatch, [huge_page])

    with pytest.raises(pdf_extraction.PdfExtractionError, match="character limit"):
        pdf_extraction.extract_text(b"fake-pdf-bytes")


def test_extract_text_accepts_content_right_at_max_chars(monkeypatch: pytest.MonkeyPatch) -> None:
    exact_page = _FakePage("x" * pdf_extraction.MAX_CHARS)
    _patch_reader(monkeypatch, [exact_page])

    text = pdf_extraction.extract_text(b"fake-pdf-bytes")

    assert len(text) == pdf_extraction.MAX_CHARS


def test_extract_text_raises_on_unparseable_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_file: Any) -> None:
        raise ValueError("not a pdf")

    monkeypatch.setattr(pdf_extraction, "PdfReader", _raise)

    with pytest.raises(pdf_extraction.PdfExtractionError, match="Could not read PDF"):
        pdf_extraction.extract_text(b"not-actually-a-pdf")
