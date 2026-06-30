"""Text extraction for uploaded paper PDFs (project chat's other content source).

Bounded by page count and extracted-character count so a pathological or
oversized PDF can't blow up embedding cost/time — mirrors the bounded-input
posture already established for ``question``/``keywords`` (see
``api/schemas.py``) and the per-IP rate limiter (``api/rate_limit.py``).
"""

from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader

MAX_PAGES = 200
MAX_CHARS = 200_000


class PdfExtractionError(ValueError):
    """Raised when a PDF can't be parsed or exceeds the bounds above."""


def extract_text(file_bytes: bytes) -> str:
    """Extract plain text from a PDF's pages, bounded by ``MAX_PAGES``/``MAX_CHARS``."""
    try:
        reader = PdfReader(BytesIO(file_bytes))
    except Exception as exc:  # noqa: BLE001 - any parse failure is a client-facing 422
        raise PdfExtractionError(f"Could not read PDF: {exc}") from exc

    if len(reader.pages) > MAX_PAGES:
        raise PdfExtractionError(
            f"PDF has {len(reader.pages)} pages, which exceeds the {MAX_PAGES}-page limit."
        )

    pages_text: list[str] = []
    total_chars = 0
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        total_chars += len(text)
        # Reject rather than silently truncate — consistent with the page-count
        # bound above. A silently truncated paper would otherwise leave a
        # corrupted/incomplete corpus entry the user has no way to know about.
        if total_chars > MAX_CHARS:
            raise PdfExtractionError(
                f"PDF extracted text exceeds the {MAX_CHARS:,}-character limit."
            )
        pages_text.append(text)

    extracted = "\n\n".join(pages_text)
    if not extracted.strip():
        raise PdfExtractionError(
            "No extractable text found in this PDF (it may be scanned/image-only)."
        )
    return extracted
