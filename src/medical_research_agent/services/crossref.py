"""Async CrossRef client for DOI / citation enrichment.

Failures are non-fatal: enrichment returns the study unchanged when CrossRef has
no match, so a missing DOI never breaks the pipeline.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, cast

import httpx

from medical_research_agent.config import Settings, get_settings
from medical_research_agent.logging_config import get_logger
from medical_research_agent.models.study import Study

_CROSSREF_BASE = "https://api.crossref.org"

# Below this title-similarity ratio, a CrossRef hit is treated as a non-match
# rather than risk attaching the wrong DOI/citation data to a study.
_TITLE_MATCH_THRESHOLD = 0.6

log = get_logger("crossref")


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


class CrossRefService:
    """Best-effort enrichment via the CrossRef REST API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        headers = {"User-Agent": f"medical-research-agent (mailto:{self.settings.crossref_mailto})"}
        self._client = httpx.AsyncClient(base_url=_CROSSREF_BASE, timeout=30.0, headers=headers)

    async def enrich(self, study: Study) -> Study:
        """Return a copy of ``study`` enriched with DOI/citations.

        Best-effort: any HTTP failure, malformed response, or low-confidence
        title match returns ``study`` unchanged rather than raising.
        """
        if not study.title:
            return study

        try:
            response = await self._client.get(
                "/works", params={"query.bibliographic": study.title, "rows": 1}
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("crossref.enrich_failed", pmid=study.pmid, error=str(exc))
            return study

        item = self._best_match(data, study.title)
        if item is None:
            return study

        return study.model_copy(
            update={
                "doi": item.get("DOI"),
                "citation_count": item.get("is-referenced-by-count"),
                "publisher": item.get("publisher"),
                "url": item.get("URL"),
            }
        )

    def _best_match(self, data: Any, study_title: str) -> dict[str, Any] | None:
        items = data.get("message", {}).get("items", [])
        if not items:
            return None

        item = items[0]
        doi = item.get("DOI")
        titles = item.get("title") or []
        if not doi or not titles:
            return None

        if _title_similarity(study_title, titles[0]) < _TITLE_MATCH_THRESHOLD:
            return None

        return cast(dict[str, Any], item)

    async def aclose(self) -> None:
        await self._client.aclose()
