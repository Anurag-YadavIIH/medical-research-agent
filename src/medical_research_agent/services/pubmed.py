"""Async client for NCBI E-utilities (PubMed).

esearch resolves a query to a list of PMIDs; efetch fetches and parses the
full records. ``search`` combines both behind the :class:`SearchFilters`
abstraction used by the rest of the pipeline.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime

import httpx
from lxml import etree
from tenacity import retry, stop_after_attempt, wait_exponential

from medical_research_agent.config import Settings, get_settings
from medical_research_agent.models.query import SearchFilters
from medical_research_agent.models.study import Study

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# NCBI rate limits: 3 req/sec without an API key, 10 req/sec with one.
_RATE_NO_KEY = 3.0
_RATE_WITH_KEY = 10.0


class PubMedService:
    """Thin async wrapper over NCBI E-utilities."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = httpx.AsyncClient(base_url=_EUTILS_BASE, timeout=30.0)
        rate = _RATE_WITH_KEY if self.settings.ncbi_api_key else _RATE_NO_KEY
        self._min_interval = 1.0 / rate
        self._last_request_at = 0.0
        self._rate_lock = asyncio.Lock()

    def _common_params(self) -> dict[str, str]:
        params = {"tool": self.settings.ncbi_tool, "email": self.settings.ncbi_email}
        if self.settings.ncbi_api_key:
            params["api_key"] = self.settings.ncbi_api_key
        return params

    async def _throttle(self) -> None:
        async with self._rate_lock:
            elapsed = time.monotonic() - self._last_request_at
            wait = self._min_interval - elapsed
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_at = time.monotonic()

    def _build_term(self, query: str, filters: SearchFilters) -> str:
        clauses = [f"({query})"]

        if filters.year_min or filters.year_max:
            year_min = filters.year_min or 1900
            year_max = filters.year_max or datetime.now(UTC).year
            clauses.append(
                f'("{year_min}/01/01"[Date - Publication] : "{year_max}/12/31"[Date - Publication])'
            )

        if filters.article_types:
            type_clause = " OR ".join(f'"{t}"[Publication Type]' for t in filters.article_types)
            clauses.append(f"({type_clause})")

        if filters.humans_only:
            clauses.append('"humans"[MeSH Terms]')

        return " AND ".join(clauses)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def esearch(self, query: str, retmax: int) -> list[str]:
        """Return a list of PMIDs for a query."""
        await self._throttle()
        params = {
            **self._common_params(),
            "db": "pubmed",
            "term": query,
            "retmax": str(retmax),
            "retmode": "json",
        }
        response = await self._client.get("/esearch.fcgi", params=params)
        response.raise_for_status()
        data = response.json()
        return list(data.get("esearchresult", {}).get("idlist", []))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def efetch(self, pmids: list[str]) -> list[Study]:
        """Fetch and parse study records for PMIDs."""
        if not pmids:
            return []
        await self._throttle()
        params = {
            **self._common_params(),
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        response = await self._client.get("/efetch.fcgi", params=params)
        response.raise_for_status()
        return self._parse_articles(response.content)

    def _parse_articles(self, xml_bytes: bytes) -> list[Study]:
        root = etree.fromstring(xml_bytes)
        return [self._parse_article(article) for article in root.findall(".//PubmedArticle")]

    def _parse_article(self, article: etree._Element) -> Study:
        pmid = article.findtext(".//MedlineCitation/PMID") or ""
        title = article.findtext(".//Article/ArticleTitle") or ""
        journal = article.findtext(".//Article/Journal/Title") or ""

        year_text = article.findtext(".//Article/Journal/JournalIssue/PubDate/Year")
        if not year_text:
            medline_date = article.findtext(".//Article/Journal/JournalIssue/PubDate/MedlineDate")
            year_text = medline_date[:4] if medline_date else None
        publication_year = int(year_text) if year_text and year_text.isdigit() else None

        authors: list[str] = []
        for author in article.findall(".//AuthorList/Author"):
            last_name = author.findtext("LastName")
            initials = author.findtext("Initials")
            collective_name = author.findtext("CollectiveName")
            if last_name:
                authors.append(f"{last_name} {initials}" if initials else last_name)
            elif collective_name:
                authors.append(collective_name)

        abstract_parts: list[str] = []
        for abstract_text in article.findall(".//Article/Abstract/AbstractText"):
            text = "".join(abstract_text.itertext()).strip()
            if not text:
                continue
            label = abstract_text.get("Label")
            abstract_parts.append(f"{label}: {text}" if label else text)

        publication_types = [
            pt.text for pt in article.findall(".//PublicationTypeList/PublicationType") if pt.text
        ]

        return Study(
            pmid=pmid,
            title=title,
            authors=authors,
            journal=journal,
            publication_year=publication_year,
            abstract="\n".join(abstract_parts),
            publication_types=publication_types,
        )

    async def search(self, query: str, filters: SearchFilters) -> list[Study]:
        """High-level search: build the query with filters, esearch, then efetch."""
        term = self._build_term(query, filters)
        pmids = await self.esearch(term, filters.max_papers)
        return await self.efetch(pmids)

    async def aclose(self) -> None:
        await self._client.aclose()
