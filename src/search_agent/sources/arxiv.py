"""Adapter arXiv: API Atom (feedparser) → RawPaper.

A normalização (`normalize_entry`) é uma função pura sobre uma entrada já
parseada, separada da chamada HTTP — assim o teste de normalização do adapter
roda sem rede (DoD da E0).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Iterable

import feedparser
import httpx

from .base import RawPaper

ARXIV_API = "https://export.arxiv.org/api/query"

# IDs arXiv modernos: "2603.07670" ou "2603.07670v2", opcionalmente com prefixo
# de categoria no estilo antigo ("cs.AI/0601001"). Pegamos o trecho final.
_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?$")


def _extract_arxiv_id(entry_id: str) -> str | None:
    """De 'http://arxiv.org/abs/2603.07670v1' tira '2603.07670' (sem versão)."""
    if not entry_id:
        return None
    tail = entry_id.rstrip("/").split("/")[-1]
    m = _ARXIV_ID_RE.search(tail)
    if m:
        return m.group(1)
    # IDs antigos com categoria: 'cs.AI/0601001'
    return tail or None


def _year_from_published(published: str | None) -> int | None:
    if not published:
        return None
    # Formato Atom: '2026-03-15T17:00:00Z'
    m = re.match(r"(\d{4})-", published)
    return int(m.group(1)) if m else None


def _clean(text: str | None) -> str | None:
    if text is None:
        return None
    # arXiv quebra abstract/título em múltiplas linhas; colapsa em espaços.
    return re.sub(r"\s+", " ", text).strip() or None


def normalize_entry(entry: Any) -> RawPaper:
    """Converte uma entrada feedparser do arXiv num RawPaper canônico."""
    authors = [
        _clean(a.get("name", "")) or ""
        for a in getattr(entry, "authors", []) or []
    ]
    authors = [a for a in authors if a]

    arxiv_id = _extract_arxiv_id(getattr(entry, "id", "") or "")
    source_ids: dict[str, str] = {}
    if arxiv_id:
        source_ids["arxiv_id"] = arxiv_id

    # arXiv às vezes expõe DOI em arxiv_doi (quando o autor publicou em journal).
    doi = _clean(getattr(entry, "arxiv_doi", None))

    return RawPaper(
        title=_clean(getattr(entry, "title", "")) or "",
        abstract=_clean(getattr(entry, "summary", None)),
        authors=authors,
        year=_year_from_published(getattr(entry, "published", None)),
        doi=doi,
        source_ids=source_ids,
        source_name="arxiv",
    )


def _build_query(area: str, categories: list[str]) -> str:
    """Monta a query da API arXiv: termo da área + OR das categorias."""
    cat_clause = " OR ".join(f"cat:{c}" for c in categories)
    term = area.strip().replace('"', "")
    parts = []
    if term:
        parts.append(f'(abs:"{term}" OR ti:"{term}")')
    if cat_clause:
        parts.append(f"({cat_clause})")
    return " AND ".join(parts) if parts else "all:*"


class ArxivSource:
    """Source para o arXiv via API Atom."""

    name = "arxiv"

    def __init__(
        self,
        categories: list[str] | None = None,
        *,
        max_results: int = 10,
        client: httpx.Client | None = None,
    ) -> None:
        self.categories = categories or ["cs.AI", "cs.CL", "cs.LG"]
        self.max_results = max_results
        self._client = client

    def fetch(self, area: str, since: datetime | None = None) -> Iterable[RawPaper]:
        params = {
            "search_query": _build_query(area, self.categories),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": str(self.max_results),
        }
        client = self._client or httpx.Client(timeout=30.0, follow_redirects=True)
        try:
            resp = client.get(ARXIV_API, params=params)
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
        finally:
            if self._client is None:
                client.close()

        for entry in feed.entries:
            paper = normalize_entry(entry)
            if since is not None and paper.year is not None and paper.year < since.year:
                continue
            yield paper
