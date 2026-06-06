"""Testes da identidade canônica (sem DB)."""

from __future__ import annotations

from search_agent.memory.identity import canonical_key, last_name, normalize_doi, normalize_title
from search_agent.sources.base import RawPaper


def _raw(**kw) -> RawPaper:
    base = dict(
        title="Memory for Autonomous LLM Agents",
        abstract="...",
        authors=["Pengfei Du", "Jane Roe"],
        year=2026,
        doi=None,
        source_ids={"arxiv_id": "2603.07670"},
        source_name="arxiv",
    )
    base.update(kw)
    return RawPaper(**base)


def test_normalize_doi_strips_prefixes_and_lowercases():
    assert normalize_doi("https://doi.org/10.1234/AbCd") == "10.1234/abcd"
    assert normalize_doi("doi:10.1234/abcd") == "10.1234/abcd"
    assert normalize_doi("  10.1234/ABCD ") == "10.1234/abcd"


def test_normalize_title_collapses_punctuation_and_case():
    assert normalize_title("Memory for  Autonomous, LLM-Agents!") == "memory for autonomous llm agents"


def test_last_name():
    assert last_name("Pengfei Du") == "du"
    assert last_name(None) == ""


def test_doi_wins_when_present():
    key = canonical_key(_raw(doi="10.1234/abcd"))
    assert key == "doi:10.1234/abcd"


def test_same_doi_different_source_same_key():
    a = _raw(doi="10.1/x", source_ids={"arxiv_id": "2601.001"}, source_name="arxiv")
    b = _raw(doi="https://doi.org/10.1/X", source_ids={"s2_id": "abc"}, source_name="s2")
    assert canonical_key(a) == canonical_key(b)  # dedup cross-source pela chave


def test_no_doi_falls_back_to_hash_and_is_stable():
    a = _raw(doi=None, title="Memory for Autonomous LLM Agents")
    b = _raw(doi=None, title="memory for   autonomous llm agents!!")  # mesma essência
    assert canonical_key(a).startswith("h:")
    assert canonical_key(a) == canonical_key(b)


def test_different_papers_get_different_keys():
    a = _raw(doi=None, title="Paper A")
    b = _raw(doi=None, title="Paper B")
    assert canonical_key(a) != canonical_key(b)
