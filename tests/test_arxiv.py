"""Teste de normalização do adapter arXiv (DoD da E0).

Parseia um Atom de exemplo (sem rede) e verifica que o RawPaper sai canônico:
id sem versão, abstract colapsado, ano extraído, autores na ordem.
"""

from __future__ import annotations

import feedparser

from search_agent.sources.arxiv import _build_query, _extract_arxiv_id, normalize_entry

SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2603.07670v2</id>
    <published>2026-03-15T17:00:00Z</published>
    <title>Memory for Autonomous
      LLM Agents</title>
    <summary>  We survey memory mechanisms
      for LLM agents.  </summary>
    <author><name>Pengfei Du</name></author>
    <author><name>Jane Roe</name></author>
    <arxiv:doi>10.1234/abcd.2026</arxiv:doi>
  </entry>
</feed>
"""


def _entry():
    return feedparser.parse(SAMPLE_ATOM).entries[0]


def test_normalize_basic_fields():
    paper = normalize_entry(_entry())
    assert paper.title == "Memory for Autonomous LLM Agents"  # multilinha colapsada
    assert paper.abstract == "We survey memory mechanisms for LLM agents."  # trim + colapso
    assert paper.year == 2026
    assert paper.authors == ["Pengfei Du", "Jane Roe"]
    assert paper.first_author == "Pengfei Du"
    assert paper.source_name == "arxiv"


def test_normalize_ids_and_doi():
    paper = normalize_entry(_entry())
    # arxiv_id sem o sufixo de versão (v2)
    assert paper.source_ids["arxiv_id"] == "2603.07670"
    assert paper.doi == "10.1234/abcd.2026"


def test_embed_text_joins_title_and_abstract():
    paper = normalize_entry(_entry())
    assert paper.embed_text == "Memory for Autonomous LLM Agents\nWe survey memory mechanisms for LLM agents."


def test_extract_arxiv_id_strips_version():
    assert _extract_arxiv_id("http://arxiv.org/abs/2603.07670v1") == "2603.07670"
    assert _extract_arxiv_id("http://arxiv.org/abs/2603.07670") == "2603.07670"


def test_extract_arxiv_id_handles_empty():
    assert _extract_arxiv_id("") is None


def test_build_query_includes_term_and_categories():
    q = _build_query("LLM Agents", ["cs.AI", "cs.CL"])
    assert 'abs:"LLM Agents"' in q
    assert "cat:cs.AI OR cat:cs.CL" in q


def test_missing_abstract_yields_none_and_title_only_embed_text():
    atom = SAMPLE_ATOM.replace("<summary>  We survey memory mechanisms\n      for LLM agents.  </summary>", "")
    entry = feedparser.parse(atom).entries[0]
    paper = normalize_entry(entry)
    assert paper.abstract is None
    assert paper.embed_text == "Memory for Autonomous LLM Agents"
