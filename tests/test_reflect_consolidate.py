"""DoD da E2 (precisa de Postgres — fixture `session`):

- reflection grounding: statement sem evidência real do run é descartado;
- feedback move um statement (confidence sobe) e o ranking reflete isso;
- expiração: statement vencido não entra no perfil ativo (anti self-reinforcing).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, text

from search_agent.db.models import Reflection, UserProfile
from search_agent.embeddings import FakeEmbedder
from search_agent.llm import FakeLLM
from search_agent.memory.consolidate import (
    CONF_CAP,
    active_profile,
    consolidate,
)
from search_agent.memory.read_path import rerank_by_profile
from search_agent.memory.reflect import ProposedStatement, reflect
from search_agent.memory.write_path import create_run, ingest, link_to_run
from search_agent.sources.base import RawPaper

EMB = FakeEmbedder(1024)


def _raw(arxiv_id, *, title, abstract="abstract here"):
    return RawPaper(
        title=title, abstract=abstract, authors=["Ada Lovelace"], year=2026,
        doi=None, source_ids={"arxiv_id": arxiv_id}, source_name="arxiv",
    )


def _run_with(session, papers, highlight_arxiv=None):
    run = create_run(session, "AI", {})
    ids = {}
    for rank, raw in enumerate(papers, start=1):
        pid = ingest(session, EMB, raw, min_year=2000)
        link_to_run(session, run.id, pid, rank=rank, was_highlight=(raw.source_ids["arxiv_id"] == highlight_arxiv))
        ids[raw.source_ids["arxiv_id"]] = pid
    session.flush()
    return run, ids


# ── Reflection grounding ────────────────────────────────────────────────────


def test_reflect_keeps_only_grounded_statements(session):
    raw = _raw("2606.00001", title="Efficient inference for LLMs")
    run, ids = _run_with(session, [raw])

    canned = json.dumps({
        "note": "Eficiência apareceu forte.",
        "statements": [
            {"statement": "valoriza eficiência de inferência", "evidence_arxiv_ids": ["2606.00001"]},  # grounded
            {"statement": "gosta de teoria de categorias", "evidence_arxiv_ids": ["9999.99999"]},      # id fora do run → descartado
        ],
    })
    proposed = reflect(session, FakeLLM(canned), run.id, model="x")

    assert [p.statement for p in proposed] == ["valoriza eficiência de inferência"]
    assert proposed[0].evidence_ids == [ids["2606.00001"]]
    refl = session.execute(select(Reflection).where(Reflection.run_id == run.id)).scalar_one()
    assert refl.grounded_ids == [ids["2606.00001"]]


def test_reflect_drops_everything_when_no_evidence(session):
    run, _ = _run_with(session, [_raw("2606.00002", title="Some paper")])
    canned = json.dumps({"note": "x", "statements": [
        {"statement": "preferência sem base", "evidence_arxiv_ids": ["0000.00000"]}
    ]})
    assert reflect(session, FakeLLM(canned), run.id, model="x") == []
    # nada persistido sem grounding
    assert session.execute(select(Reflection)).first() is None


def test_reflect_handles_garbage_json(session):
    run, _ = _run_with(session, [_raw("2606.00003", title="P")])
    assert reflect(session, FakeLLM("not json at all"), run.id, model="x") == []


# ── Consolidação: confidence + expiry ───────────────────────────────────────


def _stmt(text_, ev):
    return ProposedStatement(statement=text_, evidence_ids=ev)


def test_confidence_starts_and_rises_with_repetition(session):
    consolidate(session, [_stmt("valoriza eficiência", [1])])
    row = session.execute(select(UserProfile)).scalar_one()
    assert abs(row.confidence - 0.5) < 1e-6  # base

    consolidate(session, [_stmt("valoriza eficiência", [2])])  # repetição
    session.refresh(row)
    assert abs(row.confidence - 0.65) < 1e-6
    assert set(row.evidence_ids) == {1, 2}  # evidência acumula


def test_feedback_boosts_confidence(session):
    consolidate(session, [_stmt("prefere grafos", [10])], highlighted_ids={10})
    row = session.execute(select(UserProfile)).scalar_one()
    assert abs(row.confidence - 0.65) < 1e-6  # base + highlight


def test_confidence_capped(session):
    for _ in range(20):
        consolidate(session, [_stmt("x", [1])], highlighted_ids={1})
    row = session.execute(select(UserProfile)).scalar_one()
    assert row.confidence <= CONF_CAP


def test_expired_statements_excluded_from_active_profile(session):
    past = datetime.now(timezone.utc) - timedelta(days=1)
    consolidate(session, [_stmt("vencida", [1])], now=past, ttl_days=0)  # expira no passado
    consolidate(session, [_stmt("viva", [2])])  # expira no futuro
    active = active_profile(session, EMB)
    statements = {a.statement for a in active}
    assert "viva" in statements
    assert "vencida" not in statements


# ── Feedback move o statement → ranking reflete (DoD central) ────────────────


def test_feedback_moves_statement_and_reranks(session):
    a = _raw("2606.10001", title="Paper Alpha", abstract="alpha distinct text")
    b = _raw("2606.10002", title="Paper Beta", abstract="beta distinct text")
    _run, ids = _run_with(session, [a, b])

    # Perfil com dois statements cujo texto == embed_text de cada paper (determinístico
    # com FakeEmbedder): a afinidade de cada paper ao "seu" statement é máxima.
    sa = _stmt(a.embed_text, [ids["2606.10001"]])
    sb = _stmt(b.embed_text, [ids["2606.10002"]])

    # Feedback inicial em A → confidence de A > B; digest ranqueia A na frente.
    consolidate(session, [sa], highlighted_ids={ids["2606.10001"]})  # conf 0.65
    consolidate(session, [sb])                                       # conf 0.50
    order1 = rerank_by_profile(session, EMB, [ids["2606.10001"], ids["2606.10002"]])
    assert order1[0] == ids["2606.10001"]

    # Agora o feedback vai pra B (repetição + highlight) → conf de B passa A.
    consolidate(session, [sb], highlighted_ids={ids["2606.10002"]})  # 0.50 → 0.80
    order2 = rerank_by_profile(session, EMB, [ids["2606.10001"], ids["2606.10002"]])
    assert order2[0] == ids["2606.10002"]  # o ranking refletiu o feedback
