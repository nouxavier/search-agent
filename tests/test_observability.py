"""DoD da E5 (precisa de Postgres+pgvector — via fixture `session`):

- Toda op de memória deixa rastro em memory_events (write no ingest, read no recall).
- memory_diff localiza o que mudou no store entre dois runs.
"""

from __future__ import annotations

from sqlalchemy import text

from search_agent.embeddings import FakeEmbedder
from search_agent.memory.read_path import recall
from search_agent.memory.write_path import create_run, ingest
from search_agent.observability.diff import memory_diff
from search_agent.observability.events import record_event
from search_agent.sources.base import RawPaper

EMB = FakeEmbedder(1024)


def _raw(arxiv_id, *, title):
    return RawPaper(
        title=title, abstract=f"abstract {title}", authors=["Ada Lovelace"], year=2026,
        doi=None, source_ids={"arxiv_id": arxiv_id}, source_name="arxiv",
    )


def _events(session, op=None):
    sql = "SELECT count(*) FROM memory_events" + (" WHERE op=:op" if op else "")
    return int(session.execute(text(sql), {"op": op} if op else {}).scalar() or 0)


def test_write_emits_event(session):
    ingest(session, EMB, _raw("5001.001", title="memory for agents"), min_year=2000)
    session.flush()
    assert _events(session, "write") == 1


def test_read_emits_event(session):
    ingest(session, EMB, _raw("5001.002", title="retrieval augmented generation"), min_year=2000)
    session.flush()
    recall(session, EMB, "retrieval augmented generation", k=5)
    assert _events(session, "read") == 1


def test_record_event_is_best_effort(session):
    # Contexto com tipo não-serializável (set) não derruba — vira string.
    record_event(session, "read", "recall", {"weird": {1, 2, 3}, "k": 5})
    row = session.execute(
        text("SELECT trigger_ctx FROM memory_events ORDER BY id DESC LIMIT 1")
    ).scalar()
    assert row["k"] == 5
    assert isinstance(row["weird"], str)  # serializado como string


def test_memory_diff_between_runs(session):
    # now() é o horário de INÍCIO da transação no Postgres — commitamos entre os
    # passos pra cada um cair numa transação distinta (como na vida real, em que
    # cada `agent run` é seu próprio session_scope). Sem isso, os timestamps
    # colidiriam e a janela do diff sairia vazia.
    run_a = create_run(session, "AI", {})
    session.commit()
    # Paper criado DEPOIS do run_a e ANTES do run_b → aparece no diff.
    ingest(session, EMB, _raw("5002.001", title="paper no meio"), min_year=2000)
    session.commit()
    run_b = create_run(session, "AI", {})
    session.commit()

    d = memory_diff(session, run_a.id, run_b.id)
    assert d.earlier_run == run_a.id and d.later_run == run_b.id
    titles = [t for _, t in d.papers_added]
    assert "paper no meio" in titles
    assert d.events_by_op.get("write", 0) >= 1  # o ingest gerou evento na janela


def test_memory_diff_unknown_run(session):
    run = create_run(session, "AI", {})
    session.flush()
    try:
        memory_diff(session, run.id, 999999)
        assert False, "deveria ter levantado ValueError"
    except ValueError:
        pass
