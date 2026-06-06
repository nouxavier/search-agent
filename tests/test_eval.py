"""DoD da E4 (precisa de Postgres+pgvector — via fixture `session`):

- Governance (§7.5): deletar um paper apaga, EM CASCATA, tudo que aponta pra ele —
  embedding (coluna do próprio paper), aliases, sources, run_papers, edges, feedback.
- Metric stack (§5): as contagens batem com o estado da memória.
"""

from __future__ import annotations

from sqlalchemy import text

from search_agent.db.models import Feedback
from search_agent.embeddings import FakeEmbedder
from search_agent.eval.ablation import run_ablation
from search_agent.eval.metrics import compute_metrics
from search_agent.memory.write_path import create_run, ingest, link_to_run
from search_agent.sources.base import RawPaper

EMB = FakeEmbedder(1024)


def _raw(arxiv_id, *, title, author="Ada Lovelace"):
    return RawPaper(
        title=title, abstract=f"abstract {title}", authors=[author], year=2026,
        doi=None, source_ids={"arxiv_id": arxiv_id}, source_name="arxiv",
    )


def _count(session, sql, **p) -> int:
    return int(session.execute(text(sql), p).scalar() or 0)


def test_governance_delete_cascade(session):
    # Dois papers do mesmo autor → aresta same_author; um deles ganha feedback e run.
    a = ingest(session, EMB, _raw("4001.001", title="Analytical engine notes"), min_year=2000)
    b = ingest(session, EMB, _raw("4001.002", title="On computable numbers"), min_year=2000)
    run = create_run(session, "AI", {})
    link_to_run(session, run.id, a, rank=1)
    session.add(Feedback(paper_id=a, run_id=run.id, signal="up"))
    session.flush()

    # Pré-condição: há registros filhos apontando pro paper A.
    assert _count(session, "SELECT count(*) FROM external_ids WHERE paper_id=:i", i=a) > 0
    assert _count(session, "SELECT count(*) FROM edges WHERE src_id=:i OR dst_id=:i", i=a) == 1
    assert _count(session, "SELECT count(*) FROM feedback WHERE paper_id=:i", i=a) == 1

    # Deleção no nível do banco — dispara os ON DELETE CASCADE das FKs.
    session.execute(text("DELETE FROM papers WHERE id=:i"), {"i": a})
    session.flush()

    # Tudo que era do A sumiu (incl. o embedding, que é coluna do papers).
    assert _count(session, "SELECT count(*) FROM papers WHERE id=:i", i=a) == 0
    assert _count(session, "SELECT count(*) FROM external_ids WHERE paper_id=:i", i=a) == 0
    assert _count(session, "SELECT count(*) FROM sources WHERE paper_id=:i", i=a) == 0
    assert _count(session, "SELECT count(*) FROM run_papers WHERE paper_id=:i", i=a) == 0
    assert _count(session, "SELECT count(*) FROM edges WHERE src_id=:i OR dst_id=:i", i=a) == 0
    assert _count(session, "SELECT count(*) FROM feedback WHERE paper_id=:i", i=a) == 0
    # O outro paper sobrevive.
    assert _count(session, "SELECT count(*) FROM papers WHERE id=:i", i=b) == 1


def test_metrics_count_state(session):
    a = ingest(session, EMB, _raw("4002.001", title="paper one"), min_year=2000)
    ingest(session, EMB, _raw("4002.002", title="paper two", author="Alan Turing"), min_year=2000)
    run = create_run(session, "AI", {})
    link_to_run(session, run.id, a, rank=1)
    session.add(Feedback(paper_id=a, run_id=run.id, signal="up"))
    session.flush()

    m = compute_metrics(session)
    assert m.papers == 2
    assert m.surfaced == 1          # só A foi linkado a um run
    assert m.up_star == 1
    assert m.task_effectiveness == 1.0   # 1 de 1 surfaceado virou up
    assert m.repeated == 0          # nada re-surfaceado (G2)


def test_metrics_down_signal(session):
    a = ingest(session, EMB, _raw("4003.001", title="useless retrieval"), min_year=2000)
    run = create_run(session, "AI", {})
    link_to_run(session, run.id, a, rank=1)
    session.add(Feedback(paper_id=a, run_id=run.id, signal="down"))
    session.flush()

    m = compute_metrics(session)
    assert m.down == 1
    assert m.up_star == 0
    assert m.down_rate == 1.0


def test_ablation_runs_with_ground_truth(session):
    a = ingest(session, EMB, _raw("4004.001", title="retrieval augmented generation"), min_year=2000)
    run = create_run(session, "AI", {})
    link_to_run(session, run.id, a, rank=1)
    session.add(Feedback(paper_id=a, run_id=run.id, signal="star"))
    session.flush()

    ab = run_ablation(session, EMB, "retrieval augmented generation", k=10)
    assert ab.relevant_total == 1
    assert ab.found_on >= 1          # o paper relevante aparece no ranking
