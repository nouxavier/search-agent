"""DoD da E1 (precisa de Postgres+pgvector — via fixture `session`):

1. Dois papers com arxiv_id diferentes mas mesmo DOI fundem num registro.
2. Segundo run da mesma área não re-surfacea o que já apareceu.
+ read path responde "o que já vi sobre X?".
"""

from __future__ import annotations

from sqlalchemy import select

from search_agent.db.models import ExternalId, Paper, Source
from search_agent.db.queries import search_similar
from search_agent.embeddings import FakeEmbedder
from search_agent.memory.read_path import recall
from search_agent.memory.write_path import (
    count_papers,
    create_run,
    ingest,
    link_to_run,
    previously_seen_ids,
)
from search_agent.sources.base import RawPaper

EMB = FakeEmbedder(1024)


def _raw(arxiv_id, *, doi=None, title="A Paper on Memory", source="arxiv", year=2026):
    return RawPaper(
        title=title,
        abstract="some abstract about agent memory",
        authors=["Ada Lovelace"],
        year=year,
        doi=doi,
        source_ids={"arxiv_id": arxiv_id} if source == "arxiv" else {"s2_id": arxiv_id},
        source_name=source,
    )


def test_same_doi_different_arxiv_merges(session):
    # DoD 1: mesma DOI, arxiv_ids distintos, fontes distintas → UM paper, aliases todos.
    a = _raw("2601.001", doi="10.1/x", source="arxiv")
    b = _raw("9999.999", doi="https://doi.org/10.1/X", source="s2")  # mesmo DOI normalizado

    id_a = ingest(session, EMB, a, min_year=2000)
    id_b = ingest(session, EMB, b, min_year=2000)
    session.flush()

    assert id_a == id_b
    assert count_papers(session) == 1

    aliases = set(
        session.execute(select(ExternalId.kind, ExternalId.value).where(ExternalId.paper_id == id_a)).all()
    )
    assert ("arxiv_id", "2601.001") in aliases
    assert ("s2_id", "9999.999") in aliases
    assert ("doi", "10.1/x") in aliases

    sources = set(session.execute(select(Source.source_name).where(Source.paper_id == id_a)).scalars())
    assert sources == {"arxiv", "s2"}


def test_filter_drops_low_signal(session):
    no_abstract = RawPaper(
        title="No abstract", abstract=None, authors=["X"], year=2026,
        doi=None, source_ids={"arxiv_id": "2601.555"}, source_name="arxiv",
    )
    assert ingest(session, EMB, no_abstract, min_year=2000) is None
    old = _raw("2601.556", year=2000)
    assert ingest(session, EMB, old, min_year=2023) is None
    assert count_papers(session) == 0


def test_embedding_written_once(session):
    raw = _raw("2601.010")
    pid = ingest(session, EMB, raw, min_year=2000)
    session.flush()
    emb = session.execute(select(Paper.embedding).where(Paper.id == pid)).scalar_one()
    assert emb is not None and len(emb) == 1024


def test_second_run_does_not_resurface(session):
    # DoD 2: paper surfaceado no run 1 não volta no digest do run 2.
    raw = _raw("2601.777")
    run1 = create_run(session, "AI", {})
    pid1 = ingest(session, EMB, raw, min_year=2000)
    link_to_run(session, run1.id, pid1, rank=1)
    session.flush()

    run2 = create_run(session, "AI", {})
    pid2 = ingest(session, EMB, raw, min_year=2000)  # mesmo paper_key → mesmo id
    assert pid2 == pid1

    seen = previously_seen_ids(session, exclude_run_id=run2.id)
    assert pid1 in seen  # já foi visto no run1 → fora do digest do run2

    digest = [pid2] if pid2 not in seen else []
    assert digest == []


def test_recall_finds_seen_paper(session):
    raw = _raw("2601.020", title="Retrieval augmented generation for agents")
    ingest(session, EMB, raw, min_year=2000)
    session.flush()
    # recall acha o paper certo no topo ("o que já vi sobre X?")
    hits = recall(session, EMB, "retrieval augmented generation for agents", k=5)
    assert hits
    assert hits[0].title == "Retrieval augmented generation for agents"
    # FakeEmbedder é determinístico: consultar com o MESMO texto embedado (título+
    # abstract) → distância ~0. Prova que o vetor gravado é o que a busca recupera.
    hits_exact = recall(session, EMB, raw.embed_text, k=5)
    assert hits_exact[0].distance < 1e-6


def test_exclude_seen_in_search(session):
    raw = _raw("2601.030", title="seen paper")
    run = create_run(session, "AI", {})
    pid = ingest(session, EMB, raw, min_year=2000)
    link_to_run(session, run.id, pid, rank=1)
    session.flush()
    qvec = EMB.embed(["seen paper"])[0]
    assert search_similar(session, qvec, k=5, exclude_seen=True) == []
    assert search_similar(session, qvec, k=5, exclude_seen=False)  # sem o filtro, aparece
