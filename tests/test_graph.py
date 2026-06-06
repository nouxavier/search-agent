"""DoD da E3 (precisa de Postgres+pgvector — via fixture `session`):

O read path retorna ≥1 vizinho relacional que a busca vetorial pura NÃO traria —
"este paper se conecta a X que você viu há 3 semanas".

A prova: dois papers do mesmo autor mas de tópicos sem relação. Os embeddings ficam
longe (o kNN não os juntaria), mas a aresta same_author os conecta — e é ela que
`relational_neighbors` surfacea.
"""

from __future__ import annotations

from sqlalchemy import select

from search_agent.db.models import Edge
from search_agent.db.queries import search_similar
from search_agent.embeddings import FakeEmbedder
from search_agent.memory.graph import SUBAREA_MAX_DIST, populate_edges, relational_neighbors
from search_agent.memory.write_path import ingest
from search_agent.sources.base import RawPaper

EMB = FakeEmbedder(1024)


def _raw(arxiv_id, *, title, author="Ada Lovelace", year=2026):
    return RawPaper(
        title=title,
        abstract=f"abstract: {title}",
        authors=[author],
        year=year,
        doi=None,
        source_ids={"arxiv_id": arxiv_id},
        source_name="arxiv",
    )


def test_relational_neighbor_surfaces_distant_paper(session):
    # Mesmo autor, tópicos sem relação → embeddings distantes, mas ligados por autoria.
    a = _raw("3001.001", title="Lattice gauge theory of quark confinement")
    b = _raw("3001.002", title="Sourdough fermentation kinetics in rye flour")
    id_a = ingest(session, EMB, a, min_year=2000)
    id_b = ingest(session, EMB, b, min_year=2000)
    session.flush()

    # 1. A aresta same_author existe e está canônica (src < dst).
    edge = session.execute(
        select(Edge).where(Edge.kind == "same_author")
    ).scalar_one()
    assert {edge.src_id, edge.dst_id} == {id_a, id_b}
    assert edge.src_id < edge.dst_id

    # 2. O kNN NÃO juntaria os dois: B está bem além do corte de "mesma subárea".
    qa = EMB.embed([a.embed_text])[0]
    dist_b = next(h.distance for h in search_similar(session, qa, k=10) if h.paper_id == id_b)
    assert dist_b > SUBAREA_MAX_DIST  # logo, não há aresta same_subarea entre eles

    # 3. DoD: a travessia traz B como vizinho relacional de A (B já visto antes).
    notes = relational_neighbors(session, [id_a], candidate_ids=[id_b])
    assert id_a in notes
    assert notes[id_a].neighbor_id == id_b
    assert notes[id_a].kind == "same_author"


def test_relational_neighbors_filters_by_candidates(session):
    # Só conta como ponte quem está no conjunto de candidatos (ex.: 'já visto').
    a = _raw("3002.001", title="Topological insulators and edge states")
    b = _raw("3002.002", title="Beekeeping and colony collapse disorder")
    id_a = ingest(session, EMB, a, min_year=2000)
    id_b = ingest(session, EMB, b, min_year=2000)
    session.flush()

    assert relational_neighbors(session, [id_a], candidate_ids=[id_b])  # B é candidato → ponte
    assert relational_neighbors(session, [id_a], candidate_ids=[]) == {}  # ninguém candidato → nada


def test_no_edges_between_unrelated_papers(session):
    # Autores diferentes + tópicos diferentes → nenhuma aresta (nem autoria, nem subárea).
    a = _raw("3003.001", title="Riemann hypothesis and the zeta function", author="Bernhard Riemann")
    b = _raw("3003.002", title="Photosynthesis in C4 plants", author="Melvin Calvin")
    ingest(session, EMB, a, min_year=2000)
    ingest(session, EMB, b, min_year=2000)
    session.flush()

    assert session.execute(select(Edge)).first() is None


def test_same_subarea_fires_for_near_identical(session):
    # Mesmo conteúdo (embedding idêntico), autores distintos → liga por subárea, não autoria.
    title = "Retrieval augmented generation survey"
    a = RawPaper(title=title, abstract="x", authors=["A One"], year=2026,
                 doi=None, source_ids={"arxiv_id": "3004.001"}, source_name="arxiv")
    b = RawPaper(title=title, abstract="x", authors=["B Two"], year=2025,  # ano difere → chave difere
                 doi=None, source_ids={"arxiv_id": "3004.002"}, source_name="arxiv")
    id_a = ingest(session, EMB, a, min_year=2000)
    id_b = ingest(session, EMB, b, min_year=2000)
    session.flush()
    assert id_a != id_b  # não fundiram

    kinds = set(session.execute(select(Edge.kind)).scalars())
    assert "same_subarea" in kinds
    assert "same_author" not in kinds


def test_populate_edges_is_idempotent(session):
    a = _raw("3005.001", title="Graph neural networks for molecules")
    b = _raw("3005.002", title="Graph theory in social networks")  # mesmo autor (Lovelace)
    id_a = ingest(session, EMB, a, min_year=2000)
    ingest(session, EMB, b, min_year=2000)
    session.flush()

    before = session.execute(select(Edge)).all()
    populate_edges(session, id_a)  # roda de novo → ON CONFLICT DO NOTHING
    after = session.execute(select(Edge)).all()
    assert len(before) == len(after)
