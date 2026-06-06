"""Substrate relacional da E3 (RFC §3.2 / §5.5 / §9.2).

Liga papers por RELAÇÃO, não por similaridade de embedding. Duas pontas:
- write path: `populate_edges` cria arestas do paper novo pro que já existe.
- read path: `relational_neighbors` faz a travessia (1 hop) a partir dos
  candidatos e devolve vizinhos relacionais — a ponte que o kNN puro não dá.

Tipos de aresta:
- same_author  — match exato de first_author (limite do schema: só o 1º autor).
- same_subarea — vizinhança de embedding abaixo de SUBAREA_MAX_DIST (o "cluster"
  do §3.2). Densifica o grafo; sozinho não satisfaz o DoD (redundante com o kNN).
- cites        — direcionada e mais valiosa (§9.2), mas adiada: o feed Atom do
  arXiv não traz referências. Entra quando uma fonte fornecer (Semantic Scholar).

same_author é o que entrega o DoD: relação ortogonal ao embedding, então surfacea
um paper que a busca vetorial não traria.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..logging_setup import get_logger

log = get_logger("search_agent.graph")

# Corte de distância de cosseno pra same_subarea: < este valor = "mesma subárea".
# Apertado de propósito (≈ sim > 0.6) pra não ligar tudo com tudo.
SUBAREA_MAX_DIST = 0.40

# Prioridade ao escolher UMA relação pra mostrar por paper (cites > autor > subárea).
_KIND_RANK = {"cites": 3, "same_author": 2, "same_subarea": 1}


# ── WRITE: popula arestas do paper novo ─────────────────────────────────────

# Canoniza (src<dst) pra aresta não-direcionada não duplicar nos dois sentidos.
_SAME_AUTHOR_SQL = text(
    """
    INSERT INTO edges (src_id, dst_id, kind, weight)
    SELECT LEAST(:pid, o.id), GREATEST(:pid, o.id), 'same_author', 1.0
    FROM papers o, papers p
    WHERE p.id = :pid AND o.id <> :pid
      AND p.first_author IS NOT NULL AND o.first_author = p.first_author
    ON CONFLICT DO NOTHING
    """
)

_SAME_SUBAREA_SQL = text(
    """
    INSERT INTO edges (src_id, dst_id, kind, weight)
    SELECT LEAST(:pid, o.id), GREATEST(:pid, o.id), 'same_subarea',
           (1.0 - (o.embedding <=> p.embedding))::real
    FROM papers o, papers p
    WHERE p.id = :pid AND p.embedding IS NOT NULL
      AND o.id <> :pid AND o.embedding IS NOT NULL
      AND (o.embedding <=> p.embedding) < :max_dist
    ON CONFLICT DO NOTHING
    """
)


def populate_edges(session: Session, paper_id: int) -> int:
    """Cria as arestas do paper recém-persistido pro resto do store. Idempotente
    (ON CONFLICT DO NOTHING). Retorna quantas arestas novas entraram."""
    n_author = session.execute(_SAME_AUTHOR_SQL, {"pid": paper_id}).rowcount
    n_sub = session.execute(
        _SAME_SUBAREA_SQL, {"pid": paper_id, "max_dist": SUBAREA_MAX_DIST}
    ).rowcount
    # cites: adiado (sem fonte de referências) — §3.2 "quando disponível".
    if n_author or n_sub:
        log.info(
            "graph.edges",
            extra={"paper_id": paper_id, "same_author": n_author, "same_subarea": n_sub},
        )
    return n_author + n_sub


# ── READ: travessia de 1 hop a partir dos candidatos ────────────────────────


@dataclass(frozen=True)
class Relation:
    seed_id: int          # candidato do read path (origem da ponte)
    neighbor_id: int      # paper relacionado encontrado pela travessia
    neighbor_title: str
    kind: str
    weight: float


# Aresta não-direcionada: o vizinho é o "outro lado" em relação ao seed.
_EXPAND_SQL = text(
    """
    SELECT e.kind, e.weight, p.title,
           CASE WHEN e.src_id = ANY(:seeds) THEN e.src_id ELSE e.dst_id END AS seed_id,
           CASE WHEN e.src_id = ANY(:seeds) THEN e.dst_id ELSE e.src_id END AS neighbor_id
    FROM edges e
    JOIN papers p
      ON p.id = CASE WHEN e.src_id = ANY(:seeds) THEN e.dst_id ELSE e.src_id END
    WHERE e.src_id = ANY(:seeds) OR e.dst_id = ANY(:seeds)
    """
)


def relational_neighbors(
    session: Session,
    seed_ids: Iterable[int],
    candidate_ids: Iterable[int],
) -> dict[int, Relation]:
    """Para cada seed, a MELHOR relação cujo vizinho está em `candidate_ids`.

    `candidate_ids` é tipicamente o conjunto 'já visto em runs anteriores' — é o que
    dá o "se conecta a X que você viu há 3 semanas". Escolhe uma relação por seed
    (cites > same_author > same_subarea, desempate por peso)."""
    seeds = list(seed_ids)
    if not seeds:
        return {}
    candidates = set(candidate_ids)
    rows = session.execute(_EXPAND_SQL, {"seeds": seeds}).all()

    best: dict[int, tuple[tuple[int, float], Relation]] = {}
    for kind, weight, title, seed_id, neighbor_id in rows:
        if neighbor_id not in candidates:
            continue
        score = (_KIND_RANK.get(kind, 0), float(weight))
        cur = best.get(seed_id)
        if cur is None or score > cur[0]:
            best[seed_id] = (score, Relation(seed_id, neighbor_id, title, kind, float(weight)))
    return {seed_id: rel for seed_id, (_, rel) in best.items()}
