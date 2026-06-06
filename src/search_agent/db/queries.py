"""SQL cru de retrieval (RFC §8). Transparência > esperteza: a busca vetorial +
filtro de metadata é o que estamos internalizando, então fica em SQL explícito,
não atrás de abstração do ORM.

O operador `<=>` do pgvector é distância de cosseno (menor = mais parecido). O
vetor de consulta entra como literal '[...]' e é convertido com CAST(... AS vector)
— sem registrar adaptadores, o SQL fala por si.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session


@dataclass(frozen=True)
class Hit:
    paper_id: int
    title: str
    first_author: str | None
    year: int | None
    distance: float


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


# CANDIDATE GEN — vetor kNN + filtro de metadata no MESMO SELECT (RFC §8, passo 1).
_SEARCH_SQL = text(
    """
    SELECT p.id, p.title, p.first_author, p.year,
           (p.embedding <=> CAST(:qvec AS vector)) AS dist
    FROM papers p
    WHERE p.embedding IS NOT NULL
      AND (CAST(:area AS text) IS NULL OR p.title ILIKE '%' || :area || '%')
      AND (CAST(:exclude_seen AS boolean) = false
           OR p.id NOT IN (SELECT paper_id FROM run_papers))
    ORDER BY p.embedding <=> CAST(:qvec AS vector)
    LIMIT :k
    """
)


def search_similar(
    session: Session,
    qvec: list[float],
    *,
    k: int = 10,
    area: str | None = None,
    exclude_seen: bool = False,
) -> list[Hit]:
    rows = session.execute(
        _SEARCH_SQL,
        {"qvec": _vec_literal(qvec), "area": area, "exclude_seen": exclude_seen, "k": k},
    ).all()
    return [Hit(paper_id=r[0], title=r[1], first_author=r[2], year=r[3], distance=r[4]) for r in rows]
