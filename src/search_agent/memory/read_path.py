"""Read path (RFC §7.2): embeda a consulta e recupera do histórico.

Na E1 é só o candidate gen (vetor kNN + metadata). Graph expand (E3) e rank by
profile (E2) entram depois — por isso `recall` já existe como ponto único onde
essas etapas vão se somar.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from ..db.queries import Hit, search_similar
from ..embeddings import Embedder


def recall(
    session: Session,
    embedder: Embedder,
    query_text: str,
    *,
    k: int = 10,
    area: str | None = None,
    exclude_seen: bool = False,
) -> list[Hit]:
    """'O que já vi sobre X?' — papers do store mais próximos da consulta."""
    qvec = embedder.embed([query_text])[0]
    return search_similar(session, qvec, k=k, area=area, exclude_seen=exclude_seen)
