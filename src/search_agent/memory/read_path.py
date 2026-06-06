"""Read path (RFC §7.2): candidate gen (vetor kNN + metadata) → rank by profile (E2).

`relevance ≠ similarity` (§4.2): o kNN é só o começo. Quando há user_profile ativo,
reordenamos os candidatos misturando a similaridade à consulta com a afinidade ao
perfil — é assim que a memória semântica influencia o ranking. (Graph expand entra na E3.)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from ..db.queries import Hit, search_similar
from ..embeddings import Embedder
from ..observability.events import READ, record_event
from .consolidate import active_profile, affinity, embeddings_for

# Peso do perfil na mistura final (0 = só similaridade à consulta).
PROFILE_WEIGHT = 0.35


@dataclass(frozen=True)
class RankedHit:
    hit: Hit
    base_sim: float       # 1 - distância de cosseno à consulta
    profile_affinity: float
    score: float          # mistura final usada para ordenar


def recall(
    session: Session,
    embedder: Embedder,
    query_text: str,
    *,
    k: int = 10,
    area: str | None = None,
    exclude_seen: bool = False,
    use_profile: bool = True,
    now: datetime | None = None,
) -> list[RankedHit]:
    """'O que já vi sobre X?' — kNN reordenado pelo perfil semântico (E2)."""
    qvec = embedder.embed([query_text])[0]
    # Pool maior que k pra o re-rank ter o que reordenar.
    pool = search_similar(session, qvec, k=max(k * 3, 30), area=area, exclude_seen=exclude_seen)
    if not pool:
        return []

    profile = active_profile(session, embedder, now=now) if use_profile else []
    embs = embeddings_for(session, [h.paper_id for h in pool]) if profile else {}

    ranked: list[RankedHit] = []
    for h in pool:
        base = 1.0 - h.distance
        aff = affinity(embs.get(h.paper_id), profile) if profile else 0.0
        score = base if not profile else (1 - PROFILE_WEIGHT) * base + PROFILE_WEIGHT * aff
        ranked.append(RankedHit(hit=h, base_sim=base, profile_affinity=aff, score=score))

    ranked.sort(key=lambda r: r.score, reverse=True)
    # E5: registra a leitura (consulta, tamanho do pool, se o perfil entrou).
    record_event(
        session, READ, "recall",
        {"query": query_text[:120], "k": k, "pool": len(pool), "profile": bool(profile)},
    )
    return ranked[:k]


def rerank_by_profile(
    session: Session,
    embedder: Embedder,
    paper_ids: list[int],
    *,
    now: datetime | None = None,
) -> list[int]:
    """Ordena uma lista de paper_ids (ex.: o digest) por afinidade ao perfil.
    Sem perfil ativo, devolve a ordem original (estável)."""
    profile = active_profile(session, embedder, now=now)
    if not profile:
        return list(paper_ids)
    embs = embeddings_for(session, paper_ids)
    return sorted(paper_ids, key=lambda pid: affinity(embs.get(pid), profile), reverse=True)
