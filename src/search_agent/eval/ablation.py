"""Harness de ablation da E4 (RFC §5). DoD: número antes/depois atribuível a um
componente.

O componente isolado aqui é o **re-rank por perfil** (E2) — o toggle `use_profile`
do read path. A pergunta: ligar o perfil sobe os papers que o usuário marcou como
relevantes (up/star)?

Ground truth = papers com feedback up/star. Pra uma consulta, medimos a posição
média desses papers no ranking COM e SEM o perfil. Perfil ajudando ⇒ posição média
menor (mais perto do topo) com ele ligado.

Limite honesto: com pouco feedback o número é ruidoso — é um harness, não um
benchmark. Serve pra detectar regressão e atribuir ganho a um componente.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..embeddings import Embedder
from ..memory.read_path import recall


@dataclass(frozen=True)
class Ablation:
    query: str
    relevant_total: int          # papers up/star no store
    found_on: int                # quantos dos relevantes apareceram no top-k (perfil ON)
    found_off: int               # idem, perfil OFF
    mean_rank_on: float | None   # posição média dos relevantes encontrados (ON)
    mean_rank_off: float | None  # idem (OFF)

    @property
    def delta(self) -> float | None:
        """Quão melhor o perfil deixou a posição média (positivo = subiu no ranking)."""
        if self.mean_rank_on is None or self.mean_rank_off is None:
            return None
        return self.mean_rank_off - self.mean_rank_on


def _relevant_ids(session: Session) -> set[int]:
    rows = session.execute(
        text("SELECT DISTINCT paper_id FROM feedback WHERE signal IN ('up','star')")
    )
    return {r[0] for r in rows}


def _mean_rank(ranked, relevant: set[int]) -> tuple[int, float | None]:
    positions = [i for i, r in enumerate(ranked, start=1) if r.hit.paper_id in relevant]
    if not positions:
        return 0, None
    return len(positions), sum(positions) / len(positions)


def run_ablation(
    session: Session, embedder: Embedder, query: str, *, k: int = 20
) -> Ablation:
    relevant = _relevant_ids(session)
    on = recall(session, embedder, query, k=k, use_profile=True)
    off = recall(session, embedder, query, k=k, use_profile=False)
    found_on, mean_on = _mean_rank(on, relevant)
    found_off, mean_off = _mean_rank(off, relevant)
    return Ablation(
        query=query,
        relevant_total=len(relevant),
        found_on=found_on,
        found_off=found_off,
        mean_rank_on=mean_on,
        mean_rank_off=mean_off,
    )
