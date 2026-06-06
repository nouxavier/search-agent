"""Metric stack da E4 (RFC §5). Quatro dimensões, lidas direto do estado da memória.

- Task effectiveness: % de papers surfaceados que viraram feedback up/star.
- Memory quality: taxa de repetição indevida (G2: um paper não deve re-surfacear)
  e taxa de 'down' (recuperado mas inútil).
- Efficiency: tamanho do store, digest médio por run, nº de arestas. (Latência por
  op e tokens/run não são lidos aqui — latência o `metrics` mede ao vivo; tokens
  ficam pra E5/memory_events, ainda não instrumentado.)
- Governance: é um TESTE (deletar paper apaga tudo + embedding), não uma métrica de
  runtime — veja tests/test_eval.py::test_governance_delete_cascade.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session


@dataclass(frozen=True)
class Metrics:
    papers: int                 # tamanho do store
    runs: int                   # runs que surfacearam algo
    surfaced: int               # papers distintos já surfaceados
    edges: int                  # arestas no grafo (E3)
    avg_digest: float           # papers por run, em média
    up_star: int                # papers distintos com feedback up/star
    down: int                   # papers distintos com feedback down
    repeated: int               # papers surfaceados em >1 run (repetição indevida)

    @property
    def task_effectiveness(self) -> float:
        return self.up_star / self.surfaced if self.surfaced else 0.0

    @property
    def down_rate(self) -> float:
        return self.down / self.surfaced if self.surfaced else 0.0

    @property
    def repeat_rate(self) -> float:
        return self.repeated / self.surfaced if self.surfaced else 0.0


def _scalar(session: Session, sql: str) -> int:
    return int(session.execute(text(sql)).scalar() or 0)


def compute_metrics(session: Session) -> Metrics:
    surfaced = _scalar(session, "SELECT count(DISTINCT paper_id) FROM run_papers")
    runs_with_digest = _scalar(session, "SELECT count(DISTINCT run_id) FROM run_papers")
    total_links = _scalar(session, "SELECT count(*) FROM run_papers")
    return Metrics(
        papers=_scalar(session, "SELECT count(*) FROM papers"),
        runs=runs_with_digest,
        surfaced=surfaced,
        edges=_scalar(session, "SELECT count(*) FROM edges"),
        avg_digest=(total_links / runs_with_digest) if runs_with_digest else 0.0,
        up_star=_scalar(
            session,
            "SELECT count(DISTINCT paper_id) FROM feedback WHERE signal IN ('up','star')",
        ),
        down=_scalar(
            session, "SELECT count(DISTINCT paper_id) FROM feedback WHERE signal='down'"
        ),
        repeated=_scalar(
            session,
            """
            SELECT count(*) FROM (
                SELECT paper_id FROM run_papers
                GROUP BY paper_id HAVING count(DISTINCT run_id) > 1
            ) t
            """,
        ),
    )
