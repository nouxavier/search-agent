"""Memory diff (RFC §7.7): o que mudou no store entre dois runs.

Tipo um `git diff`, mas da memória. A janela é o intervalo de tempo entre os dois
runs; dentro dela, o que nasceu (papers, arestas, preferências) e o que morreu
(preferências que expiraram). Útil pra responder "o que essa semana de runs mudou
no que o agente sabe?".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.orm import Session


@dataclass(frozen=True)
class MemoryDiff:
    earlier_run: int
    later_run: int
    papers_added: list[tuple[int, str]] = field(default_factory=list)
    edges_added: int = 0
    profile_added: list[str] = field(default_factory=list)
    profile_expired: list[str] = field(default_factory=list)
    events_by_op: dict[str, int] = field(default_factory=dict)


def _run_ts(session: Session, run_id: int):
    ts = session.execute(text("SELECT ts FROM runs WHERE id=:i"), {"i": run_id}).scalar()
    if ts is None:
        raise ValueError(f"run {run_id} não existe")
    return ts


def memory_diff(session: Session, run_a: int, run_b: int) -> MemoryDiff:
    """Diff do store na janela entre os dois runs (ordem não importa — ordenamos por ts)."""
    ta, tb = _run_ts(session, run_a), _run_ts(session, run_b)
    (earlier, e_ts), (later, l_ts) = sorted(
        [(run_a, ta), (run_b, tb)], key=lambda x: x[1]
    )
    win = {"lo": e_ts, "hi": l_ts}

    papers = session.execute(
        text(
            "SELECT id, title FROM papers WHERE created_at > :lo AND created_at <= :hi "
            "ORDER BY id"
        ),
        win,
    ).all()
    edges = session.execute(
        text("SELECT count(*) FROM edges WHERE created_at > :lo AND created_at <= :hi"), win
    ).scalar()
    prof_added = session.execute(
        text(
            "SELECT statement FROM user_profile WHERE created_at > :lo AND created_at <= :hi"
        ),
        win,
    ).scalars().all()
    prof_expired = session.execute(
        text(
            "SELECT statement FROM user_profile WHERE expires_at > :lo AND expires_at <= :hi"
        ),
        win,
    ).scalars().all()
    events = session.execute(
        text(
            "SELECT op, count(*) FROM memory_events WHERE ts > :lo AND ts <= :hi GROUP BY op"
        ),
        win,
    ).all()

    return MemoryDiff(
        earlier_run=earlier,
        later_run=later,
        papers_added=[(r[0], r[1]) for r in papers],
        edges_added=int(edges or 0),
        profile_added=list(prof_added),
        profile_expired=list(prof_expired),
        events_by_op={op: int(n) for op, n in events},
    )
