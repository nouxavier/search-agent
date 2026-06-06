"""Consolidação episodic → semantic (RFC §9.1) + perfil para re-rank.

Statements propostos pela reflexão viram/atualizam o user_profile:
- confidence sobe com repetição e com feedback (paper marcado [HIGHLIGHT]);
- expires_at dá validade — uma reflexão errada não cristaliza (§4.3).

O perfil ativo (não expirado) re-ranqueia o digest/consulta: a afinidade de um
paper é a média das similaridades às statements, ponderada por confidence.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..db.models import Paper, UserProfile
from ..embeddings import Embedder
from ..observability.events import UPDATE, record_event
from .reflect import ProposedStatement

NEW_BASE = 0.5
HIGHLIGHT_BONUS = 0.15  # feedback move o statement
REPEAT_STEP = 0.15  # confidence sobe com repetição
CONF_CAP = 0.95
DEFAULT_TTL_DAYS = 30

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(timezone.utc)


def statement_key(statement: str) -> str:
    """Texto normalizado p/ dedup de consolidação (mesma essência → mesmo statement)."""
    return _NON_ALNUM.sub(" ", statement.lower()).strip()


def consolidate(
    session: Session,
    proposed: list[ProposedStatement],
    *,
    highlighted_ids: set[int] | None = None,
    now: datetime | None = None,
    ttl_days: int = DEFAULT_TTL_DAYS,
) -> None:
    """Funde os statements propostos no user_profile. Idempotente por statement_key."""
    highlighted_ids = highlighted_ids or set()
    now = _now(now)
    expires = now + timedelta(days=ttl_days)

    for ps in proposed:
        key = statement_key(ps.statement)
        has_feedback = any(pid in highlighted_ids for pid in ps.evidence_ids)
        existing = session.execute(
            select(UserProfile).where(UserProfile.statement_key == key)
        ).scalar_one_or_none()

        if existing is None:
            conf = NEW_BASE + (HIGHLIGHT_BONUS if has_feedback else 0.0)
            session.add(
                UserProfile(
                    statement=ps.statement,
                    statement_key=key,
                    evidence_ids=sorted(set(ps.evidence_ids)),
                    confidence=min(conf, CONF_CAP),
                    expires_at=expires,
                )
            )
            action = "new"
        else:
            bump = REPEAT_STEP + (HIGHLIGHT_BONUS if has_feedback else 0.0)
            existing.confidence = min(CONF_CAP, existing.confidence + bump)
            existing.evidence_ids = sorted(set(existing.evidence_ids) | set(ps.evidence_ids))
            existing.expires_at = expires  # refresh: continua vivo
            action = "reinforced"
        # E5: registra a mudança no perfil (o que entrou/subiu, e por quê).
        record_event(
            session, UPDATE, "user_profile",
            {"action": action, "feedback": has_feedback, "evidence": str(ps.evidence_ids)},
        )
    session.flush()


# ── Perfil ativo + afinidade (re-rank) ──────────────────────────────────────


@dataclass(frozen=True)
class ActiveStatement:
    statement: str
    confidence: float
    embedding: list[float]


def active_profile(
    session: Session, embedder: Embedder, *, now: datetime | None = None
) -> list[ActiveStatement]:
    """Statements não expirados, já com embedding (computado on-the-fly)."""
    now = _now(now)
    rows = (
        session.execute(
            select(UserProfile).where(
                or_(UserProfile.expires_at.is_(None), UserProfile.expires_at > now)
            )
        )
        .scalars()
        .all()
    )
    if not rows:
        return []
    embs = embedder.embed([r.statement for r in rows])
    return [
        ActiveStatement(statement=r.statement, confidence=float(r.confidence), embedding=e)
        for r, e in zip(rows, embs)
    ]


def _cosine(a, b) -> float:
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    return dot / (na * nb) if na and nb else 0.0


def affinity(paper_embedding, profile: list[ActiveStatement]) -> float:
    """Afinidade do paper com o perfil: média (sobre statements) de confidence × similaridade.

    Normaliza por nº de statements (não pela soma das confidences) de propósito — assim
    a confidence pesa em magnitude e mover uma statement por feedback muda o ranking."""
    if paper_embedding is None or not profile:
        return 0.0
    return sum(s.confidence * _cosine(paper_embedding, s.embedding) for s in profile) / len(profile)


def embeddings_for(session: Session, paper_ids: list[int]) -> dict[int, list[float]]:
    """Embeddings dos papers (via ORM/Vector), pra calcular afinidade no re-rank."""
    if not paper_ids:
        return {}
    rows = session.execute(
        select(Paper.id, Paper.embedding).where(Paper.id.in_(paper_ids))
    ).all()
    return {pid: (list(emb) if emb is not None else None) for pid, emb in rows}
