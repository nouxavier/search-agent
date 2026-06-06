"""Recorder do event log (RFC §7.7). Uma chamada, uma linha em memory_events.

`record_event` é deliberadamente fino e tolerante: instrumentar a memória não pode
derrubar a operação que está sendo instrumentada. Por isso valores não-serializáveis
no contexto viram string, e a escrita não levanta pro chamador.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from ..db.models import MemoryEvent
from ..logging_setup import get_logger

log = get_logger("search_agent.observability")

# Operações canônicas (RFC §7.7). Não é enum no banco pra não travar evolução.
WRITE, READ, UPDATE, DELETE = "write", "read", "update", "delete"


def record_event(
    session: Session, op: str, target: str, ctx: dict[str, Any] | None = None
) -> None:
    """Anexa um evento ao log. Best-effort: nunca propaga erro pro caller."""
    try:
        session.add(MemoryEvent(op=op, target=target, trigger_ctx=_safe(ctx)))
        session.flush()
    except Exception as exc:  # observability não pode quebrar a operação observada
        log.warning("event.record_failed", extra={"op": op, "target": target, "err": str(exc)})


def _safe(ctx: dict[str, Any] | None) -> dict[str, Any]:
    """Garante JSONB serializável: tipos exóticos viram str."""
    if not ctx:
        return {}
    out: dict[str, Any] = {}
    for k, v in ctx.items():
        out[k] = v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
    return out
