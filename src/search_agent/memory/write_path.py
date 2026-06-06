"""Write path (RFC §7.1): filter → resolve identity → dedup/merge → tag → persist.

O UPSERT por `paper_key` é o coração da dedup: o mesmo paper vindo de fontes
diferentes cai na mesma chave e funde num registro, acumulando aliases em
external_ids e fontes em sources. Embedding só é gerado na primeira vez que vemos
o paper (quando `embedding IS NULL`) — não re-embeda o que já está no store.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from ..embeddings import Embedder
from ..logging_setup import get_logger
from ..sources.base import RawPaper
from .graph import populate_edges
from .identity import canonical_key, normalize_doi
from ..db.models import ExternalId, Paper, Run, RunPaper, Source

log = get_logger("search_agent.write_path")


def passes_filter(raw: RawPaper, *, min_year: int) -> bool:
    """Descarta sinal baixo antes de gastar embedding/IO (RFC §7.1, passo FILTER)."""
    if not raw.abstract:
        return False
    if raw.year is not None and raw.year < min_year:
        return False
    return True


def create_run(session: Session, area: str, params_snapshot: dict[str, Any]) -> Run:
    run = Run(area=area, params_snapshot=params_snapshot)
    session.add(run)
    session.flush()  # popula run.id
    return run


def ingest(
    session: Session,
    embedder: Embedder,
    raw: RawPaper,
    *,
    min_year: int,
) -> int | None:
    """Grava um RawPaper na memória episódica. Retorna o paper_id, ou None se filtrado.

    Idempotente por `paper_key`: chamar de novo com o mesmo paper não duplica.
    """
    # 1. FILTER
    if not passes_filter(raw, min_year=min_year):
        log.info("write.skip", extra={"reason": "low_signal", "title": raw.title[:80]})
        return None

    # 2. RESOLVE IDENTITY
    key = canonical_key(raw)

    # 3. DEDUP / MERGE — UPSERT por paper_key; o DO UPDATE no-op garante RETURNING id
    upsert = (
        pg_insert(Paper)
        .values(
            paper_key=key,
            title=raw.title,
            abstract=raw.abstract,
            first_author=raw.first_author,
            year=raw.year,
        )
        .on_conflict_do_update(index_elements=["paper_key"], set_={"paper_key": key})
        .returning(Paper.id)
    )
    paper_id = session.execute(upsert).scalar_one()

    # 4. EMBED + TAG — só quando ainda não há vetor (primeira vez que vemos o paper)
    current_embedding = session.execute(
        select(Paper.embedding).where(Paper.id == paper_id)
    ).scalar_one()
    is_new = current_embedding is None
    if is_new:
        vec = embedder.embed([raw.embed_text])[0]
        session.execute(
            Paper.__table__.update().where(Paper.id == paper_id).values(embedding=vec)
        )
        # E3: liga o paper novo ao resto do store (same_author + same_subarea).
        populate_edges(session, paper_id)

    # external_ids: doi (normalizado) + todos os ids por fonte, como aliases
    aliases: dict[str, str] = {}
    if raw.doi:
        aliases["doi"] = normalize_doi(raw.doi)
    for kind, value in raw.source_ids.items():
        aliases[kind] = value
    for kind, value in aliases.items():
        session.execute(
            pg_insert(ExternalId)
            .values(paper_id=paper_id, kind=kind, value=value)
            .on_conflict_do_nothing(index_elements=["kind", "value"])
        )

    # sources: acrescenta a fonte onde foi visto (lista, não valor único)
    session.execute(
        pg_insert(Source)
        .values(paper_id=paper_id, source_name=raw.source_name)
        .on_conflict_do_nothing(index_elements=["paper_id", "source_name"])
    )

    log.info(
        "write.persist",
        extra={"paper_id": paper_id, "key": key, "new": is_new, "source": raw.source_name},
    )
    return paper_id


def previously_seen_ids(session: Session, *, exclude_run_id: int) -> set[int]:
    """paper_ids que algum run ANTERIOR já surfaceou — base da não-repetição (G2)."""
    rows = session.execute(
        select(RunPaper.paper_id).where(RunPaper.run_id != exclude_run_id).distinct()
    )
    return {r[0] for r in rows}


def link_to_run(
    session: Session, run_id: int, paper_id: int, *, rank: int, was_highlight: bool = False
) -> None:
    session.execute(
        pg_insert(RunPaper)
        .values(run_id=run_id, paper_id=paper_id, rank=rank, was_highlight=was_highlight)
        .on_conflict_do_nothing(index_elements=["run_id", "paper_id"])
    )


def count_papers(session: Session) -> int:
    return session.execute(select(func.count()).select_from(Paper)).scalar_one()
