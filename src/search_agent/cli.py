"""CLI (Typer).

E0: `run` imprimia em memória. E1: `run` agora **persiste** (write path com dedup)
e o digest **não repete** papers já surfaceados em runs anteriores; `query`
responde "o que já vi sobre X?" via read path. `smoke` testa o LLM.
"""

from __future__ import annotations

import typer

from .config import get_settings
from .db.session import session_scope
from .embeddings import make_embedder
from .llm import make_llm
from .logging_setup import get_logger, setup_logging
from .memory.read_path import recall
from .memory.write_path import (
    count_papers,
    create_run,
    ingest,
    link_to_run,
    previously_seen_ids,
)
from .sources.arxiv import ArxivSource

app = typer.Typer(help="search-agent — camada de memória para um agente de pesquisa.")
log = get_logger("search_agent.cli")


@app.command()
def run(
    area: str = typer.Option(None, "--area", "-a", help="Área de pesquisa (default do config)."),
    limit: int = typer.Option(None, "--limit", "-n", help="Quantos papers buscar."),
) -> None:
    """E1: arXiv → write path (dedup/persist) → digest sem repetir runs anteriores."""
    setup_logging()
    cfg = get_settings()
    area = area or cfg.agent.default_area
    limit = limit or cfg.agent.papers_per_run

    source = ArxivSource(categories=cfg.source.categories, max_results=limit)
    embedder = make_embedder(cfg.embedder)
    params = {
        "area": area,
        "limit": limit,
        "embedder": cfg.embedder.provider,
        "embed_model": cfg.embedder.model,
        "min_year": cfg.agent.min_year,
    }

    log.info("run.start", extra=params)

    with session_scope() as session:
        run_row = create_run(session, area, params)
        candidates = list(source.fetch(area))

        # WRITE PATH: grava todos os candidatos (dedup por identidade canônica).
        ingested: dict[int, object] = {}  # paper_id → RawPaper (1º visto vence)
        for raw in candidates:
            pid = ingest(session, embedder, raw, min_year=cfg.agent.min_year)
            if pid is not None and pid not in ingested:
                ingested[pid] = raw

        # DIGEST: só o que nenhum run anterior surfaceou (não-repetição, G2).
        seen = previously_seen_ids(session, exclude_run_id=run_row.id)
        digest = [(pid, raw) for pid, raw in ingested.items() if pid not in seen]

        for rank, (pid, raw) in enumerate(digest, start=1):
            link_to_run(session, run_row.id, pid, rank=rank)

        total = count_papers(session)
        run_id = run_row.id

        _print_digest(area, digest, ingested_count=len(ingested), total=total)
        log.info(
            "run.done",
            extra={"run_id": run_id, "ingested": len(ingested), "digest": len(digest), "store_total": total},
        )


@app.command()
def query(
    text: str = typer.Argument(..., help="O que você quer recuperar do histórico."),
    k: int = typer.Option(10, "--k", "-k", help="Quantos resultados."),
    area: str = typer.Option(None, "--area", "-a", help="Filtro opcional por área (no título)."),
) -> None:
    """Read path: 'o que já vi sobre X?' — busca por similaridade no histórico."""
    setup_logging()
    cfg = get_settings()
    embedder = make_embedder(cfg.embedder)

    with session_scope() as session:
        hits = recall(session, embedder, text, k=k, area=area)

    if not hits:
        typer.secho("Nada no histórico ainda. Rode `agent run` primeiro.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)

    typer.secho(f'\nMais próximos de "{text}":', bold=True)
    for i, h in enumerate(hits, start=1):
        sim = 1.0 - h.distance  # cosseno: 1 = idêntico
        typer.secho(f"\n[{i}] {h.title}", fg=typer.colors.CYAN, bold=True)
        typer.echo(f"    {h.first_author or '—'}  ({h.year or '—'})  sim≈{sim:.3f}")


@app.command()
def smoke() -> None:
    """Smoke test do LLM: uma chamada curta ao Claude pra validar credencial/SDK."""
    setup_logging()
    cfg = get_settings()
    llm = make_llm(cfg.llm)
    out = llm.complete(
        system="Responda em uma frase curta, em português.",
        prompt="Confirme que você está acessível respondendo 'ok, search-agent'.",
        model=cfg.llm.model_fast,
        max_tokens=64,
    )
    typer.secho(f"LLM ({cfg.llm.model_fast}): {out.strip()}", fg=typer.colors.GREEN)


def _print_digest(area: str, digest, *, ingested_count: int, total: int) -> None:
    if not digest:
        typer.secho(
            f'\nNada novo em "{area}" desde o último run '
            f"({ingested_count} candidatos, todos já vistos).",
            fg=typer.colors.YELLOW,
        )
    else:
        for rank, (pid, raw) in enumerate(digest, start=1):
            arxiv_id = raw.source_ids.get("arxiv_id", "—")
            typer.secho(f"\n[{rank}] {raw.title}", fg=typer.colors.CYAN, bold=True)
            typer.echo(f"    id={pid}  arxiv_id={arxiv_id}  year={raw.year}  {raw.first_author}")
    typer.secho(
        f"\n✓ digest: {len(digest)} novos de {ingested_count} candidatos · store agora tem {total} papers.",
        fg=typer.colors.GREEN,
        bold=True,
    )


if __name__ == "__main__":
    app()
