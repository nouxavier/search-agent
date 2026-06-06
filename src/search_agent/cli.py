"""CLI (Typer). Na E0: `run` (pipeline em memória) e `smoke` (testa o LLM).

Comandos futuros (query/reflect/metrics) entram nas fases seguintes; ficam de
fora aqui para não prometer o que ainda não roda.
"""

from __future__ import annotations

import typer

from .config import get_settings
from .embeddings import make_embedder
from .llm import make_llm
from .logging_setup import get_logger, setup_logging
from .sources.arxiv import ArxivSource

app = typer.Typer(help="search-agent — camada de memória para um agente de pesquisa.")
log = get_logger("search_agent.cli")


@app.command()
def run(
    area: str = typer.Option(None, "--area", "-a", help="Área de pesquisa (default do config)."),
    limit: int = typer.Option(None, "--limit", "-n", help="Quantos papers buscar."),
) -> None:
    """E0: busca no arXiv → normaliza → gera embedding → imprime (em memória, sem gravar)."""
    setup_logging()
    cfg = get_settings()

    area = area or cfg.agent.default_area
    limit = limit or cfg.agent.papers_per_run

    source = ArxivSource(categories=cfg.source.categories, max_results=limit)
    embedder = make_embedder(cfg.embedder)

    log.info("run.start", extra={"area": area, "limit": limit, "embedder": cfg.embedder.provider})

    papers = list(source.fetch(area))
    if not papers:
        typer.secho("Nenhum paper retornado do arXiv.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    vectors = embedder.embed([p.embed_text for p in papers])

    for i, (paper, vec) in enumerate(zip(papers, vectors), start=1):
        arxiv_id = paper.source_ids.get("arxiv_id", "—")
        typer.secho(f"\n[{i}] {paper.title}", fg=typer.colors.CYAN, bold=True)
        typer.echo(f"    arxiv_id={arxiv_id}  year={paper.year}  author={paper.first_author}")
        typer.echo(f"    embedding: dim={len(vec)}  ‖v‖≈{_norm(vec):.3f}")

    log.info("run.done", extra={"area": area, "count": len(papers), "dim": embedder.dim})
    typer.secho(
        f"\n✓ {len(papers)} papers normalizados com embedding (dim={embedder.dim}).",
        fg=typer.colors.GREEN,
        bold=True,
    )


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


def _norm(vec: list[float]) -> float:
    return sum(v * v for v in vec) ** 0.5


if __name__ == "__main__":
    app()
