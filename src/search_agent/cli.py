"""CLI (Typer).

E1: `run` persiste (write path, dedup) e o digest não repete runs anteriores;
`query` é o read path. E2: `reflect` gera nota grounded pós-run, `feedback` marca
um paper como relevante (sinal que move o perfil), `consolidate`/`reflect`
atualizam o `user_profile`, e `run`/`query` re-ranqueiam pelo perfil. `profile`
lista as preferências vigentes. `smoke` testa o LLM.
"""

from __future__ import annotations

from time import perf_counter

import typer
from dotenv import load_dotenv
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .config import REPO_ROOT, get_settings

# Carrega .env (ex.: ANTHROPIC_API_KEY) para os.environ — é de lá que o SDK lê.
load_dotenv(REPO_ROOT / ".env")
from .db.models import Feedback, UserProfile
from .db.queries import search_similar
from .db.session import session_scope
from .embeddings import make_embedder
from .eval.ablation import run_ablation
from .eval.metrics import compute_metrics
from .llm import make_llm
from .logging_setup import get_logger, setup_logging
from .memory.consolidate import consolidate
from .memory.graph import relational_neighbors
from .memory.read_path import recall, rerank_by_profile
from .memory.reflect import reflect
from .observability.diff import memory_diff
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
    """arXiv → write path (dedup/persist) → digest sem repetir, re-ranqueado pelo perfil (E2)."""
    setup_logging()
    cfg = get_settings()
    area = area or cfg.agent.default_area
    limit = limit or cfg.agent.papers_per_run

    source = ArxivSource(categories=cfg.source.categories, max_results=limit)
    embedder = make_embedder(cfg.embedder)
    params = {"area": area, "limit": limit, "embedder": cfg.embedder.provider, "min_year": cfg.agent.min_year}
    log.info("run.start", extra=params)

    with session_scope() as session:
        run_row = create_run(session, area, params)
        candidates = list(source.fetch(area))

        ingested: dict[int, object] = {}
        for raw in candidates:
            pid = ingest(session, embedder, raw, min_year=cfg.agent.min_year)
            if pid is not None and pid not in ingested:
                ingested[pid] = raw

        seen = previously_seen_ids(session, exclude_run_id=run_row.id)
        digest_ids = [pid for pid in ingested if pid not in seen]
        # RANK BY PROFILE (E2): o perfil semântico reordena o digest.
        digest_ids = rerank_by_profile(session, embedder, digest_ids)

        for rank, pid in enumerate(digest_ids, start=1):
            link_to_run(session, run_row.id, pid, rank=rank)

        # GRAPH EXPAND (E3): pra cada paper do digest, uma ponte a algo já visto antes.
        notes = relational_neighbors(session, digest_ids, seen)

        total = count_papers(session)
        digest = [(pid, ingested[pid]) for pid in digest_ids]
        _print_digest(area, digest, ingested_count=len(ingested), total=total, notes=notes)
        typer.secho(f"  (run #{run_row.id} · use `agent reflect {run_row.id}` para refletir)", dim=True)
        log.info("run.done", extra={"run_id": run_row.id, "ingested": len(ingested), "digest": len(digest)})


@app.command()
def query(
    text_: str = typer.Argument(..., metavar="TEXT", help="O que recuperar do histórico."),
    k: int = typer.Option(10, "--k", "-k"),
    area: str = typer.Option(None, "--area", "-a", help="Filtro opcional por área (no título)."),
    no_profile: bool = typer.Option(False, "--no-profile", help="Ignora o perfil (só similaridade)."),
) -> None:
    """Read path: 'o que já vi sobre X?' — kNN reordenado pelo perfil (E2)."""
    setup_logging()
    cfg = get_settings()
    embedder = make_embedder(cfg.embedder)

    with session_scope() as session:
        ranked = recall(session, embedder, text_, k=k, area=area, use_profile=not no_profile)

    if not ranked:
        typer.secho("Nada no histórico ainda. Rode `agent run` primeiro.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)

    typer.secho(f'\nMais próximos de "{text_}"' + ("" if no_profile else " (re-ranqueado pelo perfil):"), bold=True)
    for i, r in enumerate(ranked, start=1):
        typer.secho(f"\n[{i}] {r.hit.title}", fg=typer.colors.CYAN, bold=True)
        typer.echo(
            f"    {r.hit.first_author or '—'} ({r.hit.year or '—'})  "
            f"sim≈{r.base_sim:.3f}  perfil≈{r.profile_affinity:.3f}  score={r.score:.3f}"
        )
        snippet = _snippet(r.hit.abstract)
        if snippet:
            typer.secho(f"    {snippet}", dim=True)


@app.command()
def reflect_(
    run_id: int = typer.Argument(..., metavar="RUN_ID", help="Run a refletir."),
) -> None:
    """Reflexão grounded pós-run → atualiza o user_profile (E2)."""
    setup_logging()
    cfg = get_settings()
    llm = make_llm(cfg.llm)

    with session_scope() as session:
        proposed = reflect(session, llm, run_id, model=cfg.llm.model_reason)
        if not proposed:
            typer.secho("Nenhuma reflexão grounded (sem evidência suficiente).", fg=typer.colors.YELLOW)
            raise typer.Exit(code=0)
        highlighted = _highlighted_ids(session, run_id)
        consolidate(session, proposed, highlighted_ids=highlighted)
        typer.secho(f"✓ {len(proposed)} statements consolidados no perfil:", fg=typer.colors.GREEN)
        for ps in proposed:
            typer.echo(f"  · {ps.statement}  (evidência: {ps.evidence_ids})")


# `reflect` é palavra reservada do módulo importado; registra o comando com o nome certo.
app.command(name="reflect")(reflect_)


@app.command()
def feedback(
    paper_id: int = typer.Argument(..., help="paper_id (aparece no digest)."),
    signal: str = typer.Option("up", "--signal", "-s", help="up | down | star (down = recuperado mas inútil)."),
    run_id: int = typer.Option(None, "--run", "-r", help="Run onde marcar (default: o mais recente que surfaceou o paper)."),
) -> None:
    """Registra utilidade de um paper (E4): up/down/star. up|star também viram
    highlight, o sinal que move o perfil na consolidação (E2)."""
    setup_logging()
    if signal not in ("up", "down", "star"):
        typer.secho(f"signal inválido: {signal!r} (use up|down|star).", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    with session_scope() as session:
        if run_id is None:
            run_id = session.execute(
                text("SELECT run_id FROM run_papers WHERE paper_id=:p ORDER BY run_id DESC LIMIT 1"),
                {"p": paper_id},
            ).scalar()
        if run_id is None:
            typer.secho("Esse paper não foi surfaceado em nenhum run.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        # E4: grava o sinal explícito (idempotente por (paper, run, signal)).
        session.execute(
            pg_insert(Feedback)
            .values(paper_id=paper_id, run_id=run_id, signal=signal)
            .on_conflict_do_nothing(index_elements=["paper_id", "run_id", "signal"])
        )
        # Ponte E2: relevância positiva também marca highlight pra consolidação.
        if signal in ("up", "star"):
            session.execute(
                text("UPDATE run_papers SET was_highlight=true WHERE run_id=:r AND paper_id=:p"),
                {"r": run_id, "p": paper_id},
            )
    typer.secho(f"✓ feedback '{signal}' em paper {paper_id} (run {run_id}).", fg=typer.colors.GREEN)


@app.command()
def profile() -> None:
    """Lista as preferências vigentes (user_profile), por confidence."""
    setup_logging()
    with session_scope() as session:
        rows = session.execute(select(UserProfile).order_by(UserProfile.confidence.desc())).scalars().all()
    if not rows:
        typer.secho("Perfil vazio. Rode `agent reflect <run_id>`.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)
    typer.secho("\nuser_profile:", bold=True)
    for r in rows:
        exp = r.expires_at.date().isoformat() if r.expires_at else "—"
        typer.secho(f"\n· {r.statement}", fg=typer.colors.CYAN)
        typer.echo(f"    confidence={r.confidence:.2f}  expira={exp}  evidência={r.evidence_ids}")


@app.command()
def metrics() -> None:
    """Metric stack da E4 (RFC §5): mede se a memória ajuda, com número."""
    setup_logging()
    cfg = get_settings()
    with session_scope() as session:
        m = compute_metrics(session)
        # Efficiency — latência medida AO VIVO de uma op de read (kNN) representativa.
        embedder = make_embedder(cfg.embedder)
        qvec = embedder.embed(["retrieval augmented generation"])[0]
        t0 = perf_counter()
        search_similar(session, qvec, k=10)
        knn_ms = (perf_counter() - t0) * 1000

    typer.secho("\nMetric stack (E4)", bold=True)
    typer.secho("\n• Task effectiveness", fg=typer.colors.CYAN)
    typer.echo(f"    {m.up_star}/{m.surfaced} papers surfaceados viraram up/star → {m.task_effectiveness:.1%}")
    typer.secho("\n• Memory quality", fg=typer.colors.CYAN)
    typer.echo(f"    repetição indevida: {m.repeated}/{m.surfaced} ({m.repeat_rate:.1%})  ← G2 espera ~0")
    typer.echo(f"    'down' (recuperado mas inútil): {m.down}/{m.surfaced} ({m.down_rate:.1%})")
    typer.secho("\n• Efficiency", fg=typer.colors.CYAN)
    typer.echo(f"    store={m.papers} papers · {m.edges} arestas · digest médio={m.avg_digest:.1f}/run")
    typer.echo(f"    latência kNN (k=10): {knn_ms:.1f} ms · tokens/run: não instrumentado (E5)")
    typer.secho("\n• Governance", fg=typer.colors.CYAN)
    typer.echo("    deleção em cascata coberta por test_governance_delete_cascade.")
    typer.secho(
        "\nDica: `agent ablation \"<consulta>\"` compara o ranking com/sem o perfil.",
        dim=True,
    )


@app.command()
def ablation(
    query: str = typer.Argument(..., metavar="QUERY", help="Consulta pra avaliar o re-rank."),
    k: int = typer.Option(20, "--k", "-k"),
) -> None:
    """Ablation (E4): isola o componente 'perfil' — posição média dos papers
    relevantes (up/star) no ranking COM vs SEM o re-rank por perfil."""
    setup_logging()
    cfg = get_settings()
    embedder = make_embedder(cfg.embedder)
    with session_scope() as session:
        ab = run_ablation(session, embedder, query, k=k)

    if ab.relevant_total == 0:
        typer.secho(
            "Sem ground truth: nenhum paper com feedback up/star. "
            "Marque alguns com `agent feedback <id> -s up` e rode de novo.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=0)

    def _fmt(v):
        return f"{v:.1f}" if v is not None else "—"

    typer.secho(f'\nAblation do perfil — consulta: "{ab.query}"', bold=True)
    typer.echo(f"  relevantes (up/star) no store: {ab.relevant_total}")
    typer.echo(f"  perfil ON : {ab.found_on} no top-{k}, posição média {_fmt(ab.mean_rank_on)}")
    typer.echo(f"  perfil OFF: {ab.found_off} no top-{k}, posição média {_fmt(ab.mean_rank_off)}")
    if ab.delta is None:
        typer.secho("  Δ: indeterminado (relevantes não apareceram nos dois rankings).", fg=typer.colors.YELLOW)
    elif ab.delta > 0:
        typer.secho(f"  Δ: perfil SOBE os relevantes em {ab.delta:.1f} posições (ajuda).", fg=typer.colors.GREEN)
    elif ab.delta < 0:
        typer.secho(f"  Δ: perfil DESCE os relevantes em {abs(ab.delta):.1f} posições (atrapalha aqui).", fg=typer.colors.RED)
    else:
        typer.secho("  Δ: sem diferença de posição.", dim=True)


@app.command()
def events(
    n: int = typer.Option(20, "--n", "-n", help="Quantos eventos recentes mostrar."),
    op: str = typer.Option(None, "--op", help="Filtra por op: write|read|update|delete."),
) -> None:
    """Event log da E5 (RFC §7.7): as últimas operações de memória, com o contexto."""
    setup_logging()
    sql = "SELECT ts, op, target, trigger_ctx FROM memory_events"
    params: dict = {"n": n}
    if op:
        sql += " WHERE op=:op"
        params["op"] = op
    sql += " ORDER BY ts DESC LIMIT :n"
    with session_scope() as session:
        rows = session.execute(text(sql), params).all()
    if not rows:
        typer.secho("Nenhum evento registrado ainda. Rode um `agent run`/`query`.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)
    _colors = {"write": typer.colors.GREEN, "read": typer.colors.BLUE,
               "update": typer.colors.MAGENTA, "delete": typer.colors.RED}
    typer.secho(f"\nÚltimos {len(rows)} eventos de memória:", bold=True)
    for ts, ev_op, target, ctx in rows:
        when = ts.strftime("%m-%d %H:%M:%S")
        typer.secho(f"  {when}  {ev_op:<6}", fg=_colors.get(ev_op), nl=False)
        typer.echo(f" {target}  {ctx or {}}")


@app.command()
def diff(
    run_a: int = typer.Argument(..., metavar="RUN_A"),
    run_b: int = typer.Argument(..., metavar="RUN_B"),
) -> None:
    """Memory diff (E5): o que mudou no store entre dois runs (papers/arestas/perfil)."""
    setup_logging()
    with session_scope() as session:
        try:
            d = memory_diff(session, run_a, run_b)
        except ValueError as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(code=1)

    typer.secho(f"\nMemory diff: run {d.earlier_run} → run {d.later_run}", bold=True)
    typer.secho(f"\n+ {len(d.papers_added)} papers novos", fg=typer.colors.GREEN)
    for pid, title in d.papers_added[:15]:
        typer.echo(f"    id={pid}  {title[:70]}")
    if len(d.papers_added) > 15:
        typer.echo(f"    … (+{len(d.papers_added) - 15})")
    typer.secho(f"+ {d.edges_added} arestas novas", fg=typer.colors.GREEN)
    if d.profile_added:
        typer.secho(f"+ {len(d.profile_added)} preferências novas", fg=typer.colors.MAGENTA)
        for s in d.profile_added:
            typer.echo(f"    · {s}")
    if d.profile_expired:
        typer.secho(f"- {len(d.profile_expired)} preferências expiraram", fg=typer.colors.YELLOW)
        for s in d.profile_expired:
            typer.echo(f"    · {s}")
    if d.events_by_op:
        resumo = " · ".join(f"{op}={n}" for op, n in sorted(d.events_by_op.items()))
        typer.secho(f"\neventos na janela: {resumo}", dim=True)


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


def _snippet(abstract: str | None, *, limit: int = 200) -> str:
    """Resumo de uma linha: colapsa quebras e trunca em ~limit chars (corte por palavra)."""
    if not abstract:
        return ""
    text_ = " ".join(abstract.split())
    if len(text_) <= limit:
        return text_
    return text_[:limit].rsplit(" ", 1)[0] + "…"


def _highlighted_ids(session, run_id: int) -> set[int]:
    rows = session.execute(
        text("SELECT paper_id FROM run_papers WHERE run_id=:r AND was_highlight=true"),
        {"r": run_id},
    )
    return {row[0] for row in rows}


def _print_digest(area: str, digest, *, ingested_count: int, total: int, notes=None) -> None:
    notes = notes or {}
    if not digest:
        typer.secho(
            f'\nNada novo em "{area}" desde o último run ({ingested_count} candidatos, todos já vistos).',
            fg=typer.colors.YELLOW,
        )
    else:
        for rank, (pid, raw) in enumerate(digest, start=1):
            arxiv_id = raw.source_ids.get("arxiv_id", "—")
            typer.secho(f"\n[{rank}] {raw.title}", fg=typer.colors.CYAN, bold=True)
            typer.echo(f"    id={pid}  arxiv_id={arxiv_id}  year={raw.year}  {raw.first_author}")
            rel = notes.get(pid)
            if rel:  # E3: ponte relacional pra algo já visto antes
                typer.secho(
                    f'    ↳ conecta-se a "{rel.neighbor_title[:60]}" (id={rel.neighbor_id}) via {rel.kind}',
                    fg=typer.colors.MAGENTA,
                )
    typer.secho(
        f"\n✓ digest: {len(digest)} novos de {ingested_count} candidatos · store agora tem {total} papers.",
        fg=typer.colors.GREEN,
        bold=True,
    )


if __name__ == "__main__":
    app()
