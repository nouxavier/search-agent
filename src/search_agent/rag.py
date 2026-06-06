"""RAG mínimo (retrieval + generation) sobre o store de papers.

O read path já faz o R (recall por kNN + re-rank por perfil, E2). Aqui ligamos o
G: o Claude responde a uma pergunta usando SÓ os abstracts recuperados, citando
arxiv_ids concretos. Mesmo princípio de grounding da reflexão (§4.3): sem papers
no contexto, não há resposta — o LLM não "inventa" do conhecimento dele.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from .embeddings import Embedder
from .llm import LLMClient
from .memory.read_path import RankedHit, recall

ASK_SYS = (
    "Você responde perguntas de pesquisa usando APENAS os papers fornecidos no prompt.\n"
    "Regras:\n"
    "- Baseie-se SÓ nos abstracts dados. NÃO use conhecimento externo.\n"
    "- Cite cada afirmação com o arxiv_id entre colchetes, ex.: [2401.12345].\n"
    "- Se os papers não cobrem a pergunta, diga isso claramente — não invente.\n"
    "- Responda em português, de forma concisa (2-5 frases)."
)


@dataclass(frozen=True)
class Source:
    arxiv_id: str | None
    title: str
    score: float


@dataclass(frozen=True)
class AskResult:
    answer: str
    sources: list[Source]


def _arxiv_ids(session: Session, paper_ids: list[int]) -> dict[int, str]:
    if not paper_ids:
        return {}
    rows = session.execute(
        text(
            "SELECT paper_id, value FROM external_ids "
            "WHERE kind = 'arxiv_id' AND paper_id = ANY(:ids)"
        ),
        {"ids": list(paper_ids)},
    ).all()
    return {r[0]: r[1] for r in rows}


def _render_context(ranked: list[RankedHit], arxiv: dict[int, str]) -> str:
    blocks = []
    for r in ranked:
        aid = arxiv.get(r.hit.paper_id) or f"paper#{r.hit.paper_id}"
        blocks.append(
            f"[{aid}] {r.hit.title} ({r.hit.year or '—'})\n{(r.hit.abstract or '').strip()[:600]}"
        )
    return "Papers recuperados:\n\n" + "\n\n".join(blocks)


def ask(
    session: Session,
    embedder: Embedder,
    llm: LLMClient,
    question: str,
    *,
    model: str,
    k: int = 5,
    use_profile: bool = True,
) -> AskResult | None:
    """Recupera os k papers mais relevantes e pede ao LLM uma resposta grounded.

    Retorna None se o store não tiver nada a recuperar.
    """
    ranked = recall(session, embedder, question, k=k, use_profile=use_profile)
    if not ranked:
        return None

    arxiv = _arxiv_ids(session, [r.hit.paper_id for r in ranked])
    prompt = f"{_render_context(ranked, arxiv)}\n\nPergunta: {question}"
    answer = llm.complete(system=ASK_SYS, prompt=prompt, model=model, max_tokens=600)

    sources = [
        Source(arxiv_id=arxiv.get(r.hit.paper_id), title=r.hit.title, score=r.score)
        for r in ranked
    ]
    return AskResult(answer=answer.strip(), sources=sources)
