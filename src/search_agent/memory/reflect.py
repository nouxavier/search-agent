"""Reflexão pós-run (RFC §4.3 / §9.1).

O LLM (Sonnet) lê os papers de um run e propõe statements de preferência — mas
cada um precisa apontar para arxiv_ids **concretos daquele run**. Statements sem
evidência são descartados: sem grounding, não vira preferência. Isso é o que
evita o agente "achar" preferências do nada (reflection grounding).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..db.models import Reflection
from ..llm import LLMClient

REFLECT_SYS = (
    "Você analisa um digest de papers de pesquisa e abstrai PREFERÊNCIAS do leitor.\n"
    "Responda SÓ com JSON válido, no formato:\n"
    '{"note": "<1-2 frases>", "statements": ['
    '{"statement": "<preferência curta>", "evidence_arxiv_ids": ["<id>", ...]}]}\n'
    "Regras:\n"
    "- Cada statement DEVE citar pelo menos um arxiv_id que esteja na lista fornecida.\n"
    "- NÃO invente arxiv_ids. Use apenas os que aparecem no prompt.\n"
    "- Prefira papers marcados como [HIGHLIGHT] — são o sinal mais forte.\n"
    "- No máximo 3 statements. Se não houver evidência clara, devolva statements: []."
)


@dataclass(frozen=True)
class ProposedStatement:
    statement: str
    evidence_ids: list[int]  # paper_ids (não arxiv_ids) — já resolvidos contra o run


def _run_papers(session: Session, run_id: int) -> list[dict]:
    rows = session.execute(
        text(
            """
            SELECT p.id AS paper_id, e.value AS arxiv_id, p.title, p.abstract,
                   rp.was_highlight
            FROM run_papers rp
            JOIN papers p ON p.id = rp.paper_id
            LEFT JOIN external_ids e ON e.paper_id = p.id AND e.kind = 'arxiv_id'
            WHERE rp.run_id = :run_id
            ORDER BY rp.rank
            """
        ),
        {"run_id": run_id},
    ).mappings().all()
    return [dict(r) for r in rows]


def _render(papers: list[dict]) -> str:
    lines = []
    for p in papers:
        tag = " [HIGHLIGHT]" if p["was_highlight"] else ""
        lines.append(f"- arxiv_id={p['arxiv_id']}{tag}: {p['title']}\n  {(p['abstract'] or '')[:300]}")
    return "Papers deste run:\n" + "\n".join(lines)


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}


def reflect(
    session: Session, llm: LLMClient, run_id: int, *, model: str
) -> list[ProposedStatement]:
    """Gera a reflexão do run e persiste a nota grounded. Retorna os statements
    propostos (já com paper_ids resolvidos) para a consolidação."""
    papers = _run_papers(session, run_id)
    if not papers:
        return []

    raw = llm.complete(system=REFLECT_SYS, prompt=_render(papers), model=model, max_tokens=800)
    data = _parse_json(raw)

    by_arxiv = {p["arxiv_id"]: p["paper_id"] for p in papers if p["arxiv_id"]}
    proposed: list[ProposedStatement] = []
    grounded: set[int] = set()

    for st in data.get("statements", []):
        stmt = (st.get("statement") or "").strip()
        ev_ids = [by_arxiv[a] for a in (st.get("evidence_arxiv_ids") or []) if a in by_arxiv]
        if not stmt or not ev_ids:
            continue  # GROUNDING: sem evidência real do run, descarta (§4.3)
        proposed.append(ProposedStatement(statement=stmt, evidence_ids=sorted(set(ev_ids))))
        grounded.update(ev_ids)

    if not grounded:
        return []  # nenhuma reflexão grounded → nada é persistido

    note = (data.get("note") or "").strip() or "(sem nota)"
    session.add(Reflection(run_id=run_id, note=note, grounded_ids=sorted(grounded)))
    session.flush()
    return proposed
