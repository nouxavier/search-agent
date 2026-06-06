"""RAG mínimo (retrieval + generation):

- o contexto montado pro LLM contém os abstracts recuperados e cita arxiv_ids;
- as fontes devolvidas batem com os papers do store;
- store vazio → None (sem alucinação por falta de contexto).
"""

from __future__ import annotations

from search_agent.embeddings import FakeEmbedder
from search_agent.llm import FakeLLM
from search_agent.memory.write_path import create_run, ingest, link_to_run
from search_agent.rag import ask
from search_agent.sources.base import RawPaper

EMB = FakeEmbedder(1024)


def _raw(arxiv_id, *, title, abstract):
    return RawPaper(
        title=title, abstract=abstract, authors=["Ada Lovelace"], year=2026,
        doi=None, source_ids={"arxiv_id": arxiv_id}, source_name="arxiv",
    )


def _seed(session, papers):
    run = create_run(session, "AI", {})
    for rank, raw in enumerate(papers, start=1):
        pid = ingest(session, EMB, raw, min_year=2000)
        link_to_run(session, run.id, pid, rank=rank)
    session.flush()


def test_ask_returns_none_on_empty_store(session):
    llm = FakeLLM("não deveria ser chamado")
    assert ask(session, EMB, llm, "qualquer coisa", model="x") is None
    assert llm.calls == []  # sem contexto, nem chama o LLM


def test_ask_grounds_context_and_returns_sources(session):
    _seed(session, [
        _raw("2606.20001", title="Retrieval augmented generation", abstract="RAG reduces hallucination."),
        _raw("2606.20002", title="Unrelated topic", abstract="something about category theory"),
    ])

    llm = FakeLLM("RAG reduz alucinação [2606.20001].")
    result = ask(session, EMB, llm, "retrieval augmented generation", model="reason-model", k=2)

    assert result is not None
    assert result.answer == "RAG reduz alucinação [2606.20001]."

    # O LLM recebeu o contexto grounded: arxiv_id + abstract + a pergunta.
    call = llm.calls[-1]
    assert call["model"] == "reason-model"
    assert "2606.20001" in call["prompt"]
    assert "RAG reduces hallucination" in call["prompt"]
    assert "Pergunta: retrieval augmented generation" in call["prompt"]

    # Fontes devolvidas carregam os arxiv_ids resolvidos do store.
    arxiv_ids = {s.arxiv_id for s in result.sources}
    assert "2606.20001" in arxiv_ids
