# RFC: Camada de Memória do `search-agent`

| | |
|---|---|
| **Status** | Draft (proposta para revisão) |
| **Autor** | Noua |
| **Data** | 2026-06-05 |
| **Escopo** | Sistema completo de memória (E0→E5), com spec técnica |
| **Base conceitual** | Du, P. (2026). *Memory for Autonomous LLM Agents.* arXiv:2603.07670 |
| **Docs relacionados** | [fase-0.md](fase-0.md) (diagnóstico), [search-project.md](search-project.md) (plano conceitual), [plano-implementacao.md](plano-implementacao.md) (fases de engenharia) |

> Convenção: `(§X.Y)` referencia seções de Du (2026); todo o conteúdo é parafraseado. Blocos de código são **ilustrativos da intenção de design**, não a implementação final.

---

## 1. Resumo (TL;DR)

O `search-agent` hoje é **stateless**: cada run busca papers no arXiv, cura ~10–12 e renderiza um widget — e esquece tudo. Este RFC propõe uma **camada de memória persistente** que transforma o agente de *working + procedural memory* (o que cabe no prompt + o pipeline) em um sistema com **episodic** (histórico de papers/runs) e **semantic** (perfil de preferências que evolui), sobre um substrate **structured + vector** num único PostgreSQL + pgvector.

A entrega é faseada (E0→E5), começando pelo **Pattern B** (context + retrieval store, controle heurístico, §7.6) e só adicionando complexidade — grafo relacional, avaliação, observability — quando cada camada anterior estiver sólida e medida.

**Decisão central:** um store só (Postgres+pgvector), identidade de paper **canônica e source-agnostic**, write/read path **explícitos e auditáveis**, controle **heurístico** até dados justificarem o contrário.

---

## 2. Contexto e motivação

O diagnóstico da [fase-0.md](fase-0.md) classificou o sistema atual nos três eixos do paper (§3):

| Eixo | Estado atual | Buraco |
|------|--------------|--------|
| Temporal scope | working + procedural presentes; **episodic ausente**, **semantic fora do sistema** | sem histórico, sem perfil que evolui |
| Substrate | context-resident text (prompt + widget transitório) | nada sobrevive ao run |
| Control policy | heurístico ponta a ponta | ok manter — mas não há memória para controlar ainda |

Os itens do roadmap são, na taxonomia, pedidos por essas camadas ausentes:

| Desejo | Na taxonomia |
|--------|--------------|
| Histórico persistente | episodic + substrate persistente |
| Favoritos/ratings (👍/👎/⭐) | sinal para consolidation episódico→semantic |
| Comparação semana a semana | coerência cross-session (§5.5) |
| Perfil de preferências | semantic memory dentro do sistema |

**Por que agora:** o paper conclui (§10) que arquitetura de memória recebe uma fração do cuidado dado à escolha do modelo, e que inverter isso é a maior alavanca disponível. O agente já funciona; o gargalo de utilidade não é o modelo, é a ausência de memória.

---

## 3. Goals / Non-goals

**Goals**
- G1. Persistir cada paper e cada run como memória episódica, com identidade canônica que funde duplicatas cross-source.
- G2. Read path que responde *"o que já vi sobre X?"* e **não repete** papers entre runs.
- G3. Perfil semântico (`user_profile`) que evolui a partir de reflexões grounded e influencia o ranking.
- G4. Ligar papers por relação (autor, citação, sub-área), não só similaridade.
- G5. Medir se a memória ajuda (metric stack), não chutar.
- G6. Tornar toda operação de memória auditável (observability) e a deleção completa (governance, §7.5).

**Non-goals (explicitamente fora deste RFC)**
- N1. Controle aprendido (RL) sobre operações de memória (§4.5) — só se os dados de E4 justificarem; alto custo, baixa interpretabilidade.
- N2. Reescrever a camada de output/render — ela permanece, só muda a *fonte* dos dados (prompt → banco).
- N3. Serviço de embeddings/LLM SOTA — "o ponto é o pipeline, não o SOTA".
- N4. Infra distribuída (fila, cache dedicado, segundo banco) — nada entra antes de uma fase precisar.
- N5. Multi-fonte completo — o *schema* já é source-agnostic, mas só o adapter arXiv entra agora.

---

## 4. Visão de design

### 4.1 Princípio: Pattern B primeiro

O paper define três architecture patterns (§7.6). Adotamos o **Pattern B** — context window + retrieval store externo com controle simples — e o tratamos como destino, não estação de passagem. Pattern C (tiered/learned control) só se E4 mostrar ganho empírico no nosso workload.

### 4.2 Substrate: um store só

Retrieval aqui é inerentemente **vetor + metadata + relação no mesmo query** ("papers de eficiência, da área X, dos últimos 30 dias, que ainda não vi, e o que eles citam"). PostgreSQL + pgvector dá os três num `SELECT` transacional, com deleção atômica (paper + embedding juntos). Alternativas em §11.

### 4.3 Identidade canônica (source-agnostic)

O mesmo paper aparece em fontes diferentes com IDs diferentes. A identidade do registro é **canônica**:

```
paper_key = DOI normalizado, se existir
          → senão  sha1(normalize(title) | first_author_lastname | year)
```

IDs por fonte (`arxiv_id`, futuro `s2_id`, `openalex_id`, `doi`) são **aliases** numa tabela à parte; `sources` é a **lista** de fontes onde o paper foi visto. Isso torna a dedup cross-source uma propriedade do schema, não do código de cada adapter — adicionar fonte = escrever um adapter, zero migração.

### 4.4 Diagrama de fluxo

```
                 ┌──────────── WRITE PATH (§7.1) ────────────┐
 Source adapter  │  filter → resolve identity → dedup/merge  │
 (arXiv → norm)  │   → metadata tag → embed → persist        │──► Postgres
                 └───────────────────────────────────────────┘     + pgvector
                                                                      │
 LLM (reflect/   ┌──────────── READ PATH (§7.2) ─────────────┐       │
 consolidate) ◄──│ candidate gen (vector kNN + metadata SQL) │◄──────┤
                 │  → graph expand (E3) → rank by profile (E2)│       │
                 └───────────────────────────────────────────┘       │
                                  │                                   │
                          render (widget) ◄── run_papers ────────────┘
```

---

## 5. Spec — Modelo de dados (DDL)

Postgres 16 + extensão `vector`. DDL gerada via Alembic (schema versionado — §6.6 schema drift).

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- ── E1: episodic core ───────────────────────────────────────────
CREATE TABLE papers (
    id           BIGSERIAL PRIMARY KEY,
    paper_key    TEXT NOT NULL UNIQUE,          -- identidade canônica (§4.3)
    title        TEXT NOT NULL,
    abstract     TEXT,
    first_author TEXT,
    year         SMALLINT,
    embedding    VECTOR(384),                   -- MiniLM-L6-v2; trocável
    schema_ver   SMALLINT NOT NULL DEFAULT 1,   -- versiona o registro (§6.6)
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX papers_embedding_hnsw
    ON papers USING hnsw (embedding vector_cosine_ops);

CREATE TABLE external_ids (
    paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    kind     TEXT   NOT NULL,                   -- 'doi' | 'arxiv_id' | 's2_id' | ...
    value    TEXT   NOT NULL,
    PRIMARY KEY (kind, value)                   -- um alias pertence a um paper só
);

CREATE TABLE sources (
    paper_id    BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    source_name TEXT   NOT NULL,                -- 'arxiv' | ...
    seen_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (paper_id, source_name)
);

CREATE TABLE runs (
    id              BIGSERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
    area            TEXT NOT NULL,
    params_snapshot JSONB NOT NULL              -- defaults vigentes no run (config)
);

CREATE TABLE run_papers (
    run_id       BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    paper_id     BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    rank         INT,
    was_highlight BOOLEAN NOT NULL DEFAULT false,
    PRIMARY KEY (run_id, paper_id)
);

-- ── E2: semantic ────────────────────────────────────────────────
CREATE TABLE reflections (
    id          BIGSERIAL PRIMARY KEY,
    run_id      BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    note        TEXT NOT NULL,
    grounded_ids BIGINT[] NOT NULL,             -- papers que sustentam a nota (§4.3)
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE user_profile (
    id          BIGSERIAL PRIMARY KEY,
    statement   TEXT NOT NULL,                  -- "valoriza eficiência de inferência"
    evidence_ids BIGINT[] NOT NULL,             -- grounding obrigatório
    confidence  REAL NOT NULL DEFAULT 0.5,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ                     -- anti self-reinforcing error (§4.3)
);

-- ── E3: relational substrate ────────────────────────────────────
CREATE TABLE edges (
    src_paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    dst_paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    kind         TEXT   NOT NULL,               -- 'same_author'|'same_subarea'|'cites'
    weight       REAL   NOT NULL DEFAULT 1.0,
    PRIMARY KEY (src_paper_id, dst_paper_id, kind)
);

-- ── E4: feedback / eval ─────────────────────────────────────────
CREATE TABLE feedback (
    paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    run_id   BIGINT REFERENCES runs(id) ON DELETE SET NULL,
    signal   TEXT NOT NULL,                     -- 'up'|'down'|'star'
    ts       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (paper_id, run_id, signal)
);

-- ── E5: observability ───────────────────────────────────────────
CREATE TABLE memory_events (
    id          BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    op          TEXT NOT NULL,                  -- 'write'|'read'|'update'|'delete'
    target      TEXT NOT NULL,                  -- tabela/registro afetado
    trigger_ctx JSONB                           -- o que disparou a operação
);
```

**Governance (§7.5):** `ON DELETE CASCADE` em `papers` garante que apagar um paper remove embeddings, aliases, sources, edges e feedback numa transação. Critério de E4: provar isso com teste.

---

## 6. Spec — Interfaces

Tudo atrás de interface para manter componentes trocáveis (embedder local↔Voyage, LLM, fontes).

```python
# sources/base.py
@dataclass(frozen=True)
class RawPaper:
    title: str
    abstract: str | None
    authors: list[str]
    year: int | None
    doi: str | None
    source_ids: dict[str, str]      # {'arxiv_id': '2603.07670'}
    source_name: str

class Source(Protocol):
    name: str
    def fetch(self, area: str, since: datetime | None) -> Iterable[RawPaper]: ...

# embeddings.py
class Embedder(Protocol):
    dim: int
    def embed(self, texts: list[str]) -> list[list[float]]: ...

# llm.py
class LLMClient(Protocol):
    def complete(self, *, system: str, prompt: str,
                 model: str, cache_system: bool = False) -> str: ...
    # impl Anthropic: claude-haiku-4-5 (tag/filter alto volume),
    #                 claude-sonnet-4-6 (reflexão/consolidação);
    #                 cache_system=True usa prompt caching no user_profile.
```

---

## 7. Spec — Write path (E1, §7.1)

```python
# memory/write_path.py
def ingest(raw: RawPaper, run_id: int, *, db, embedder) -> int:
    # 1. FILTER — descarta sinal baixo antes de gastar embedding/IO
    if not passes_filter(raw):          # ex.: sem abstract, fora de escopo
        log_event("write", "papers", {"skipped": "low_signal"})
        return  # noqa

    # 2. RESOLVE IDENTITY — chave canônica (§4.3)
    key = canonical_key(raw)            # DOI → senão hash(title|author|year)

    # 3. DEDUP / MERGE — UPSERT pela chave; funde aliases e sources
    paper_id = db.execute(
        """INSERT INTO papers (paper_key, title, abstract, first_author, year)
           VALUES (:key, :title, :abstract, :author, :year)
           ON CONFLICT (paper_key) DO UPDATE SET title = papers.title  -- no-op
           RETURNING id""", ...).scalar()
    upsert_external_ids(db, paper_id, raw.source_ids, raw.doi)
    upsert_source(db, paper_id, raw.source_name)

    # 4. EMBED + TAG — só na primeira vez que vemos o paper
    if is_new(paper_id):
        vec = embedder.embed([f"{raw.title}\n{raw.abstract}"])[0]
        db.execute("UPDATE papers SET embedding = :v WHERE id = :id", ...)

    # 5. LINK ao run
    db.execute("INSERT INTO run_papers (run_id, paper_id) VALUES (...)")
    log_event("write", f"papers/{paper_id}", {"run_id": run_id, "key": key})
    return paper_id
```

**Invariante testável (DoD E1):** dois `RawPaper` com `arxiv_id` distintos mas mesmo DOI → **um** registro em `papers`, dois em `external_ids`.

---

## 8. Spec — Read path (E1→E3, §7.2)

```python
# memory/read_path.py
def recall(area: str, query_text: str, *, db, embedder,
           k: int = 30, exclude_seen: bool = True) -> list[Paper]:
    qvec = embedder.embed([query_text])[0]

    # 1. CANDIDATE GEN — vetor kNN + filtros de metadata no MESMO SQL
    candidates = db.execute("""
        SELECT p.*, p.embedding <=> :qvec AS dist
        FROM papers p
        WHERE (:area IS NULL OR p.title ILIKE '%' || :area || '%')   -- placeholder p/ tag real
          AND (NOT :exclude_seen OR p.id NOT IN (
                SELECT paper_id FROM run_papers))                     -- não repetir (G2)
        ORDER BY p.embedding <=> :qvec                                -- HNSW cosine
        LIMIT :k
    """, qvec=qvec, area=area, exclude_seen=exclude_seen, k=k)

    # 2. GRAPH EXPAND (E3) — vizinhos relacionais que similaridade não traria
    expanded = graph_expand(db, candidates, kinds=("cites", "same_author"))

    # 3. RANK BY PROFILE (E2) — reordena por afinidade com user_profile
    return rank_by_profile(db, candidates + expanded)
```

`relevance ≠ similarity` (§4.2): o candidate gen é só o começo; graph expand + profile rank é o que faz a query ser *relevante*, não só *parecida*.

---

## 9. Spec — Fases E2–E5

### E2 — Reflexão & consolidação (episodic → semantic, §4.3 / §9.1)

```python
def reflect(run_id: int, *, db, llm):
    papers = papers_of_run(db, run_id)
    note = llm.complete(model="claude-sonnet-4-6", system=REFLECT_SYS,
                        prompt=render(papers))         # nota curta + IDs citados
    grounded = extract_grounded_ids(note, papers)      # GROUNDING obrigatório
    if not grounded:
        return  # sem evidência → não vira reflexão (§4.3)
    db.insert("reflections", run_id=run_id, note=note, grounded_ids=grounded)

def consolidate(*, db, llm):
    # episódios + reflexões → statements do user_profile, com evidence + expiry
    # confidence sobe com repetição; expires_at evita cristalizar erro
    ...
```

- **Reflection grounding:** todo `statement` aponta para `evidence_ids` concretos. Sem evidência, não entra.
- **Anti self-reinforcing error (§4.3):** `expires_at` + revisão periódica; uma conclusão errada não enviesa todos os runs futuros.
- **DoD:** um sinal de `feedback` move um `statement`, e o ranking do run seguinte reflete a mudança.

### E3 — Substrate relacional (§3.2, §5.5, §9.2)

- Write path popula `edges` (same_author por match de autor; same_subarea por cluster de embedding; cites por parse de referências quando disponível).
- `graph_expand` faz travessia de 1–2 hops a partir dos candidatos do read path.
- **DoD:** o read path retorna ≥1 vizinho relacional que a busca vetorial pura não traria ("este paper cita um que você viu há 3 semanas").

### E4 — Metric stack (§5)

| Dimensão | Métrica |
|----------|---------|
| Task effectiveness | % de papers surfaceados que viraram `feedback=up/star` |
| Memory quality | taxa de repetição indevida; registros recuperados mas inúteis |
| Efficiency | latência por op de memória; tokens de memória por run |
| Governance | teste: deletar paper apaga de todas as tabelas + embedding |

- Harness de **ablation**: rodar com memória on/off e comparar. **DoD:** número antes/depois atribuível a um componente.

### E5 — Observability (§7.7)

- `memory_events` registra toda op com `trigger_ctx`; migração para **structlog** (JSON).
- `agent diff <run_a> <run_b>`: o que mudou no store entre dois runs (memory diff).
- **DoD:** dado um digest ruim, localizar se a falha foi write / read / compressão / raciocínio do LLM.

---

## 10. Plano de rollout

Detalhado em [plano-implementacao.md](plano-implementacao.md). Resumo:

| Fase | Entrega que roda | Pronto quando |
|------|------------------|---------------|
| E0 | pipeline em memória (arXiv→norm→embed→print) | comando retorna N papers normalizados; pytest verde |
| E1 | Pattern B: dedup canônica + "o que já vi sobre X?" | dois runs não repetem; merge cross-source testado |
| E2 | perfil semântico que evolui e re-ranqueia | feedback move statement; ranking muda |
| E3 | conexões relacionais entre runs | ≥1 vizinho relacional não-trivial |
| E4 | métricas + ablation | número antes/depois atribuível |
| E5 | event log + memory diff | falha localizável por etapa |

**Regra:** não avançar antes do "pronto quando". A maioria do ganho mora em E1 bem feito (§7.6, §10).

---

## 11. Alternativas consideradas

| Decisão | Escolha | Alternativa | Por que não |
|---------|---------|-------------|-------------|
| Vector store | **pgvector** | Pinecone/Qdrant/Chroma dedicado | Retrieval é vetor+metadata+relação no mesmo query; volume (poucos milhares) não justifica sincronizar dois bancos nem transações distribuídas. pgvector já tem HNSW. |
| Acesso a dados | **SQLAlchemy + SQL cru no retrieval** | ORM 100% / SQL 100% cru | ORM puro esconde o read path (que queremos *internalizar*); SQL puro perde migrações disciplinadas. Híbrido: ORM no CRUD, SQL explícito onde transparência importa. |
| Embeddings | **MiniLM local** (dev) | Voyage/OpenAI desde já | Grátis, offline, determinístico p/ testes. Interface `Embedder` permite trocar sem tocar o pipeline. |
| Identidade | **canônica DOI→hash** | dedup por `arxiv_id` | `arxiv_id` quebra no multi-fonte; canônica é pré-requisito do schema source-agnostic. |
| Control policy | **heurístico** | learned/RL desde já | Alto custo de treino, risco de learned forgetting, baixa interpretabilidade — sem justificativa antes de E4 (§4.5). |
| Store de config | **config.toml versionado** | manter no prompt | Defaults no prompt foram diagnosticados como buraco na fase-0; viram dado auditável. |

---

## 12. Riscos e modos de falha (§4, §6.6)

| Risco | Seção | Mitigação |
|-------|-------|-----------|
| **Summarization drift** — compressão descarta o detalhe raro | §4.1 | manter registro bruto em fidelidade total no store; nunca comprimir o original |
| **Relevance ≠ similarity** — kNN traz parecido, não relevante | §4.2 | graph expand + profile rank no read path, não só cosseno |
| **Self-reinforcing error** — preferência errada cristaliza | §4.3 | reflection grounding + `expires_at` + revisão |
| **Silent orchestration failure** — decisão de memória falha sem stack trace | §4.4 | observability desde E5; `memory_events` |
| **Schema drift** — formato do arXiv/registro muda | §6.6 | Alembic + coluna `schema_ver` por registro |
| **Stale records / contradição** quando houver histórico | §7.3 | timestamps + revisão de perfil; preferir registro mais recente com evidência |

---

## 13. Questões em aberto

1. **Confirmar no código** a premissa "nenhuma persistência hoje" e onde os defaults realmente moram (item aberto da fase-0 §11). Se já houver storage, ajustar E0/E1.
2. **Threshold de filtragem** do write path: fixo no `config.toml` ou função do volume do run? (decidir em E1; medir em E4).
3. **`same_subarea` em E3**: clustering de embeddings (HDBSCAN/k-means) ou tag declarada? Começar com tag, evoluir se preciso.
4. **Citações (`cites`)**: arXiv não dá grafo de citação direto. Adiar `cites` para quando entrar um adapter (Semantic Scholar) que forneça, ou parsear referências do PDF? (não bloqueia E3; `same_author`/`same_subarea` já entregam o DoD).
5. **Dimensão do embedding** acoplada à coluna `VECTOR(384)`: trocar de embedder = migração. Aceitável agora; documentar.

---

## 14. Referência

Du, P. (2026). *Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers.* arXiv:2603.07670 — §2.1 (loop write–manage–read), §2.3 (design objectives), §3 (taxonomia), §4.1–4.5 (failure modes / control), §5 (avaliação), §5.5 (cross-session), §6.6 (schema drift), §7.1–7.7 (write/read path, patterns, governance, observability), §9.1–9.4 (frontiers), §10 (conclusão).
