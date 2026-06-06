# Plano de Implementação — Stack e Fases de Código

| | |
|---|---|
| **Projeto** | Camada de Memória para o Agente de Pesquisa (`search-agent`) |
| **Documento** | Plano de engenharia (stack + fases de código + entregáveis) |
| **Complementa** | [search-project.md](search-project.md) (o *quê* / conceito por fase) e [fase-0.md](fase-0.md) (diagnóstico nos três eixos) |
| **Objetivo** | Definir uma stack mínima e uma sequência de fases **de código** onde cada fase entrega algo que roda e tem critério de pronto verificável. |

> Este doc é o "como construir". Os conceitos do paper (Du, 2026, arXiv:2603.07670) já estão mapeados nos outros dois; aqui só referencio as seções `(§X.Y)` quando uma decisão técnica decorre delas. As fases de código (`E0…E5`) mapeiam para as fases conceituais (`0…5`), mas adicionam uma fundação de infra que o plano conceitual não detalha.

---

## 1. Princípios que guiam a stack

1. **Um store só.** PostgreSQL + pgvector cobre estruturado *e* vetorial no mesmo banco — sem segundo serviço de infra (§3.2, substrate híbrido sem o custo operacional de dois bancos).
2. **Começar pelo Pattern B.** Context window + retrieval store com controle heurístico. Nada de controle aprendido até os dados justificarem (§7.6).
3. **Transparência sobre esperteza.** O write/read path tem que ser legível linha a linha — é o que estamos tentando *internalizar*. Preferir SQL explícito a abstração mágica.
4. **Cada dependência paga aluguel.** Só entra na stack o que uma fase precisa. Nada de adicionar Redis/fila/grafo dedicado "para o futuro".
5. **Schema versionado desde o dia 1.** O paper trata *schema drift* como modo de falha real (§6.6); migrações gerenciadas (Alembic) não são opcional.

---

## 2. Stack

| Camada | Escolha | Por quê |
|--------|---------|---------|
| Linguagem | **Python 3.12** | Ecossistema de papers/embeddings/LLM. |
| Gerência de deps / venv | **uv** | Rápido, lockfile reprodutível, um binário só. |
| Banco | **PostgreSQL 16 + pgvector** | Estruturado + vetorial num store só (§3.2). Sobe via `docker-compose`. |
| Acesso a dados | **SQLAlchemy 2.0 (Core/ORM) + Alembic** | ORM para o CRUD comum; **SQL cru** para as queries de retrieval (similaridade + filtros) onde transparência importa. Alembic versiona o schema. |
| Embeddings | **BGE-M3 via Ollama** (`bge-m3`, 1024d) | Multilíngue, contexto 8k (abstract inteiro cabe), local/grátis/offline, sem API key — "o ponto é o pipeline, não o SOTA". Pré-req: `ollama pull bge-m3`. Trocável por **Voyage AI** (`voyage-3`) atrás da interface `Embedder` quando quiser qualidade. |
| LLM (reflexão/consolidação) | **Claude via Anthropic SDK** | `claude-haiku-4-5` para tagging/filtragem de alto volume e baixo custo; `claude-sonnet-4-6` para reflexão/consolidação (raciocínio). **Prompt caching** no `user_profile` para baratear runs repetidos. Tudo atrás de uma interface `LLMClient`. |
| Fontes (ingestão) | **httpx** + adapters por fonte; arXiv via API Atom (`feedparser`) | Cada fonte é um *adapter* que normaliza para o schema canônico (ver §4). |
| Config | **pydantic-settings** + `config.toml` | Defaults (idioma, domínios, papers/área, threshold) saem do prompt e viram dado versionável (resolve o buraco "defaults moram no prompt" da fase-0). |
| CLI | **Typer** | `agent run`, `agent query`, `agent reflect`, `agent metrics`. |
| Logging | stdlib `logging` (JSON) → **structlog** na Fase E5 | Começa simples; vira observability estruturada quando a fase pedir (§7.7). |
| Testes | **pytest** + Postgres efêmero (testcontainers ou compose de teste) | Write/read path têm que ter teste de dedup e de não-repetição. |
| Saída | render atual (widget/HTML) alimentado **pelo banco**, não pelo prompt | A camada de output (§ prompt.md) permanece; muda a *fonte* dos dados. |

**Stack mínima para começar (Fase E0):** `uv`, Postgres+pgvector no compose, SQLAlchemy+Alembic, Ollama com `bge-m3`, Anthropic SDK, httpx, Typer, pydantic-settings, pytest. Nada além disso entra antes de uma fase precisar.

---

## 3. Layout do repositório

```
search-agent/
├── docs/                      # os 3 planos + prompts
├── pyproject.toml             # uv + deps
├── docker-compose.yml         # postgres + pgvector
├── alembic/                   # migrações versionadas
├── config.toml                # defaults (idioma, domínios, thresholds)
└── src/search_agent/
    ├── cli.py                 # Typer: run/query/reflect/metrics
    ├── config.py              # pydantic-settings
    ├── db/
    │   ├── models.py          # tabelas SQLAlchemy
    │   └── queries.py         # SQL cru de retrieval (transparente)
    ├── sources/
    │   ├── base.py            # interface Source + dataclass normalizada
    │   └── arxiv.py           # adapter arXiv
    ├── memory/
    │   ├── write_path.py      # filter → dedup → tag → persist (§7.1)
    │   ├── read_path.py       # retrieval: vetor + metadata (§7.2)
    │   ├── identity.py        # chave canônica (DOI → título+autor+ano)
    │   ├── reflect.py         # reflexão pós-run (§4.3)  [E2]
    │   ├── consolidate.py     # episódico → user_profile (§9.1) [E2]
    │   └── graph.py           # arestas + travessia relacional (§3.2) [E3]
    ├── embeddings.py          # interface Embedder + impl local/Voyage
    ├── llm.py                 # interface LLMClient + impl Anthropic
    ├── eval/                  # metric stack + ablation [E4]
    └── observability/         # event log + memory diff [E5]
```

---

## 4. Modelo de dados canônico (introduzido em E1, evolui)

```
runs(id, ts, area, params_snapshot)
papers(id, paper_id_canonico, title, abstract, year, first_author, embedding vector, created_at)
external_ids(paper_id, kind, value)        # kind ∈ {doi, arxiv_id, s2_id, openalex_id}
sources(paper_id, source_name, seen_at)    # LISTA de fontes onde foi visto
run_papers(run_id, paper_id, rank, was_highlight)
-- E2:
user_profile(id, statement, evidence_ids[], confidence, created_at, expires_at)
reflections(id, run_id, note, grounded_ids[])
-- E3:
edges(src_paper_id, dst_paper_id, kind, weight)   # kind ∈ {same_author, same_subarea, cites}
-- E4:
feedback(paper_id, run_id, signal)         # 👍/👎/⭐
-- E5:
memory_events(id, ts, op, target, trigger_ctx)    # write/read/update/delete
```

**Identidade canônica** (`identity.py`): `paper_id_canonico = DOI quando existir → senão hash(título normalizado + 1º autor + ano)`. IDs por fonte vivem em `external_ids`; `sources` é lista. É isso que torna a dedup cross-source possível sem migração ao adicionar uma fonte nova — só se escreve um adapter (§ Fase 1 do plano conceitual).

---

## 5. Fases de implementação

Cada fase: **objetivo → entregável que roda → componentes → conceito → pronto quando**. Não avançar antes do "pronto quando".

### Fase E0 — Fundação (scaffold & infra)
*Não tem fase conceitual equivalente; é a base que as outras assumem.*

- **Objetivo:** ter o esqueleto rodando ponta a ponta, sem persistência.
- **Entregável:** `uv run agent run --area "LLM Agents"` busca no arXiv, normaliza, gera embedding e **imprime** os papers — tudo em memória. Postgres sobe no compose mas ainda não é usado para gravar.
- **Componentes:** `pyproject.toml`/uv, `docker-compose.yml` (postgres+pgvector), Alembic inicializado, `config.toml` + `config.py`, `cli.py` (Typer), `sources/base.py` + `sources/arxiv.py`, `embeddings.py`, `llm.py` (smoke test de chamada ao Claude), logging JSON.
- **Pronto quando:** o comando retorna N registros normalizados do arXiv com embedding, e `pytest` roda verde com um teste de normalização do adapter.

### Fase E1 — Camada episodic: write + read path  *(↔ conceitual Fase 1)*
- **Objetivo:** Pattern B funcionando — gravar e recuperar com dedup canônica.
- **Entregável:** runs sucessivos **não repetem** o mesmo paper (mesmo vindo de fontes distintas); `agent query "o que já vi sobre X?"` retorna do histórico.
- **Componentes:** migração com o schema do §4 (papers, external_ids, sources, runs, run_papers); `identity.py`; `write_path.py` (**filter → dedup → metadata tag → persist**, §7.1); `read_path.py` (similaridade pgvector + filtro de metadata, §7.2); `db/queries.py` com o SQL de retrieval explícito.
- **Conceito:** Pattern B + write/read path (§7.1, §7.2, §7.6).
- **Pronto quando:** teste prova que dois papers com `arxiv_id` diferentes mas mesmo DOI fundem num registro; segundo run da mesma área não re-surfacea o que já apareceu.

### Fase E2 — Reflexão e consolidação: episodic → semantic  *(↔ conceitual Fase 2)*
- **Objetivo:** o agente abstrai preferências dos runs e elas influenciam o ranking.
- **Entregável:** `agent reflect` gera uma nota pós-run; o `user_profile` muda ao longo de alguns runs e re-ranqueia o próximo digest.
- **Componentes:** `reflect.py` (Claude Sonnet gera nota curta, **grounded** em `arxiv_id`s concretos — sem evidência não vira preferência, §4.3); `consolidate.py` (atualiza `user_profile` com `confidence` e `expires_at`); hook no `read_path` para reordenar por perfil; prompt caching do perfil.
- **Conceito:** reflexão + consolidação + reflection grounding (§4.3, §9.1).
- **Cuidado:** *self-reinforcing error* — expiração/revisão periódica das preferências para uma reflexão errada não cristalizar.
- **Pronto quando:** existe um teste/observação de que um sinal de feedback move uma `statement` do perfil, e o ranking do run seguinte reflete isso.

### Fase E3 — Substrate relacional/grafo  *(↔ conceitual Fase 3)*
- **Objetivo:** ligar papers por relação, não só por similaridade.
- **Entregável:** o digest diz "este paper se conecta a X que você viu há 3 semanas".
- **Componentes:** `edges` (same_author, same_subarea, cites); `graph.py` (popular arestas no write path + **travessia** no read path, somando à busca vetorial).
- **Conceito:** substrate estruturado + cross-session coherence + esboço de causally grounded retrieval (§3.2, §5.5, §9.2).
- **Pronto quando:** o read path retorna pelo menos um vizinho relacional que a busca por similaridade pura não traria.

### Fase E4 — Avaliação: metric stack  *(↔ conceitual Fase 4)*
- **Objetivo:** medir se a memória ajuda, com número, não chute.
- **Entregável:** `agent metrics` reporta task effectiveness / memory quality / efficiency / governance, e um harness de ablation simples (memória on/off).
- **Componentes:** `feedback` (captura 👍/👎/⭐); `eval/` (taxa de relevância marcada, taxa de repetição indevida, latência/op, tokens de memória por run, teste de deleção total inclusive embeddings).
- **Conceito:** de recall para agentic utility (§5).
- **Pronto quando:** há números antes/depois de uma mudança e dá para atribuir o ganho a um componente.

### Fase E5 — Observability  *(↔ conceitual Fase 5)*
- **Objetivo:** memória debugável.
- **Entregável:** `memory_events` registra toda op (write/read/update/delete) com o contexto que disparou; `agent diff <run_a> <run_b>` mostra o que mudou no store.
- **Componentes:** migração para structlog; `observability/` com event log e "memory diff".
- **Conceito:** observability e debugging (§7.7).
- **Pronto quando:** dado um digest ruim, dá para localizar se a falha foi no write, no read, na compressão ou no raciocínio do LLM.

---

## 6. Stretch (fronteiras do §9 — só com base sólida)

| Stretch | Onde toca no código |
|---------|---------------------|
| *Learning to forget* (§9.4) | política em `consolidate.py`/`eval` além de expirar por data |
| *Causally grounded retrieval* (§9.2) | coluna `causal_parent` em `papers`, navegada no `read_path` |
| *Dual-buffer consolidation* (§9.1) | buffer "quente" em quarentena antes de `user_profile` virar permanente |
| Multi-fonte | novos adapters em `sources/` (S2, OpenAlex, DBLP, Papers with Code) — zero migração graças à identidade canônica |

---

## 7. Mapa fase de código → fase conceitual → entregável

| Código | Conceitual | Entrega que roda |
|--------|-----------|------------------|
| E0 | — | Pipeline em memória: arXiv → normaliza → embed → imprime |
| E1 | 1 | Pattern B: dedup canônica + "o que já vi sobre X?" |
| E2 | 2 | Perfil semântico que evolui e re-ranqueia |
| E3 | 3 | Conexões relacionais entre papers de runs diferentes |
| E4 | 4 | Métricas + ablation com número |
| E5 | 5 | Event log + memory diff |

---

## 8. Como abordar

Uma fase por vez; **parar quando o "pronto quando" estiver satisfeito**. O paper é explícito: a maioria do ganho real mora no Pattern B (E1) bem feito; complexidade (E3+, controle aprendido) só se justifica com dados na mão (§7.6, §10).
