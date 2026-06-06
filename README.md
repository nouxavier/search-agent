# search-agent

Camada de memória persistente para um agente de pesquisa — construída por fases
(`E0…E5`), implementando na prática os conceitos de *Memory for Autonomous LLM
Agents* (Du, 2026, arXiv:2603.07670).

O agente busca papers no arXiv, normaliza, gera embeddings e **lembra o que já viu
entre execuções**: deduplica, aprende suas preferências, conecta papers por relação,
mede se a memória ajuda e registra toda operação pra auditoria.

> 📚 **Novo no assunto?** O [guia de estudo — embeddings e grafos](docs/guia-estudo-embeddings-grafos.md)
> explica do zero, com analogias e o código lado a lado, como o agente acha o que é
> *parecido* (embeddings) e o que é *relacionado* (grafos).

**Status: E0→E5 completo.** O arco inteiro do plano:

| Fase | O que dá ao agente |
|---|---|
| E0–E1 | lembra (dedup canônica + *"o que já vi sobre X?"*) |
| E2 | aprende seu gosto (perfil que evolui e re-ranqueia) |
| E3 | conecta por relação (grafo de arestas) |
| E4 | **mede** se a memória ajuda (metric stack + ablation) |
| E5 | **explica** o que a memória fez (event log + diff) |

Mais a fundo em [docs/](docs/): [search-project.md](docs/search-project.md) (conceito),
[plano-implementacao.md](docs/plano-implementacao.md) (engenharia),
[rfc-memory-layer.md](docs/rfc-memory-layer.md) (spec técnica).

## O que o Ollama faz aqui

O Ollama roda um modelo de IA **localmente** (sem API, sem chave, sem nuvem) com uma
função só: **transformar texto em embedding** — um vetor de 1024 números que
representa o *significado* do paper (título + abstract). Esses vetores viabilizam a
busca por similaridade (*"o que já vi sobre X?"*) — comparando significado, não
palavras exatas. O modelo é o **bge-m3** (multilíngue, 1024d, grátis/offline);
trocável por outro embedder (ex.: Voyage AI) atrás da interface `Embedder` sem mexer
no pipeline.

---

## Pré-requisitos

| Ferramenta | Para quê | Instalar (macOS) |
|---|---|---|
| **Python 3.12+** | runtime | já vem / `brew install python@3.12` |
| **uv** | deps + venv | `brew install uv` |
| **Ollama** | embeddings (bge-m3, local/offline) | `brew install --cask ollama-app` — **veja a nota abaixo** |
| **Docker** | Postgres + pgvector | Docker Desktop |
| **Claude** (LLM) | `smoke` / `reflect` | assinatura **Pro/Max** via `claude` (padrão, sem key) **ou** API key — [detalhes](#como-o-agente-fala-com-o-claude) |

> ### ⚠️ Ollama: use o **cask**, não a formula
> A formula do Homebrew (`brew install ollama`) instala mas o runtime vem
> quebrado — `/api/embed` retorna `500: llama-server binary not found`. Use o
> **cask**:
> ```bash
> brew install --cask ollama-app
> ```
> Se você já instalou a formula por engano: `brew uninstall ollama` e depois o cask.

---

## Setup (uma vez)

```bash
git clone <repo> && cd search-agent

# 1. dependências + venv (uv lê o pyproject.toml + uv.lock)
uv sync

# 2. baixar o modelo de embedding (~1.2 GB, uma vez)
ollama pull bge-m3

# 3. subir o Postgres e aplicar o schema
docker compose up -d
uv run alembic upgrade head
```

---

## Os comandos

Pensa no agente como um **assistente de pesquisa** com memória. Cada comando é uma
ordem que você dá pra ele:

| Comando | Em uma frase | Papel |
|---|---|---|
| `agent run` | *"Vai ao arXiv, traz papers novos e arquiva."* | **escreve** na memória |
| `agent query` | *"Me lembra o que já vimos sobre X."* | **lê** da memória |
| `agent feedback` | *"Esse aqui me serviu / é ruído."* (up·down·star) | **ensina** o gosto |
| `agent reflect` | *"Pensa no run e anota meus interesses."* | **ensina** (via Claude) |
| `agent profile` | *"Me diz o que entendeu que eu gosto."* | **mostra** o perfil |
| `agent metrics` | *"A memória está ajudando? Com número."* | **mede** |
| `agent ablation` | *"O perfil puxa peso no ranking?"* | **mede** (isola componente) |
| `agent diff` | *"O que mudou no store entre dois runs?"* | **audita** |
| `agent events` | *"Mostra as últimas operações de memória."* | **audita** |
| `agent smoke` | *"O Claude tá me ouvindo?"* | testa o LLM |

O fio condutor: **`run` põe coisa na memória, `query` tira. `feedback` e `reflect`
moldam o *gosto* (o `user_profile`), e esse gosto re-ordena os próximos `run`/`query`.**
`metrics`/`ablation` dizem se está valendo; `diff`/`events` explicam o que aconteceu.

---

## Operação no dia a dia

### Antes de cada sessão — ligar a infra

```bash
docker compose up -d                         # Postgres
ollama serve &                               # ou abra o app Ollama (menu bar)
curl -s localhost:11434/api/version          # confirma o Ollama: {"version":"..."}
```

O Claude você já está logado (Max), então `reflect` funciona sem API key.

### De manhã — buscar o que há de novo

```bash
uv run agent run -a "retrieval augmented generation"   # -n N limita a quantidade
uv run agent run -a "intelligent systems"

```

Traz papers novos (sem repetir o que já viu), re-ranqueados pelo seu perfil, com as
pontes relacionais do grafo. A saída imprime os `id=` e o `run #` que você usa depois:

```text
[1] Graph-Augmented Retrieval for Knowledge Graphs
    id=27  arxiv_id=2606.06003  year=2026  Grama Chethan
    ↳ conecta-se a "QCFuse: Query-Aware Cache Fusion…" (id=29) via same_subarea
...
✓ digest: 5 novos de 5 candidatos · store agora tem 26 papers.
  (run #9 · use `agent reflect 9` para refletir)
```

### Ao longo do dia — ensinar e consultar

```bash
uv run agent feedback 27 -s up      # "me serviu"  (-s star pros TOP, -s down pra ruído)
uv run agent query "o que já vi sobre memória de agentes?"   # busca no que já leu
uv run agent query "graph retrieval" -k 5 --no-profile       # top-5, só similaridade
```

`query` mostra `sim≈` (à consulta), `perfil≈` (afinidade ao perfil) e `score` (mistura).

### Fim do dia/semana — consolidar e revisar

```bash
uv run agent reflect 9        # o Claude lê o run #9 e atualiza seu perfil (grounded)
uv run agent profile          # confere o que ele aprendeu (preferências + confidence)
```

Cada preferência expira (anti *self-reinforcing error*) e ganha confidence quando
reaparece ou recebe feedback.

### Manutenção/insight (semanal)

```bash
uv run agent metrics                       # effectiveness, ruído, tamanho, latência
uv run agent ablation "<consulta típica>"  # o perfil sobe os papers relevantes?
uv run agent diff 5 9                       # o que mudou no store entre os runs 5 e 9
uv run agent events --op update            # auditar: por que o perfil mudou?
```

### Cola rápida

| Ritmo | Comandos |
|---|---|
| Todo dia | `run` → `feedback` → `query` (quando precisar) |
| Semanal | `reflect` → `profile` → `metrics` |
| Investigar algo estranho | `diff`, `events`, `ablation` |

**Notas de operação**
- **Onde mora seu conhecimento:** tudo no Postgres do compose. Backup = `pg_dump`.
  Apagar um paper apaga tudo dele em cascata (governança, E4).
- **Custo:** `run`/`query` não tocam o LLM (só arXiv + Ollama, grátis). Só `reflect`
  chama o Claude — e pela sua **Max**, sem gastar API.
- **Automação:** o `run` diário é candidato a agendamento — um cron chamando
  `uv run agent run`, ou os fluxos `/schedule`/`/loop` do Claude Code.

---

## Como o agente fala com o Claude

O `reflect` e o `smoke` precisam do Claude. Tem três modos, escolhidos pelo
`provider` em `[llm]` no [config.toml](config.toml):

| `provider` | Como cobra | Precisa de quê |
|---|---|---|
| `claude_cli` *(padrão)* | sua **assinatura Pro/Max** (sem API key) | estar logado no Claude Code (`claude`) |
| `anthropic` | crédito **pay-as-you-go** da API | `ANTHROPIC_API_KEY` no `.env` |
| `fake` | — | nada (resposta fixa, pra testes) |

O modo padrão roteia pelo binário `claude -p`, usando o login da sua assinatura —
**não gasta API key**. Teste com `uv run agent smoke` (→ `Ok, search-agent.`).

**Usar a API paga** em vez da assinatura:

```bash
cp .env.example .env                   # edite e cole ANTHROPIC_API_KEY=sk-ant-...
SEARCH_AGENT__LLM__PROVIDER=anthropic uv run agent smoke   # pontual
# ou fixe provider = "anthropic" no config.toml
```

> O `.env` é carregado automaticamente e fica fora do git. No modo `claude_cli` a
> `ANTHROPIC_API_KEY` é ignorada de propósito (pra cobrança ir pra assinatura).

---

## Configuração

Os defaults (idioma, área, threshold, modelos, URL do banco) vivem em
[config.toml](config.toml) — versionado, não no prompt. Sobrescreva por variável de
ambiente com o prefixo `SEARCH_AGENT__` e `__` como separador aninhado (o env
**sobrepõe** o toml):

```bash
SEARCH_AGENT__EMBEDDER__PROVIDER=fake \
SEARCH_AGENT__AGENT__PAPERS_PER_RUN=20 \
uv run agent run
```

`EMBEDDER__PROVIDER=fake` dá um embedder offline (vetor determinístico, não-semântico)
pra rodar o pipeline sem Ollama — útil em testes.

### Quais temas o agente busca

Dois níveis no [config.toml](config.toml) controlam o que entra na busca:

```toml
[agent]
default_area = "LLM Agents"          # o ASSUNTO — vira a query no arXiv (sem -a)

[source]
categories = ["cs.AI", "cs.CL", "cs.LG"]   # o UNIVERSO — categorias do arXiv onde procurar
```

A `default_area` é *o quê* (o tema); as `categories` são *o onde* (cs.AI = IA,
cs.CL = linguagem/NLP, cs.LG = machine learning). **Ver** o tema atual = olhar o
`config.toml`. **Mudar**:

```bash
uv run agent run -a "retrieval augmented generation"   # só neste run (não edita nada)
# ou edite default_area no config.toml pra fixar de vez
```

> A *área* diz **onde** o agente pesca; o *perfil* (`user_profile`, construído por
> `feedback`/`reflect`) diz **quais peixes** ele te mostra primeiro. Veja o perfil
> com `agent profile`.

#### Exemplos de assuntos — Sistemas Inteligentes

Termos em **inglês** (o arXiv casa a frase no título/abstract dos papers, que são em
inglês). Use com `-a`, ou fixe seu favorito em `default_area`:

| Assunto (`-a "…"`) | Cobre | Categorias arXiv |
|---|---|---|
| `intelligent systems` | guarda-chuva amplo | cs.AI |
| `multi-agent systems` | coordenação/competição entre agentes | cs.MA, cs.AI |
| `autonomous agents` | agentes que agem sozinhos | cs.AI, cs.MA |
| `reinforcement learning` | aprendizado por recompensa | cs.LG, cs.AI |
| `neuro-symbolic reasoning` | une redes neurais + lógica | cs.AI, cs.LG |
| `knowledge representation and reasoning` | KR&R clássico | cs.AI |
| `self-adaptive systems` | sistemas que se ajustam sozinhos | cs.AI, cs.SE |
| `swarm intelligence` | inteligência coletiva/enxame | cs.NE, cs.MA |
| `evolutionary computation` | algoritmos evolutivos | cs.NE |
| `explainable AI` | IA interpretável | cs.AI, cs.LG |
| `cognitive architectures` | arquiteturas cognitivas | cs.AI |
| `intelligent robotics` | robôs autônomos | cs.RO, cs.AI |

```bash
uv run agent run -a "multi-agent systems"
uv run agent run -a "neuro-symbolic reasoning"
```

> **Saiu de IA/NLP/ML?** Aí ajuste também as `categories` no `config.toml` —
> ex.: `cs.MA` (multi-agente), `cs.NE` (computação neural/evolutiva), `cs.RO`
> (robótica), `cs.SE` (engenharia de software). Sem isso, a busca fica restrita a
> cs.AI/cs.CL/cs.LG e pode perder papers da nova área.

---

## Testes e infra

```bash
uv run pytest                # suíte completa (usa Postgres efêmero via testcontainers)

docker compose up -d         # subir o banco
docker compose down          # parar
```

---

## Estrutura

```text
src/search_agent/
├── cli.py            # Typer: run/query/feedback/reflect/profile/metrics/ablation/diff/events/smoke
├── config.py         # pydantic-settings (config.toml + env; env sobrepõe)
├── sources/          # adapters de fonte → RawPaper canônico (base.py + arxiv.py)
├── db/
│   ├── models.py     # ORM: papers, external_ids, sources, runs, run_papers,
│   │                 #   reflections, user_profile, edges, feedback, memory_events
│   ├── session.py    # engine + session_scope
│   └── queries.py    # SQL cru de retrieval (vetor kNN + metadata)
├── memory/
│   ├── identity.py   # chave canônica (DOI → hash) — dedup cross-source
│   ├── write_path.py # filter → dedup/merge → persist → edges → link run (§7.1)
│   ├── read_path.py  # recall + re-rank por perfil (§7.2 / E2)
│   ├── graph.py      # arestas + travessia relacional (§3.2 / E3)
│   ├── reflect.py    # reflexão grounded pós-run (§4.3 / E2)
│   └── consolidate.py# reflexões → user_profile + afinidade (§9.1 / E2)
├── eval/             # E4: metrics.py (metric stack) + ablation.py (perfil on/off)
├── observability/    # E5: events.py (event log) + diff.py (memory diff)
├── embeddings.py     # Embedder: OllamaEmbedder (bge-m3) | FakeEmbedder
├── llm.py            # LLMClient: AnthropicClient | ClaudeCliClient (Max) | FakeLLM
└── logging_setup.py  # logging JSON
alembic/versions/     # 0001_e1_episodic … 0005_e5_observability
docker-compose.yml    # Postgres 16 + pgvector
config.toml           # defaults versionados
```

## Troubleshooting

| Sintoma | Causa / fix |
|---|---|
| `500 ... llama-server binary not found` | Formula do ollama em vez do cask — veja a nota acima |
| `Could not connect ... 11434` | Ollama não está rodando — `ollama serve &` ou abra o app |
| `301 Moved Permanently` (arXiv) | já tratado (https + follow_redirects); se aparecer, atualize o repo |
| `Cannot connect to the Docker daemon` | Docker Desktop não está aberto |
| `agent smoke` falha com `invalid x-api-key` | provider `anthropic` sem `ANTHROPIC_API_KEY` válida — use `claude_cli` (Max) ou ponha a key no `.env` |
| `agent smoke` (claude_cli) falha | binário `claude` não encontrado ou não logado — instale o Claude Code e rode `claude` uma vez |
