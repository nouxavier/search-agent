# search-agent

Camada de memória persistente para um agente de pesquisa — construída por fases
(`E0…E5`), implementando na prática os conceitos de *Memory for Autonomous LLM
Agents* (Du, 2026, arXiv:2603.07670).

O agente busca papers no arXiv, normaliza, gera embeddings e — a partir da E1 —
lembra o que já viu entre execuções. Veja [docs/](docs/) para o plano completo:
[search-project.md](docs/search-project.md) (conceito), [plano-implementacao.md](docs/plano-implementacao.md)
(engenharia) e [rfc-memory-layer.md](docs/rfc-memory-layer.md) (spec técnica).

> 📚 **Novo no assunto?** O [guia de estudo — embeddings e grafos](docs/guia-estudo-embeddings-grafos.md)
> explica do zero, com analogias e o código lado a lado, como o agente acha o que é
> *parecido* (embeddings) e o que é *relacionado* (grafos).

**Fase atual: E2** — reflexão e consolidação (episodic → semantic). Sobre a E1
(dedup canônica + write/read path), o agente agora **reflete** sobre cada run —
com *grounding* obrigatório em arxiv_ids concretos — e abstrai um `user_profile`
de preferências que **evolui** (confidence + expiração) e **re-ranqueia** o
digest/consulta. `agent feedback` marca um paper como relevante, sinal que move o
perfil. Requer Postgres + migração aplicada (veja Setup); `agent reflect` usa o
Claude — por padrão pela sua **assinatura Pro/Max** (sem API key), ou pela API paga
(veja [Como o agente fala com o Claude](#como-o-agente-fala-com-o-claude)).

## O que o Ollama faz aqui

O Ollama roda um modelo de IA **localmente** (sem API, sem chave, sem nuvem) e tem
uma função só no projeto: **transformar texto em embedding** — um vetor de 1024
números que representa o *significado* do paper (título + abstract). É o `embedding:
dim=1024` que aparece na saída. Esses vetores são o que vai permitir, a partir da
E1, busca por similaridade (*"o que já vi sobre X?"*, *"esse paper é parecido com
aquele de 3 semanas atrás"*) — comparando significado, não palavras exatas. O
modelo usado é o **bge-m3** (multilíngue, 1024d, grátis/offline); é trocável por
outro embedder (ex.: Voyage AI) atrás da interface `Embedder` sem mexer no pipeline.

---

## Pré-requisitos

| Ferramenta | Para quê | Instalar (macOS) |
|---|---|---|
| **Python 3.12+** | runtime | já vem / `brew install python@3.12` |
| **uv** | deps + venv | `brew install uv` |
| **Ollama** | embeddings (bge-m3, local/offline) | `brew install --cask ollama-app` — **veja a nota abaixo** |
| **Docker** | Postgres + pgvector (usado a partir da E1) | Docker Desktop |
| **Claude** (LLM) | `smoke` / `reflect` (E2+) | assinatura **Pro/Max** via `claude` (padrão, sem key) **ou** API key — [detalhes](#como-o-agente-fala-com-o-claude) |

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

# 3. subir o Postgres e aplicar o schema (E1 em diante)
docker compose up -d
uv run alembic upgrade head
```

---

## Rodar

Antes de cada sessão, garanta que o Ollama está no ar:

```bash
# abra o app Ollama (menu bar) OU rode o servidor manualmente:
ollama serve &

# confirme:
curl -s localhost:11434/api/version          # → {"version":"..."}
```

### Os comandos numa olhada

Pensa no agente como um **assistente de pesquisa** com memória. Cada comando é uma
ordem que você dá pra ele:

| Comando | Em uma frase | O que faz com a memória |
|---|---|---|
| `agent run` | *"Vai ao arXiv, traz papers novos e arquiva."* | **escreve** — entra coisa nova |
| `agent query` | *"Me lembra o que já vimos sobre X."* | **lê** — só consulta o que já está guardado |
| `agent feedback` | *"Esse paper aqui me foi útil, anota."* | **ensina** — marca um paper como relevante |
| `agent reflect` | *"Pensa no último run e anota meus interesses."* | **ensina** — o Claude resume teus gostos |
| `agent profile` | *"Me diz o que você entendeu que eu gosto."* | **mostra** — lista as preferências acumuladas |
| `agent smoke` | *"Teste de microfone: o Claude tá me ouvindo?"* | **nada** — só checa a conexão com o LLM |

O fio condutor: **`run` põe coisa na memória, `query` tira. `feedback` e `reflect`
moldam o *gosto* do agente (o `user_profile`), e esse gosto re-ordena os próximos
`run`/`query` — papers alinhados ao perfil sobem.** `profile` é só a janela pra ver
esse gosto; `smoke` é um teste técnico que não mexe em nada.

Ordem típica de uma sessão. O passo 1 **imprime os ids** que os passos 2 e 3 usam —
cada paper sai com um `id=` e o run inteiro com um `run #`:

```text
[3] Beyond Vector Similarity: A Structural Analysis of Graph-Augmented Retrieval...
    id=27  arxiv_id=2606.06003  year=2026  Grama Chethan          ← esse 27 é o <paper_id>
...
✓ digest: 5 novos de 5 candidatos · store agora tem 11 papers.
  (run #5 · use `agent reflect 5` para refletir)                 ← esse 5 é o <run_id>
```

```bash
uv run agent run -a "retrieval augmented generation" -n 5   # 1. traz e arquiva → imprime os ids (veja acima)
uv run agent feedback <paper_id>                            # 2. (opcional) "gostei desse" — um id do passo 1, ex.: 27
uv run agent reflect <run_id>                               # 3. reflete sobre aquele run (o run #) → atualiza o perfil
uv run agent profile                                        # 4. confere o que ele aprendeu
uv run agent query "graph retrieval"                        # 5. consulta — já reordenado pelo perfil
```

### Pipeline principal — `agent run`

Busca no arXiv, **grava** com dedup canônica e mostra só o que é novo (papers já
surfaceados em runs anteriores não voltam):

```bash
uv run agent run                                    # área default (config.toml)
uv run agent run --area "memory for LLM agents" --limit 10
uv run agent run -a "retrieval augmented generation" -n 5
```

Saída: um bloco por paper novo, terminando em `✓ digest: N novos de M candidatos ·
store agora tem T papers`. Rodar de novo na mesma área → `Nada novo desde o último run`.

### Recuperar do histórico — `agent query`

Read path: "o que já vi sobre X?" por similaridade de embedding.

```bash
uv run agent query "trustworthy memory search for agents"
uv run agent query "retrieval augmented generation" -k 5
uv run agent query "long-horizon agents" --area "memory"   # filtro por título
uv run agent query "agent memory" --no-profile             # só similaridade, sem re-rank
```

A saída traz `sim≈` (similaridade à consulta), `perfil≈` (afinidade ao
`user_profile`) e `score` (mistura usada na ordenação).

### Reflexão e perfil — `reflect` / `feedback` / `profile` (E2)

```bash
# 1. marca um paper do digest como relevante (sinal que move o perfil)
uv run agent feedback <paper_id>           # paper_id aparece no `run`/`query`

# 2. reflete sobre um run → gera nota grounded e atualiza o user_profile
uv run agent reflect <run_id>               # usa o Claude (run_id é impresso no `run`)
                                            # provider em config.toml — Max por padrão, sem key

# 3. inspeciona as preferências vigentes
uv run agent profile
```

O perfil passa a influenciar o ranking do próximo `run`/`query`. Cada statement
expira (anti *self-reinforcing error*) e ganha confidence quando reaparece ou
recebe feedback.

### Sem Ollama (offline) — embedder fake

Vetor determinístico, não-semântico — pro pipeline rodar sem o modelo:

```bash
SEARCH_AGENT__EMBEDDER__PROVIDER=fake uv run agent run -n 3
```

### Como o agente fala com o Claude

O `reflect` e o `smoke` precisam do Claude. Tem dois jeitos, escolhidos pelo
`provider` em `[llm]` no [config.toml](config.toml):

| `provider` | Como cobra | Precisa de quê |
|---|---|---|
| `claude_cli` *(padrão)* | sua **assinatura Pro/Max** (sem API key) | estar logado no Claude Code (`claude`) |
| `anthropic` | crédito **pay-as-you-go** da API | `ANTHROPIC_API_KEY` no `.env` |
| `fake` | — | nada (resposta fixa, pra testes) |

O modo padrão (`claude_cli`) roteia pelo binário `claude -p`, então usa o login da
sua assinatura — **não gasta API key**. Pré-requisito: ter o Claude Code instalado e
logado.

```bash
# teste rápido de que o LLM responde (usa o provider do config.toml):
uv run agent smoke                     # → LLM (claude-haiku-4-5): Ok, search-agent.
```

**Usar a API paga em vez da assinatura** — copie `.env.example` para `.env`, ponha a
chave, e troque o provider:

```bash
cp .env.example .env                   # edite e cole ANTHROPIC_API_KEY=sk-ant-...
SEARCH_AGENT__LLM__PROVIDER=anthropic uv run agent smoke   # pontual
# ou fixe provider = "anthropic" no config.toml
```

> O `.env` é carregado automaticamente e fica fora do git. No modo `claude_cli` a
> `ANTHROPIC_API_KEY` é ignorada de propósito (pra cobrança ir pra assinatura).

### Testes

```bash
uv run pytest                # suíte do adapter arXiv
```

### Infra (Postgres + pgvector)

Sobe na E0, mas só é usado para gravar a partir da E1:

```bash
docker compose up -d
docker compose exec db psql -U search -d search_agent \
  -c "SELECT extversion FROM pg_extension WHERE extname='vector';"
docker compose down          # parar
```

---

## Configuração

Os defaults (idioma, área, threshold, modelo de embedding/LLM, URL do banco)
vivem em [config.toml](config.toml) — versionado, não no prompt. Sobrescreva por
variável de ambiente com o prefixo `SEARCH_AGENT__` e `__` como separador
aninhado:

```bash
SEARCH_AGENT__EMBEDDER__PROVIDER=fake \
SEARCH_AGENT__AGENT__PAPERS_PER_RUN=20 \
uv run agent run
```

---

## Estrutura

```
src/search_agent/
├── cli.py            # Typer: run / query / reflect / feedback / profile / smoke
├── config.py         # pydantic-settings (lê config.toml + env)
├── sources/          # adapters de fonte → RawPaper canônico (base.py + arxiv.py)
├── db/
│   ├── models.py     # schema ORM (papers, external_ids, sources, runs, run_papers,
│   │                 #   reflections, user_profile)
│   ├── session.py    # engine + session_scope
│   └── queries.py    # SQL cru de retrieval (vetor kNN + metadata)
├── memory/
│   ├── identity.py   # chave canônica (DOI → hash) — dedup cross-source
│   ├── write_path.py # filter → dedup/merge → tag → persist → link run (§7.1)
│   ├── read_path.py  # recall + re-rank por perfil (§7.2 / E2)
│   ├── reflect.py    # reflexão grounded pós-run (§4.3)
│   └── consolidate.py# reflexões → user_profile + afinidade (§9.1)
├── embeddings.py     # Embedder: OllamaEmbedder (bge-m3) | FakeEmbedder
├── llm.py            # LLMClient: AnthropicClient | ClaudeCliClient (Max) | FakeLLM
└── logging_setup.py  # logging JSON
alembic/              # migrações versionadas (0001_e1_episodic, 0002_e2_semantic)
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
