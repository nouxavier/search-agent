# search-agent

Camada de memória persistente para um agente de pesquisa — construída por fases
(`E0…E5`), implementando na prática os conceitos de *Memory for Autonomous LLM
Agents* (Du, 2026, arXiv:2603.07670).

O agente busca papers no arXiv, normaliza, gera embeddings e — a partir da E1 —
lembra o que já viu entre execuções. Veja [docs/](docs/) para o plano completo:
[search-project.md](docs/search-project.md) (conceito), [plano-implementacao.md](docs/plano-implementacao.md)
(engenharia) e [rfc-memory-layer.md](docs/rfc-memory-layer.md) (spec técnica).

**Fase atual: E0** — pipeline ponta a ponta em memória (arXiv → normaliza → embed
→ imprime). Ainda **não persiste** nada; o Postgres sobe mas só passa a gravar na E1.

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
| Anthropic API key | só pro `agent smoke` / fases E2+ | `export ANTHROPIC_API_KEY=sk-ant-...` |

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

### Pipeline principal — `agent run`

```bash
uv run agent run                                    # área default (config.toml), 10 papers
uv run agent run --area "memory for LLM agents" --limit 10
uv run agent run -a "retrieval augmented generation" -n 5
```

Saída esperada: um bloco por paper (título, `arxiv_id`, ano, autor, `embedding:
dim=1024`), terminando em `✓ N papers normalizados com embedding (dim=1024)`.

### Sem Ollama (offline) — embedder fake

Vetor determinístico, não-semântico — pro pipeline rodar sem o modelo:

```bash
SEARCH_AGENT__EMBEDDER__PROVIDER=fake uv run agent run -n 3
```

### Smoke test do LLM (Claude) — precisa de API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run agent smoke
```

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
├── cli.py            # Typer: agent run / smoke
├── config.py         # pydantic-settings (lê config.toml + env)
├── sources/          # adapters de fonte → RawPaper canônico (base.py + arxiv.py)
├── embeddings.py     # Embedder: OllamaEmbedder (bge-m3) | FakeEmbedder
├── llm.py            # LLMClient: AnthropicClient (usado a partir da E2)
└── logging_setup.py  # logging JSON
alembic/              # migrações versionadas (schema entra na E1)
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
| `agent smoke` falha com auth | falta `ANTHROPIC_API_KEY` no ambiente |
```
