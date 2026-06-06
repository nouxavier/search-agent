# Projeto: Camada de Memória para um Agente de Pesquisa

**Objetivo:** construir, do zero e por fases, uma camada de memória persistente para um agente de LLM — usando o domínio dos seus digests de pesquisa como substrato — implementando na prática os conceitos do survey *"Memory for Autonomous LLM Agents"* (Pengfei Du, 2026, arXiv:2603.07670).

A meta não é o produto final perfeito, é **internalizar cada mecanismo do paper escrevendo o código dele**. Cada fase entrega algo que roda.

---

## Por que este projeto

O paper define memória como um loop **write–manage–read** e organiza tudo em três eixos (temporal scope, substrate, control policy). Este projeto te faz passar pela evolução que o paper descreve: começar simples (Pattern B) e só adicionar complexidade quando a anterior estiver sólida.

Domínio escolhido: um agente que, a cada semana, busca papers no arXiv e gera um digest. É o caso ideal porque é **multi-sessão** e **interdependente** (a semana 5 deveria "lembrar" o que apareceu na semana 1) — exatamente o cenário onde memória faz diferença.

---

## Stack sugerida (simples de propósito)

- **Python** como linguagem.
- **PostgreSQL + pgvector** como store único (estruturado + vetorial no mesmo banco — evita infra extra).
- Um **modelo de embeddings** (qualquer um; o ponto é o pipeline, não o SOTA).
- A **API de um LLM** para reflexão/consolidação.
- Logging estruturado (a stdlib `logging` já basta para começar).

Regra de ouro do paper: **comece pelo Pattern B** (context window + retrieval store) e só evolua para controle aprendido quando os dados justificarem.

---

## Fase 0 — Mapear o sistema nos três eixos

**O que fazer:** antes de qualquer código, escreva um documento de 1 página classificando seu agente nos três eixos do Capítulo 3.

- *Temporal scope:* o que é working / episodic / semantic / procedural hoje?
- *Substrate:* o que você vai guardar em tabela estruturada vs vetorial?
- *Control policy:* hoje é heuristic (busca fixa, top-N) — anote isso como ponto de partida.

**Conceito do paper:** taxonomia de três dimensões (Cap. 3).
**Pronto quando:** você consegue dizer, para cada registro que o agente cria, em qual eixo ele cai.

---

## Fase 1 — Baseline Pattern B (write path + read path)

**O que fazer:** modele o banco e implemente o ciclo básico.

- Tabela `runs` (cada execução do digest) e `papers` (registros episódicos) com metadata: `timestamp`, `area`, `embedding`, mais a **identidade canônica** e as **fontes** (ver abaixo).
- **Identidade do paper (source-agnostic):** a única fonte hoje é o arXiv, mas o schema já assume múltiplas fontes. Cada `paper` tem um `paper_id` canônico resolvido por **DOI quando existir → senão título normalizado + 1º autor + ano**. Os IDs por fonte (`arxiv_id`, futuro `s2_id`, `openalex_id`, `doi`) vivem numa tabela de **aliases** (`external_ids`), e `sources` é a **lista** de fontes onde o paper foi visto — não um valor único.
- **Write path** (Cap. 7.1): antes de gravar, faça `filtering` (descartar registro de baixo sinal), `deduplication` (resolver pela chave canônica acima — o mesmo paper vindo de fontes diferentes funde num só registro, com os IDs como aliases) e `metadata tagging`.
- **Read path** (Cap. 7.2): dada a área do dia, recupere os papers relevantes do histórico por similaridade + filtro de metadata.

**Conceito do paper:** Pattern B + write/read path (Cap. 7.1, 7.2, 7.6).
**Pronto quando:** dois runs seguidos não repetem o mesmo paper (mesmo se vierem de fontes distintas), e você consegue perguntar "o que já vi sobre X?".

> **Nota de escopo (multi-fonte):** começamos só com arXiv, mas o design já contempla receber outras (Semantic Scholar, OpenAlex, DBLP, Papers with Code). Adicionar uma fonte deve ser só escrever um *adapter* que normaliza o resultado para o schema canônico — sem migração. Por isso a dedup é por identidade canônica, e não por `arxiv_id`.

---

## Fase 2 — Reflexão e consolidação (episodic → semantic)

**O que fazer:** depois de cada run, gere uma nota de reflexão e atualize um perfil de preferências.

- *Reflection* (Cap. 4.3): após o run, peça ao LLM uma nota curta — "o que foi relevante nesta semana e por quê".
- *Consolidation* (Cap. 3.1 / 9.1): mantenha um registro semântico `user_profile` que se atualiza a partir das reflexões (ex.: "tende a marcar como relevante papers de eficiência de inferência").
- *Reflection grounding* (Cap. 4.3): exija que cada item da reflexão aponte para `arxiv_id`s concretos. Sem evidência, não vira preferência.

**Conceito do paper:** memória reflexiva + consolidação (Cap. 4.3, 3.1, 9.1).
**Pronto quando:** o perfil semântico muda ao longo de alguns runs e influencia o ranking do próximo digest.

**Cuidado a implementar:** *self-reinforcing error*. Não deixe uma reflexão errada se cristalizar — adicione expiração ou revisão periódica das preferências.

---

## Fase 3 — Substrate estruturado / grafo (coerência entre sessões)

**O que fazer:** ligue papers entre runs por relações, não só por similaridade.

- Crie arestas: mesmo autor, mesma sub-área, citação cruzada.
- No read path, além da busca vetorial, **navegue pelas relações** (ex.: "este paper cita um que apareceu há 3 semanas").

**Conceito do paper:** substrate estruturado + cross-session coherence + esboço de causally grounded retrieval (Cap. 3.2, 5.5, 9.2).
**Pronto quando:** o digest consegue dizer "este paper se conecta a X que você viu antes" — algo que busca por similaridade pura não daria.

---

## Fase 4 — Avaliação (metric stack)

**O que fazer:** defina como medir se a memória está ajudando, sem chutar.

Monte um mini metric stack (Cap. 5.4):
- *Task effectiveness:* taxa de "papers que eu de fato marquei como relevantes".
- *Memory quality:* taxa de repetição indevida, registros recuperados mas inúteis.
- *Efficiency:* latência por operação de memória, tokens gastos com conteúdo de memória por run.
- *Governance:* dá pra deletar um paper/preferência de todas as tabelas (incluindo embeddings)?

**Conceito do paper:** de recall para agentic utility (Cap. 5).
**Pronto quando:** você tem números antes/depois de uma mudança e sabe atribuir o ganho a um componente (ablation simples).

---

## Fase 5 — Observability

**O que fazer:** torne a memória debugável.

- Logue **toda** operação (write/read/update/delete) com o contexto que disparou.
- Implemente um "memory diff": o que mudou no store entre dois runs.

**Conceito do paper:** observability e debugging (Cap. 7.7).
**Pronto quando:** dado um digest ruim, você consegue dizer se o problema foi no write, no read, na compressão ou no raciocínio do LLM.

---

## Stretch goals (as fronteiras do Cap. 9)

- *Learning to forget* (9.4): política de esquecimento seletivo em vez de só expirar por data.
- *Causally grounded retrieval* (9.2): anotar um "causal parent" em cada registro no momento da escrita e navegar por isso na leitura.
- *Dual-buffer consolidation* (9.1): registros novos ficam num buffer "quente" em quarentena e só viram permanentes após passar por checagem.

---

## Mapa: fase → o que você aprende

| Fase | Conceito do paper internalizado |
|------|----------------------------------|
| 0 | Taxonomia de três eixos |
| 1 | Pattern B, write/read path |
| 2 | Reflexão, consolidação, reflection grounding |
| 3 | Substrate estruturado, coerência entre sessões |
| 4 | Avaliação de utilidade (não só recall) |
| 5 | Observability de memória |
| Stretch | Forgetting, causal retrieval, consolidação principiada |

---

## Como abordar

Faça uma fase por vez e **pare quando o "pronto quando" estiver satisfeito** — não pule pro controle aprendido só porque é mais interessante. O paper é explícito: a maioria dos ganhos reais mora no Pattern B bem feito, e a complexidade só se justifica com dados na mão.