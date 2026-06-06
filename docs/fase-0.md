# Fase 0 — Mapeamento da Memória nos Três Eixos

| | |
|---|---|
| **Projeto** | Camada de Memória para o Agente de Pesquisa (`search-agent`) |
| **Fase** | 0 — Diagnóstico (sem código) |
| **Base conceitual** | Du, P. (2026). *Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers.* arXiv:2603.07670 |
| **Objetivo** | Classificar o sistema atual nos três eixos da taxonomia do paper, com a justificativa de cada classificação, para localizar os buracos **antes** de construir. |

> Convenção: referências ao paper aparecem como `(§X.Y)` indicando a seção correspondente em Du (2026). Todo o conteúdo é parafraseado.

---

## 0. Por que esta fase existe

Antes de construir memória, é preciso entender o que já se tem. O paper dá três motivos que tornam esse diagnóstico não-opcional:

1. **Memória é um sistema, não um depósito.** O paper formaliza memória como um loop *write–manage–read* acoplado à percepção e à ação (§2.1): a cada passo o agente lê da memória para decidir, e escreve/gerencia a memória depois de agir. Construir esse loop sem saber o que hoje é lido e escrito é construir às cegas.

2. **Memória é o *belief state* do agente.** Formalmente, o sistema é um POMDP e a memória faz o papel do estado de crença — um resumo *suficiente* da história que substitui o estado real do mundo, sob orçamento de computação e storage (§2.2). Não se projeta um belief state sem antes descrever qual é o resumo suficiente atual.

3. **Não existe memória "certa", só trade-offs.** O paper mostra cinco objetivos que se opõem — utility, efficiency, adaptivity, faithfulness e governance (§2.3). Projetar memória é escolher o ponto de equilíbrio para o caso de uso, e isso exige saber onde o sistema está hoje em cada eixo.

A taxonomia de três dimensões (§3) é o instrumento desse diagnóstico. A Fase 0 é, portanto, "localizar-se no mapa antes de navegar" — e é também a intervenção mais barata do projeto: o paper conclui (§10) que arquitetura de memória costuma receber uma fração do cuidado dado à escolha do modelo, e que inverter isso é a maior alavanca disponível. Uma hora de classificação honesta aqui evita semanas de retrabalho adiante.

---

## 1. Fundamento conceitual — os três eixos

O paper organiza qualquer sistema de memória em três dimensões **ortogonais** (independentes entre si). A inspiração vem da ciência cognitiva: designers de agentes acabam, muitas vezes sem perceber, espelhando a estrutura da memória humana (§3).

### 1.1 Temporal scope — *que tipo de coisa, e por quanto tempo* (§3.1)

Quatro tipos, emprestados da psicologia (Atkinson & Shiffrin; Tulving; Baddeley; Squire — citados em §3.1):

- **Working memory** — o que cabe no context window agora. O LLM é o "executivo central" e a janela é o buffer; ambos compartilham o mesmo gargalo: capacidade limitada.
- **Episodic memory** — registros de experiências concretas: uma chamada de ferramenta, um turno, uma observação, com timestamp e embedding para recuperação posterior.
- **Semantic memory** — conhecimento abstraído e des-contextualizado. Vários episódios consolidam num fato semântico (ex.: três correções de data viram "o usuário prefere DD/MM/YYYY").
- **Procedural memory** — skills reutilizáveis e executáveis (o exemplo do paper é a skill library do Voyager).

**Por que importa:** o paper destaca que a pergunta difícil é a *transition policy* — quando um registro episódico "se forma" em semântico (consolidation). E o diagnóstico honesto do campo: a maioria dos sistemas implementa bem só duas dessas camadas e trata as transições com heurísticas grosseiras, sendo a consolidação a parte mais mal-resolvida (§3.1, retomado como problema em aberto em §9.1).

### 1.2 Representational substrate — *como é fisicamente guardado* (§3.2)

O substrate **restringe o que o agente consegue fazer** com a memória:

- **Context-resident text** — resumos, scratchpads, no próprio prompt. Transparente, zero infraestrutura, mas limitadíssimo em capacidade.
- **Vector-indexed stores** — registros viram embeddings; busca por similaridade. Escala para milhões, mas perde relações: responde "o que é mais parecido?", não "o que causou o quê?".
- **Structured stores** — SQL, key-value, knowledge graphs. Preservam relações e permitem queries precisas ("todas as falhas do serviço X nos últimos 7 dias"), ao custo de projetar um schema.
- **Executable repositories** — bibliotecas de skills/código que o agente invoca direto.
- **Hybrid** — o padrão em produção (o MemGPT empilha context + recall DB + arquivo vetorial).

### 1.3 Control policy — *quem decide o quê* (§3.3)

O eixo que o paper chama de mais consequente e menos discutido — quem decide o que guardar, recuperar e descartar:

- **Heuristic** — regras fixas (top-k, resumir a cada n turnos, expirar após d dias). Previsível, fácil de debugar, cego ao contexto.
- **Prompted self-control** — operações de memória viram tool calls e o LLM decide quando chamá-las (ex.: `core_memory_append` do MemGPT).
- **Learned** — operações de memória como ações de uma política treinada por RL (ex.: AgeMem, §4.5). Mais teto, custo de treino alto, baixa interpretabilidade.

---

## 2. Sistema sob análise

Agente que, sob comando curto, busca papers recentes no arXiv em **4 domínios** (LLM Agents, Model Efficiency, AI Safety, MLOps/Inference Infra), cura ~10–12 papers (~3 por domínio, 3 marcados como highlights) e renderiza um **widget interativo** com cards (contribution, method, evidence, practical relevance, limitations, tags, autores, link). Saída em inglês.

> **Direção de evolução (multi-fonte):** hoje a única fonte é o arXiv, mas o roadmap prevê buscar em **várias bases de engenharia/CS** (Semantic Scholar, OpenAlex, DBLP, Papers with Code). Isso não muda os três eixos deste diagnóstico, mas torna a **deduplicação cross-source** parte central do write path da Fase 1: o mesmo paper aparece em fontes diferentes com IDs diferentes, então a identidade do registro passa a ser canônica (DOI → título+autor+ano), com os IDs por fonte guardados como aliases. Ver §9 e Fase 1.

**Premissa do estado atual (a confirmar no código):** cada run é independente; os resultados vivem apenas no widget daquela execução; não há persistência entre runs; os defaults (idioma, número de domínios, papers por área, critério de highlight) vivem no prompt/instruções, não em código de dados.

---

## 3. Classificação — Eixo 1: Temporal scope

| Tipo | Definição (§3.1) | No sistema | Estado | Por quê / consequência |
|------|------------------|------------|--------|------------------------|
| **Working** | O que cabe no context window | Prompt + resultados de busca do run | **Presente e dominante** | É quase toda a memória atual. Sozinha, é stateless: nada sobrevive ao run. |
| **Episodic** | Registros de experiências concretas | Cada run; cada paper; cada busca | **Ausente (não persistido)** | Sem episodic não há como evitar repetir papers nem manter coerência entre sessões — o paper mostra que coerência cross-session é um desafio distinto e em grande parte não resolvido (§5.5). **Buraco principal.** |
| **Semantic** | Conhecimento abstraído | "O que a Noua valoriza"; defaults | **Fora do sistema** | Existe, mas no loop humano/prompt — o agente não lê nem atualiza. Transformar episódios em semântico é justamente a consolidation que o paper diz ser a parte mal-resolvida (§3.1, §9.1). |
| **Procedural** | Skills/procedimentos reutilizáveis | Pipeline busca→curadoria→render; schema do card | **Presente e estável** | Ponto forte. Bem definido no workflow. |

**Detalhe do "por quê" para cada buraco:**

- **Episodic ausente → repetição e amnésia entre semanas.** O paper é explícito que long context não substitui isso (§5.5): mesmo uma janela enorme não dá armazenamento persistente entre sessões nem recuperação seletiva de meses de histórico (§8). O desejo de "histórico" e "comparação semana a semana" é, na taxonomia, o pedido por episodic memory de verdade.

- **Semantic fora do sistema → personalização que não evolui sozinha.** Hoje, para o digest refletir suas preferências, você precisa estar no loop. O paper trata a consolidação (episódico → semântico) como underserved e propõe processos principiados como o *dual-buffer* (registros novos em quarentena antes de virar permanentes), inspirado no hipocampo (§9.1). O desejo de "favoritos/ratings" é o gatilho natural dessa consolidação.

---

## 4. Classificação — Eixo 2: Representational substrate

- **Atual:** *context-resident text* (§3.2). Tudo vive no prompt e num widget transitório que embute resultados como conteúdo estático.

- **Por que é limitante:** é o substrate mais transparente, mas também o mais limitado em capacidade (§3.2). E o paper alerta para duas patologias do context-resident quando a história cresce: *summarization drift* — cada passada de compressão descarta silenciosamente detalhes raros e críticos (§4.1) — e *attentional dilution* / "lost in the middle" — info no meio de uma janela longa é recuperada de forma menos confiável (§4.1). Para você, o sintoma equivalente é simples: hoje nada sobrevive ao run.

- **Por que o alvo é structured + vector:** a busca vetorial pura responde "o que é parecido?" mas não "o que se relaciona com o quê?" (§3.2). Como parte do seu objetivo é **ligar papers ao longo do tempo** (mesmo autor, citação cruzada, mesma sub-área), você precisa de um substrate estruturado/grafo — o paper coloca isso na fronteira do *causally grounded retrieval* (§9.2), onde se busca pelo que é causalmente/relacionalmente relevante, não só pelo mais similar. PostgreSQL + pgvector dá os dois no mesmo banco.

---

## 5. Classificação — Eixo 3: Control policy

- **Atual:** *heuristic* (§3.3), ponta a ponta — busca fixa nos 4 domínios, ~3 papers por área, janela de "últimas semanas", schema de card fixo. A escolha dos highlights é julgamento do LLM, mas não é política aprendida nem auditável.

- **Operações de memória:** nenhuma ainda — não há memória persistente a gerenciar. Este eixo só ganha conteúdo a partir da Fase 1.

- **Por que está certo manter heurístico agora:** o paper recomenda explicitamente **começar pelo Pattern B** (context + retrieval store com controle simples) e só evoluir para tiered/learned control (Pattern C) quando dados empíricos mostrarem ganho no seu workload (§7.6). Pular direto para controle aprendido (RL) traria custo de treino alto, risco de *learned forgetting* deletar info importante e baixa interpretabilidade (§4.5) — sem justificativa nesta fase.

---

## 6. Diagnóstico (síntese)

**Estado atual em uma linha:** o `search-agent` é hoje **working + procedural memory, sobre um substrate context-resident, com control heurístico.** As camadas **episodic e semantic estão ausentes ou vivem fora do sistema.**

Isso explica, de forma direta, os itens do roadmap:

| Desejo do roadmap | O que é, na taxonomia |
|-------------------|------------------------|
| Histórico persistente | Episodic memory + substrate persistente |
| Favoritos / ratings (👍/👎/⭐) | Sinal para consolidation episódico → semantic |
| Comparação semana a semana | Coerência cross-session (§5.5) |
| Perfil de preferências | Semantic memory dentro do sistema |

---

## 7. Os cinco design objectives aplicados ao sistema (§2.3)

Onde o sistema está hoje em cada objetivo, e a tensão que as próximas fases vão enfrentar:

| Objetivo | Definição (§2.3) | Hoje | Tensão a gerenciar |
|----------|------------------|------|--------------------|
| **Utility** | A memória melhora o resultado? | Baixa: sem memória persistente, cada run parte do zero | Guardar mais aumenta utility mas pressiona efficiency e governance |
| **Efficiency** | Custo (tokens/latência/storage) por utilidade | Alta hoje (não guarda nada), mas à custa de utility | Retrieval e storage vão adicionar latência; medir custo é obrigatório (§5.5) |
| **Adaptivity** | Atualiza incrementalmente sem re-treinar? | Nula no sistema (só via loop humano) | Reflexão/consolidação (Fase 2) introduzem adaptatividade |
| **Faithfulness** | O recuperado é correto e atual? | N/A (não recupera) | Risco de *stale records* e contradições quando houver histórico (§7.3) |
| **Governance** | Privacidade, deleção, conformidade | N/A | Ao persistir, surge o dever de deletar de todas as camadas, inclusive embeddings (§7.5) |

---

## 8. Failure modes a que o sistema está ou estará exposto

Mapeando os modos de falha do paper ao seu caso — útil como checklist nas próximas fases:

- **Summarization drift (§4.1):** se em algum momento você comprimir histórico para caber no prompt, o detalhe raro (um paper de nicho que você valorizou) tende a sumir. *Mitigação:* manter o registro bruto no store externo, em fidelidade total.
- **Relevance ≠ similarity (§4.2):** "me mostre o que se conecta ao que vi" não é a mesma query que "o mais parecido". O gargalo será *retrieval quality*, não storage. *Mitigação:* formulação de query e filtros de metadata, não só similaridade.
- **Self-reinforcing error (§4.3):** quando houver perfil de preferências, uma conclusão errada ("Noua não liga para AI Safety") pode se cristalizar e enviesar todos os runs seguintes. *Mitigação:* reflection grounding (cada preferência aponta para evidência concreta) + expiração/revisão.
- **Silent orchestration failures (§4.4):** decisões de memória falham sem stack trace — só geram um digest um pouco pior, que acumula. *Mitigação:* observability desde cedo (Fase 5).
- **Schema drift (§6.6):** o formato de resposta do arXiv ou do seu pipeline pode mudar; registros antigos viram inválidos. *Mitigação:* versionar o schema dos registros.

---

## 9. Classificação dos registros que o agente cria

| Registro produzido | Eixo | Estado |
|--------------------|------|--------|
| Resultados de busca do run atual | Working | Presente |
| Um run (execução do digest) | Episodic | A criar (Fase 1) |
| Um paper individual surfaceado | Episodic | A criar (Fase 1) |
| Identidade canônica + IDs por fonte (`doi`, `arxiv_id`, …) | Episodic/structured | A criar (Fase 1, multi-fonte) |
| Relações entre papers (autor, citação, tema) | Episodic/structured | A criar (Fase 3) |
| Nota de reflexão pós-run | Semantic (insumo) | A criar (Fase 2) |
| "Áreas/temas que a Noua valoriza" | Semantic | A criar (Fase 2) |
| Defaults (inglês, 4 domínios, ~3/área) | Semantic (hoje implícito) | Externo ao sistema |
| Pipeline busca→curadoria→render | Procedural | Presente |
| Schema do card | Procedural | Presente |

---

## 10. Implicações para as próximas fases

- **Fase 1 — criar a camada episodic.** Modelar `runs` e `papers` num substrate structured+vector (Pattern B, §7.6), com **write path** disciplinado: filtering, deduplication por `arxiv_id`, e metadata tagging (§7.1). Read path com retrieval por similaridade + filtro de metadata (§7.2).
- **Fase 2 — criar a camada semantic.** Reflexão pós-run e consolidação para um perfil de preferências, com reflection grounding (§4.3) e atenção à consolidation principiada (§9.1).
- **Fases seguintes** atacam substrate estruturado/relacional (§3.2, §9.2), avaliação (§5) e observability (§7.7).

---

## 11. Critério de pronto (Definition of Done)

A Fase 0 está concluída quando:

- [x] O sistema está classificado nos três eixos, com justificativa por seção do paper.
- [x] Para qualquer registro que o agente produz, sei dizer em qual eixo ele cai (tabela §9).
- [x] Os buracos estão nomeados (episodic ausente, semantic fora do sistema) e ligados ao roadmap.
- [ ] **A confirmar no código:** validar a premissa de "nenhuma persistência hoje" e onde os defaults realmente moram. Se já houver storage ou defaults hardcoded, atualizar as tabelas dos §3 e §9.

---

## 12. Referência

Du, P. (2026). *Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers.* arXiv:2603.07670. https://arxiv.org/abs/2603.07670

Seções citadas neste documento: §2.1 (loop write–manage–read), §2.2 (POMDP / belief state), §2.3 (cinco design objectives), §3.1 (temporal scope), §3.2 (substrate), §3.3 (control policy), §4.1 (summarization drift), §4.2 (relevance vs similarity), §4.3 (self-reinforcing error / reflection grounding), §4.4 (silent orchestration failures), §4.5 (policy-learned / AgeMem), §5.5 (long context ≠ memória; cross-session coherence; custo), §6.6 (schema drift), §7.1 (write path), §7.2 (read path), §7.3 (staleness/contradictions), §7.5 (governance/deleção), §7.6 (três architecture patterns), §7.7 (observability), §9.1 (principled consolidation / dual-buffer), §9.2 (causally grounded retrieval), §10 (conclusão).
