# Guia de estudo — embeddings e grafos no search-agent

> Um passeio do **zero à intuição** sobre as duas ideias que fazem a memória do
> agente funcionar: **embeddings** (achar o que é *parecido*) e **grafos** (achar o
> que é *relacionado*). Cada conceito vem com a analogia, o desenho, e o ponto exato
> do código onde ele vive — pra você ler, mexer e entender.

**Como usar:** leia na ordem. Onde aparecer 🧪 **Experimente**, rode o comando — ver
o número mudar fixa mais que ler. Onde aparecer 📍 **No código**, abra o arquivo
citado: a teoria e a implementação lado a lado.

Pré-requisito pra rodar os exemplos: `docker compose up -d` e o Ollama no ar (veja o
[README](../README.md)).

---

## Parte 0 — O problema que estamos resolvendo

O agente lê papers do arXiv e precisa **lembrar** o que já viu, pra responder coisas
como *"o que eu já li sobre RAG?"* ou *"esse paper novo se conecta com algo de 3
semanas atrás?"*.

O computador não entende "RAG" nem "parecido". Ele entende **números**. Então o
truque do projeto inteiro é: **transformar texto em números de um jeito que
‘significado parecido’ vire ‘números parecidos’**. Esses números são o *embedding*.

A partir daí, "buscar por significado" vira "achar os vetores mais próximos" — uma
conta de geometria. É isso que vamos destrinchar.

---

## Parte 1 — Embeddings: transformar significado em geometria

### 1.1 A intuição: o mapa

Imagine um **mapa**. Cada cidade é um ponto com duas coordenadas (latitude,
longitude). Cidades próximas no mapa tendem a ser parecidas (clima, cultura). Você
resume "uma cidade" em **2 números** e, de quebra, ganha uma noção de distância.

Um **embedding** é a mesma ideia, com duas diferenças:

1. Em vez de 2 coordenadas, são **1024**. (Significado é complicado; 2 números não
   bastam, 1024 dão "espaço" pra separar assuntos.)
2. Em vez de cidades, são **textos**. Dois papers sobre RAG caem **perto** nesse
   espaço de 1024 dimensões; um paper de RAG e um de culinária caem **longe**.

Você não consegue desenhar 1024 dimensões, mas a intuição de "perto = parecido"
continua valendo. É só um mapa com muito mais eixos.

```
   espaço de significado (imagine 1024 eixos, aqui só 2)

        RAG sobre grafos •   • RAG e alucinação
                          \ /
                           •  ← "retrieval augmented generation" (sua busca)
                          /
   memória de agentes •         •  culinária (bem longe)
```

### 1.2 O que É, concretamente, um embedding

Uma **lista de 1024 números decimais**. No projeto, normalizados (comprimento 1 —
vamos ver por quê). Exemplo encurtado:

```
"retrieval augmented generation"  →  [0.013, -0.058, 0.021, ..., 0.004]   (1024 números)
```

Quem produz esses números é um **modelo de IA treinado** pra isso. No projeto é o
**bge-m3** (multilíngue, 1024 dimensões), rodando **localmente** via Ollama — sem
nuvem, sem API key. Você manda texto, ele devolve o vetor.

📍 **No código:** [embeddings.py](../src/search_agent/embeddings.py).
A interface é mínima — qualquer embedder é "texto entra, lista de vetores sai":

```python
class Embedder(Protocol):
    dim: int
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

O `OllamaEmbedder` é literalmente um POST pro Ollama:

```python
resp = client.post(f"{self.host}/api/embed", json={"model": "bge-m3", "input": texts})
return resp.json()["embeddings"]
```

> **Por que uma interface?** Pra trocar o motor sem mexer no resto. Tem também um
> `FakeEmbedder` (vetor derivado do hash do texto) que deixa o pipeline rodar
> offline e nos testes. Ele **não é semântico** — textos diferentes viram vetores
> quase aleatórios. Útil pra testar encanamento, inútil pra significado.

🧪 **Experimente** — veja um embedding nascendo:
```bash
SEARCH_AGENT__EMBEDDER__PROVIDER=fake uv run agent run -n 1   # offline, vetor fake
uv run agent run -a "retrieval augmented generation" -n 1     # real (bge-m3): repare em "embedding: dim=1024"
```

### 1.3 Medindo "perto": similaridade de cosseno

Temos pontos num espaço. Como medir se dois estão "perto"? O projeto usa **distância
de cosseno**, que olha o **ângulo** entre dois vetores (não o tamanho).

- Vetores apontando **pra mesma direção** → ângulo ≈ 0° → **similaridade ≈ 1**
  (muito parecidos).
- Vetores **perpendiculares** → ângulo 90° → **similaridade ≈ 0** (sem relação).
- Sentidos **opostos** → similaridade ≈ -1.

```
        ^  B
        |               A e B: ângulo pequeno → similares
        | /A            A e C: ~90°          → não relacionados
        |/
        +--------> C
```

A conta exata (produto escalar sobre os tamanhos):

```
similaridade(A, B) = (A · B) / (‖A‖ · ‖B‖)
```

**Por que os vetores vêm normalizados** (‖v‖ = 1)? Porque aí o denominador é 1 e a
similaridade vira só o produto escalar `A · B` — mais barato e estável. (No
`FakeEmbedder` você vê a normalização explícita na última linha: divide cada
componente por `norm`.)

E **distância vs. similaridade** — não confunda, elas andam ao contrário:

```
distância de cosseno = 1 − similaridade
   parecido   → similaridade alta (~1) → distância baixa (~0)
   sem relação→ similaridade baixa (~0)→ distância alta (~1)
```

📍 **No código:** em [read_path.py:54](../src/search_agent/memory/read_path.py#L54) a
conversão aparece como `base = 1.0 - h.distance` — o `sim≈` que você vê na saída do
`agent query` é exatamente isso.

---

## Parte 2 — Busca por similaridade (o "kNN")

Com embeddings + uma medida de distância, "buscar por significado" vira: **dado o
vetor da sua pergunta, ache os k papers de vetor mais próximo**. Isso se chama
**k-Nearest Neighbors (kNN)** — os k vizinhos mais próximos.

Quem guarda os vetores e faz essa conta rápido é o **Postgres + pgvector** (uma
extensão que adiciona o tipo `vector` e operadores de distância).

📍 **No código:** [queries.py](../src/search_agent/db/queries.py). O operador `<=>` do
pgvector é a distância de cosseno:

```sql
SELECT p.id, p.title, (p.embedding <=> CAST(:qvec AS vector)) AS dist
FROM papers p
WHERE p.embedding IS NOT NULL
ORDER BY p.embedding <=> CAST(:qvec AS vector)   -- menor distância primeiro
LIMIT :k
```

Leia em voz alta: *"ordene os papers pela distância do meu vetor de busca, pegue os k
primeiros"*. É a busca semântica inteira — o resto do projeto é construído em cima
disso.

🧪 **Experimente:**
```bash
uv run agent query "graph retrieval for knowledge graphs" -k 5
```
Olhe a coluna `sim≈`: quanto maior, mais perto do que você pediu. Note que ele acha
papers **mesmo sem as palavras exatas** baterem — é significado, não busca textual.

---

## Parte 3 — A virada: por que embedding NÃO basta

Embeddings respondem *"o que é **parecido**?"*. Mas relevância nem sempre é
parecença. Dois exemplos que o cosseno **não** captura:

- **Mesmo autor, temas diferentes.** A pesquisadora que você acompanha publicou um
  paper de física e outro de biologia. Os textos são distantes no embedding (assuntos
  diferentes!), mas pra você eles se conectam — *mesma autora*.
- **Citação.** Um paper novo **cita** um que você leu mês passado. Pode usar palavras
  bem diferentes, mas a relação é real e importante.

Essas são relações **estruturais**, ortogonais ao significado do texto. Pra capturá-
las, precisamos de outra estrutura além do "mapa": um **grafo**.

> Esse é o pulo da **Fase E3** do projeto. O próprio read path já avisava:
> *"relevância ≠ similaridade; o kNN é só o começo"*
> ([read_path.py:3](../src/search_agent/memory/read_path.py#L3)).

---

## Parte 4 — Grafos: do "parecido" ao "relacionado"

### 4.1 A intuição: pontos e linhas

Um **grafo** é só isto: **nós** (pontos) ligados por **arestas** (linhas). Aqui:

- **nó** = um paper.
- **aresta** = uma relação concreta entre dois papers.

Enquanto o embedding diz "esses dois estão *perto* no mapa", a aresta diz "esses dois
estão *ligados* por um motivo nomeável".

```
   (papel A) ───same_author─── (papel B)        "mesmo autor"
       │
   same_subarea                                 "mesma subárea (embeddings próximos)"
       │
   (papel C) ───cites──────────► (papel D)      "C cita D" (direcionada)
```

### 4.2 Grafo num banco relacional: é só uma tabela de arestas

Antes de "tipos de aresta", desmistifique o "grafo no banco". **Não existe um tipo
`grafo` no Postgres.** O grafo é só *duas tabelas comuns*: a de **nós** (`papers`,
que você já tem) e uma de **arestas** (`edges`), onde cada linha é uma ligação entre
dois papers.

```
papers (nós)                      edges (arestas)
┌────┬───────────────┐            ┌────────┬────────┬─────────────┐
│ id │ first_author  │            │ src_id │ dst_id │ kind        │
├────┼───────────────┤            ├────────┼────────┼─────────────┤
│ 1  │ Ada Lovelace  │            │   1    │   2    │ same_author │
│ 2  │ Ada Lovelace  │            └────────┴────────┴─────────────┘
│ 3  │ Alan Turing   │              ↑ "o paper 1 e o 2 estão ligados por autoria"
└────┴───────────────┘
```

Uma aresta é uma linha com **dois ponteiros** (`src_id`, `dst_id`, ambos apontando pra
`papers.id`) e um rótulo (`kind`). **Escrever** o grafo é `INSERT`; **caminhar** no
grafo é `SELECT`. Nada além de SQL sobre uma tabela bem modelada.

O comando que cria as arestas de autoria é este (de
[graph.py](../src/search_agent/memory/graph.py)) — roda quando um paper novo entra,
com `:pid` = o id dele:

```sql
INSERT INTO edges (src_id, dst_id, kind, weight)
SELECT LEAST(:pid, o.id), GREATEST(:pid, o.id), 'same_author', 1.0
FROM papers o, papers p
WHERE p.id = :pid AND o.first_author = p.first_author AND o.id <> :pid
ON CONFLICT DO NOTHING
```

Lê-se: *"pra todo paper `o` que tem o mesmo autor do paper novo `p`, grave uma aresta
`same_author`"*. As peças:

- **`INSERT ... SELECT`** — insere **o resultado de uma busca**, não valores fixos. Se
  o SELECT achar 3 papers, nascem 3 arestas de um golpe só.
- **`FROM papers o, papers p`** — a mesma tabela com dois apelidos: `p` = o paper novo,
  `o` = "os outros" (varre todos).
- **`WHERE p.id = :pid`** — **trava `p`** numa linha só (o paper novo); agora
  `p.first_author` é um valor fixo pra comparar contra todo `o`.
- **`o.first_author = p.first_author`** — **a regra da aresta**. Troque esta linha e
  muda o tipo de relação.
- **`o.id <> :pid`** — não liga o paper a si mesmo (evita laço).
- **`LEAST/GREATEST`** — põe o menor id em `src_id`. Como "mesmo autor" não tem
  direção, isso garante **uma linha por ligação** (A–B nunca vira também B–A).

#### Exemplos pra fixar (1–10 escrita, 11–19 leitura)

Use sempre este estado inicial de `papers` (e mude só o que cada exemplo disser):

```
papers:  id=1 Ada | id=2 Ada | id=3 Turing | id=4 Ada | id=5 Hopper
```

**1. A aresta nasce (caso feliz).** Entra o paper `id=2` (`:pid=2`). `p` = (2, Ada).
`o` varre todos; só `id=1` (Ada) casa e não é ele mesmo.
→ insere **uma** linha: `(src=1, dst=2, same_author)` (o `LEAST` botou o 1 na frente).

**2. Sem par → nenhuma aresta (e isso não é erro).** Entra `id=3` (Turing). Nenhum
outro paper é do Turing. O SELECT retorna **zero linhas**, então o INSERT
**não insere nada** — `INSERT...SELECT` vazio é silencioso, não estoura.

**3. Um paper, várias arestas de uma vez.** Entra `id=4` (Ada), com 1 e 2 (Ada) já no
banco. O SELECT casa **dois** `o` → nascem **duas** arestas no mesmo comando:
`(1,4)` e `(2,4)`. É o poder do `INSERT...SELECT`: N relações num único `INSERT`.

**4. Nada de laço (o papel do `o.id <> :pid`).** Sem essa cláusula, ao entrar `id=4`
o próprio 4 casaria (ele tem o mesmo autor que ele mesmo!) e geraria a aresta absurda
`(4,4)`. A linha `o.id <> :pid` é o que poda esse laço.

**5. Ordem canônica mata a aresta gêmea.** Imagine sem `LEAST/GREATEST`. Ao entrar o
2, geraria `(2,1)`. Se um dia reprocessasse o 1, geraria `(1,2)` — **duas linhas pra
mesma ligação**. Com `LEAST/GREATEST`, ambas viram `(1,2)`: uma ligação, uma linha.

**6. Rodar de novo não duplica (`ON CONFLICT DO NOTHING`).** A PK de `edges` é
`(src_id, dst_id, kind)`. Se `(1,2,same_author)` já existe e o comando roda de novo,
o banco **ignora** essa linha em vez de dar erro de chave duplicada. Por isso a
operação é *idempotente* (reprocessar é seguro) — é o que o teste
`test_populate_edges_is_idempotent` verifica.

**7. Mesma estrutura, outra regra: `same_subarea`.** Troque só a condição do `WHERE`:
em vez de comparar autor, compare **distância de embedding**:

```sql
INSERT INTO edges (src_id, dst_id, kind, weight)
SELECT LEAST(:pid, o.id), GREATEST(:pid, o.id), 'same_subarea',
       (1.0 - (o.embedding <=> p.embedding))::real        -- peso = similaridade
FROM papers o, papers p
WHERE p.id = :pid AND (o.embedding <=> p.embedding) < 0.40 AND o.id <> :pid
ON CONFLICT DO NOTHING
```

O esqueleto é idêntico ao de autoria; o **tipo de relação é só a regra do WHERE**.
Note que aqui o `weight` carrega a similaridade (em `same_author` era fixo `1.0`).

**8. Aresta COM direção: `cites`.** "A cita B" **tem** sentido (≠ "B cita A"). Então
`cites` **não** usa `LEAST/GREATEST` — grava na ordem real: se o paper `5` cita o `1`,
a linha é `(src=5, dst=1, cites)`. Aqui `(5,1)` e `(1,5)` significam coisas
diferentes. (No projeto, `cites` está adiado — o arXiv não fornece referências.)

**9. Caminhando 1 hop: os vizinhos de um nó.** Com `edges = {(1,2),(1,4)}`
same_author, quem são os vizinhos do paper `1`? Como a relação é não-direcionada, o
vizinho é "o outro lado":

```sql
SELECT CASE WHEN src_id = 1 THEN dst_id ELSE src_id END AS vizinho
FROM edges
WHERE (src_id = 1 OR dst_id = 1) AND kind = 'same_author';
-- → 2, 4
```

**Ler** o grafo é isto: um `SELECT` na tabela de arestas. O `CASE` resolve o "outro
lado" sem precisar saber se o 1 caiu em `src` ou `dst`.

**10. De vários nós ao mesmo tempo (`ANY(:seeds)`).** No read path você parte de
**vários** papers (as "sementes" do kNN), não de um. Em vez de `= 1`, use
`= ANY(:seeds)` com uma lista:

```sql
-- seeds = [1, 5] → todas as arestas que tocam 1 ou 5
SELECT src_id, dst_id, kind
FROM edges
WHERE src_id = ANY(:seeds) OR dst_id = ANY(:seeds);
```

É exatamente o que `relational_neighbors` faz
([graph.py](../src/search_agent/memory/graph.py)) — uma travessia de 1 hop a partir de
um punhado de sementes, numa query só.

Os próximos exemplos são de **leitura** (consultar o grafo). Use este estado de
`edges` já montado (e `papers` agora com `id=6 Babbage`):

```
edges:  (1,2,same_author)  (1,4,same_author)  (2,4,same_author)   ← cluster Ada
        (4,5,same_subarea, w=0.78)                                 ← 4 e 5 perto no embedding
        (6,1,cites)                                                ← paper 6 cita o 1 (direcionada)
```

**11. Travessia filtrada por tipo.** Vizinhos do paper `4` **só por autoria** (ignora
a aresta de subárea com o 5):

```sql
SELECT CASE WHEN src_id = 4 THEN dst_id ELSE src_id END AS vizinho
FROM edges
WHERE (src_id = 4 OR dst_id = 4) AND kind = 'same_author';
-- → 1, 2   (o 5 não vem: é same_subarea)
```

**12. Não devolver a própria semente.** Ao expandir um conjunto, o resultado pode
incluir um nó que já era semente — filtre fora com `<> ALL(:seeds)`:

```sql
SELECT DISTINCT CASE WHEN src_id = ANY(:seeds) THEN dst_id ELSE src_id END AS vizinho
FROM edges
WHERE (src_id = ANY(:seeds) OR dst_id = ANY(:seeds))
  AND CASE WHEN src_id = ANY(:seeds) THEN dst_id ELSE src_id END <> ALL(:seeds);
-- seeds = [1,2] → traz 4 (vizinho de ambos), mas não 1 nem 2
```

**13. Grau de um nó (quantos vizinhos tem).** Conte as arestas que tocam cada paper.
Truque: junte as duas colunas numa só com `UNION ALL`, depois `GROUP BY`:

```sql
SELECT id, count(*) AS grau
FROM (SELECT src_id AS id FROM edges
      UNION ALL
      SELECT dst_id AS id FROM edges) t
GROUP BY id ORDER BY grau DESC;
-- → 1:3, 4:3, 2:2, 5:1, 6:1
```

**14. Os papers mais conectados (os "hubs").** É o exemplo 13 + `LIMIT`. Útil pra achar
o paper central de um cluster:

```sql
SELECT id, count(*) AS grau
FROM (SELECT src_id AS id FROM edges UNION ALL SELECT dst_id AS id FROM edges) t
GROUP BY id ORDER BY grau DESC LIMIT 3;
-- → 1 e 4 empatados no topo (grau 3)
```

**15. Dois hops (amigos de amigos).** Aqui o grafo brilha. Junte `edges` **consigo
mesma**: a 1ª aresta vai da semente ao meio, a 2ª do meio ao destino. Com `cites`
(direcionada, SQL mais limpo) — "quem o paper 6 cita, e quem *esse* cita":

```sql
SELECT e2.dst_id AS dois_hops
FROM edges e1
JOIN edges e2 ON e2.src_id = e1.dst_id        -- o "meio" conecta as duas arestas
WHERE e1.src_id = 6 AND e1.kind = 'cites' AND e2.kind = 'cites';
-- 6 cita 1; se o 1 citasse alguém, viria aqui. Cada JOIN extra = mais um hop.
```

**16. O vizinho mais forte (usando o peso).** `same_subarea` guarda a similaridade em
`weight`. Pra pegar o vizinho de subárea mais próximo do paper `4`:

```sql
SELECT CASE WHEN src_id = 4 THEN dst_id ELSE src_id END AS vizinho, weight
FROM edges
WHERE (src_id = 4 OR dst_id = 4) AND kind = 'same_subarea'
ORDER BY weight DESC LIMIT 1;
-- → 5 (w=0.78)
```

É assim que `relational_neighbors` desempata quando há várias relações: maior peso vence.

**17. Só vizinhos dentro de um conjunto (o filtro real do read path).** O
`relational_neighbors` não quer *qualquer* vizinho — quer os que estão entre os
"candidatos" (tipicamente *já vistos em runs anteriores*). Isso é um `AND ... = ANY`:

```sql
SELECT CASE WHEN src_id = ANY(:seeds) THEN dst_id ELSE src_id END AS vizinho
FROM edges
WHERE (src_id = ANY(:seeds) OR dst_id = ANY(:seeds))
  AND CASE WHEN src_id = ANY(:seeds) THEN dst_id ELSE src_id END = ANY(:candidatos);
-- só sobram vizinhos que estão na lista :candidatos
```

É exatamente o `candidate_ids` da função — o que dá o *"se conecta a X que você já viu"*.

**18. Apagar um paper → as arestas somem sozinhas (`ON DELETE CASCADE`).** As FKs de
`edges` apontam pra `papers(id)` com `ON DELETE CASCADE`. Então:

```sql
DELETE FROM papers WHERE id = 4;
-- o banco remove em cascata (1,4), (2,4) e (4,5) — nenhuma aresta "órfã" sobra
```

Integridade do grafo de graça: você nunca fica com uma aresta apontando pra um nó que
não existe mais.

**19. Por que existe o índice `idx_edges_dst`.** A PK `(src_id, dst_id, kind)` deixa
rápida a busca por `src_id`. Mas a travessia também filtra por `dst_id` (o "outro
lado"). Sem um índice nesse lado, `WHERE ... OR dst_id = :x` faria varredura da tabela
inteira. Por isso a migração cria `idx_edges_dst (dst_id, kind)` — travessia rápida nos
dois sentidos. (Veja em [0003_e3_edges.py](../alembic/versions/0003_e3_edges.py).)

> **Resumo:** grafo em SQL é isto — toda pergunta de grafo (vizinhos, grau, caminhos,
> hubs) vira um `SELECT` / `JOIN` / `GROUP BY` sobre a tabela `edges`. Você já sabe SQL,
> então você já sabe consultar grafos.
>
> **Quando vale um banco de grafo de verdade (Neo4j & cia)?** Quando você precisa de
> travessias **profundas** (5, 10, 20 hops: "amigos de amigos de amigos…"), onde fazer
> `JOIN` repetido fica caro. Pra 1–2 hops como aqui, a tabela `edges` indexada resolve
> — e você não carrega um segundo banco.

### 4.3 Os três tipos de aresta do projeto

📍 **No código:** [graph.py](../src/search_agent/memory/graph.py).

| Tipo | Significado | Como é detectado | Status |
|---|---|---|---|
| `same_author` | mesmo autor | `first_author` igual | ✅ ativo |
| `same_subarea` | mesma subárea | embeddings com distância `< 0.40` | ✅ ativo |
| `cites` | um cita o outro | parse de referências | ⏸ adiado (arXiv não dá referências) |

Repare na sutileza: **`same_subarea` usa o embedding** — é o grafo *reaproveitando* a
similaridade. Já **`same_author` é ortogonal ao embedding** — é justamente a relação
que o kNN sozinho **nunca** acharia. Por isso `same_author` é a aresta que "prova" o
valor do grafo (e é a que o teste de DoD usa).

### 4.4 Escrevendo arestas (no write path)

Quando um paper **novo** é gravado, o projeto o liga ao resto do store. É uma única
chamada dentro do `ingest`:

📍 [write_path.py:86](../src/search_agent/memory/write_path.py#L86) →
`populate_edges(session, paper_id)`.

Por baixo, dois `INSERT ... SELECT` (um por tipo). O de autoria, em essência:

```sql
INSERT INTO edges (src_id, dst_id, kind, weight)
SELECT LEAST(:pid, o.id), GREATEST(:pid, o.id), 'same_author', 1.0
FROM papers o, papers p
WHERE p.id = :pid AND o.first_author = p.first_author AND o.id <> :pid
```

> **Por que `LEAST`/`GREATEST`?** A relação "mesmo autor" não tem direção — A–B é o
> mesmo que B–A. Pra não guardar a aresta duas vezes, gravamos sempre na ordem
> canônica `src_id < dst_id`. (Já `cites` **tem** direção: quem cita → citado.)

### 4.5 Lendo arestas: a travessia (no read path)

Buscar no grafo é **caminhar pelas arestas**. Partindo de um conjunto de nós
("sementes" — os papers que o kNN trouxe, ou o digest do run), pego os **vizinhos a 1
passo** (1 *hop*):

```
   sementes (do kNN)        vizinhos a 1 hop (via arestas)
   ┌─────────┐
   │ paper 27│──same_subarea──► paper 28   "você viu isso semana passada"
   │ paper 25│──same_author───► paper 12
   └─────────┘
```

📍 **No código:** `relational_neighbors(session, seed_ids, candidate_ids)` em
[graph.py](../src/search_agent/memory/graph.py). Ele faz a travessia e, pra cada
semente, escolhe **uma** relação pra mostrar (prioridade `cites > same_author >
same_subarea`, desempate pelo peso). O `candidate_ids` é tipicamente "papers já vistos
em runs anteriores" — é o que dá o *"se conecta a X que você viu há 3 semanas"*.

🧪 **Experimente** — veja a ponte aparecer no digest:
```bash
uv run agent run -a "retrieval augmented generation" -n 20
```
Cada paper novo ganha uma linha `↳ conecta-se a "..." (id=N) via same_subarea`. E
note: alguns papers **não** ganham ponte — ficaram acima do corte de distância. O
grafo não liga tudo com tudo; isso é o `SUBAREA_MAX_DIST = 0.40` fazendo seu papel.

---

## Parte 5 — Os dois juntos: similaridade + relação

A memória do agente combina as duas lentes:

```
  pergunta / paper novo
        │
        ▼
  ┌───────────────┐   "o que é PARECIDO?"        ┌──────────────────┐
  │  EMBEDDINGS   │ ─────────────────────────►   │   candidatos kNN │
  │ (bge-m3 +     │                              └────────┬─────────┘
  │  pgvector <=> │                                       │ sementes
  └───────────────┘                                       ▼
  ┌───────────────┐   "o que é RELACIONADO?"      ┌──────────────────┐
  │    GRAFO      │ ─────────────────────────►    │ vizinhos por     │
  │ (edges +      │                               │ aresta (1 hop)   │
  │  travessia)   │                               └────────┬─────────┘
  └───────────────┘                                        ▼
                                            resultado: parecidos  +  conectados
```

Cada uma cobre o ponto cego da outra:
- **Embedding** acha o assunto certo, mas é cego a autor/citação.
- **Grafo** acha a relação concreta, mas (no caso `same_subarea`) ainda se apoia no
  embedding pra saber o que é "mesma subárea".

Juntas, o digest deixa de ser só "papers parecidos com sua busca" e vira "papers
parecidos **e** papers conectados ao que você já conhece".

---

## Glossário rápido

- **Embedding / vetor:** lista de 1024 números que representa o *significado* de um
  texto. Perto = parecido.
- **bge-m3:** o modelo (via Ollama) que gera os embeddings. Multilíngue, 1024d, local.
- **Similaridade de cosseno:** mede parecença pelo ângulo entre vetores (1 = igual,
  0 = sem relação).
- **Distância de cosseno:** `1 − similaridade`. É o que o pgvector calcula com `<=>`.
- **kNN:** "k vizinhos mais próximos" — a busca por similaridade.
- **pgvector:** extensão do Postgres que guarda vetores e calcula distância.
- **Grafo / nó / aresta:** papers (nós) ligados por relações (arestas).
- **Travessia (1 hop):** caminhar uma aresta a partir de um nó pra achar vizinhos.
- **DoD (Definition of Done):** o critério que fecha cada fase. O da E3: o read path
  trazer ≥1 vizinho relacional que a similaridade pura não traria.

## Exercícios pra fixar

1. **Cosseno na mão.** Por que normalizar os vetores (‖v‖=1) deixa a similaridade
   igual ao produto escalar? (Dica: olhe a fórmula da §1.3 e o que vira o
   denominador.)
2. **Por que `same_author` e não `same_subarea` prova o grafo?** Construa o argumento:
   o que `same_subarea` herda do embedding que `same_author` não herda?
3. **Leia o teste do DoD** em [tests/test_graph.py](../tests/test_graph.py). Por que
   ele usa dois papers do mesmo autor com **títulos sem relação**? O que ele estaria
   deixando de provar se os títulos fossem parecidos?
4. **Mude o corte.** Baixe `SUBAREA_MAX_DIST` pra `0.20` em
   [graph.py](../src/search_agent/memory/graph.py), rode um `run -n 20` e compare
   quantas pontas `↳` aparecem. O que um corte mais apertado troca?

## Pra ir além (a fundo no projeto)

- [rfc-memory-layer.md](rfc-memory-layer.md) — a spec técnica (§3.2 grafo, §6 interfaces, §8 retrieval).
- [plano-implementacao.md](plano-implementacao.md) — as fases E0…E5 e o que cada uma entrega.
- [search-project.md](search-project.md) — o conceito de memória por trás de tudo.
