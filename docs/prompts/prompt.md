# Autonomous Research Intelligence Agent — System Design

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RESEARCH INTELLIGENCE AGENT                   │
├──────────────────┬──────────────────────────────────────────────┤
│  INGESTION LAYER │  PROCESSING LAYER  │  INTELLIGENCE LAYER     │
│                  │                    │                          │
│  ┌────────────┐  │  ┌──────────────┐  │  ┌──────────────────┐  │
│  │ API Crawlers│  │  │ Deduplication│  │  │ Relevance Scorer │  │
│  │ RSS/Atom   │  │  │ Normalization│  │  │ Impact Analyzer  │  │
│  │ Web Scrapers│  │  │ Embedding Gen│  │  │ Trend Detector   │  │
│  └────────────┘  │  └──────────────┘  │  └──────────────────┘  │
│                  │                    │                          │
│  STORAGE LAYER   │  ANALYSIS LAYER    │  OUTPUT LAYER           │
│                  │                    │                          │
│  ┌────────────┐  │  ┌──────────────┐  │  ┌──────────────────┐  │
│  │ Vector DB  │  │  │ LLM Pipeline │  │  │ Report Generator │  │
│  │ SQL Store  │  │  │ Citation Graph│  │  │ Notification Sys │  │
│  │ Cache      │  │  │ Cluster Engine│  │  │ Dashboard        │  │
│  └────────────┘  │  └──────────────┘  │  └──────────────────┘  │
└──────────────────┴──────────────────────────────────────────────┘
```

Five horizontal concerns cutting across all layers:

- **Scheduler** — orchestrates periodic runs (daily crawl, weekly report)
- **Config Store** — interest profiles, scoring weights, source list
- **Audit Log** — every decision made by the agent is traceable
- **Feedback Loop** — you rate papers; the agent reweights its scoring
- **Secret/Key Manager** — API keys, model credentials

---

## 2. Step-by-Step Agent Workflow

### Phase 0 — Discovery (runs daily)

```
For each configured source:
  1. Fetch new papers since last run timestamp
  2. Normalize metadata → {title, abstract, authors, date, doi, source}
  3. Deduplicate against seen-papers store (hash on doi + title)
  4. Compute text embedding (abstract + title)
  5. Store in vector DB + relational store
```

### Phase 1 — Filtering & Scoring (runs after Discovery)

```
For each new paper:
  1. Semantic similarity score  → cosine sim against your interest profile embeddings
  2. Citation velocity score    → citations/week from Semantic Scholar API
  3. Source prestige score      → weighted score per venue (arXiv cs.AI > random blog)
  4. Novelty score              → distance from existing paper clusters
  5. Author reputation score    → h-index or institutional affiliation
  6. Composite score = weighted sum → threshold filter (drop bottom 60%)
```

### Phase 2 — Deep Analysis (runs on top-N scored papers)

```
For each paper passing threshold:
  1. Full-text fetch (PDF → text via pdfplumber or Docling)
  2. LLM structured extraction:
     - Core contribution (1 sentence)
     - Method novelty (what's different from prior work)
     - Experimental evidence quality
     - Limitations stated by authors
     - Potential failure modes / risks
  3. Industry implication analysis:
     - Which engineering domains are affected?
     - Time-to-production estimate (research / near-term / deployable now)
     - Who benefits? (startups, big tech, academics)
  4. Trend tagging → assign to one or more tracked trend clusters
```

### Phase 3 — Trend Detection (runs weekly)

```
1. Cluster all paper embeddings from the last 30 days (HDBSCAN or k-means)
2. Compare cluster sizes to previous week → detect growing clusters
3. Identify papers that bridge multiple clusters (cross-domain signals)
4. Score trend momentum = (cluster growth rate) × (avg paper quality score)
5. Flag "emerging" if a cluster grew >40% week-over-week
```

### Phase 4 — Report Generation (weekly)

```
1. Select top papers by composite score
2. Group by trend cluster
3. LLM synthesis pass: "What does this week's research collectively suggest?"
4. Generate actionable section: "What should you read / experiment with / share?"
5. Output to: Markdown file, email, Notion page, or Slack message
```

---

## 3. Recommended Tools, APIs, and Technologies

### Data Sources & APIs

| Source | Method | Notes |
|---|---|---|
| **arXiv** | REST API + RSS | Free, daily updates, filter by category |
| **Semantic Scholar** | Official API | Citation counts, author data, paper graph |
| **Papers With Code** | REST API | Links papers to GitHub repos — huge signal |
| **OpenAlex** | REST API | Open, comprehensive, replaces MAG |
| **Hugging Face Papers** | RSS/scrape | Community-curated ML papers |
| **DeepMind / OpenAI blogs** | RSS | High-signal org research |
| **ACM DL / IEEE Xplore** | API (limited) | For SE and systems papers |

### Core Infrastructure

| Component | Recommended Choice | Why |
|---|---|---|
| **Orchestration** | Prefect or Dagster | Better than Airflow for Python-native pipelines |
| **Vector DB** | Qdrant (self-hosted) or Pinecone | Efficient semantic search |
| **Relational DB** | PostgreSQL | Paper metadata, scores, history |
| **Embedding model** | `text-embedding-3-large` (OpenAI) or `nomic-embed-text` (local) | Quality vs cost tradeoff |
| **LLM for analysis** | Claude claude-sonnet-4-6 via API | Long context for full papers |
| **PDF parsing** | Docling (IBM) or pdfplumber | Docling preserves structure better |
| **Caching** | Redis | Rate-limit compliance, dedup |
| **Scheduler** | Prefect schedules or cron + Python | Keep it simple initially |

### Runtime Options

- **Minimal / local**: Python + SQLite + Ollama (nomic-embed + llama3) — zero API cost
- **Production**: Claude API + Qdrant cloud + PostgreSQL + Prefect cloud
- **Hybrid**: Local embeddings + Claude API only for analysis pass (cost-effective)

---

## 4. Filtering and Ranking Strategy

### Relevance Scoring Model

```python
composite_score = (
    0.35 * semantic_similarity    # cosine sim to your interest profile
  + 0.20 * citation_velocity      # citations gained in last 30 days (normalized)
  + 0.15 * source_prestige        # venue weight table
  + 0.15 * novelty_score          # distance from existing cluster centroids
  + 0.10 * author_reputation      # h-index percentile
  + 0.05 * recency_bonus          # exponential decay, half-life = 14 days
)
```

### Interest Profile (seed embeddings)

```
"autonomous agent systems using LLMs for reasoning"
"distributed training infrastructure for large models"
"impact of AI automation on software engineering labor"
"novel transformer architectures with improved efficiency"
"multi-agent coordination and emergent behavior"
"AI safety and alignment in deployed systems"
```

### Hard Filters (before scoring)

- Published within the last N days (configurable per source)
- Language: English only
- Category: cs.AI, cs.LG, cs.SE, cs.DC, cs.NE, stat.ML (arXiv codes)
- Minimum abstract length > 100 words
- Not a workshop abstract only

### Disruptive Research Signals

A paper is flagged as potentially disruptive if **3+ of these are true**:

1. Novelty score > 0.8 (far from existing clusters)
2. Replicates or challenges a result from a top-5 venue paper
3. Authors from 3+ different institutions
4. Public GitHub repo with >100 stars within 7 days of publication
5. Cited within 14 days by papers from recognized research labs
6. Breaks a known benchmark by >10% margin
7. Introduces a new benchmark or evaluation framework
8. Claims to replace or subsume a widely-used method

---

## 5. Structured Paper Analysis Pipeline

Each paper that passes the threshold gets analyzed via LLM into structured JSON:

```json
{
  "core_contribution": "One sentence description of the main idea",
  "prior_work_delta": "What specifically is different from previous approaches",
  "methodology": {
    "approach": "...",
    "datasets_used": ["..."],
    "evaluation_metrics": ["..."],
    "reproducibility": "high | medium | low"
  },
  "evidence_quality": {
    "score": 0.0-1.0,
    "reasoning": "Ablations present? Baselines fair? Error bars reported?"
  },
  "limitations": ["author-stated", "reviewer-identified"],
  "industry_implications": {
    "domains_affected": ["MLOps", "SWE tooling", "..."],
    "deployment_readiness": "research | 1-2 years | deployable now",
    "who_benefits_most": "...",
    "disruption_potential": "low | medium | high | transformative"
  },
  "recommended_action": "read | skim | skip | share | experiment",
  "related_papers": ["doi1", "doi2"],
  "trend_tags": ["agent-systems", "inference-efficiency", "..."]
}
```

---

## 6. Quality Filtering — Removing Low-Signal Content

### Automated Quality Signals

| Signal | Low Quality Indicator |
|---|---|
| Abstract structure | No hypothesis, no results, pure announcement |
| Author count | 1 author on a "breakthrough" paper |
| Venue | Predatory journal list (Beall's list lookup) |
| Citation pattern | Zero citations after 60 days (for non-new papers) |
| GitHub repo | Repo exists but has no code, only README |
| Benchmark claims | Claims >50% improvement without ablations |
| Writing quality | LLM perplexity score too low (may indicate AI-generated fluff) |

### LLM Self-Critique Pass

After extraction, run a second prompt:

```
"You just analyzed this paper. Now act as a skeptical reviewer.
List 3 reasons this paper might be less significant than it appears.
Rate confidence in your previous analysis: high / medium / low."
```

Papers where confidence = low get flagged for human review rather than auto-included.

---

## 7. Converting Research into Actionable Insights

### Insight Taxonomy

| Type | Example Output |
|---|---|
| **Read** | "This week's must-read: paper X introduces concept Y which directly applies to your work on Z" |
| **Experiment** | "Paper X's technique can be tested in 1 weekend with this OSS repo. Estimated effort: low." |
| **Share** | "This finding challenges common wisdom in ML engineering — good for a blog post or tweet thread" |
| **Track** | "Early signal — not mature yet but this research direction will matter in 12-18 months" |
| **Hire/Learn** | "This skill set is growing in research — consider adding to your learning roadmap" |
| **Invest attention** | "Three papers this week converge on the same conclusion — this trend is worth deep study" |

### Weekly Report Structure

```markdown
# Research Intelligence Report — Week of [DATE]

## TL;DR (60 seconds)
- [3 bullet points: biggest signals of the week]

## Must-Read Papers (Top 3-5)
For each:
- Title + link
- Why it matters to you specifically
- Core insight in 2 sentences
- Recommended action

## Emerging Trends
- Trend name + momentum score
- Supporting papers
- What this means for engineering practice in 12-24 months

## Signals Worth Watching
- Papers that scored medium but have high novelty — early stage

## Industry Implications
- Connects research to: job market, tool adoption, infra changes

## Your Learning Agenda (suggested)
- Concept to study this week based on recurring themes

## Stats
- Papers scanned: N | Filtered in: M | Analyzed: K
```

---

## 8. Innovative Ideas for a Personal Research Intelligence Assistant

### Bi-directional Learning
- You rate each recommended paper (thumbs up / down / star)
- Ratings update scoring weights via Bayesian update on each interest dimension
- After 4 weeks, the agent has a personalized model of your taste

### Citation Graph Navigation
- Traverse references and forward citations of high-scoring papers
- Surfaces influential older papers and detects "research lineages"
- Visualize as a graph in weekly report appendix

### Serendipity Mode
- Reserve 10% of report space for low-relevance but high-novelty papers
- Explicit goal: surface things outside your current mental model
- Label as "wild card" — avoids filter bubbles

### Conference Radar
- Track upcoming deadlines for NeurIPS, ICML, ICLR, OSDI, SOSP, FSE
- Flag papers from authors who submitted to those venues as "potential preview"
- Preview what the community will be talking about in 6 months

### Trend Half-Life Tracking
- Track each trend cluster's growth curve over 12 weeks
- Classify as: emerging / accelerating / plateauing / declining
- Alert when a tracked trend starts declining

### "Explain Like I'll Implement It" Mode
- For papers with a GitHub repo, generate:
  - "What would you need to reproduce this in a weekend?"
  - "What's the minimum viable experiment to test the core claim?"

### Research-to-Career Signal
- Tag papers citing skills growing in job postings (connect to LinkedIn/Indeed data)
- Surface: "Papers this week suggest growing demand for X — 23% increase in job postings"
- Directly relevant for tracking technology labor markets

### Personal Knowledge Graph
- Every analyzed paper writes nodes/edges to a local knowledge graph (Obsidian or Neo4j)
- Nodes: concepts, methods, authors, institutions, findings
- Edges: "builds on", "contradicts", "enables", "cited by"
- Queryable second brain: "What does my research history say about efficient inference?"

---

## Implementation Roadmap

| Phase | Scope | Effort |
|---|---|---|
| **Phase 1 — MVP** | arXiv RSS + Semantic Scholar API + SQLite + Claude API for summaries + Markdown report | 1-2 weekends |
| **Phase 2 — Semantic** | Add vector DB, embedding-based scoring, interest profile | +1 weekend |
| **Phase 3 — Analysis** | Full PDF parsing, structured JSON extraction, quality filtering | +2 weekends |
| **Phase 4 — Trends** | Clustering, trend detection, weekly trend report | +1 weekend |
| **Phase 5 — Feedback** | Rating UI (CLI or Telegram bot), adaptive scoring | +1 weekend |
| **Phase 6 — Knowledge Graph** | Obsidian/Neo4j integration, concept graph | ongoing |

> Core principle: the agent should save you reading time, not create more of it. Every component is oriented toward giving you a Tuesday morning 10-minute brief that's more valuable than 8 hours of ad-hoc browsing.

---

## Engineering Detail

> **Language: Go 1.22**
> Go handles the I/O-heavy pipeline (parallel API calls, DB writes) naturally with goroutines.
> The only ML component without a native Go equivalent is HDBSCAN clustering — handled via a lightweight Python sidecar in Phase 4.

### Project Structure

```text
research-agent/
├── go.mod
├── go.sum
├── docker-compose.yml
├── .env.example
│
├── cmd/
│   └── agent/
│       └── main.go                 # Entry point + cron scheduler
│
├── internal/
│   ├── config/
│   │   └── config.go               # Load settings from env / .env
│   ├── db/
│   │   ├── db.go                   # SQLite open + schema init + query methods
│   │   └── migrations/             # Plain .sql files, applied at startup
│   ├── ingestion/
│   │   ├── source.go               # Source interface + RawPaper struct
│   │   ├── arxiv.go                # arXiv Atom feed via gofeed
│   │   ├── s2.go                   # Semantic Scholar REST API
│   │   └── rss.go                  # Generic RSS for org blogs
│   ├── processing/
│   │   └── dedup.go                # Filter already-seen source IDs
│   ├── scoring/
│   │   └── composite.go            # Keyword + prestige + recency scorer
│   ├── analysis/
│   │   └── anthropic.go            # Raw HTTP client for Claude API (tool use)
│   ├── report/
│   │   ├── generator.go            # Render report via text/template
│   │   └── templates/
│   │       └── weekly.md.tmpl      # Go template → Markdown
│   └── pipeline/
│       └── discovery.go            # Orchestrates all steps end-to-end
│
└── internal/scoring/
    └── composite_test.go           # Unit tests (go test ./...)
```

---

### Tech Stack

#### Language & Runtime

| Layer | Choice | Rationale |
|---|---|---|
| Language | Go 1.22 | Fast, low memory, excellent concurrency for I/O pipelines |
| Module management | Go modules (`go mod`) | Built-in, no extra tooling |
| Config / env | `github.com/joho/godotenv` + `os.Getenv` | Lightweight, no reflection overhead |
| HTTP client | `net/http` (stdlib) | Zero dependencies, full control |
| Retry logic | Simple exponential backoff helper | ~15 lines, no external dep needed |
| JSON | `encoding/json` (stdlib) | Handles all API request/response marshaling |
| Concurrency | `sync.WaitGroup` + goroutines | Parallel source fetching, idiomatic Go |

#### Storage

| Component | Tech | Notes |
|---|---|---|
| Relational DB (MVP) | SQLite via `modernc.org/sqlite` | Pure Go — no CGO, single file, zero ops |
| Relational DB (Prod) | PostgreSQL 16 + pgvector | Switch by changing `DATABASE_URL` |
| DB driver | `database/sql` stdlib interface | Same code works with both SQLite and Postgres |
| Migrations | Plain `.sql` files applied at startup | No migration framework needed at this scale |
| Vector DB (Phase 2+) | Qdrant — official Go client `qdrant/go-client` | For embedding-based scoring |
| Cache / dedup (Phase 2+) | Redis via `redis/go-redis/v9` | Rate-limit windows + dedup hashes |

#### AI / ML

| Purpose | Choice | Notes |
|---|---|---|
| Embeddings (Phase 2) | OpenAI via `sashabaranov/go-openai` | `text-embedding-3-small` |
| Paper analysis | Claude API — raw `net/http` | No official Go SDK; tool use via JSON |
| Critique pass | Claude Haiku — same client | ~20x cheaper per call |
| Clustering (Phase 4) | Python sidecar (FastAPI + HDBSCAN) | Go calls `/cluster` via HTTP |
| Report templating | `text/template` stdlib | Renders `.tmpl` → Markdown |

#### Orchestration & Infrastructure

| Component | Choice | Notes |
|---|---|---|
| Scheduler (MVP) | `robfig/cron` | Lightweight, idiomatic Go cron |
| Scheduler (Prod) | Temporal (`go.temporal.io/sdk`) | Go-native workflow engine, durable execution |
| PDF parsing | `exec.Command("pdftotext", ...)` | Shell out to poppler-utils in container |
| RSS / Atom parsing | `mmcdole/gofeed` | arXiv Atom feed + org blogs |

---

### Database Schema

```sql
-- Works on both SQLite (MVP) and PostgreSQL (prod)

CREATE TABLE IF NOT EXISTS papers (
    id                   TEXT PRIMARY KEY,
    doi                  TEXT UNIQUE,
    source               TEXT NOT NULL,      -- 'arxiv' | 'semantic_scholar' | 'rss'
    source_id            TEXT NOT NULL,
    title                TEXT NOT NULL,
    abstract             TEXT,
    authors              TEXT,               -- JSON array: ["Name", ...]
    arxiv_cats           TEXT,               -- JSON array: ["cs.AI", ...]
    published_at         DATETIME,
    fetched_at           DATETIME DEFAULT CURRENT_TIMESTAMP,
    pdf_url              TEXT,
    keyword_score        REAL,
    prestige_score       REAL,
    recency_score        REAL,
    composite_score      REAL,
    passed_threshold     INTEGER DEFAULT 0,
    core_contribution    TEXT,
    recommended_action   TEXT,               -- 'read'|'skim'|'skip'|'share'|'experiment'
    disruption_potential TEXT,               -- 'low'|'medium'|'high'|'transformative'
    trend_tags           TEXT,               -- JSON array
    industry_insight     TEXT,
    analyzed             INTEGER DEFAULT 0,
    analyzed_at          DATETIME,
    UNIQUE(source, source_id)
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id               TEXT PRIMARY KEY,
    started_at       DATETIME NOT NULL,
    finished_at      DATETIME,
    papers_fetched   INTEGER DEFAULT 0,
    papers_new       INTEGER DEFAULT 0,
    papers_scored    INTEGER DEFAULT 0,
    papers_analyzed  INTEGER DEFAULT 0,
    status           TEXT,                   -- 'running'|'success'|'failed'
    error            TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    id         TEXT PRIMARY KEY,
    paper_id   TEXT REFERENCES papers(id),
    rating     INTEGER CHECK (rating BETWEEN -1 AND 2),
    note       TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

### Key Go Interfaces

```go
// internal/ingestion/source.go

type RawPaper struct {
    Source          string
    SourceID        string
    Title           string
    Abstract        string
    Authors         []string
    PublishedAt     time.Time
    DOI             string
    PDFURL          string
    ArxivCategories []string
}

type Source interface {
    FetchSince(ctx context.Context, since time.Time) ([]RawPaper, error)
}
```

```go
// internal/scoring/composite.go

type Weights struct {
    Keyword  float64 // 0.55 — MVP: no embeddings yet
    Prestige float64 // 0.25
    Recency  float64 // 0.20
}

type Scores struct {
    Keyword   float64
    Prestige  float64
    Recency   float64
    Composite float64
}

type Scorer struct {
    weights       Weights
    threshold     float64
    interestVocab map[string]bool
}

func NewScorer(profile []string, threshold float64) *Scorer { ... }
func (s *Scorer) Score(title, abstract, source string, publishedAt time.Time) Scores { ... }
func (s *Scorer) Passes(scores Scores) bool { return scores.Composite >= s.threshold }
```

```go
// internal/analysis/anthropic.go
// No official Go SDK — raw HTTP with tool use

type AnalysisResult struct {
    CoreContribution    string   `json:"core_contribution"`
    RecommendedAction  string   `json:"recommended_action"`
    DisruptionPotential string  `json:"disruption_potential"`
    TrendTags          []string `json:"trend_tags"`
    IndustryInsight    string   `json:"industry_insight"`
}

type Client struct {
    apiKey     string
    model      string
    httpClient *http.Client
}

func (c *Client) Analyze(ctx context.Context, title, abstract string) (*AnalysisResult, error) {
    // POST https://api.anthropic.com/v1/messages
    // tool_choice: {"type": "tool", "name": "submit_analysis"}
    // parse response.content[0].input as AnalysisResult
}
```

---

### Docker Compose

```yaml
services:
  qdrant:                               # Phase 2+ (vector search)
    image: qdrant/qdrant:latest
    restart: unless-stopped
    volumes: [qdrant_data:/qdrant/storage]
    ports: ["6333:6333"]

  cluster-sidecar:                      # Phase 4 (HDBSCAN — only ML part in Python)
    build: ./sidecar
    restart: unless-stopped
    ports: ["8001:8001"]

  agent:
    build: .
    restart: unless-stopped
    depends_on: [qdrant]
    environment:
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      DATABASE_PATH: /data/research_agent.db
      REPORTS_DIR: /reports
      CLUSTER_SIDECAR_URL: http://cluster-sidecar:8001
    volumes:
      - ./data:/data
      - ./reports:/reports

volumes:
  qdrant_data:
```

> MVP runs without Docker at all: just `go run ./cmd/agent` with a `.env` file.

---

### Testing Strategy

```text
internal/
├── scoring/
│   └── composite_test.go       # go test — weight invariants, keyword matching
├── processing/
│   └── dedup_test.go           # filter logic, collision edge cases
└── ingestion/
    └── arxiv_test.go           # golden file test against recorded Atom XML
```

```go
// internal/scoring/composite_test.go

func TestWeightsSumToOne(t *testing.T) {
    w := DefaultWeights()
    total := w.Keyword + w.Prestige + w.Recency
    if math.Abs(total-1.0) > 1e-9 {
        t.Fatalf("weights sum to %f, want 1.0", total)
    }
}

func TestRelevantPaperPasses(t *testing.T) {
    scorer := NewScorer([]string{"autonomous LLM agent reasoning"}, 0.30)
    scores := scorer.Score(
        "Autonomous LLM Agent Reasoning System",
        "We present an autonomous agent using LLM reasoning for distributed tasks.",
        "arxiv",
        time.Now(),
    )
    if !scorer.Passes(scores) {
        t.Fatalf("expected paper to pass, composite=%.2f", scores.Composite)
    }
}

func TestIrrelevantPaperFiltered(t *testing.T) {
    scorer := NewScorer([]string{"autonomous LLM agent reasoning"}, 0.30)
    scores := scorer.Score(
        "Novel Pasta Fermentation Techniques",
        "We explore artisan yeast fermentation methods for pasta production.",
        "arxiv",
        time.Now().Add(-30*24*time.Hour),
    )
    if scorer.Passes(scores) {
        t.Fatalf("expected paper to be filtered, composite=%.2f", scores.Composite)
    }
}
```

Run with: `go test ./... -v -count=1`

---

### CI/CD (GitHub Actions)

```yaml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with: { go-version: "1.22" }
      - run: go mod download
      - run: go vet ./...
      - run: go test ./... -race -count=1 -coverprofile=coverage.out
      - uses: codecov/codecov-action@v4

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with: { go-version: "1.22" }
      - run: go build -o bin/agent ./cmd/agent
      - uses: actions/upload-artifact@v4
        with: { name: agent-binary, path: bin/agent }
```

> No external services needed in CI — SQLite runs in-process, no Postgres/Redis containers required.

---

### Cost Breakdown (~500 papers/week ingested, ~150 analyzed)

| Item | Volume / month | Cost |
|---|---|---|
| `text-embedding-3-small` (Phase 2) | ~2.4M tokens | ~$0.05 |
| `claude-sonnet-4-6` analysis | 150 papers × 8K tokens × 4 weeks | ~$18 |
| `claude-haiku-4-5` critique | 150 papers × 2K tokens × 4 weeks | ~$0.40 |
| `claude-sonnet-4-6` weekly synthesis | 4 reports × 12K tokens | ~$2 |
| Infrastructure (small VPS — Go binary is ~10MB, low RAM) | — | ~$4 |
| **Total** | | **~$24 / month** |

**Cost reduction levers:**

- Switch analysis to `claude-haiku-4-5` → drops to ~$5/month total
- Use `nomic-embed-text` via Ollama locally → eliminates embedding cost
- Reduce analyzed paper count from 150 → 50/week → ~$8/month with Sonnet