"""E1 — camada episodic core (RFC §5).

papers + external_ids + sources + runs + run_papers, com a extensão pgvector e o
índice HNSW para busca por similaridade. DDL escrita à mão (SQL explícito) em vez
de autogenerate — o write/read path é o que estamos internalizando, então o schema
fica legível linha a linha.

Revision ID: 0001_e1_episodic
Revises:
"""

from alembic import op

revision = "0001_e1_episodic"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute(
        """
        CREATE TABLE papers (
            id           BIGSERIAL PRIMARY KEY,
            paper_key    TEXT NOT NULL UNIQUE,          -- identidade canônica (§4.3)
            title        TEXT NOT NULL,
            abstract     TEXT,
            first_author TEXT,
            year         SMALLINT,
            embedding    VECTOR(1024),                  -- BGE-M3; trocável (RFC §13.5)
            schema_ver   SMALLINT NOT NULL DEFAULT 1,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX papers_embedding_hnsw "
        "ON papers USING hnsw (embedding vector_cosine_ops)"
    )

    op.execute(
        """
        CREATE TABLE external_ids (
            paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            kind     TEXT   NOT NULL,                   -- 'doi'|'arxiv_id'|'s2_id'|...
            value    TEXT   NOT NULL,
            PRIMARY KEY (kind, value)                   -- um alias pertence a um paper só
        )
        """
    )

    op.execute(
        """
        CREATE TABLE sources (
            paper_id    BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            source_name TEXT   NOT NULL,
            seen_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (paper_id, source_name)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE runs (
            id              BIGSERIAL PRIMARY KEY,
            ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
            area            TEXT NOT NULL,
            params_snapshot JSONB NOT NULL
        )
        """
    )

    op.execute(
        """
        CREATE TABLE run_papers (
            run_id        BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            paper_id      BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            rank          INT,
            was_highlight BOOLEAN NOT NULL DEFAULT false,
            PRIMARY KEY (run_id, paper_id)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS run_papers")
    op.execute("DROP TABLE IF EXISTS runs")
    op.execute("DROP TABLE IF EXISTS sources")
    op.execute("DROP TABLE IF EXISTS external_ids")
    op.execute("DROP TABLE IF EXISTS papers")
    # a extensão vector é deixada — pode ser usada por outros schemas
