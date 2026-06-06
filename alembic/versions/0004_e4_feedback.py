"""E4 — avaliação: tabela feedback (RFC §5, schema E4).

Sinal explícito de utilidade por paper: up | down | star. Alimenta o metric stack
(task effectiveness = % surfaceado que virou up/star; memory quality = down =
recuperado mas inútil). `run_id` é ON DELETE SET NULL — o sinal sobrevive ao run.

`was_highlight` (E2) continua existindo como ponte pra consolidação; o comando
`feedback` grava nos dois (up/star → highlight).

Revision ID: 0004_e4_feedback
Revises: 0003_e3_edges
"""

from alembic import op

revision = "0004_e4_feedback"
down_revision = "0003_e3_edges"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Desvio consciente da RFC: ela põe run_id no PK *e* ON DELETE SET NULL — mas
    # coluna de PK é NOT NULL, então o SET NULL falharia ao apagar um run. Usamos
    # PK surrogate + UNIQUE(paper_id, run_id, signal) pra manter "um sinal por
    # (paper, run)" sem prender run_id como NOT NULL.
    op.execute(
        """
        CREATE TABLE feedback (
            id       BIGSERIAL PRIMARY KEY,
            paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            run_id   BIGINT REFERENCES runs(id) ON DELETE SET NULL,
            signal   TEXT NOT NULL CHECK (signal IN ('up', 'down', 'star')),
            ts       TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (paper_id, run_id, signal)
        )
        """
    )
    op.execute("CREATE INDEX idx_feedback_signal ON feedback (signal)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS feedback")
