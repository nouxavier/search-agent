"""E2 — semantic (reflexão → consolidação) (RFC §5 / §9.1).

reflections (nota grounded pós-run) e user_profile (preferências que evoluem,
com confidence + expires_at). `statement_key` (texto normalizado) é detalhe de
implementação para o UPSERT de consolidação — não está no §5, é o que permite
"confidence sobe com repetição" sem duplicar statements.

Revision ID: 0002_e2_semantic
Revises: 0001_e1_episodic
"""

from alembic import op

revision = "0002_e2_semantic"
down_revision = "0001_e1_episodic"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE reflections (
            id           BIGSERIAL PRIMARY KEY,
            run_id       BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            note         TEXT NOT NULL,
            grounded_ids BIGINT[] NOT NULL,          -- papers que sustentam a nota (§4.3)
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE user_profile (
            id            BIGSERIAL PRIMARY KEY,
            statement     TEXT NOT NULL,
            statement_key TEXT NOT NULL UNIQUE,       -- texto normalizado (dedup de consolidação)
            evidence_ids  BIGINT[] NOT NULL,          -- grounding obrigatório
            confidence    REAL NOT NULL DEFAULT 0.5,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            expires_at    TIMESTAMPTZ                 -- anti self-reinforcing error (§4.3)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS user_profile")
    op.execute("DROP TABLE IF EXISTS reflections")
