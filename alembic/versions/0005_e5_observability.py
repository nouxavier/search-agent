"""E5 — observability: memory_events (RFC §7.7).

Registra toda operação de memória (write/read/update/delete) com o contexto que a
disparou (`trigger_ctx`, JSONB). É a "câmera de segurança": dado um digest ruim, dá
pra reconstruir o que aconteceu e localizar a falha (write? read? raciocínio?).

Revision ID: 0005_e5_observability
Revises: 0004_e4_feedback
"""

from alembic import op

revision = "0005_e5_observability"
down_revision = "0004_e4_feedback"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE memory_events (
            id          BIGSERIAL PRIMARY KEY,
            ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
            op          TEXT NOT NULL,              -- write | read | update | delete
            target      TEXT NOT NULL,              -- tabela/registro afetado
            trigger_ctx JSONB                       -- o que disparou a operação
        )
        """
    )
    # Consultas típicas: por tempo (tail recente) e por tipo de op.
    op.execute("CREATE INDEX idx_memory_events_ts ON memory_events (ts DESC)")
    op.execute("CREATE INDEX idx_memory_events_op ON memory_events (op)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS memory_events")
