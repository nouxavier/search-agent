"""E3 — substrate relacional: arestas entre papers (RFC §3.2 / §5.5 / §9.2).

`edges` liga papers por RELAÇÃO, não por similaridade de embedding:
- same_author / same_subarea: não-direcionadas, guardadas canonicamente (src_id < dst_id).
- cites: direcionada (quem cita → citado); adiada — o feed Atom do arXiv não traz
  referências (§3.2, "quando disponível").

O ganho da fase é o read path trazer um vizinho relacional que o kNN puro não
traria ("este paper se conecta a X que você viu há 3 semanas").

Revision ID: 0003_e3_edges
Revises: 0002_e2_semantic
"""

from alembic import op

revision = "0003_e3_edges"
down_revision = "0002_e2_semantic"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE edges (
            src_id     BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            dst_id     BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            kind       TEXT   NOT NULL,                 -- same_author | same_subarea | cites
            weight     REAL   NOT NULL DEFAULT 1.0,     -- força da relação (sim p/ same_subarea)
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (src_id, dst_id, kind),
            CHECK (src_id <> dst_id)
        )
        """
    )
    # PK cobre lookup por src_id; índice extra pro lado dst (travessia não-direcionada).
    op.execute("CREATE INDEX idx_edges_dst ON edges (dst_id, kind)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS edges")
