"""Schema episodic da E1 (RFC §5). ORM SQLAlchemy 2.0.

A DDL real é versionada via Alembic (alembic/versions/0001_e1_episodic.py); estes
models são a visão da aplicação sobre as mesmas tabelas. Os dois precisam concordar
— a migração é a fonte da verdade para o banco, os models para o código.

Identidade canônica e dedup cross-source vêm de `paper_key` (UNIQUE): o mesmo paper
vindo de fontes diferentes funde num registro só, com os IDs por fonte em external_ids.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Dimensão do embedding (BGE-M3). Acoplada à coluna; trocar de modelo com outra
# dimensão = migração + re-embed de todo o store (RFC §13, questão aberta 5).
EMBED_DIM = 1024


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    paper_key: Mapped[str] = mapped_column(Text, unique=True, nullable=False)  # identidade canônica
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str | None] = mapped_column(Text)
    first_author: Mapped[str | None] = mapped_column(Text)
    year: Mapped[int | None] = mapped_column(SmallInteger)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM))
    schema_ver: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    external_ids: Mapped[list["ExternalId"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    sources: Mapped[list["Source"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )


class ExternalId(Base):
    """Aliases por fonte (doi, arxiv_id, s2_id, ...). PK (kind, value): um alias
    pertence a um paper só — é o que viabiliza a dedup cross-source."""

    __tablename__ = "external_ids"

    paper_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[str] = mapped_column(Text, primary_key=True)

    paper: Mapped[Paper] = relationship(back_populates="external_ids")


class Source(Base):
    """Lista de fontes onde o paper foi visto (não um valor único)."""

    __tablename__ = "sources"

    paper_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True
    )
    source_name: Mapped[str] = mapped_column(Text, primary_key=True)
    seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    paper: Mapped[Paper] = relationship(back_populates="sources")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    area: Mapped[str] = mapped_column(Text, nullable=False)
    params_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)


class RunPaper(Base):
    """Quais papers cada run *surfaceou* (digest). É o registro de 'já visto' que
    faz dois runs seguidos não repetirem o mesmo paper."""

    __tablename__ = "run_papers"

    run_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True
    )
    paper_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True
    )
    rank: Mapped[int | None] = mapped_column(Integer)
    was_highlight: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
