"""Fixtures de teste. `session` dá uma sessão SQLAlchemy contra um Postgres+pgvector
efêmero (testcontainers) — pula com mensagem clara se Docker/testcontainers não
estiverem disponíveis, pra não travar os testes que não precisam de banco.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from search_agent.db.models import Base

PG_IMAGE = "pgvector/pgvector:pg16"


@pytest.fixture(scope="session")
def pg_url():
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        pytest.skip("testcontainers não instalado (uv sync --group dev)")

    try:
        container = PostgresContainer(PG_IMAGE, driver="psycopg")
        container.start()
    except Exception as exc:  # Docker fora do ar, imagem ausente, etc.
        pytest.skip(f"Postgres efêmero indisponível: {exc}")

    try:
        yield container.get_connection_url()
    finally:
        container.stop()


@pytest.fixture
def session(pg_url) -> Session:
    engine = create_engine(pg_url, future=True)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    s = factory()
    try:
        yield s
    finally:
        s.close()
        Base.metadata.drop_all(engine)
        engine.dispose()
