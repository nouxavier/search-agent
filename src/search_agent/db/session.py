"""Engine + factory de sessão. URL vem do config.toml (config.db.url)."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import get_settings


@lru_cache
def get_engine(url: str | None = None) -> Engine:
    url = url or get_settings().db.url
    return create_engine(url, future=True)


@lru_cache
def _factory(url: str | None = None) -> sessionmaker:
    return sessionmaker(bind=get_engine(url), expire_on_commit=False, future=True)


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    """Sessão transacional: commit no sucesso, rollback no erro."""
    session = _factory(url)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
