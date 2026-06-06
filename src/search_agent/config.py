"""Config tipada (pydantic-settings) lida de config.toml + env.

Defaults vivem em config.toml (versionado), não no prompt — resolve o buraco
diagnosticado na fase-0. Override por env usa o prefixo SEARCH_AGENT__ com '__'
como separador aninhado (ex.: SEARCH_AGENT__EMBEDDER__PROVIDER=fake).
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raiz do repo = dois níveis acima deste arquivo (src/search_agent/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.toml"


class AgentCfg(BaseModel):
    language: str = "pt"
    default_area: str = "LLM Agents"
    papers_per_run: int = 10
    min_year: int = 2023


class SourceCfg(BaseModel):
    name: str = "arxiv"
    categories: list[str] = ["cs.AI", "cs.CL", "cs.LG"]


class EmbedderCfg(BaseModel):
    provider: str = "ollama"   # "ollama" | "fake"
    model: str = "bge-m3"
    dim: int = 1024
    ollama_host: str = "http://localhost:11434"


class LLMCfg(BaseModel):
    provider: str = "anthropic"
    model_fast: str = "claude-haiku-4-5"
    model_reason: str = "claude-sonnet-4-6"


class DBCfg(BaseModel):
    url: str = "postgresql+psycopg://search:search@localhost:5432/search_agent"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SEARCH_AGENT__",
        env_nested_delimiter="__",
        extra="ignore",
    )

    agent: AgentCfg = AgentCfg()
    source: SourceCfg = SourceCfg()
    embedder: EmbedderCfg = EmbedderCfg()
    llm: LLMCfg = LLMCfg()
    db: DBCfg = DBCfg()


def _load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


@lru_cache
def get_settings(config_path: Path | None = None) -> Settings:
    """Carrega config.toml e aplica overrides de env por cima.

    Precedência: defaults dos modelos < config.toml < env (SEARCH_AGENT__...).
    """
    data = _load_toml(config_path or CONFIG_PATH)
    # Env tem a última palavra: BaseSettings já lê env; passamos o toml como base.
    return Settings(**data)
