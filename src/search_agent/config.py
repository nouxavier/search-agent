"""Config tipada (pydantic-settings) lida de config.toml + env.

Defaults vivem em config.toml (versionado), não no prompt — resolve o buraco
diagnosticado na fase-0. Override por env usa o prefixo SEARCH_AGENT__ com '__'
como separador aninhado (ex.: SEARCH_AGENT__EMBEDDER__PROVIDER=fake).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

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
    provider: str = "anthropic"   # "anthropic" (API key) | "claude_cli" (assinatura Max) | "fake"
    model_fast: str = "claude-haiku-4-5"
    model_reason: str = "claude-sonnet-4-6"
    cli_bin: str = "claude"       # binário usado pelo provider claude_cli
    cli_timeout_s: int = 120      # timeout do subprocess `claude -p`


class DBCfg(BaseModel):
    url: str = "postgresql+psycopg://search:search@localhost:5432/search_agent"


# Caminho do toml lido pela source — get_settings o ajusta antes de instanciar.
_TOML_PATH: Path = CONFIG_PATH


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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedência (primeiro ganha): init < env < dotenv < toml < defaults.
        # toml DEPOIS do env => env (SEARCH_AGENT__...) sobrepõe o config.toml,
        # como o docstring do módulo promete. Antes, o toml vinha como init kwargs
        # e ganhava do env — bug que silenciava os overrides do README.
        toml = TomlConfigSettingsSource(settings_cls, toml_file=_TOML_PATH)
        return (init_settings, env_settings, dotenv_settings, toml, file_secret_settings)


@lru_cache
def get_settings(config_path: Path | None = None) -> Settings:
    """Carrega config.toml e aplica overrides de env por cima.

    Precedência: defaults dos modelos < config.toml < env (SEARCH_AGENT__...).
    """
    global _TOML_PATH
    _TOML_PATH = config_path or CONFIG_PATH
    return Settings()
