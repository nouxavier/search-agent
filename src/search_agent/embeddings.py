"""Interface Embedder + impls (RFC §6).

OllamaEmbedder (bge-m3, 1024d) é a impl MVP; FakeEmbedder dá um vetor
determinístico para rodar offline/em teste. Trocar de provider não toca o
pipeline — só muda config.embedder.provider (e, para outro modelo, a dim).
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable

import httpx

from .config import EmbedderCfg


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    """POST {host}/api/embed {"model": "bge-m3", "input": texts} → embeddings.

    Pré-requisito: `ollama pull bge-m3`.
    """

    def __init__(self, cfg: EmbedderCfg, *, client: httpx.Client | None = None) -> None:
        self.dim = cfg.dim
        self.model = cfg.model
        self.host = cfg.ollama_host.rstrip("/")
        self._client = client

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._client or httpx.Client(timeout=120.0)
        try:
            resp = client.post(
                f"{self.host}/api/embed",
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
        finally:
            if self._client is None:
                client.close()
        embeddings = data.get("embeddings")
        if not embeddings or len(embeddings) != len(texts):
            raise RuntimeError(
                f"Ollama retornou {len(embeddings or [])} embeddings para {len(texts)} textos"
            )
        return embeddings


class FakeEmbedder:
    """Vetor determinístico derivado do hash do texto — offline, sem deps.

    Não é semântico; serve para o pipeline da E0 rodar sem Ollama e para testes.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            seed = hashlib.sha256(text.encode("utf-8")).digest()
            # Expande o digest até dim floats em [-1, 1], depois normaliza (L2).
            raw = [
                (seed[i % len(seed)] / 127.5) - 1.0
                for i in range(self.dim)
            ]
            norm = math.sqrt(sum(v * v for v in raw)) or 1.0
            out.append([v / norm for v in raw])
        return out


def make_embedder(cfg: EmbedderCfg) -> Embedder:
    if cfg.provider == "ollama":
        return OllamaEmbedder(cfg)
    if cfg.provider == "fake":
        return FakeEmbedder(cfg.dim)
    raise ValueError(f"embedder provider desconhecido: {cfg.provider!r}")
