"""Interface Source + o registro normalizado RawPaper (RFC §6).

RawPaper é o schema canônico para onde TODA fonte normaliza. A identidade
canônica (DOI → senão hash) e a dedup cross-source são derivadas dele na E1;
aqui (E0) ele é só o contrato de normalização que o adapter precisa cumprir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Protocol, runtime_checkable


@dataclass(frozen=True)
class RawPaper:
    title: str
    abstract: str | None
    authors: list[str]
    year: int | None
    doi: str | None
    # IDs específicos da fonte, ex.: {"arxiv_id": "2603.07670"}. Viram aliases
    # em external_ids na E1.
    source_ids: dict[str, str] = field(default_factory=dict)
    source_name: str = ""

    @property
    def first_author(self) -> str | None:
        return self.authors[0] if self.authors else None

    @property
    def embed_text(self) -> str:
        """Texto que vira embedding: título + abstract (abstract inteiro cabe em bge-m3, ctx 8k)."""
        parts = [self.title.strip()]
        if self.abstract:
            parts.append(self.abstract.strip())
        return "\n".join(parts)


@runtime_checkable
class Source(Protocol):
    name: str

    def fetch(self, area: str, since: datetime | None = None) -> Iterable[RawPaper]:
        """Busca papers da área e os normaliza para RawPaper."""
        ...
