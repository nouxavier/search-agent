"""Identidade canônica, source-agnostic (RFC §4.3).

paper_key = DOI normalizado, se existir
          → senão  sha1(normalize(title) | sobrenome_1º_autor | ano)

É o que funde o mesmo paper vindo de fontes diferentes num registro só. A chave
é prefixada ('doi:' / 'h:') para deixar explícito como foi resolvida.
"""

from __future__ import annotations

import hashlib
import re

from ..sources.base import RawPaper

_DOI_PREFIX = re.compile(r"^(https?://(dx\.)?doi\.org/|doi:)", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_doi(doi: str) -> str:
    """Tira prefixo de URL/'doi:' e baixa pra minúsculas. DOI é case-insensitive."""
    return _DOI_PREFIX.sub("", doi.strip()).lower()


def normalize_title(title: str) -> str:
    """Minúsculas + colapsa qualquer não-alfanumérico em espaço único."""
    return _NON_ALNUM.sub(" ", title.lower()).strip()


def last_name(author: str | None) -> str:
    if not author:
        return ""
    return author.strip().split()[-1].lower()


def canonical_key(raw: RawPaper) -> str:
    if raw.doi:
        return "doi:" + normalize_doi(raw.doi)
    basis = f"{normalize_title(raw.title)}|{last_name(raw.first_author)}|{raw.year or ''}"
    return "h:" + hashlib.sha1(basis.encode("utf-8")).hexdigest()
