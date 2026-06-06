"""Interface LLMClient + impl Anthropic (RFC §6).

Na E0 só existe um smoke test (`agent smoke` na CLI). O uso real — reflexão e
consolidação — entra na E2. `cache_system=True` aplica prompt caching no bloco
de system (usado pro user_profile na E2, barateando runs repetidos).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .config import LLMCfg


@runtime_checkable
class LLMClient(Protocol):
    def complete(
        self,
        *,
        system: str,
        prompt: str,
        model: str,
        cache_system: bool = False,
        max_tokens: int = 1024,
    ) -> str: ...


class AnthropicClient:
    """Wrapper fino sobre o Anthropic SDK (messages API).

    Resolve a API key do ambiente (ANTHROPIC_API_KEY) como o SDK faz por padrão.
    """

    def __init__(self, cfg: LLMCfg) -> None:
        self.cfg = cfg
        self._client = None  # lazy: só importa/instancia o SDK quando usado

    def _ensure_client(self):
        if self._client is None:
            import anthropic  # import tardio: E0 roda sem chamar o LLM

            self._client = anthropic.Anthropic()
        return self._client

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        model: str,
        cache_system: bool = False,
        max_tokens: int = 1024,
    ) -> str:
        client = self._ensure_client()
        if cache_system:
            system_param = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        else:
            system_param = system

        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_param,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


def make_llm(cfg: LLMCfg) -> LLMClient:
    if cfg.provider == "anthropic":
        return AnthropicClient(cfg)
    raise ValueError(f"llm provider desconhecido: {cfg.provider!r}")
