"""Interface LLMClient + impls Anthropic / Claude CLI (RFC §6).

Na E0 só existe um smoke test (`agent smoke` na CLI). O uso real — reflexão e
consolidação — entra na E2. `cache_system=True` aplica prompt caching no bloco
de system (usado pro user_profile na E2, barateando runs repetidos).

Dois jeitos de falar com o Claude:
- `anthropic` (default): SDK da API, cobra do crédito pay-as-you-go (ANTHROPIC_API_KEY).
- `claude_cli`: roteia pelo binário `claude -p`, usando o login da assinatura
  Pro/Max — sem API key. Trade-off: sem `cache_control` estruturado e sem
  `max_tokens` de saída (a CLI não expõe). Bom pra rodar com a Max.
"""

from __future__ import annotations

import os
import shutil
import subprocess
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


class ClaudeCliClient:
    """Fala com o Claude via `claude -p` — usa o login da assinatura Pro/Max, sem API key.

    Resolve o binário (cfg.cli_bin, default "claude") e remove ANTHROPIC_API_KEY
    do ambiente do subprocess, pra forçar a cobrança na assinatura (OAuth) em vez
    da API paga. `cache_system` e `max_tokens` são aceitos por compatibilidade de
    interface, mas a CLI não os expõe — então são ignorados.
    """

    def __init__(self, cfg: LLMCfg) -> None:
        self.cfg = cfg
        self._bin: str | None = None

    def _ensure_bin(self) -> str:
        if self._bin is None:
            found = shutil.which(self.cfg.cli_bin) or os.path.expanduser(
                f"~/.local/bin/{self.cfg.cli_bin}"
            )
            if not (shutil.which(self.cfg.cli_bin) or os.path.exists(found)):
                raise RuntimeError(
                    f"binário {self.cfg.cli_bin!r} não encontrado no PATH; "
                    "instale o Claude Code e faça login (claude) com sua conta Pro/Max."
                )
            self._bin = shutil.which(self.cfg.cli_bin) or found
        return self._bin

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        model: str,
        cache_system: bool = False,
        max_tokens: int = 1024,
    ) -> str:
        bin_path = self._ensure_bin()
        # Sem ANTHROPIC_API_KEY: força o uso do login da assinatura, não da API paga.
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        proc = subprocess.run(
            [bin_path, "-p", "--model", model, "--system-prompt", system, "--output-format", "text"],
            input=prompt,
            capture_output=True,
            text=True,
            env=env,
            timeout=self.cfg.cli_timeout_s,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude -p falhou (exit {proc.returncode}): {proc.stderr.strip()}")
        return proc.stdout.strip()


class FakeLLM:
    """Retorna uma resposta fixa — pra reflexão rodar offline e em testes, sem API.

    `canned` é o texto devolvido por `complete` (ex.: um JSON de reflexão).
    """

    def __init__(self, canned: str = "{}") -> None:
        self.canned = canned
        self.calls: list[dict] = []

    def complete(
        self, *, system: str, prompt: str, model: str, cache_system: bool = False, max_tokens: int = 1024
    ) -> str:
        self.calls.append({"system": system, "prompt": prompt, "model": model})
        return self.canned


def make_llm(cfg: LLMCfg) -> LLMClient:
    if cfg.provider == "anthropic":
        return AnthropicClient(cfg)
    if cfg.provider == "claude_cli":
        return ClaudeCliClient(cfg)
    if cfg.provider == "fake":
        return FakeLLM()
    raise ValueError(f"llm provider desconhecido: {cfg.provider!r}")
