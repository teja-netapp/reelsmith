"""AI editor providers. 'auto' tries Claude -> Gemini -> local Ollama -> offline rules."""
from __future__ import annotations

from ..config import LLM_PROVIDER
from .base import Provider, ProviderError

__all__ = ["Provider", "ProviderError", "PROVIDERS", "CHOICES", "chain", "statuses"]
from .claude import ClaudeProvider
from .gemini import GeminiProvider
from .ollama import OllamaProvider

PROVIDERS: dict[str, Provider] = {
    "claude": ClaudeProvider(),
    "gemini": GeminiProvider(),
    "ollama": OllamaProvider(),
}
AUTO_ORDER = ["claude", "gemini", "ollama"]
CHOICES = ["auto", "gemini", "ollama", "claude", "heuristic"]


def statuses() -> dict:
    out = {k: p.status() for k, p in PROVIDERS.items()}
    out["heuristic"] = {"available": True, "detail": "built-in rules, no AI"}
    return out


def chain(choice: str | None) -> list[Provider]:
    """Providers to try, in order. The heuristic planner is always the final fallback."""
    choice = (choice or LLM_PROVIDER or "auto").lower()
    if choice == "heuristic":
        return []
    if choice in PROVIDERS:
        return [PROVIDERS[choice]]
    return [PROVIDERS[k] for k in AUTO_ORDER if PROVIDERS[k].status().get("available")]
