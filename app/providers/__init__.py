"""AI editor providers. 'auto' tries Claude -> Gemini -> local Ollama -> offline rules."""
from __future__ import annotations

from .. import config as C
from .base import Provider, ProviderError

__all__ = ["Provider", "ProviderError", "PROVIDERS", "CHOICES", "chain", "chain_with_skips", "statuses"]
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


def chain_with_skips(choice: str | None) -> tuple[list[Provider], list[dict]]:
    """Providers to try, in order (the heuristic planner is always the final fallback), plus the
    editors 'auto' skipped and why - recorded in the edit report."""
    choice = (choice or C.LLM_PROVIDER or "auto").lower()
    if choice == "heuristic":
        return [], []
    if choice in PROVIDERS:
        return [PROVIDERS[choice]], []
    use, skipped = [], []
    for k in AUTO_ORDER:
        st = PROVIDERS[k].status()
        if st.get("available"):
            use.append(PROVIDERS[k])
        else:
            skipped.append({"provider": k, "reason": st.get("detail", "not available")})
    return use, skipped


def chain(choice: str | None) -> list[Provider]:
    return chain_with_skips(choice)[0]
