"""Global configuration for Reelsmith."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:  # optional .env support
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:  # pragma: no cover
    pass

WORKSPACE = Path(os.environ.get("REELSMITH_WORKSPACE", ROOT / "workspace"))
MODELS_DIR = ROOT / "models"
FONTS_DIR = ROOT / "assets" / "fonts"
WEB_DIR = ROOT / "web"
FACE_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"

def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else default


# Which AI plans the edit: auto | gemini | ollama | claude | heuristic
LLM_PROVIDER = _env("LLM_PROVIDER", "auto").lower()

CLAUDE_MODEL = _env("CLAUDE_MODEL", "claude-sonnet-5-5")

GEMINI_MODEL = _env("GEMINI_MODEL")  # empty = auto-detect (prefers gemini-4-hydrogen)

OLLAMA_HOST = _env("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = _env("OLLAMA_MODEL", "qwen3-vl:8b-instruct")       # vision: logs every shot
OLLAMA_PLAN_MODEL = _env("OLLAMA_PLAN_MODEL", OLLAMA_MODEL)          # writes the edit (same model = no swapping)
OLLAMA_NUM_CTX = int(_env("OLLAMA_NUM_CTX", "16384"))
LOCAL_MAX_SHOTS = int(_env("LOCAL_MAX_SHOTS", "30"))

WHISPER_MODEL = _env("WHISPER_MODEL", "base")
FPS = 30
MAX_SHOTS_FOR_PLANNER = int(_env("REELSMITH_MAX_SHOTS", "72"))


def has_api_key() -> bool:
    """Claude credentials present."""
    return bool(_env("ANTHROPIC_API_KEY") or _env("ANTHROPIC_AUTH_TOKEN"))


def gemini_key() -> str:
    return _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")


WORKSPACE.mkdir(parents=True, exist_ok=True)
