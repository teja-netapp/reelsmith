"""Global configuration for Reelsmith.

AI-editor settings (keys, models, hosts) can change while the app runs - from the Settings panel -
so other modules read them as `config.NAME` (live) rather than `from config import NAME` (frozen).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = Path(os.environ.get("REELSMITH_ENV_FILE", ROOT / ".env"))

# keys exported in the shell (not from .env) win over .env on every start - the UI warns about this
SECRET_KEYS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
SHELL_VARS = {k for k in SECRET_KEYS + ("OLLAMA_HOST", "LLM_PROVIDER") if os.environ.get(k)}

try:  # optional .env support
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE)
except Exception:  # pragma: no cover
    pass

WORKSPACE = Path(os.environ.get("REELSMITH_WORKSPACE", ROOT / "workspace"))
MODELS_DIR = ROOT / "models"
FONTS_DIR = ROOT / "assets" / "fonts"
WEB_DIR = ROOT / "web"
FACE_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
FPS = 30


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else default


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def reload() -> None:
    """(Re)read AI settings from the environment. Called at import and after the UI saves settings."""
    global LLM_PROVIDER, CLAUDE_MODEL, GEMINI_MODEL, OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_PLAN_MODEL
    global OLLAMA_NUM_CTX, LOCAL_MAX_SHOTS, WHISPER_MODEL, MAX_SHOTS_FOR_PLANNER, OLLAMA_PLAN_DEPTH
    # Which AI plans the edit: auto | gemini | ollama | claude | heuristic
    LLM_PROVIDER = _env("LLM_PROVIDER", "auto").lower()
    CLAUDE_MODEL = _env("CLAUDE_MODEL", "claude-sonnet-5-5")
    GEMINI_MODEL = _env("GEMINI_MODEL")  # empty = auto-detect (prefers gemini-4-hydrogen)
    OLLAMA_HOST = _env("OLLAMA_HOST", "http://127.0.0.1:11434")
    OLLAMA_MODEL = _env("OLLAMA_MODEL", "qwen3-vl:8b-instruct")       # vision: logs every shot
    OLLAMA_PLAN_MODEL = _env("OLLAMA_PLAN_MODEL", OLLAMA_MODEL)          # writes the edit
    OLLAMA_NUM_CTX = _int("OLLAMA_NUM_CTX", 16384)
    OLLAMA_PLAN_DEPTH = _env("OLLAMA_PLAN_DEPTH", "best").lower()  # best = story outline + edit, fast = 1 pass
    LOCAL_MAX_SHOTS = _int("LOCAL_MAX_SHOTS", 30)
    WHISPER_MODEL = _env("WHISPER_MODEL", "base")
    MAX_SHOTS_FOR_PLANNER = _int("REELSMITH_MAX_SHOTS", 72)


reload()


def has_api_key() -> bool:
    """Claude credentials present."""
    return bool(_env("ANTHROPIC_API_KEY") or _env("ANTHROPIC_AUTH_TOKEN"))


def gemini_key() -> str:
    return _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")


def mask(secret: str) -> str:
    s = (secret or "").strip()
    if not s:
        return ""
    return f"{s[:4]}…{s[-4:]}" if len(s) > 12 else "••••"


# ----------------------------------------------------------------------------------------------
# .env writer (keeps your comments and other settings; file is git-ignored and chmod 600)
# ----------------------------------------------------------------------------------------------
_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


def _quote(v: str) -> str:
    if v == "" or re.fullmatch(r"[A-Za-z0-9_\-.:/@+]*", v):
        return v
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


def update_env_file(updates: dict[str, str], path: Path | None = None) -> Path:
    path = Path(path or ENV_FILE)
    if path.exists():
        lines = path.read_text().splitlines()
    elif (ROOT / ".env.example").exists():
        lines = (ROOT / ".env.example").read_text().splitlines()
    else:
        lines = []
    done: set[str] = set()
    out: list[str] = []
    for ln in lines:
        m = _LINE.match(ln)
        if m and m.group(1) in updates:
            k = m.group(1)
            if k in done:  # drop duplicate definitions
                continue
            out.append(f"{k}={_quote(updates[k])}")
            done.add(k)
        else:
            out.append(ln)
    for k, v in updates.items():
        if k not in done:
            out.append(f"{k}={_quote(v)}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(out) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    for k, v in updates.items():
        os.environ[k] = v
    reload()
    return path


WORKSPACE.mkdir(parents=True, exist_ok=True)
