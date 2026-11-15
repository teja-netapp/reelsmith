#!/usr/bin/env bash
# Start Reelsmith at http://127.0.0.1:8765
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then echo "Run ./setup.sh first"; exit 1; fi
# shellcheck disable=SC1091
source .venv/bin/activate

# make sure the local AI runtime is up (if it's installed)
if command -v ollama >/dev/null 2>&1 && ! curl -fsS -m 1 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  echo "Starting Ollama…"
  open -a Ollama >/dev/null 2>&1 || brew services start ollama >/dev/null 2>&1 || (nohup ollama serve >/dev/null 2>&1 &)
  for _ in $(seq 1 15); do curl -fsS -m 1 http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break; sleep 1; done
fi

PORT="${PORT:-8765}"
( sleep 2; open "http://127.0.0.1:$PORT" >/dev/null 2>&1 || true ) &
exec python -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT"
