#!/usr/bin/env bash
# One-time setup for Reelsmith on macOS (Apple Silicon or Intel). Safe to re-run.
#   SKIP_OLLAMA=1 ./setup.sh   -> skip installing Ollama / pulling the local model
set -euo pipefail
cd "$(dirname "$0")"

say() { printf "\n\033[1;33m▸ %s\033[0m\n" "$*"; }
envval() { [ -f .env ] && grep -E "^$1=" .env | tail -1 | cut -d= -f2- | tr -d '"' || true; }

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required. Install it from https://brew.sh and re-run ./setup.sh"; exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  say "Installing FFmpeg (video engine)…"; brew install ffmpeg
fi

PY=""
for c in python3.12 python3.11 python3.13; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  say "Installing Python 3.12…"; brew install python@3.12; PY=python3.12
fi

if [ ! -d .venv ]; then
  say "Creating virtual environment with $PY"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel >/dev/null
say "Installing Python packages (first run takes a few minutes)…"
pip install -r requirements.txt

mkdir -p models assets/fonts workspace
[ -f .env ] || cp .env.example .env

say "Downloading face-tracking model (YuNet, 230 KB)…"
YUNET=models/face_detection_yunet_2023mar.onnx
if [ ! -s "$YUNET" ]; then
  curl -fsSL -o "$YUNET" "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
    || curl -fsSL -o "$YUNET" "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
    || echo "  (could not download - smart reframe will use saliency tracking)"
  if [ -f "$YUNET" ] && [ "$(wc -c <"$YUNET")" -lt 50000 ]; then rm -f "$YUNET"; echo "  (download incomplete - skipped)"; fi
fi

say "Downloading open-source fonts (OFL) for titles & captions…"
G=https://github.com/google/fonts/raw/main/ofl
for f in "poppins/Poppins-ExtraBold.ttf" "poppins/Poppins-Bold.ttf" "poppins/Poppins-Light.ttf" \
         "bebasneue/BebasNeue-Regular.ttf" "anton/Anton-Regular.ttf"; do
  out="assets/fonts/$(basename "$f")"
  [ -s "$out" ] || curl -fsSL -o "$out" "$G/$f" || echo "  (skipped $f - system fonts will be used)"
done

WM="$(envval WHISPER_MODEL)"; WM="${WM:-base}"
say "Pre-downloading the speech model (Whisper '$WM')…"
python - "$WM" <<'EOF' || echo "  (will download on first use)"
import sys
from faster_whisper import WhisperModel
WhisperModel(sys.argv[1], device="cpu", compute_type="int8")
print("  ok")
EOF

# ---------------------------------------------------------------------------------------------
# Local AI editor (Ollama)
# ---------------------------------------------------------------------------------------------
if [ "${SKIP_OLLAMA:-0}" != "1" ]; then
  OM="$(envval OLLAMA_MODEL)"; OM="${OM:-qwen3-vl:8b-instruct}"
  if ! command -v ollama >/dev/null 2>&1; then
    say "Installing Ollama (local AI runtime)…"
    brew install ollama
  fi
  if ! curl -fsS -m 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    say "Starting Ollama…"
    open -a Ollama >/dev/null 2>&1 || brew services start ollama >/dev/null 2>&1 || (nohup ollama serve >/dev/null 2>&1 &)
    for _ in $(seq 1 20); do curl -fsS -m 1 http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break; sleep 1; done
  fi
  say "Pulling the local vision model $OM (one-time download, ~6 GB for the 8B model)…"
  if ! ollama pull "$OM"; then
    echo "  Could not pull $OM. If Ollama says it needs a newer version: brew upgrade ollama (or update the Ollama app)."
  fi
fi

say "Done!"
echo "  • Gemini (optional, recommended): in the app click ⚙ Settings and paste your key (https://aistudio.google.com/apikey)"
echo "  • Local AI works out of the box with Ollama ($(envval OLLAMA_MODEL))"
echo "  • Start the app:  ./run.sh"
