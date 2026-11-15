"""Test setup: isolated workspace, fake Ollama, synthetic media generated with FFmpeg."""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

FAKE_OLLAMA_PORT = 11987
_WS = tempfile.mkdtemp(prefix="reelsmith-test-")
# configure BEFORE the app is imported (config is read at import time)
os.environ["REELSMITH_WORKSPACE"] = _WS
os.environ["OLLAMA_HOST"] = f"http://127.0.0.1:{FAKE_OLLAMA_PORT}"
os.environ["OLLAMA_MODEL"] = "qwen3-vl:8b-instruct"
os.environ["LLM_PROVIDER"] = "auto"
os.environ["REELSMITH_NO_TRANSCRIBE"] = "1"  # don't download Whisper in CI
for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OLLAMA_PLAN_MODEL"):
    os.environ.pop(k, None)


def _ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="session")
def media(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    d = tmp_path_factory.mktemp("media")
    land = d / "land.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=3", "-f", "lavfi", "-i", "mandelbrot=s=1280x720:r=30",
        "-f", "lavfi", "-i", "color=c=0x203040:s=1280x720:r=30:d=3", "-f", "lavfi", "-i", "sine=f=330:d=9",
        "-filter_complex", "[1:v]trim=duration=3,setpts=PTS-STARTPTS[m];[0:v][m][2:v]concat=n=3:v=1:a=0[v]",
        "-map", "[v]", "-map", "3:a", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(land))
    track = d / "track.mp4"
    _ff("-f", "lavfi", "-i", "color=c=0x223344:s=1280x720:r=30:d=5", "-f", "lavfi",
        "-i", "color=c=0xffcc33:s=120x200:r=30:d=5", "-filter_complex",
        "[0:v][1:v]overlay=x='100+t*180':y=260:shortest=1[v]", "-map", "[v]", "-c:v", "libx264",
        "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(track))
    music = d / "music.mp3"
    expr = "0.8*sin(2*PI*55*t)*exp(-18*mod(t,0.5))"
    _ff("-f", "lavfi", "-i", f"aevalsrc='{expr}|{expr}':s=44100:d=30", "-c:a", "libmp3lame", "-q:a", "5", str(music))
    return {"land": str(land), "track": str(track), "music": str(music)}


@pytest.fixture(scope="session")
def fake_ollama():
    import fake_ollama as fo

    srv, log = fo.start(FAKE_OLLAMA_PORT)
    yield log
    srv.shutdown()
