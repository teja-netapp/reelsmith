"""Command line: python -m app.cli --clips a.mp4 b.mov --music song.mp3 --prompt "moody travel reel" """
from __future__ import annotations

import argparse
import json
import shutil
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description="Reelsmith - AI cinematic reel editor")
    ap.add_argument("--clips", nargs="+", required=True, help="video files (clips or full-length videos)")
    ap.add_argument("--music", help="optional music track (mp3/m4a/wav)")
    ap.add_argument("--prompt", default="", help="describe the reel you want")
    ap.add_argument("--duration", type=float, default=30, help="target length in seconds")
    ap.add_argument("--out", default="reel.mp4")
    ap.add_argument("--captions", action="store_true")
    ap.add_argument("--no-beat-sync", action="store_true")
    ap.add_argument("--no-autopick", action="store_true", help="use all clips in order")
    ap.add_argument("--no-reframe", action="store_true", help="centre-crop instead of subject tracking")
    ap.add_argument("--quality", choices=["preview", "draft", "final"], default="final")
    ap.add_argument("--editor", choices=["auto", "gemini", "ollama", "claude", "heuristic"], default=None,
                    help="which AI edits the reel (default: LLM_PROVIDER from .env, else auto)")
    ap.add_argument("--offline", action="store_true", help="no AI at all: built-in rules (same as --editor heuristic)")
    a = ap.parse_args(argv)

    from .config import LLM_PROVIDER
    from .pipeline import run_sync

    editor = "heuristic" if a.offline else (a.editor or LLM_PROVIDER or "auto")
    job = run_sync(a.clips, a.music, a.prompt, {
        "target_duration": a.duration, "captions": a.captions, "beat_sync": not a.no_beat_sync,
        "auto_pick": not a.no_autopick, "smart_reframe": not a.no_reframe, "quality": a.quality,
        "planner": editor,
    })
    if job.status != "done":
        print("FAILED:", job.error, file=sys.stderr)
        print("\n".join(job.logs[-15:]), file=sys.stderr)
        return 1
    shutil.copy(job.output, a.out)
    print(json.dumps({"output": a.out, "title": job.plan.get("title"), "planner": job.plan.get("planner"),
                      "reasoning": job.plan.get("reasoning"), "duration": job.timeline_summary["duration"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
