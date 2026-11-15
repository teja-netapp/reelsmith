"""Local web server: python -m uvicorn app.main:app --port 8765"""
from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .audio import whisper_available
from .config import FACE_MODEL, LLM_PROVIDER, WEB_DIR
from .effects import CAPTION_STYLES, LOOK_DESCRIPTIONS, TEXT_STYLES
from .pipeline import DEFAULT_OPTIONS, Job, Runner
from .providers import CHOICES, statuses
from .reframe import MOTIONS
from .transitions import TRANSITIONS
from .video_io import check_ffmpeg, video_encoder

app = FastAPI(title="Reelsmith")
runner = Runner()
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mts", ".m2ts", ".3gp", ".hevc"}


def _safe_name(name: str) -> str:
    base = os.path.basename(name or "file")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)[:120] or "file"


def _clean_options(raw: dict) -> dict:
    o = dict(DEFAULT_OPTIONS)
    if not isinstance(raw, dict):
        return o
    if "target_duration" in raw:
        try:
            o["target_duration"] = float(min(180, max(5, float(raw["target_duration"]))))
        except Exception:
            pass
    for k in ("auto_pick", "beat_sync", "captions", "smart_reframe"):
        if k in raw:
            o[k] = bool(raw[k])
    if raw.get("quality") in ("preview", "draft", "final"):
        o["quality"] = raw["quality"]
    if raw.get("planner") in CHOICES:
        o["planner"] = raw["planner"]
    return o


def _get(job_id: str) -> Job:
    job = runner.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


async def _save_upload(f: UploadFile, dest: str):
    with open(dest, "wb") as out:
        while True:
            chunk = await f.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


@app.get("/")
def index():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/api/config")
def config():
    return {
        "version": __version__,
        "providers": statuses(),
        "default_planner": LLM_PROVIDER if LLM_PROVIDER in CHOICES else "auto",
        "whisper": whisper_available(),
        "face_model": FACE_MODEL.exists(),
        "ffmpeg": check_ffmpeg(),
        "encoder": video_encoder(),
        "looks": LOOK_DESCRIPTIONS,
        "transitions": {k: v[1] for k, v in TRANSITIONS.items()},
        "motions": MOTIONS,
        "text_styles": list(TEXT_STYLES),
        "caption_styles": list(CAPTION_STYLES),
        "defaults": DEFAULT_OPTIONS,
    }


@app.post("/api/jobs")
async def create_job(
    clips: Optional[List[UploadFile]] = File(None),
    music: Optional[UploadFile] = File(None),
    description: str = Form(""),
    options: str = Form("{}"),
    paths: str = Form(""),
):
    try:
        opts = json.loads(options or "{}")
    except Exception:
        opts = {}
    job = Job()
    up = job.dir / "uploads"
    up.mkdir(exist_ok=True)
    clip_paths = []
    for i, f in enumerate(clips or []):
        if not f.filename:
            continue
        dest = str(up / f"{i:02d}_{_safe_name(f.filename)}")
        await _save_upload(f, dest)
        clip_paths.append(dest)
    # optional: absolute paths to files already on this Mac (avoids copying huge videos)
    for line in (paths or "").splitlines():
        p = os.path.expanduser(line.strip().strip('"').strip("'"))
        if not p:
            continue
        if not os.path.isfile(p):
            raise HTTPException(400, f"File not found: {p}")
        clip_paths.append(os.path.abspath(p))
    if not clip_paths:
        raise HTTPException(400, "Add at least one video clip")
    if music is not None and music.filename:
        mdest = str(up / f"music_{_safe_name(music.filename)}")
        await _save_upload(music, mdest)
        job.music = mdest
    job.clips = clip_paths
    job.description = description.strip()[:4000]
    job.options = _clean_options(opts)
    runner.submit(job, "new")
    return {"id": job.id}


@app.get("/api/jobs")
def list_jobs():
    out = []
    for j in sorted(runner.jobs.values(), key=lambda j: j.created, reverse=True)[:30]:
        out.append({"id": j.id, "status": j.status, "title": (j.plan or {}).get("title"),
                    "description": j.description[:120], "created": j.created, "has_output": bool(j.output)})
    return out


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _get(job_id).to_public()


@app.get("/api/jobs/{job_id}/output")
def get_output(job_id: str, download: int = 0):
    job = _get(job_id)
    if not job.output or not os.path.exists(job.output):
        raise HTTPException(404, "no output yet")
    if download:
        title = re.sub(r"[^A-Za-z0-9]+", "-", (job.plan or {}).get("title") or "reel").strip("-").lower() or "reel"
        return FileResponse(job.output, media_type="video/mp4", filename=f"{title}-v{job.version}.mp4")
    return FileResponse(job.output, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/thumb/{shot_id}")
def get_thumb(job_id: str, shot_id: str):
    job = _get(job_id)
    if not re.fullmatch(r"c\d+s\d+", shot_id):
        raise HTTPException(400, "bad shot id")
    p = job.dir / "thumbs" / f"{shot_id}.jpg"
    if not p.exists():
        raise HTTPException(404, "no thumbnail")
    return FileResponse(str(p), media_type="image/jpeg")


@app.post("/api/jobs/{job_id}/refine")
def refine(job_id: str, body: dict = Body(...)):
    job = _get(job_id)
    if job.status in ("running", "queued"):
        raise HTTPException(409, "job is busy")
    if not job.analysis or not job.plan:
        raise HTTPException(409, "nothing to refine yet")
    fb = str(body.get("feedback", "")).strip()
    if not fb:
        raise HTTPException(400, "feedback is empty")
    if isinstance(body.get("options"), dict):
        job.options = _clean_options({**job.options, **body["options"]})
    runner.submit(job, "refine", fb[:2000])
    return {"ok": True}


@app.post("/api/jobs/{job_id}/render")
def rerender(job_id: str, body: dict = Body(...)):
    job = _get(job_id)
    if job.status in ("running", "queued"):
        raise HTTPException(409, "job is busy")
    if not job.analysis:
        raise HTTPException(409, "job has no analysis")
    plan = body.get("plan") or job.plan
    if not isinstance(plan, dict):
        raise HTTPException(400, "plan must be an object")
    if isinstance(body.get("options"), dict):
        job.options = _clean_options({**job.options, **body["options"]})
    runner.submit(job, "plan", plan)
    return {"ok": True}
