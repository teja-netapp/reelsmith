"""Local web server: python -m uvicorn app.main:app --port 8765"""
from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from urllib.parse import urlparse

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from . import config as C
from . import settings as S
from .audio import whisper_available
from .config import FACE_MODEL, WEB_DIR
from .effects import CAPTION_STYLES, TEXT_STYLES
from .grading import (LOOK_CATEGORIES, apply_to_image, look_category, look_descriptions, look_family, look_label,
                      look_names)
from .learning import get_learner
from .pipeline import DEFAULT_OPTIONS, Job, Runner
from .providers import CHOICES, statuses
from .reframe import MOTIONS
from .report import report_markdown
from .transitions import TRANSITIONS
from .video_io import check_ffmpeg, video_encoder

app = FastAPI(title="Reelsmith")
runner = Runner()
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

# The app holds your API keys, so it only answers requests addressed to this Mac (blocks DNS-rebinding)
# and refuses changes coming from other websites open in your browser (blocks cross-site requests).
_LOCAL = {"127.0.0.1", "localhost", "::1", "testserver"} | {
    h.strip() for h in os.environ.get("REELSMITH_ALLOWED_HOSTS", "").split(",") if h.strip()}


def _hostname(value: str) -> str:
    try:
        return (urlparse("//" + value).hostname or "").lower()
    except ValueError:
        return ""


@app.middleware("http")
async def local_only(request: Request, call_next):
    if _hostname(request.headers.get("host", "")) not in _LOCAL:
        return JSONResponse({"detail": "Reelsmith only accepts requests to localhost."}, status_code=403)
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin and origin != "null" and (urlparse(origin).hostname or "").lower() not in _LOCAL:
            return JSONResponse({"detail": "Cross-site request blocked."}, status_code=403)
        if origin == "null":
            return JSONResponse({"detail": "Cross-site request blocked."}, status_code=403)
    resp = await call_next(request)
    if request.url.path.startswith("/api/settings"):
        resp.headers["Cache-Control"] = "no-store"
    return resp

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
    for k in ("auto_pick", "beat_sync", "captions", "smart_reframe", "auto_color", "grading"):
        if k in raw:
            o[k] = bool(raw[k])
    if "grade_strength" in raw:
        try:
            o["grade_strength"] = round(float(min(1.5, max(0.0, float(raw["grade_strength"])))), 3)
        except Exception:
            pass
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
        "default_planner": C.LLM_PROVIDER if C.LLM_PROVIDER in CHOICES else "auto",
        "whisper": whisper_available(),
        "face_model": FACE_MODEL.exists(),
        "ffmpeg": check_ffmpeg(),
        "encoder": video_encoder(),
        "looks": look_descriptions(),
        "transitions": {k: v[1] for k, v in TRANSITIONS.items()},
        "motions": MOTIONS,
        "text_styles": list(TEXT_STYLES),
        "caption_styles": list(CAPTION_STYLES),
        "defaults": DEFAULT_OPTIONS,
    }


@app.get("/api/settings")
def get_settings():
    """AI editor settings. API keys are only ever returned masked."""
    return S.get_settings()


@app.post("/api/settings/verify")
def verify_settings(body: dict = Body(...)):
    """Check a key (typed but not saved yet, or the saved one if 'key' is empty) / the Ollama connection."""
    try:
        return S.verify(str(body.get("provider", "")), key=(body.get("key") or "").strip() or None,
                        host=body.get("host") or None, model=body.get("model") or None,
                        remember=not body.get("key"))
    except S.SettingsError as e:
        raise HTTPException(400, str(e))


@app.post("/api/settings")
def save_settings(body: dict = Body(...)):
    """Save keys/models (verified first; a rejected key is never saved). Applies immediately, no restart."""
    try:
        return S.save_settings(body)
    except S.SettingsError as e:
        raise HTTPException(400, str(e))


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


def _slug(job: Job) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", (job.plan or {}).get("title") or "reel").strip("-").lower() or "reel"


@app.get("/api/jobs/{job_id}/output")
def get_output(job_id: str, download: int = 0, v: int = 0):
    job = _get(job_id)
    path = job.output_for(v) or job.output
    if not path or not os.path.exists(path):
        raise HTTPException(404, "no output yet")
    ver = v if job.output_for(v) else job.version
    if download:
        try:  # downloading a version is a (weak) "I like this one" signal
            get_learner().downloaded(job.id, ver)
        except Exception:
            pass
        return FileResponse(path, media_type="video/mp4", filename=f"{_slug(job)}-v{ver}.mp4")
    return FileResponse(path, media_type="video/mp4")


def _report(job: Job, v: int) -> dict:
    p = job.report_path(v or job.version)
    if not p.exists():
        raise HTTPException(404, "no report for this version")
    return json.loads(p.read_text())


@app.get("/api/jobs/{job_id}/report")
def get_report(job_id: str, v: int = 0):
    return _report(_get(job_id), v)


@app.get("/api/jobs/{job_id}/report.md")
def get_report_md(job_id: str, v: int = 0):
    job = _get(job_id)
    r = _report(job, v)
    return PlainTextResponse(report_markdown(r), media_type="text/markdown",
                             headers={"Content-Disposition": f'attachment; filename="{_slug(job)}-v{r["version"]}-report.md"'})


@app.get("/api/jobs/{job_id}/ai_plan")
def get_ai_plan(job_id: str, v: int = 0):
    """The editor's raw plan exactly as the AI returned it, before validation / assistant-editor fixes."""
    job = _get(job_id)
    p = job.dir / f"ai_plan_v{v or job.version}.json"
    if not p.exists():
        raise HTTPException(404, "no raw plan for this version")
    return FileResponse(str(p), media_type="application/json", filename=f"{_slug(job)}-v{v or job.version}-ai-plan.json")


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


# ----------------------------------------------------------------------------------------------
# Learning: ratings, Look Lab, taste stats
# ----------------------------------------------------------------------------------------------
@app.post("/api/jobs/{job_id}/rate")
def rate_version(job_id: str, body: dict = Body(...)):
    job = _get(job_id)
    try:
        v, stars = int(body.get("version") or job.version), float(body.get("stars"))
    except Exception:
        raise HTTPException(400, "version and stars (1-5) are required")
    if not 1 <= stars <= 5:
        raise HTTPException(400, "stars must be 1-5")
    r = get_learner().rate(job.id, v, stars)
    for x in job.versions:
        if x["version"] == v:
            x["rating"] = stars
    job.save()
    return {**r, "stats": get_learner().stats(job.description)}


def _hero_thumb(job: Job, v: int = 0):
    import cv2
    plan = job.plan or {}
    if v and (job.dir / f"plan_v{v}.json").exists():
        plan = json.loads((job.dir / f"plan_v{v}.json").read_text())
    sid = next((s.get("shot_id") for s in plan.get("segments", []) if re.fullmatch(r"c\d+s\d+", s.get("shot_id") or "")), None)
    p = job.dir / "thumbs" / f"{sid}.jpg" if sid else None
    img = cv2.imread(str(p)) if p and p.exists() else None
    return img, plan


@app.get("/api/jobs/{job_id}/looks")
def list_looks(job_id: str):
    job = _get(job_id)
    d = look_descriptions()
    prof = get_learner().profile(job.description)
    liked = prof.get("looks") or []
    return {"current": (job.plan or {}).get("look"), "liked": liked,
            "grading": (job.plan or {}).get("grading", True) is not False,
            "strength": float(job.options.get("grade_strength", 1.0)),
            "categories": LOOK_CATEGORIES + (["Your LUTs"] if any(n.startswith("lut:") for n in look_names()) else []),
            "looks": [{"name": n, "label": look_label(n), "category": look_category(n), "family": look_family(n),
                       "description": d.get(n, ""), "liked": n in liked}
                      for n in look_names() if n != "none"]}


@app.get("/api/jobs/{job_id}/look_preview")
def look_preview(job_id: str, look: str, strength: float = 1.0):
    import cv2
    job = _get(job_id)
    if look not in look_names():
        raise HTTPException(404, "unknown look")
    strength = round(min(1.5, max(0.0, float(strength))), 2)
    out = job.dir / "looks" / f"{re.sub(r'[^A-Za-z0-9_-]', '_', look)}_{int(strength * 100)}.jpg"
    if not out.exists():
        img, plan = _hero_thumb(job)
        if img is None:
            raise HTTPException(404, "no frame to preview")
        out.parent.mkdir(exist_ok=True)
        graded = apply_to_image(img, look, float(plan.get("look_intensity", 1.0)) * strength)
        cv2.imwrite(str(out), graded, [cv2.IMWRITE_JPEG_QUALITY, 86])
    return FileResponse(str(out), media_type="image/jpeg")


@app.post("/api/jobs/{job_id}/look")
def apply_look(job_id: str, body: dict = Body(...)):
    """Look Lab: re-render the current edit with another grade (also teaches the taste model)."""
    job = _get(job_id)
    if job.status in ("running", "queued"):
        raise HTTPException(409, "job is busy")
    look = str(body.get("look", ""))
    if look not in look_names() or not job.plan:
        raise HTTPException(400, "unknown look")
    from .grading import look_texture
    plan = {k: v for k, v in job.plan.items() if not k.startswith("_")}
    tex = look_texture(look)
    plan.update({"look": look, "halation": tex["halation"], "bloom": tex["bloom"], "_source": "look_lab"})
    if isinstance(body.get("options"), dict):
        job.options = _clean_options({**job.options, **body["options"]})
    runner.submit(job, "plan", plan)
    return {"ok": True}


@app.get("/api/transitions")
def transitions_catalog():
    """Every transition (by CapCut-style category) and clip effect, for the per-cut pickers."""
    from .clipfx import BEAT_EFFECTS, CLIP_EFFECTS
    from .transitions import CATEGORY_LABELS, REGISTRY, by_category
    cats = [{"key": c, "label": CATEGORY_LABELS.get(c, c),
             "items": [{"name": n, "description": REGISTRY[n].description, "energy": REGISTRY[n].energy,
                        "duration": REGISTRY[n].duration} for n in names]}
            for c, names in by_category().items()]
    cats.insert(0, {"key": "cut", "label": "Cut", "items": [{"name": "cut", "description": "hard cut",
                                                            "energy": 1, "duration": 0.0}]})
    return {"categories": cats,
            "effects": [{"name": k, "description": v, "beat": k in BEAT_EFFECTS} for k, v in CLIP_EFFECTS.items()]}


def _preview_img(job_id: str | None, shot: str | None):
    import cv2
    if not (job_id and shot) or job_id not in runner.jobs or not re.fullmatch(r"c\d+s\d+", shot):
        return None
    p = runner.jobs[job_id].dir / "thumbs" / f"{shot}.jpg"
    return cv2.imread(str(p)) if p.exists() else None


def _preview_dir(job_id: str | None):
    if job_id and job_id in runner.jobs:
        return runner.jobs[job_id].dir / "previews"
    return C.WORKSPACE / "previews"


@app.get("/api/transition_preview")
def transition_preview(name: str, job: str | None = None, a: str | None = None, b: str | None = None):
    from .previews import transition_clip
    from .transitions import REGISTRY
    if name not in REGISTRY:
        raise HTTPException(404, "unknown transition")
    ia, ib = _preview_img(job, a), _preview_img(job, b)
    tag = f"{a}_{b}" if ia is not None and ib is not None else "demo"
    out = transition_clip(name, ia if tag != "demo" else None, ib if tag != "demo" else None,
                          _preview_dir(job) / f"tr_{name}_{tag}.mp4")
    return FileResponse(str(out), media_type="video/mp4", headers={"Cache-Control": "max-age=86400"})


@app.get("/api/effect_preview")
def effect_preview(name: str, job: str | None = None, shot: str | None = None):
    from .clipfx import CLIP_EFFECTS
    from .previews import effect_clip
    if name not in CLIP_EFFECTS:
        raise HTTPException(404, "unknown effect")
    img = _preview_img(job, shot)
    out = effect_clip(name, img, _preview_dir(job) / f"fx_{name}_{shot if img is not None else 'demo'}.mp4")
    return FileResponse(str(out), media_type="video/mp4", headers={"Cache-Control": "max-age=86400"})


@app.post("/api/directives")
def detect_directives(body: dict = Body(...)):
    """What Reelsmith will treat as hard rules in this text (shown live under the brief / revision box)."""
    from .planner import instructions
    text = str(body.get("text", ""))[:4000]
    opts = _clean_options(body.get("options") or {})
    jid = body.get("job_id")
    if body.get("revision") and not jid:  # a revision changes only what it mentions ("just add X" = add X)
        from . import directives as D
        from .grading import look_names as _ln
        rules = D.collect("", [text], {}, _ln())
    elif jid and jid in runner.jobs:
        job = runner.jobs[jid]
        rules = instructions(job.description, {**opts, **job.options, "_revisions": list(job.instructions)},
                             text or None)
    else:
        rules = instructions(text, opts)
    pub = rules.to_public()
    pub["text"] = rules.summary_lines()
    return pub


@app.get("/api/learning")
def learning_stats():
    return get_learner().stats()


@app.post("/api/learning")
def learning_toggle(body: dict = Body(...)):
    if "enabled" in body:
        get_learner().set_enabled(bool(body["enabled"]))
    return get_learner().stats()


@app.post("/api/learning/reset")
def learning_reset():
    get_learner().reset()
    return get_learner().stats()


@app.get("/api/learning/export")
def learning_export():
    return PlainTextResponse(get_learner().export(), media_type="application/x-ndjson",
                             headers={"Content-Disposition": 'attachment; filename="reelsmith-taste-dataset.jsonl"'})
