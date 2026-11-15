"""End-to-end orchestration: analyse -> transcribe -> music -> plan (Claude) -> timeline -> render."""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid

from . import analyze as A
from .audio import analyze_music, transcribe, whisper_available
from . import config as C
from .config import FACE_MODEL, WORKSPACE
from .planner import plan_edit, refine_edit, sanitize_plan
from .reframe import FaceDetector
from .render import RenderSettings, render
from .learning import get_learner
from .report import build_report, mp4_metadata
from .timeline import build_timeline
from .video_io import probe

DEFAULT_OPTIONS = {
    "target_duration": 30,
    "auto_pick": True,
    "beat_sync": True,
    "captions": False,
    "smart_reframe": True,
    "quality": "final",  # preview | draft | final
    "planner": "auto",   # auto | gemini | ollama | claude | heuristic
    "auto_color": True,  # shot matching (per-shot white balance + exposure) before the look
}

_face_det = None


def face_detector() -> FaceDetector:
    global _face_det
    if _face_det is None:
        _face_det = FaceDetector(FACE_MODEL)
    return _face_det


class Job:
    def __init__(self, job_id: str | None = None):
        self.id = job_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.dir = WORKSPACE / "jobs" / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.status = "queued"   # queued | running | done | error
        self.stage = "Queued"
        self.progress = 0.0
        self.error = None
        self.logs: list[str] = []
        self.clips: list[str] = []
        self.music: str | None = None
        self.description = ""
        self.options = dict(DEFAULT_OPTIONS)
        self.analysis: dict | None = None
        self.music_info: dict | None = None
        self.plan: dict | None = None
        self.timeline_summary: dict | None = None
        self.version = 0
        self.output: str | None = None
        self.versions: list[dict] = []   # history: one entry per rendered version
        self.created = time.time()
        self.lock = threading.Lock()

    # -- persistence -----------------------------------------------------------------------
    def save(self):
        state = {k: getattr(self, k) for k in ("id", "status", "stage", "progress", "error", "clips", "music",
                                               "description", "options", "plan", "timeline_summary", "version",
                                               "output", "versions", "created")}
        state["logs"] = self.logs[-200:]
        (self.dir / "job.json").write_text(json.dumps(state, indent=1, default=str))
        if self.analysis is not None:
            (self.dir / "analysis.json").write_text(json.dumps(self.analysis, default=str))
        if self.music_info is not None:
            (self.dir / "music.json").write_text(json.dumps(self.music_info))

    @classmethod
    def load(cls, job_id: str) -> "Job":
        j = cls(job_id)
        d = json.loads((j.dir / "job.json").read_text())
        for k, v in d.items():
            setattr(j, k, v)
        if (j.dir / "analysis.json").exists():
            j.analysis = json.loads((j.dir / "analysis.json").read_text())
        if (j.dir / "music.json").exists():
            j.music_info = json.loads((j.dir / "music.json").read_text())
        if j.status in ("running", "queued"):
            j.status, j.stage = "error", "Interrupted (server restarted)"
        return j

    def log(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.logs.append(line)
        print(f"[{self.id}] {msg}", flush=True)

    def set(self, progress: float | None = None, stage: str | None = None):
        if progress is not None:
            self.progress = round(float(progress), 4)
        if stage is not None and stage != self.stage:
            self.stage = stage

    def report_path(self, version: int | None = None):
        return self.dir / f"report_v{version or self.version}.json"

    def output_for(self, version: int | None = None) -> str | None:
        if version:
            for v in self.versions:
                if v["version"] == version and os.path.exists(v["output"]):
                    return v["output"]
            return None
        return self.output

    def to_public(self) -> dict:
        report = None
        rp = self.report_path()
        if self.version and rp.exists():
            try:
                report = json.loads(rp.read_text())
            except Exception:
                report = None
        return {
            "id": self.id, "status": self.status, "stage": self.stage, "progress": self.progress,
            "error": self.error, "logs": self.logs[-30:], "description": self.description,
            "options": self.options, "plan": self.plan, "timeline": self.timeline_summary,
            "version": self.version,
            "versions": [{k: v.get(k) for k in ("version", "created", "trigger", "feedback", "editor", "model",
                                                "duration", "look")} for v in self.versions],
            "report": report,
            "output_url": f"/api/jobs/{self.id}/output?v={self.version}" if self.output else None,
            "clips": [os.path.basename(c) for c in self.clips],
            "music": os.path.basename(self.music) if self.music else None,
            "shots": [{"id": s["id"], "clip_id": s["clip_id"], "start": s["start"], "end": s["end"],
                       "score": s.get("score")} for s in (self.analysis or {}).get("shots", [])],
        }


# ----------------------------------------------------------------------------------------------
def run_analysis(job: Job):
    opts = job.options
    det = face_detector()
    if not det.available:
        job.log("Face model not found - smart reframe will use saliency/motion tracking only.")
    thumbs = job.dir / "thumbs"
    thumbs.mkdir(exist_ok=True)
    clips, shots = [], []
    n = len(job.clips)
    for i, path in enumerate(job.clips):
        info = probe(path)
        if not info.has_video:
            job.log(f"Skipping {os.path.basename(path)} (no video stream)")
            continue
        cid = f"c{i}"
        job.set(stage=f"Analysing clip {i + 1}/{n}: {os.path.basename(path)}")
        clip, s = A.analyze_clip(info, cid, det, thumbs,
                                 progress=lambda p, i=i: job.set(0.02 + 0.28 * (i + p) / n))
        clips.append(clip)
        shots += s
        job.log(f"{clip['name']}: {clip['duration']:.1f}s, {clip['num_scenes']} scenes, {len(s)} shots")
    if not clips:
        raise RuntimeError("No usable video clips were uploaded")
    A.score_shots(shots)
    selected = A.select_shots(shots, C.MAX_SHOTS_FOR_PLANNER, [c["id"] for c in clips])
    sheets = A.make_contact_sheets(selected, job.dir / "sheets")
    analysis = {"clips": clips, "all_shots": shots, "shots": selected, "sheets": sheets, "transcripts": {}}

    # speech
    # transcripts also help Claude choose moments; skip on very long footage unless captions are wanted
    want_tr = opts.get("captions") or sum(c["duration"] for c in clips) <= 20 * 60
    if os.environ.get("REELSMITH_NO_TRANSCRIBE") == "1":
        want_tr = False
    if want_tr and whisper_available():
        for i, c in enumerate(clips):
            if not c["has_audio"]:
                continue
            job.set(0.30 + 0.12 * i / len(clips), f"Transcribing speech in {c['name']}")
            try:
                tr = transcribe(c["path"])
                if tr is not None:
                    analysis["transcripts"][c["id"]] = tr
                    if tr["words"]:
                        job.log(f"{c['name']}: {len(tr['words'])} words of speech ({tr.get('language')})")
            except Exception as e:
                job.log(f"Transcription unavailable ({type(e).__name__}: {str(e)[:160]}); "
                        f"continuing without speech/captions. Run ./setup.sh to download the Whisper model.")
                break
    elif opts.get("captions"):
        job.log("faster-whisper is not installed - captions are disabled.")
    job.analysis = analysis

    if job.music:
        job.set(0.43, "Analysing music beats")
        try:
            job.music_info = analyze_music(job.music, float(opts["target_duration"]))
            job.log(f"Music: {job.music_info['tempo']} BPM, {len(job.music_info['beats'])} beats, "
                    f"best section starts at {job.music_info['suggested_start']:.1f}s")
        except Exception as e:
            job.log(f"Music analysis failed ({e}); continuing without beat sync")
            job.music_info = {"path": job.music, "name": os.path.basename(job.music), "duration": 600,
                              "tempo": 120, "beat_interval": 0.5, "beats": [], "downbeats": [],
                              "energy_half_seconds": [], "suggested_start": 0.0}


def run_render(job: Job, trigger: dict | None = None, ai_plan: dict | None = None, timings: dict | None = None):
    opts = job.options
    trigger = trigger or {"type": "new"}
    timings = dict(timings or {})
    job.set(0.52, "Building timeline")
    tl = build_timeline(job.plan, job.analysis, job.music_info, opts)
    job.timeline_summary = {
        "duration": tl["duration"], "frames": tl["total_frames"], "beats_snapped": tl["beats_snapped"],
        "captions": len(tl["caption_words"]),
        "segments": [{"shot_id": s["shot_id"], "start_frame": s["start_frame"], "frames": s["n_frames"],
                      "transition_in": s["transition_in"], "motion": s["motion"], "speed": s["speed"]}
                     for s in tl["segments"]],
    }
    job.log(f"Timeline: {len(tl['segments'])} segments, {tl['duration']:.1f}s"
            + (f", {tl['beats_snapped']} cuts on the beat" if tl["beats_snapped"] else ""))
    job.version += 1
    out = job.dir / f"reel_v{job.version}.mp4"
    settings = RenderSettings.preset_named(opts.get("quality", "final"))

    def prog(p, msg):
        job.set(0.55 + 0.44 * p, msg)

    t0 = time.time()
    res = render(tl, job.plan, job.music_info, str(out), str(job.dir / "tmp"), settings, face_detector(),
                 opts.get("smart_reframe", True), prog, job.log, metadata=mp4_metadata(job.plan, tl, job.version),
                 auto_color=opts.get("auto_color", True))
    timings["render"] = round(time.time() - t0, 1)
    timings["total"] = round(sum(v for k, v in timings.items() if k != "total"), 1)
    job.output = str(out)
    v = job.version
    (job.dir / f"plan_v{v}.json").write_text(json.dumps(job.plan, indent=1))
    if ai_plan is not None:
        (job.dir / f"ai_plan_v{v}.json").write_text(json.dumps(ai_plan, indent=1, default=str))
    report = build_report(job, tl, job.plan, settings, res, trigger, timings, str(out))
    report["files"].update({"plan": f"plan_v{v}.json", "ai_plan": f"ai_plan_v{v}.json" if ai_plan is not None else None})
    job.report_path(v).write_text(json.dumps(report, indent=1, default=str))
    ed = job.plan.get("editor") or {}
    job.versions.append({
        "version": v, "created": time.time(), "trigger": trigger.get("type"), "feedback": trigger.get("feedback"),
        "editor": ed.get("label") or job.plan.get("planner"), "model": ed.get("model"),
        "duration": tl["duration"], "look": job.plan.get("look"), "output": str(out), "rating": None,
    })
    try:  # learning: remember this edit (features + summary) so ratings/downloads can teach the model
        get_learner().record_edit(job.id, v, job.plan, job.analysis, job.options, job.description, trigger.get("type"))
    except Exception as e:
        job.log(f"learning: could not record this edit ({e})")
    n_auto = len(job.plan.get("craft_changes", [])) + len(job.plan.get("warnings", []))
    job.log(f"v{v} made by {ed.get('label')} ({ed.get('model') or 'no model'}); {n_auto} automatic fix(es) - "
            f"see 'How this reel was made'")


def _run(job: Job, mode: str, payload=None):
    try:
        job.status, job.error = "running", None
        def plan_prog(p, msg):
            job.set(0.46 + 0.07 * min(1.0, p), msg)

        timings = {}
        trigger = {"type": mode}
        if mode == "new":
            job.set(0.01, "Analysing footage")
            t0 = time.time()
            run_analysis(job)
            timings["analysis"] = round(time.time() - t0, 1)
            job.save()
            job.set(0.46, "Planning the edit")
            t0 = time.time()
            job.plan = plan_edit(job.analysis, job.music_info, job.options, job.description, job.log, plan_prog)
            timings["planning"] = round(time.time() - t0, 1)
        elif mode == "refine":
            job.set(0.46, "Revising the edit")
            trigger["feedback"] = payload
            try:
                deltas = get_learner().revision(job.id, job.version, payload, job.description)
                if deltas:
                    job.log("learning from your request: " + ", ".join(f"{k} {'+' if v > 0 else '-'}"
                                                                       for k, v in deltas.items()))
            except Exception as e:
                job.log(f"learning: could not parse the request ({e})")
            t0 = time.time()
            job.plan = refine_edit(job.plan, payload, job.analysis, job.music_info, job.options, job.description,
                                   job.log, plan_prog)
            timings["planning"] = round(time.time() - t0, 1)
        elif mode == "plan":
            job.set(0.46, "Validating edited plan")
            prev_editor = (job.plan or {}).get("editor", {}).get("label") or (job.plan or {}).get("planner")
            old_look = (job.plan or {}).get("look")
            source = payload.get("_source")
            clean = {k: v for k, v in payload.items() if not k.startswith("_")}
            job.plan = sanitize_plan(clean, job.analysis, job.music_info, job.options)
            job.plan["planner"] = "Manual plan edit"
            job.plan["editor"] = {"provider": "manual", "label": "Manual plan edit (JSON)", "model": None,
                                  "based_on": prev_editor}
            if source == "look_lab":
                trigger = {"type": "look", "feedback": f"Look Lab: {old_look} → {job.plan.get('look')}"}
                job.plan["editor"]["label"] = "Look Lab"
                try:
                    get_learner().look_choice(job.id, job.version, old_look, job.plan.get("look"), job.description)
                except Exception as e:
                    job.log(f"learning: {e}")
            job.plan["_ai_plan"] = clean
        ai_plan = job.plan.pop("_ai_plan", None)
        ed = job.plan.get("editor") or {}
        job.log(f"Plan by {ed.get('label') or job.plan.get('planner')}"
                + (f" [{ed['model']}]" if ed.get("model") else "")
                + f": '{job.plan.get('title')}', look={job.plan.get('look')}, {len(job.plan['segments'])} segments")
        for w in job.plan.get("warnings", []):
            job.log("validation: " + w)
        for c in job.plan.get("craft_changes", []):
            job.log(f"assistant editor [{c['rule']}]: {c['change']}")
        job.save()
        run_render(job, trigger, ai_plan, timings)
        job.status = "done"
        job.set(1.0, "Done")
    except Exception as e:
        job.status = "error"
        job.error = f"{type(e).__name__}: {e}"
        job.stage = "Failed"
        job.log(traceback.format_exc()[-1500:])
    finally:
        job.save()


class Runner:
    """Single worker thread: renders are CPU heavy, so jobs run one at a time."""

    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self._q: list = []
        self._cv = threading.Condition()
        threading.Thread(target=self._loop, daemon=True).start()
        jobs_dir = WORKSPACE / "jobs"
        if jobs_dir.exists():
            for d in sorted(jobs_dir.iterdir())[-30:]:
                if (d / "job.json").exists():
                    try:
                        self.jobs[d.name] = Job.load(d.name)
                    except Exception:
                        pass

    def submit(self, job: Job, mode: str, payload=None):
        self.jobs[job.id] = job
        job.status, job.stage = "queued", "Queued"
        job.save()
        with self._cv:
            self._q.append((job, mode, payload))
            self._cv.notify()

    def _loop(self):
        while True:
            with self._cv:
                while not self._q:
                    self._cv.wait()
                job, mode, payload = self._q.pop(0)
            _run(job, mode, payload)


def run_sync(clips, music, description, options) -> Job:
    """Blocking run (used by the CLI)."""
    job = Job()
    job.clips = [os.path.abspath(c) for c in clips]
    job.music = os.path.abspath(music) if music else None
    job.description = description
    job.options.update(options)
    _run(job, "new")
    return job
