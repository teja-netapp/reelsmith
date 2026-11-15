"""Local AI editor on Ollama (default: qwen3-vl:8b-instruct). Free, private, works offline.

Small local models are weaker than Claude/Gemini at reading dense contact sheets and at long, exact
JSON, so the work is split the way a real edit suite does it:

  1. Logging - the vision model looks at every shot on its own and writes a log entry
     (what happens, subject position, shot type, mood, quality, hook value). Cached per shot.
  2. Editing - the model reads the shot log + brief + music and writes a *compact* plan
     (shot id, seconds on screen, which part, speed, camera, transition, sound, text).
     Ollama's JSON-schema mode constrains every field to valid values (even the shot ids).
  3. Code turns the compact plan into exact in/out points (word-safe for speech) and the
     craft pass (craft.py) enforces the playbook rules a small model tends to miss.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json

from PIL import Image

from ..analyze import select_shots
from ..audio import CACHE
from .. import config as C
from ..effects import CAPTION_STYLES, LOOKS, TEXT_STYLES
from ..playbook import (FORMAT_NOTE, PRINCIPLES, ROLE, clip_lines, music_lines, settings_lines,
                        transcript_lines, words_in)
from ..reframe import MOTIONS
from ..transitions import TRANSITIONS
from .base import Provider, ProviderError, http_json, parse_json_loose

SHOT_TYPES = ["wide", "medium", "close_up", "detail", "aerial", "pov", "text_or_screen"]
MOODS = ["joyful", "calm", "epic", "romantic", "moody", "energetic", "funny", "nostalgic", "neutral"]

SHOT_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "subject": {"type": "string"},
        "subject_x": {"type": "number"},
        "shot_type": {"type": "string", "enum": SHOT_TYPES},
        "mood": {"type": "string", "enum": MOODS},
        "quality": {"type": "integer"},
        "hook": {"type": "integer"},
    },
    "required": ["description", "subject", "subject_x", "shot_type", "mood", "quality", "hook"],
}

LOG_PROMPT = """You are logging footage for a video editor. Look at this frame from a video clip and fill in:
- description: what is happening, max 15 words
- subject: the main subject, 2-4 words
- subject_x: horizontal position of the main subject: 0 = left edge, 0.5 = centre, 1 = right edge
- shot_type: one of wide, medium, close_up, detail, aerial, pov, text_or_screen
- mood: the feeling of the image
- quality: 1-10 technical quality (sharp, well exposed, well composed)
- hook: 1-10 how eye-catching this would be as the very first frame of an Instagram Reel
Answer in JSON."""

COMPACT_GUIDE = """Plan fields: each segment uses one shot from the shot log. seconds = how long it plays on screen. \
part = which part of the shot to use (start, middle or end). speed = playback speed. camera = virtual camera move. \
transition = how the segment enters (ignored for the first one). sound = true keeps the shot's own audio (speech, \
laughter, ambience). text = on-screen text, or "" for none. Global fields: concept (your plan in 1-2 sentences, \
write it first), title, pace, look, grain, letterbox, caption_style."""

PACE_LEN = {"fast": (0.6, 1.4), "medium": (1.2, 2.6), "slow": (2.2, 4.2)}


def _host() -> str:
    h = C.OLLAMA_HOST
    if not h.startswith("http"):
        h = "http://" + h
    return h.rstrip("/").replace("0.0.0.0", "127.0.0.1")


def _img_b64(path: str, max_side: int = 448) -> str:
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _guess_pace(description: str) -> str:
    d = (description or "").lower()
    if any(w in d for w in ("fast", "energetic", "hype", "punchy", "action", "party", "dance", "upbeat", "quick")):
        return "fast"
    if any(w in d for w in ("calm", "slow", "chill", "dreamy", "emotional", "peaceful", "wedding", "romantic")):
        return "slow"
    return "medium"


def compact_schema(shot_ids: list[str]) -> dict:
    seg = {
        "type": "object",
        "properties": {
            "shot": {"type": "string", "enum": shot_ids},
            "seconds": {"type": "number"},
            "part": {"type": "string", "enum": ["start", "middle", "end"]},
            "speed": {"type": "number", "enum": [0.5, 1, 1.5, 2]},
            "camera": {"type": "string", "enum": list(MOTIONS)},
            "transition": {"type": "string", "enum": list(TRANSITIONS)},
            "sound": {"type": "boolean"},
            "text": {"type": "string"},
            "text_style": {"type": "string", "enum": list(TEXT_STYLES)},
        },
        "required": ["shot", "seconds", "part", "speed", "camera", "transition", "sound", "text", "text_style"],
    }
    return {
        "type": "object",
        "properties": {
            "concept": {"type": "string"},
            "title": {"type": "string"},
            "pace": {"type": "string", "enum": list(PACE_LEN)},
            "look": {"type": "string", "enum": list(LOOKS)},
            "grain": {"type": "string", "enum": ["none", "light", "medium", "heavy"]},
            "letterbox": {"type": "boolean"},
            "caption_style": {"type": "string", "enum": list(CAPTION_STYLES)},
            "segments": {"type": "array", "items": seg, "minItems": 1, "maxItems": 60},
        },
        "required": ["concept", "title", "pace", "look", "grain", "letterbox", "caption_style", "segments"],
    }


class OllamaProvider(Provider):
    name = "ollama"
    polish_level = "full"

    @property
    def label(self):
        return f"Local AI ({C.OLLAMA_PLAN_MODEL})"

    # -- plumbing ---------------------------------------------------------------------------
    def installed_models(self) -> list[str]:
        res = http_json(f"{_host()}/api/tags", timeout=2.5, retries=0)
        return [m.get("name", "") for m in res.get("models", [])]

    @staticmethod
    def _has(models, name):
        return name in models or f"{name}:latest" in models or any(m.split(":")[0] == name for m in models)

    def status(self):
        try:
            models = self.installed_models()
        except Exception:
            return {"available": False, "detail": "Ollama is not running"}
        missing = [m for m in {C.OLLAMA_MODEL, C.OLLAMA_PLAN_MODEL} if not self._has(models, m)]
        if missing:
            return {"available": False, "detail": f"run: ollama pull {missing[0]}"}
        return {"available": True, "detail": C.OLLAMA_PLAN_MODEL}

    def chat(self, model: str, system: str, user: str, schema: dict, images=None, temperature=0.4,
             num_ctx=None, timeout=900) -> dict:
        msg = {"role": "user", "content": user}
        if images:
            msg["images"] = images
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, msg],
            "format": schema,
            "stream": False,
            "keep_alive": "10m",
            "options": {"temperature": temperature, "num_ctx": num_ctx or C.OLLAMA_NUM_CTX},
        }
        res = http_json(f"{_host()}/api/chat", body, timeout=timeout, retries=1)
        if res.get("error"):
            raise ProviderError(res["error"])
        self._last_raw = res
        return parse_json_loose(res.get("message", {}).get("content", ""))

    # -- 1. logging ---------------------------------------------------------------------------
    def log_shots(self, shots: list[dict], log, progress) -> dict:
        n = len(shots)
        stats = {"logged": 0, "cached": 0, "failed": 0}
        for i, s in enumerate(shots):
            if s.get("vlm"):
                stats["cached"] += 1
                continue
            try:
                raw = open(s["thumb"], "rb").read()
            except Exception:
                stats["failed"] += 1
                continue
            key = hashlib.sha1(raw + C.OLLAMA_MODEL.encode()).hexdigest()[:20]
            cpath = CACHE / f"vlm_{key}.json"
            if cpath.exists():
                s["vlm"] = json.loads(cpath.read_text())
                stats["cached"] += 1
                continue
            progress(0.05 + 0.45 * i / max(1, n), f"Local AI is logging shot {i + 1}/{n} ({s['id']})")
            try:
                d = self.chat(C.OLLAMA_MODEL, "You are a precise footage logger. Reply only with JSON.", LOG_PROMPT,
                              SHOT_SCHEMA, images=[_img_b64(s["thumb"])], temperature=0.1, num_ctx=4096,
                              timeout=300)
            except Exception as e:
                log(f"shot logging failed for {s['id']}: {e}")
                stats["failed"] += 1
                continue
            d["subject_x"] = float(min(1.0, max(0.0, float(d.get("subject_x", 0.5) or 0.5))))
            d["quality"] = int(min(10, max(1, int(d.get("quality", 5) or 5))))
            d["hook"] = int(min(10, max(1, int(d.get("hook", 5) or 5))))
            d["description"] = str(d.get("description", ""))[:160]
            d["model"] = C.OLLAMA_MODEL
            s["vlm"] = d
            cpath.write_text(json.dumps(d))
            stats["logged"] += 1
        return stats

    def _meta(self, stats: dict, n_shots: int) -> dict:
        r = getattr(self, "_last_raw", {}) or {}
        return {
            "model": C.OLLAMA_PLAN_MODEL,
            "shot_log_model": C.OLLAMA_MODEL,
            "shots_shown": n_shots,
            "shots_logged": stats.get("logged", 0),
            "shots_cached": stats.get("cached", 0),
            "shots_failed": stats.get("failed", 0),
            "usage": {"input_tokens": r.get("prompt_eval_count"), "output_tokens": r.get("eval_count")},
            "runs_on": "this Mac (Ollama)",
        }

    # -- 2. editing ---------------------------------------------------------------------------
    def _shot_log_lines(self, shots, analysis) -> list[str]:
        tr_all = analysis.get("transcripts", {})
        out = ["# Shot log (id | length | quality | hook | type | mood | what happens | speech)"]
        for s in shots:
            v = s.get("vlm") or {}
            q = (s["score"] * 10 + v.get("quality", s["score"] * 10)) / 2
            ws = words_in(tr_all.get(s["clip_id"]), s["start"], s["end"])
            sp = (' | says: "' + " ".join(w["word"] for w in ws[:16]) + '"') if ws else ""
            mo = " | high motion" if s.get("motion", 0) > 0.6 else ""
            out.append(f"{s['id']} | {s['duration']:.1f}s | {q:.0f}/10 | {v.get('hook', '?')}/10 | "
                       f"{v.get('shot_type', '?')} | {v.get('mood', '?')} | {v.get('description', '(not logged)')}"
                       f"{mo}{sp}")
        return out

    def _prompt(self, shots, analysis, music, options, description, pace) -> str:
        target = float(options.get("target_duration", 30))
        lo, hi = PACE_LEN[pace]
        n_lo, n_hi = max(2, int(target / hi)), max(3, int(round(target / lo)))
        L = ["# Creative brief", description.strip() or "(no brief: make a polished, engaging reel)", "",
             "# Settings", *settings_lines(options, music), "", "# Footage", *clip_lines(analysis), "",
             *self._shot_log_lines(shots, analysis)]
        tl = transcript_lines(analysis, 2500)
        if tl:
            L += [""] + tl
        L += [""] + music_lines(music, energy=False)
        beat = f" Make seconds multiples of the beat ({music['beat_interval']:.2f}s)." if music and options.get(
            "beat_sync") else ""
        L += ["", "# Your task",
              f"Write the edit. The seconds of all segments must add up to about {target:.0f}. "
              f"Use roughly {n_lo}-{n_hi} segments ({lo:.1f}-{hi:.1f} s each for a {pace} pace).{beat} "
              "Follow the brief's instructions first (e.g. which shot to open or end on). Otherwise open with the "
              "highest-hook shot. Seconds can't exceed a shot's length unless speed is below 1. "
              "Never use the same part of the same shot twice."]
        return "\n".join(L)

    def _system(self) -> str:
        return "\n\n".join([ROLE, FORMAT_NOTE, COMPACT_GUIDE, PRINCIPLES, "Reply only with JSON."])

    def _local_shots(self, analysis, extra_ids=()):
        order = [c["id"] for c in analysis["clips"]]
        shots = select_shots(analysis["shots"], C.LOCAL_MAX_SHOTS, order)
        have = {s["id"] for s in shots}
        by_id = _shot_index(analysis)
        for sid in extra_ids:
            if sid in by_id and sid not in have:
                shots.append(by_id[sid])
                have.add(sid)
        return shots

    def plan(self, analysis, music, options, description, log, progress):
        shots = self._local_shots(analysis)
        stats = self.log_shots(shots, log, progress)
        pace = _guess_pace(description)
        progress(0.55, f"Local AI ({C.OLLAMA_PLAN_MODEL}) is writing the edit - about a minute on a laptop")
        cp = self.chat(C.OLLAMA_PLAN_MODEL, self._system(), self._prompt(shots, analysis, music, options, description,
                                                                          pace),
                       compact_schema([s["id"] for s in shots]), temperature=0.5)
        self.last_meta = self._meta(stats, len(shots))
        return compact_to_raw(cp, analysis, music)

    def refine(self, prev_plan, feedback, analysis, music, options, description, log, progress):
        prev_ids = [s.get("shot_id") for s in prev_plan.get("segments", [])]
        shots = self._local_shots(analysis, prev_ids)
        stats = self.log_shots(shots, log, progress)
        pace = _guess_pace(description + " " + feedback)
        current = full_to_compact(prev_plan, analysis)
        text = (self._prompt(shots, analysis, music, options, description, pace) +
                "\n\n# Current edit (already rendered)\n" + json.dumps(current, indent=1) +
                "\n\n# Revision request from the user\n" + feedback.strip() +
                "\n\nApply the revision: keep what works, change what was asked, and return the COMPLETE new edit.")
        progress(0.55, f"Local AI ({C.OLLAMA_PLAN_MODEL}) is revising the edit")
        cp = self.chat(C.OLLAMA_PLAN_MODEL, self._system(), text, compact_schema([s["id"] for s in shots]),
                       temperature=0.5)
        self.last_meta = self._meta(stats, len(shots))
        return compact_to_raw(cp, analysis, music)


# ----------------------------------------------------------------------------------------------
# compact <-> full plan conversion
# ----------------------------------------------------------------------------------------------
_GRAIN = {"none": 0.0, "light": 0.15, "medium": 0.3, "heavy": 0.5}


def _shot_index(analysis: dict) -> dict:
    idx = {s["id"]: s for s in analysis.get("all_shots", [])}
    idx.update({s["id"]: s for s in analysis["shots"]})  # planner shots carry the 'vlm' log
    return idx


def compact_to_raw(cp: dict, analysis: dict, music: dict | None) -> dict:
    shots = _shot_index(analysis)
    segs = []
    for i, s in enumerate(cp.get("segments") or []):
        sh = shots.get(str(s.get("shot", "")))
        if not sh:
            continue
        try:
            speed = float(s.get("speed", 1) or 1)
        except Exception:
            speed = 1.0
        speed = min(2.0, max(0.5, speed))
        try:
            out = float(s.get("seconds", 2) or 2)
        except Exception:
            out = 2.0
        out = min(12.0, max(0.4, out))
        src = min(out * speed, sh["duration"])
        part = s.get("part", "middle")
        if part == "start":
            st = sh["start"]
        elif part == "end":
            st = sh["end"] - src
        else:
            st = sh["start"] + (sh["duration"] - src) / 2
        text = str(s.get("text", "") or "").strip()
        style = s.get("text_style") if s.get("text_style") in TEXT_STYLES else "bold"
        pos = {"cinematic_title": "center", "kicker": "top", "subtitle": "bottom"}.get(style,
                                                                                       "center" if i == 0 else "top")
        fx = (sh.get("vlm") or {}).get("subject_x", sh.get("focus_x"))
        segs.append({
            "shot_id": sh["id"], "start": round(st, 3), "end": round(st + src, 3), "speed": speed,
            "motion": s.get("camera", "none"), "transition_in": s.get("transition", "cut"),
            "keep_audio": bool(s.get("sound", False)), "focus_x": fx,
            "text": {"content": text[:60], "style": style, "position": pos} if text else None,
        })
    return {
        "reasoning": str(cp.get("concept", ""))[:800],
        "title": str(cp.get("title", "My reel"))[:80],
        "pace": cp.get("pace") if cp.get("pace") in PACE_LEN else "medium",
        "look": cp.get("look", "clean"),
        "look_intensity": 0.9,
        "grain": _GRAIN.get(cp.get("grain"), 0.15),
        "vignette": 0.3,
        "letterbox": bool(cp.get("letterbox", False)),
        "fade_in": True, "fade_out": True,
        "music_start": music["suggested_start"] if music else 0.0,
        "music_volume": 0.9,
        "caption_style": cp.get("caption_style", "pop_yellow"),
        "segments": segs,
    }


def full_to_compact(plan: dict, analysis: dict) -> dict:
    shots = _shot_index(analysis)
    segs = []
    for s in plan.get("segments", []):
        sh = shots.get(s.get("shot_id"))
        part = "middle"
        if sh:
            if abs(s["start"] - sh["start"]) < 0.15:
                part = "start"
            elif abs(s["end"] - sh["end"]) < 0.15:
                part = "end"
        t = s.get("text") or {}
        sp = s.get("speed", 1)
        segs.append({"shot": s.get("shot_id"), "seconds": round((s["end"] - s["start"]) / sp, 2), "part": part,
                     "speed": sp, "camera": s.get("motion"), "transition": s.get("transition_in"),
                     "sound": s.get("keep_audio", False), "text": t.get("content", ""),
                     "text_style": t.get("style", "bold")})
    g = plan.get("grain", 0)
    grain = "none" if g < 0.05 else "light" if g < 0.22 else "medium" if g < 0.4 else "heavy"
    return {"concept": plan.get("reasoning", ""), "title": plan.get("title", ""), "pace": plan.get("pace", "medium"),
            "look": plan.get("look"), "grain": grain, "letterbox": plan.get("letterbox", False),
            "caption_style": plan.get("caption_style", "pop_yellow"), "segments": segs}
