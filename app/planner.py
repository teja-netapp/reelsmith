"""The 'director': turns a brief + analysed footage into an edit decision list (EDL).

Any AI editor can write the plan (Claude, Gemini, or a free local model on Ollama - see providers/).
Every plan then goes through the same validation and the craft pass (craft.py). If no AI editor is
available, or one fails, the built-in rule-based planner still delivers a reel.
"""
from __future__ import annotations

import random
import re

from .craft import polish
from .effects import CAPTION_STYLES, LOOKS, TEXT_POSITIONS, TEXT_STYLES
from .playbook import words_in as _words_in
from .providers import chain
from .reframe import MOTIONS
from .transitions import TRANSITIONS

HEURISTIC_LABEL = "Built-in rules (no AI)"


def _noop(*_a, **_k):
    return None


def _finalize(raw, analysis, music, options, description, level, label):
    plan = sanitize_plan(raw, analysis, music, options)
    plan = polish(plan, analysis, options, level, description)
    pace, notes = plan.get("pace"), plan.get("craft_notes", [])
    warns = plan.get("warnings", [])
    final = sanitize_plan(plan, analysis, music, options)
    final["pace"] = pace
    final["planner"] = label
    if notes:
        final["craft_notes"] = notes
    w = sorted(set(warns + final.get("warnings", [])))
    if w:
        final["warnings"] = w
    return final


def plan_edit(analysis, music, options, description, log=print, progress=None) -> dict:
    progress = progress or _noop
    for prov in chain(options.get("planner")):
        try:
            raw = prov.plan(analysis, music, options, description, log, progress)
            return _finalize(raw, analysis, music, options, description, prov.polish_level, prov.label)
        except Exception as e:  # network / auth / model problems -> try the next editor
            log(f"{prov.label} could not plan the edit ({type(e).__name__}: {str(e)[:300]}); trying the next editor.")
    raw = heuristic_raw(analysis, music, options, description)
    return _finalize(raw, analysis, music, options, description, "full", HEURISTIC_LABEL)


def refine_edit(prev_plan, feedback, analysis, music, options, description, log=print, progress=None) -> dict:
    progress = progress or _noop
    for prov in chain(options.get("planner")):
        try:
            raw = prov.refine(prev_plan, feedback, analysis, music, options, description, log, progress)
            return _finalize(raw, analysis, music, options, description, prov.polish_level, prov.label + " - revised")
        except Exception as e:
            log(f"{prov.label} could not revise the edit ({type(e).__name__}: {str(e)[:300]}); trying the next editor.")
    raw = heuristic_raw(analysis, music, options, description + " " + feedback, seed=len(feedback) * 7 + 1)
    return _finalize(raw, analysis, music, options, description + " " + feedback, "full", HEURISTIC_LABEL + " - revised")


# ----------------------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------------------
def _f(v, default, lo=None, hi=None):
    try:
        x = float(v)
    except Exception:
        x = float(default)
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return x


def sanitize_plan(raw: dict, analysis: dict, music: dict | None, options: dict) -> dict:
    clips = {c["id"]: c for c in analysis["clips"]}
    shots = {s["id"]: s for s in analysis.get("all_shots", [])}
    shots.update({s["id"]: s for s in analysis["shots"]})
    warnings = []
    plan = {
        "reasoning": str(raw.get("reasoning", ""))[:1500],
        "title": str(raw.get("title", "Untitled reel"))[:80],
        "look": raw.get("look") if raw.get("look") in LOOKS else "clean",
        "look_intensity": _f(raw.get("look_intensity"), 0.9, 0, 1.2),
        "grain": _f(raw.get("grain"), 0.15, 0, 1),
        "vignette": _f(raw.get("vignette"), 0.3, 0, 1),
        "letterbox": bool(raw.get("letterbox", False)),
        "fade_in": bool(raw.get("fade_in", True)),
        "fade_out": bool(raw.get("fade_out", True)),
        "music_start": _f(raw.get("music_start"), music["suggested_start"] if music else 0, 0),
        "music_volume": _f(raw.get("music_volume"), 0.9, 0, 1),
        "original_audio_volume": _f(raw.get("original_audio_volume"), 1.0, 0, 1.5),
        "caption_style": raw.get("caption_style") if raw.get("caption_style") in CAPTION_STYLES else "pop_yellow",
        "segments": [],
    }
    if raw.get("pace") in ("fast", "medium", "slow"):
        plan["pace"] = raw["pace"]
    for s in raw.get("segments", []) or []:
        sid = str(s.get("shot_id", ""))
        shot = shots.get(sid)
        cid = s.get("clip_id") or (shot["clip_id"] if shot else None)
        if cid is None:
            m = re.match(r"^(c\d+)", sid)
            cid = m.group(1) if m else None
        if cid not in clips:
            warnings.append(f"dropped segment with unknown shot '{sid}'")
            continue
        clip = clips[cid]
        cd = clip["duration"]
        start = _f(s.get("start"), shot["start"] if shot else 0.0, 0, max(0, cd - 0.2))
        end = _f(s.get("end"), shot["end"] if shot else start + 2, start + 0.2, cd)
        if end - start < 0.2:
            continue
        tr = s.get("transition_in") if s.get("transition_in") in TRANSITIONS else "cut"
        td = _f(s.get("transition_duration") or TRANSITIONS[tr][0], TRANSITIONS[tr][0], 0.0, 1.2)
        fx = s.get("focus_x")
        if fx is None and shot:
            fx = (shot.get("vlm") or {}).get("subject_x", shot.get("focus_x"))
        text = None
        t = s.get("text")
        if isinstance(t, dict) and str(t.get("content", "")).strip():
            text = {"content": str(t["content"]).strip()[:80],
                    "style": t.get("style") if t.get("style") in TEXT_STYLES else "bold",
                    "position": t.get("position") if t.get("position") in TEXT_POSITIONS else "center"}
        plan["segments"].append({
            "shot_id": sid if shot else f"{cid}",
            "clip_id": cid,
            "start": round(start, 3),
            "end": round(end, 3),
            "speed": round(_f(s.get("speed"), 1.0, 0.25, 4.0), 3),
            "motion": s.get("motion") if s.get("motion") in MOTIONS else "none",
            "transition_in": tr,
            "transition_duration": round(td, 3),
            "focus_x": round(_f(fx, 0.5, 0, 1), 3),
            "keep_audio": bool(s.get("keep_audio", False)),
            "text": text,
        })
    if not plan["segments"]:
        raise ValueError("Plan contains no usable segments")
    plan["segments"][0]["transition_in"] = "cut"
    if not music:
        for s in plan["segments"]:
            s["keep_audio"] = True

    # duration guard: drop/trim from the end if far too long
    target = float(options.get("target_duration", 30))

    def total():
        segs = plan["segments"]
        t = sum((s["end"] - s["start"]) / s["speed"] for s in segs)
        t -= sum(s["transition_duration"] for s in segs[1:] if s["transition_in"] != "cut")
        return t

    while len(plan["segments"]) > 1 and total() > target * 1.12:
        last = plan["segments"][-1]
        over = total() - target
        ld = (last["end"] - last["start"]) / last["speed"]
        if ld - over >= 1.0:
            last["end"] = round(last["start"] + (ld - over) * last["speed"], 3)
            break
        plan["segments"].pop()
        warnings.append("trimmed trailing segments to fit the target duration")
    if music:
        plan["music_start"] = round(min(plan["music_start"], max(0.0, music["duration"] - total() - 0.3)), 3)
    plan["estimated_duration"] = round(total(), 2)
    if warnings:
        plan["warnings"] = sorted(set(warnings))
    return plan


# ----------------------------------------------------------------------------------------------
# Heuristic planner (offline fallback)
# ----------------------------------------------------------------------------------------------
_LOOK_WORDS = [
    (("black and white", "b&w", "noir", "monochrome"), "noir"),
    (("vintage", "retro", "80s", "70s", "90s", "nostalg", "vhs"), "vintage"),
    (("teal", "blockbuster", "hollywood", "epic", "action"), "teal_orange"),
    (("golden", "sunset", "sunrise", "warm"), "golden_hour"),
    (("film", "analog", "kodak", "35mm"), "warm_film"),
    (("moody", "dark", "dramatic", "rain", "gloomy"), "moody"),
    (("neon", "cyber", "night", "city lights"), "neon_night"),
    (("cool", "cold", "winter", "nordic", "snow", "blue"), "cool_blue"),
    (("vibrant", "colorful", "colourful", "summer", "bright", "fun"), "vibrant"),
]
_FAST = ("fast", "energetic", "hype", "punchy", "action", "sport", "party", "dance", "upbeat", "quick", "intense")
_SLOW = ("calm", "slow", "chill", "dreamy", "emotional", "peaceful", "wedding", "relax", "soft", "romantic")


def heuristic_plan(analysis, music, options, description, seed=0) -> dict:
    """Validated + polished offline plan (kept for scripts/tests)."""
    return _finalize(heuristic_raw(analysis, music, options, description, seed), analysis, music, options,
                     description, "full", HEURISTIC_LABEL)


def heuristic_raw(analysis, music, options, description, seed=0) -> dict:
    rnd = random.Random(seed or 1234)
    desc = (description or "").lower()
    target = float(options.get("target_duration", 30))
    look = "teal_orange" if "cinematic" in desc else "clean"
    for words, lk in _LOOK_WORDS:
        if any(w in desc for w in words):
            look = lk
            break
    pace = "fast" if any(w in desc for w in _FAST) else ("slow" if any(w in desc for w in _SLOW) else "mid")
    seg_len = {"fast": 1.1, "slow": 2.8, "mid": 1.9}[pace]
    if music:
        bi = music["beat_interval"]
        seg_len = max(1, round(seg_len / bi)) * bi
    order = {c["id"]: i for i, c in enumerate(analysis["clips"])}
    # candidate windows: every shot split into non-overlapping windows of ~seg_len
    cands = []
    for s in analysis["shots"]:
        k = max(1, int(s["duration"] // max(0.4, seg_len)))
        wl = s["duration"] / k
        for j in range(k):
            cands.append(dict(s, start=s["start"] + j * wl, end=s["start"] + (j + 1) * wl, duration=wl,
                              score=s["score"] - 0.04 * abs(j - (k - 1) / 2), _win=j))
    if options.get("auto_pick", True):
        ranked = sorted(cands, key=lambda s: s["score"], reverse=True)
        chosen, acc = [], 0.0
        for s in ranked:
            if acc >= target * 1.03:
                break
            if s["score"] < 0.2 and acc > target * 0.5:
                continue
            chosen.append(s)
            acc += min(seg_len, s["duration"])
        hook = chosen[0] if chosen else None
        chosen.sort(key=lambda s: (order[s["clip_id"]], s["start"]))
        if hook in chosen:
            chosen.remove(hook)
            chosen.insert(0, hook)
    else:
        chosen = sorted(cands, key=lambda s: (order[s["clip_id"]], s["start"]))
        need = max(1, int(round(target / seg_len)))
        if len(chosen) > need:
            step = len(chosen) / need
            chosen = [chosen[int(i * step)] for i in range(need)]
    # if footage is short, stretch segments (up to their window) to approach the target
    acc = sum(min(seg_len, s["duration"]) for s in chosen)
    stretch = min(2.5, target / acc) if acc > 0 and acc < target * 0.95 else 1.0
    palettes = {
        "fast": ["cut", "cut", "whip_left", "cut", "flash", "cut", "zoom_punch", "cut", "glitch", "whip_right"],
        "slow": ["crossfade", "cut", "light_leak", "cut", "blur_dissolve", "cut", "dip_to_black"],
        "mid": ["cut", "crossfade", "cut", "whip_right", "cut", "light_leak", "cut", "zoom_punch"],
    }[pace]
    motions = ["push_in", "drift_left", "pull_out", "drift_right", "ken_burns", "handheld"]
    if pace == "fast":
        motions = ["punch_in", "push_in", "handheld", "drift_right", "punch_in", "pull_out"]
    tr_all = analysis.get("transcripts", {})
    talky = any(w in desc for w in ("vlog", "talk", "speech", "interview", "podcast", "caption", "say", "voice"))
    segs = []
    slow_mo = "slow mo" in desc or "slow-mo" in desc or "slowmo" in desc or "slow motion" in desc
    for i, s in enumerate(chosen):
        d = min(s["duration"], seg_len * stretch * rnd.uniform(0.85, 1.2))
        speed = 1.0
        if slow_mo and s["motion"] > 0.45 and i % 2 == 1:
            speed = 0.5
        src = min(s["duration"], d * speed)
        st = s["start"] + (s["duration"] - src) / 2
        tr = "cut" if i == 0 else palettes[i % len(palettes)]
        has_speech = bool(_words_in(tr_all.get(s["clip_id"]), st, st + src))
        segs.append({
            "shot_id": s["id"], "start": round(st, 3), "end": round(st + src, 3), "speed": speed,
            "motion": motions[i % len(motions)], "transition_in": tr,
            "transition_duration": TRANSITIONS[tr][0], "focus_x": s.get("focus_x") or 0.5,
            "keep_audio": (not music) or (talky and has_speech),
        })
    titles = re.findall(r'["“]([^"”]{1,40})["”]', description or "")
    if titles and segs:
        segs[0]["text"] = {"content": titles[0], "style": "cinematic_title", "position": "center"}
    raw = {
        "reasoning": f"Offline heuristic edit: {pace} pacing, '{look}' look, best-scoring shots first as the hook, "
                     f"then in story order.",
        "title": titles[0] if titles else "My reel",
        "look": look,
        "look_intensity": 0.9,
        "grain": 0.35 if look in ("warm_film", "vintage", "noir") else 0.12,
        "vignette": 0.35,
        "letterbox": "cinematic" in desc or "film" in desc,
        "fade_in": True, "fade_out": True,
        "music_start": music["suggested_start"] if music else 0,
        "music_volume": 0.9 if not talky else 0.5,
        "caption_style": "pop_yellow",
        "pace": {"mid": "medium"}.get(pace, pace),
        "segments": segs,
    }
    return raw
