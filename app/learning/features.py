"""Describe an edit as a feature vector the reward model can learn from.

Features are style choices you can feel when watching (pace, rhythm, transitions, camera moves,
look family, texture, text, hook quality...). Context (what kind of reel the brief asks for) is
crossed with the most taste-dependent features, so it can learn e.g. "warm looks for travel,
neon for city nights" instead of one global preference.
"""
from __future__ import annotations

import re

import numpy as np

from ..grading import FAMILIES, look_family

CATEGORIES = {
    "travel": ("travel", "trip", "vacation", "holiday", "city tour", "explore", "journey", "wander", "beach", "island"),
    "hype": ("hype", "party", "festival", "concert", "club", "energetic", "dance", "night out"),
    "vlog": ("vlog", "talking", "talk", "my day", "day in", "interview", "podcast", "speech", "tutorial"),
    "romantic": ("wedding", "romantic", "love", "couple", "anniversary", "proposal"),
    "sport": ("sport", "gym", "workout", "football", "soccer", "basketball", "run", "skate", "surf", "bike", "action"),
    "food": ("food", "recipe", "cooking", "restaurant", "cafe", "coffee", "dinner", "brunch"),
    "nature": ("nature", "hike", "mountain", "forest", "lake", "sunset", "sunrise", "ocean", "camping", "landscape"),
    "city": ("city", "street", "urban", "night", "neon", "downtown"),
    "family": ("family", "kids", "baby", "birthday", "friends", "dog", "cat", "pet"),
    "product": ("product", "brand", "launch", "unboxing", "promo", "ad ", "commercial"),
}
TRANSITION_GROUPS = {
    "crossfade": ("crossfade",), "dip": ("dip_to_black",), "flash": ("flash",),
    "whip": ("whip_left", "whip_right", "whip_up"), "zoom_punch": ("zoom_punch",), "slide_up": ("slide_up",),
    "light_leak": ("light_leak",), "glitch": ("glitch",), "blur_dissolve": ("blur_dissolve",), "iris": ("iris",),
}
MOTION_GROUPS = {
    "static": ("none",), "push": ("push_in", "pull_out", "punch_in"), "drift": ("drift_left", "drift_right", "rise",
                                                                              "ken_burns"),
    "handheld": ("handheld",),
}
# features whose preference most often depends on the kind of reel
CONTEXT_CROSSED = ["pace_short", "avg_len", "styl_frac", "grain", "letterbox", "slowmo_frac"] + \
                  [f"lf_{f}" for f in FAMILIES]

FEATURE_LABELS = {
    "pace_short": "very short shots (<1 s)", "avg_len": "longer shots", "len_cv": "varied shot lengths",
    "cuts_per_10s": "more cuts", "styl_frac": "stylised transitions", "slowmo_frac": "slow motion",
    "fast_frac": "speed-ups", "look_intensity": "strong grades", "grain": "film grain", "vignette": "vignette",
    "letterbox": "letterbox bars", "halation": "halation glow", "bloom": "bloom glow", "has_title": "an opening title",
    "text_frac": "on-screen text", "captions": "captions", "hook_q": "a strong opening shot",
    "mean_q": "sharp, well-exposed shots", "speech_frac": "original speech/sound", "fade_in": "fade-in",
    "fade_out": "fade-out",
}
for _g in TRANSITION_GROUPS:
    FEATURE_LABELS[f"t_{_g}"] = f"{_g.replace('_', ' ')} transitions"
for _g in MOTION_GROUPS:
    FEATURE_LABELS[f"m_{_g}"] = {"static": "locked-off shots", "push": "push-in/pull-out moves",
                                 "drift": "slow drifts/pans", "handheld": "handheld shake"}[_g]
for _f in FAMILIES:
    FEATURE_LABELS[f"lf_{_f}"] = f"{_f} looks"


def category(brief: str) -> str:
    d = " " + (brief or "").lower() + " "
    best, hits = "general", 0
    for cat, words in CATEGORIES.items():
        n = sum(1 for w in words if re.search(r"\b" + re.escape(w.strip()) + r"\b", d))
        if n > hits:
            best, hits = cat, n
    return best


def label(key: str) -> str:
    if "×" in key:
        cat, base = key.split("×", 1)
        return f"{FEATURE_LABELS.get(base, base)} for {cat} reels"
    return FEATURE_LABELS.get(key, key)


def plan_features(plan: dict, analysis: dict, options: dict, brief: str = "") -> dict[str, float]:
    segs = plan.get("segments") or []
    if not segs:
        return {}
    shots = {s["id"]: s for s in analysis.get("all_shots", [])}
    shots.update({s["id"]: s for s in analysis.get("shots", [])})
    durs = np.array([(s["end"] - s["start"]) / max(0.05, s["speed"]) for s in segs])
    total = float(durs.sum()) or 1.0
    n = len(segs)
    trs = [s.get("transition_in", "cut") for s in segs[1:]]
    f: dict[str, float] = {
        "pace_short": float((durs < 1.0).mean()),
        "avg_len": float(min(durs.mean(), 8.0) / 4.0),
        "len_cv": float(min(1.0, durs.std() / (durs.mean() + 1e-6))),
        "cuts_per_10s": float(min(2.0, n / total * 10 / 10)),
        "styl_frac": float(np.mean([t != "cut" for t in trs])) if trs else 0.0,
        "slowmo_frac": float(np.mean([s["speed"] < 0.95 for s in segs])),
        "fast_frac": float(np.mean([s["speed"] > 1.05 for s in segs])),
        "look_intensity": float(plan.get("look_intensity", 0.9)),
        "grain": float(plan.get("grain", 0.0)),
        "vignette": float(plan.get("vignette", 0.0)),
        "letterbox": 1.0 if plan.get("letterbox") else 0.0,
        "halation": float(plan.get("halation") or 0.0),
        "bloom": float(plan.get("bloom") or 0.0),
        "has_title": 1.0 if any((s.get("text") or {}).get("style") == "cinematic_title" for s in segs) else 0.0,
        "text_frac": float(np.mean([bool(s.get("text")) for s in segs])),
        "captions": 1.0 if options.get("captions") else 0.0,
        "speech_frac": float(np.mean([bool(s.get("keep_audio")) for s in segs])),
        "fade_in": 1.0 if plan.get("fade_in", True) else 0.0,
        "fade_out": 1.0 if plan.get("fade_out", True) else 0.0,
    }
    for g, names in TRANSITION_GROUPS.items():
        f[f"t_{g}"] = float(np.mean([t in names for t in trs])) if trs else 0.0
    mos = [s.get("motion", "none") for s in segs]
    for g, names in MOTION_GROUPS.items():
        f[f"m_{g}"] = float(np.mean([m in names for m in mos]))
    fam = look_family(plan.get("look", "clean"))
    for fm in FAMILIES:
        f[f"lf_{fm}"] = 1.0 if fm == fam else 0.0
    qs = [(shots.get(s.get("shot_id")) or {}).get("score", 0.5) for s in segs]
    vlm_hook = ((shots.get(segs[0].get("shot_id")) or {}).get("vlm") or {}).get("hook")
    f["hook_q"] = float(0.5 * qs[0] + 0.5 * (vlm_hook / 10 if vlm_hook else qs[0]))
    f["mean_q"] = float(np.mean(qs))
    cat = category(brief)
    if cat != "general":
        for k in CONTEXT_CROSSED:
            if f.get(k):
                f[f"{cat}×{k}"] = f[k]
    return {k: round(v, 4) for k, v in f.items()}


def summarize(plan: dict, rating: float | None = None, downloaded: bool = False) -> dict:
    """Compact description of an edit, used as an example of your taste in later prompts."""
    segs = plan.get("segments") or []
    durs = [(s["end"] - s["start"]) / max(0.05, s["speed"]) for s in segs]
    trs: dict[str, int] = {}
    for s in segs[1:]:
        trs[s.get("transition_in", "cut")] = trs.get(s.get("transition_in", "cut"), 0) + 1
    title = next(((s.get("text") or {}).get("content") for s in segs if s.get("text")), None)
    return {"shots": len(segs), "avg_len": round(float(np.mean(durs)), 2) if durs else 0, "look": plan.get("look"),
            "transitions": trs, "slowmo": sum(1 for s in segs if s["speed"] < 0.95), "title": title,
            "grain": round(float(plan.get("grain", 0)), 2), "letterbox": bool(plan.get("letterbox")),
            "rating": rating, "downloaded": downloaded}
