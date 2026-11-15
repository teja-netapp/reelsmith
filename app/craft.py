"""The 'assistant editor' pass: enforces playbook rules on any plan, whoever wrote it.

level="light" (Claude / Gemini): only safety fixes - word-safe speech cuts, no duplicate moments,
duration fit.
level="full" (local models / heuristic): also pacing bounds, transition restraint, motion variety,
strongest hook first and a resolved ending.
"""
from __future__ import annotations

import re

from .transitions import TRANSITIONS

PACE_BOUNDS = {"fast": (0.5, 2.0), "medium": (0.8, 3.5), "slow": (1.5, 5.5)}
ALT_MOTION = {"push_in": "drift_left", "pull_out": "drift_right", "drift_left": "push_in",
              "drift_right": "pull_out", "punch_in": "push_in", "rise": "drift_right",
              "ken_burns": "pull_out", "handheld": "push_in"}
_FAST = ("fast", "energetic", "hype", "punchy", "action", "party", "dance", "upbeat", "quick")
_SLOW = ("calm", "slow", "chill", "dreamy", "emotional", "peaceful", "wedding", "romantic")


def guess_pace(text: str) -> str:
    d = (text or "").lower()
    if any(w in d for w in _FAST):
        return "fast"
    if any(w in d for w in _SLOW):
        return "slow"
    return "medium"


def _out(s):
    return (s["end"] - s["start"]) / s["speed"]


def _overlap(a, b, c, d):
    return max(0.0, min(b, d) - max(a, c))


def _hook_value(s, shots):
    sh = shots.get(s.get("shot_id")) or {}
    v = sh.get("vlm") or {}
    if "hook" in v:
        return 0.6 * v["hook"] / 10 + 0.4 * sh.get("score", 0.5)
    return sh.get("score", 0.5)


def _variety(segs, notes):
    """Transition restraint (<= ~30% stylised, no identical ones back-to-back) and camera-move variety."""
    segs[0]["transition_in"] = "cut"
    styl = [i for i, s in enumerate(segs) if i > 0 and s["transition_in"] != "cut"]
    allowed = max(1, round(0.3 * (len(segs) - 1)))
    if len(styl) > allowed:
        step = len(styl) / allowed
        keep_idx = {styl[int(k * step)] for k in range(allowed)}
        for i in styl:
            if i not in keep_idx:
                segs[i]["transition_in"] = "cut"
        notes.append("reduced stylised transitions so they land harder")
    prev = None
    for s in segs[1:]:
        if s["transition_in"] != "cut" and s["transition_in"] == prev:
            s["transition_in"] = "cut"
        prev = s["transition_in"] if s["transition_in"] != "cut" else prev
    for s in segs:
        if s["transition_in"] in TRANSITIONS and s["transition_in"] != "cut":
            s["transition_duration"] = s.get("transition_duration") or TRANSITIONS[s["transition_in"]][0]
    for i in range(1, len(segs)):
        m = segs[i]["motion"]
        if m != "none" and m == segs[i - 1]["motion"]:
            alt = ALT_MOTION.get(m, "push_in")
            nxt = segs[i + 1]["motion"] if i + 1 < len(segs) else None
            if alt == nxt:
                alt = next(a for a in ("drift_right", "pull_out", "drift_left", "push_in") if a not in (m, nxt))
            segs[i]["motion"] = alt


def total_duration(segs):
    t = sum(_out(s) for s in segs)
    t -= sum(s.get("transition_duration", 0) for s in segs[1:] if s.get("transition_in") != "cut")
    return t


def polish(plan: dict, analysis: dict, options: dict, level: str = "light", description: str = "") -> dict:
    notes: list[str] = []
    segs = [dict(s) for s in plan.get("segments", [])]
    if not segs:
        return plan
    clips = {c["id"]: c for c in analysis["clips"]}
    shots = {s["id"]: s for s in analysis.get("all_shots", [])}
    shots.update({s["id"]: s for s in analysis.get("shots", [])})
    trs = analysis.get("transcripts", {})
    pace = plan.get("pace") or guess_pace(description)
    pace = pace if pace in PACE_BOUNDS else "medium"
    target = float(options.get("target_duration", 30))
    full = level == "full"

    def shot_bounds(s):
        sh = shots.get(s.get("shot_id"))
        cd = clips[s["clip_id"]]["duration"]
        return (sh["start"], sh["end"]) if sh else (0.0, cd)

    # 1. speech: speed 1.0 and never cut mid-word
    for s in segs:
        tr = trs.get(s["clip_id"])
        if not (s.get("keep_audio") and tr and tr.get("words")):
            continue
        ws = [w for w in tr["words"] if w["end"] > s["start"] and w["start"] < s["end"]]
        if not ws:
            continue
        if abs(s["speed"] - 1.0) > 1e-3:
            s["speed"] = 1.0
            notes.append("speech segments play at normal speed")
        cd = clips[s["clip_id"]]["duration"]
        for w in ws:
            if w["start"] < s["start"] < w["end"]:
                s["start"] = round(max(0.0, w["start"] - 0.05), 3)
            if w["start"] < s["end"] < w["end"]:
                s["end"] = round(min(cd, w["end"] + 0.08), 3)

    # 2. no duplicate moments (a ramp continuing the previous segment is fine: it doesn't overlap)
    used: dict[str, list] = {}
    kept = []
    for s in segs:
        L = s["end"] - s["start"]
        prior = used.setdefault(s["clip_id"], [])
        ov = max((_overlap(s["start"], s["end"], a, b) for a, b in prior), default=0.0)
        if L > 0 and ov / L > 0.5:
            lo, hi = shot_bounds(s)
            moved = False
            t = lo
            while t + L <= hi + 1e-6:
                if max((_overlap(t, t + L, a, b) for a, b in prior), default=0.0) <= 0.2 * L:
                    s["start"], s["end"] = round(t, 3), round(t + L, 3)
                    moved = True
                    break
                t += 0.25
            if moved:
                notes.append("moved a repeated moment to a fresh part of its shot")
            elif full and len(segs) > 2:
                notes.append("dropped a segment that repeated an earlier moment")
                continue
        prior.append((s["start"], s["end"]))
        kept.append(s)
    segs = kept

    if full:
        # 3. pacing bounds (speech may run longer)
        lo_d, hi_d = PACE_BOUNDS[pace]
        for s in segs:
            cap = 8.0 if s.get("keep_audio") else hi_d
            d = _out(s)
            if d > cap * 1.15:
                s["end"] = round(s["start"] + cap * s["speed"], 3)
            elif d < lo_d * 0.8:
                _, hi = shot_bounds(s)
                s["end"] = round(min(hi, s["start"] + lo_d * s["speed"]), 3)

        # 4. hook: strongest image first - unless the user said how to open, or wants clips in order
        told_opening = re.search(r"\b(open|start|begin)\w*\s+(on|with)\b|\bfirst shot\b", (description or "").lower())
        if len(segs) > 2 and not told_opening and options.get("auto_pick", True):
            vals = [_hook_value(s, shots) for s in segs]
            best = max(range(len(segs) - 1), key=lambda i: vals[i])  # never steal the ending
            if vals[0] < vals[best] - 0.15:
                old_first = segs[0]
                first = segs.pop(best)
                first["transition_in"] = "cut"
                # an opening title belongs to the opening shot
                if old_first.get("text") and not first.get("text"):
                    first["text"], old_first["text"] = old_first["text"], None
                segs.insert(0, first)
                notes.append("moved the strongest shot to the opening hook")
        segs[0]["transition_in"] = "cut"

    # 5. duration fit: stretch segments inside their shots, then (full) add unused strong shots
    tot = total_duration(segs)
    if tot < target * 0.88:
        for s in segs:
            if s.get("keep_audio"):
                continue
            _, hi = shot_bounds(s)
            want = _out(s) * min(1.8, target / max(tot, 0.1))
            if full:
                want = min(want, PACE_BOUNDS[pace][1] * 1.2)
            s["end"] = round(min(hi, s["start"] + want * s["speed"]), 3)
        tot = total_duration(segs)
        if full and tot < target * 0.88:
            used_ids = {(s["clip_id"], round(s["start"], 1)) for s in segs}
            pool = sorted(analysis.get("shots", []), key=lambda sh: _hook_value({"shot_id": sh["id"]}, shots),
                          reverse=True)
            L = sum(PACE_BOUNDS[pace]) / 2
            motions = ["push_in", "drift_right", "pull_out", "drift_left"]
            for sh in pool:
                if tot >= target * 0.95:
                    break
                prior = used.setdefault(sh["clip_id"], [])
                ln = min(L, sh["duration"])
                st = sh["start"] + (sh["duration"] - ln) / 2
                if any(_overlap(st, st + ln, a, b) > 0.2 * ln for a, b in prior) or (sh["clip_id"], round(st, 1)) in used_ids:
                    continue
                new = {"shot_id": sh["id"], "clip_id": sh["clip_id"], "start": round(st, 3), "end": round(st + ln, 3),
                       "speed": 1.0, "motion": motions[len(segs) % 4], "transition_in": "cut",
                       "transition_duration": 0.0, "focus_x": (sh.get("vlm") or {}).get("subject_x", sh.get("focus_x") or 0.5),
                       "keep_audio": False, "text": None}
                segs.insert(max(1, len(segs) - 1), new)
                prior.append((st, st + ln))
                tot += ln
            notes.append("added strong unused shots to reach the target length")

    if full and segs:
        # 6. resolved ending
        last = segs[-1]
        if last["motion"] == "punch_in":
            last["motion"] = "push_in"
        if _out(last) < 1.2:
            _, hi = shot_bounds(last)
            last["end"] = round(min(hi, last["start"] + 1.2 * last["speed"]), 3)
        plan["fade_out"] = True
        # 7. variety (last, so it also covers reordered / added segments)
        _variety(segs, notes)

    plan["segments"] = segs
    plan["pace"] = pace
    if notes:
        plan["craft_notes"] = sorted(set(notes))
    return plan
