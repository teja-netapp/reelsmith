"""The 'assistant editor' pass: enforces playbook rules on any plan, whoever wrote it.

level="light" (Claude / Gemini): only safety fixes - word-safe speech cuts, no duplicate moments,
duration fit.
level="full" (local models / heuristic): also pacing bounds, transition restraint, motion variety,
strongest hook first and a resolved ending.

Every change is recorded in plan["craft_changes"] as {"rule", "change", "why"} so the edit report
can show exactly what was altered after the AI wrote its plan.
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


def fmt_t(t: float) -> str:
    m, s = divmod(max(0.0, float(t)), 60)
    return f"{int(m)}:{s:04.1f}"


def _out(s):
    return (s["end"] - s["start"]) / s["speed"]


def _overlap(a, b, c, d):
    return max(0.0, min(b, d) - max(a, c))


def _free(a, b, used):
    """Sub-intervals of [a, b] not covered by any used (x, y) interval."""
    out, cur = [], a
    for x, y in sorted(used):
        if y <= cur or x >= b:
            continue
        if x > cur:
            out.append((cur, min(x, b)))
        cur = max(cur, y)
        if cur >= b:
            break
    if cur < b:
        out.append((cur, b))
    return [(x, y) for x, y in out if y - x > 0.05]


def _hook_value(s, shots):
    sh = shots.get(s.get("shot_id")) or {}
    v = sh.get("vlm") or {}
    if "hook" in v:
        return 0.6 * v["hook"] / 10 + 0.4 * sh.get("score", 0.5)
    return sh.get("score", 0.5)


def total_duration(segs):
    t = sum(_out(s) for s in segs)
    t -= sum(s.get("transition_duration", 0) for s in segs[1:] if s.get("transition_in") != "cut")
    return t


class _Log:
    def __init__(self, clips):
        self.items: list[dict] = []
        self.clips = clips

    def desc(self, s) -> str:
        name = self.clips.get(s["clip_id"], {}).get("name", s["clip_id"])
        return f"{name} @ {fmt_t(s['start'])}"

    def add(self, rule: str, change: str, why: str):
        self.items.append({"rule": rule, "change": change, "why": why})


def _variety(segs, log: _Log, taste: dict | None = None):
    """Transition restraint (<= ~30% stylised, no identical ones back-to-back) and camera-move variety."""
    segs[0]["transition_in"] = "cut"
    styl = [i for i, s in enumerate(segs) if i > 0 and s["transition_in"] != "cut"]
    frac = float((taste or {}).get("styl_max", 0.3))   # your learned transition budget, default ~30%
    allowed = max(1, round(frac * (len(segs) - 1)))
    if len(styl) > allowed:
        step = len(styl) / allowed
        keep_idx = {styl[int(k * step)] for k in range(allowed)}
        for i in styl:
            if i not in keep_idx:
                log.add("transition restraint", f"{log.desc(segs[i])}: {segs[i]['transition_in']} → cut",
                        f"{len(styl)} of {len(segs) - 1} edits were stylised; keeping {allowed} so they land harder")
                segs[i]["transition_in"] = "cut"
    prev = None
    for s in segs[1:]:
        if s["transition_in"] != "cut" and s["transition_in"] == prev:
            log.add("transition restraint", f"{log.desc(s)}: {s['transition_in']} → cut",
                    "same stylised transition twice in a row")
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
            log.add("camera variety", f"{log.desc(segs[i])}: camera {m} → {alt}",
                    "same camera move as the previous shot")
            segs[i]["motion"] = alt


def polish(plan: dict, analysis: dict, options: dict, level: str = "light", description: str = "",
           taste: dict | None = None) -> dict:
    segs = [dict(s) for s in plan.get("segments", [])]
    if not segs:
        return plan
    clips = {c["id"]: c for c in analysis["clips"]}
    log = _Log(clips)
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
            log.add("speech", f"{log.desc(s)}: speed {s['speed']:g}× → 1×", "this shot keeps its speech")
            s["speed"] = 1.0
        cd = clips[s["clip_id"]]["duration"]
        for w in ws:
            if w["start"] < s["start"] < w["end"]:
                new = round(max(0.0, w["start"] - 0.05), 3)
                log.add("word-safe cut", f"{log.desc(s)}: in-point {fmt_t(s['start'])} → {fmt_t(new)}",
                        f"the cut fell in the middle of the word \"{w['word']}\"")
                s["start"] = new
            if w["start"] < s["end"] < w["end"]:
                new = round(min(cd, w["end"] + 0.08), 3)
                log.add("word-safe cut", f"{log.desc(s)}: out-point {fmt_t(s['end'])} → {fmt_t(new)}",
                        f"the cut fell in the middle of the word \"{w['word']}\"")
                s["end"] = new

    # 2. no duplicate moments (a ramp continuing the previous segment is fine: it doesn't overlap)
    used: dict[str, list] = {}
    kept = []
    for s in segs:
        L = s["end"] - s["start"]
        prior = used.setdefault(s["clip_id"], [])
        ov = sum(_overlap(s["start"], s["end"], a, b) for a, b in prior)
        if L > 0 and ov > max(0.25, 0.15 * L):
            lo, hi = shot_bounds(s)
            old = log.desc(s)
            fixed = False
            t = lo
            while t + L <= hi + 1e-6:  # 1) same length, fresh part of the same shot
                if sum(_overlap(t, t + L, a, b) for a, b in prior) <= 0.05:
                    s["start"], s["end"] = round(t, 3), round(t + L, 3)
                    log.add("no repeats", f"{old} → {fmt_t(s['start'])}", "that moment was already in the reel")
                    fixed = True
                    break
                t += 0.25
            if not fixed:  # 2) keep only the part that hasn't been used yet
                free = _free(s["start"], s["end"], prior)
                if free:
                    a, b = max(free, key=lambda x: x[1] - x[0])
                    if b - a >= max(0.6, 0.5 * L):
                        s["start"], s["end"] = round(a, 3), round(b, 3)
                        log.add("no repeats", f"{old}: trimmed to {fmt_t(a)}–{fmt_t(b)}",
                                "part of it was already in the reel")
                        fixed = True
            if not fixed and full and len(segs) > 2:
                log.add("no repeats", f"removed a second use of {old}", "that moment was already in the reel")
                continue
        prior.append((s["start"], s["end"]))
        kept.append(s)
    segs = kept

    taste = taste or {}
    # 2b. your stated dislikes apply to every editor: banned transitions, slow motion
    avoid = set(taste.get("avoid_transitions", []))
    for s in segs[1:]:
        if s["transition_in"] in avoid:
            log.add("your taste", f"{log.desc(s)}: {s['transition_in']} → cut", "you've told me you don't like it")
            s["transition_in"] = "cut"
    if taste.get("slowmo_frac") == -1:
        for s in segs:
            if s["speed"] < 0.95:
                out = _out(s)
                s["speed"] = 1.0
                s["end"] = round(min(shot_bounds(s)[1], s["start"] + out), 3)
                log.add("your taste", f"{log.desc(s)}: slow motion → normal speed", "you prefer no slow motion")

    if full:
        # 3. pacing bounds (speech may run longer) - centred on your learned shot length if known
        lo_d, hi_d = PACE_BOUNDS[pace]
        if taste.get("avg_len"):
            L = float(taste["avg_len"])
            lo_d, hi_d = max(0.4, 0.55 * L), max(1.2, 1.7 * L)
        for s in segs:
            cap = 8.0 if s.get("keep_audio") else hi_d
            d = _out(s)
            if d > cap * 1.15:
                s["end"] = round(s["start"] + cap * s["speed"], 3)
                log.add("pacing", f"{log.desc(s)}: {d:.1f} s → {cap:.1f} s on screen",
                        f"{pace} pace: shots up to ~{cap:.1f} s")
            elif d < lo_d * 0.8:
                _, hi = shot_bounds(s)
                s["end"] = round(min(hi, s["start"] + lo_d * s["speed"]), 3)
                if abs(_out(s) - d) > 0.05:
                    log.add("pacing", f"{log.desc(s)}: {d:.1f} s → {_out(s):.1f} s on screen",
                            f"{pace} pace: shots at least ~{lo_d:.1f} s")

        # 4. hook: strongest image first - unless the user said how to open, or wants clips in order
        told_opening = re.search(r"\b(open|start|begin)\w*\s+(on|with)\b|\bfirst shot\b", (description or "").lower())
        if len(segs) > 2 and not told_opening and options.get("auto_pick", True) and not options.get("_keep_order"):
            vals = [_hook_value(s, shots) for s in segs]
            best = max(range(len(segs) - 1), key=lambda i: vals[i])  # never steal the ending
            if vals[0] < vals[best] - 0.15:
                old_first = segs[0]
                first = segs.pop(best)
                first["transition_in"] = "cut"
                log.add("hook first", f"moved {log.desc(first)} to the opening (was shot {best + 1})",
                        f"strongest opening image (hook {vals[best]:.2f} vs {vals[0]:.2f})")
                if old_first.get("text") and not first.get("text"):
                    first["text"], old_first["text"] = old_first["text"], None
                    log.add("hook first", f"moved the title \"{first['text']['content']}\" to the new opening shot",
                            "an opening title belongs on the first shot")
                segs.insert(0, first)
        segs[0]["transition_in"] = "cut"

    # 5. duration fit: stretch segments inside their shots, then (full) add unused strong shots
    tot = total_duration(segs)
    if tot < target * 0.88:
        before = tot
        n_str = 0
        for s in segs:
            if s.get("keep_audio"):
                continue
            _, hi = shot_bounds(s)
            # never stretch into footage another segment already uses
            later = [o["start"] for o in segs if o is not s and o["clip_id"] == s["clip_id"]
                     and o["start"] >= s["start"] - 1e-3]
            if later:
                hi = min(hi, min(later))
            if hi <= s["end"]:
                continue
            want = _out(s) * min(1.8, target / max(tot, 0.1))
            if full:
                want = min(want, PACE_BOUNDS[pace][1] * 1.2)
            old_d = _out(s)
            s["end"] = round(min(hi, s["start"] + want * s["speed"]), 3)
            if _out(s) - old_d > 0.05:
                n_str += 1
        tot = total_duration(segs)
        if n_str:
            log.add("length fit", f"lengthened {n_str} shot(s) within their takes: {before:.1f} s → {tot:.1f} s",
                    f"the plan was shorter than the {target:.0f} s target")
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
                       "transition_duration": 0.0,
                       "focus_x": (sh.get("vlm") or {}).get("subject_x", sh.get("focus_x") or 0.5),
                       "keep_audio": False, "text": None}
                segs.insert(max(1, len(segs) - 1), new)
                prior.append((st, st + ln))
                tot += ln
                log.add("length fit", f"added {log.desc(new)} ({ln:.1f} s, unused strong shot)",
                        f"the plan was still shorter than the {target:.0f} s target")

    if full and segs:
        # 6. resolved ending
        last = segs[-1]
        if last["motion"] == "punch_in":
            last["motion"] = "push_in"
            log.add("ending", f"{log.desc(last)}: camera punch_in → push_in", "end on a calm, resolved move")
        if _out(last) < 1.2:
            _, hi = shot_bounds(last)
            d = _out(last)
            last["end"] = round(min(hi, last["start"] + 1.2 * last["speed"]), 3)
            if _out(last) - d > 0.05:
                log.add("ending", f"{log.desc(last)}: {d:.1f} s → {_out(last):.1f} s", "let the last image breathe")
        if not plan.get("fade_out", True):
            log.add("ending", "added a fade-out", "end the reel cleanly")
        plan["fade_out"] = True
        # 7. variety (last, so it also covers reordered / added segments)
        _variety(segs, log, taste)

    plan["segments"] = segs
    plan["pace"] = pace
    plan["craft_changes"] = log.items
    plan["craft_notes"] = sorted({c["rule"] for c in log.items})
    return plan
