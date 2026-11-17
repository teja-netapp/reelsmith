"""The learning loop: feedback -> reward model -> better edits next time.

How an edit improves with your feedback (no model fine-tuning needed - this runs in milliseconds):
  1. House style   - what the reward model learned is written into the AI editor's prompt, together
                     with your best-rated past edits as examples (in-context learning).
  2. Assistant     - the craft pass uses your learned pace, transition budget and banned transitions.
  3. Search        - after the AI writes its plan, variants are generated (pace, transitions, look,
                     texture, slow-mo) and scored by the reward model; the best one is rendered.
                     Occasionally a near-best variant is tried instead (exploration), so the model
                     keeps learning instead of getting stuck on its first guess.
Every signal you give (stars, downloads, Look Lab picks, revision requests) updates the model.
"""
from __future__ import annotations

import copy
import json
import math
import random
import re
import time

import numpy as np

from ..grading import FAMILIES, look_family, look_names, look_texture
from ..transitions import REGISTRY
from .features import CONTEXT_CROSSED, TRANSITION_GROUPS, category, label, plan_features, summarize
from .preferences import parse
from .reward import RewardModel
from .store import Store

PREFERRED_BY_CATEGORY = {
    "travel": ["light_leak", "whip_right", "pull_in", "lens_flare", "push_left"],
    "hype": ["zoom_flash", "zoom_punch", "shake_punch", "glitch_whip", "flash_blast", "spin"],
    "romantic": ["crossfade", "light_leak", "blur_dissolve", "dip_to_white", "heart"],
    "vlog": ["crossfade", "story_swipe", "slide_left"], "sport": ["zoom_punch", "whip_left", "shake", "speed_sweep"],
    "food": ["slide_up", "scrapbook", "whip_right", "pull_in"], "nature": ["crossfade", "light_leak", "focus_blur",
                                                                        "film_burn"],
    "city": ["whip_left", "glitch", "rgb_split", "zoom_punch", "cube_3d"], "family": ["crossfade", "light_leak",
                                                                                    "scrapbook", "paper_tear"],
    "product": ["zoom_punch", "flip_3d", "slide_up", "pull_in"], "general": ["crossfade", "whip_right", "light_leak",
                                                                           "pull_in"],
}
GROUP_DEFAULT = {"basic": "pull_in", "camera": "zoom_punch", "light": "light_leak", "blur": "focus_blur",
                 "glitch": "glitch", "distortion": "ripple", "slide": "push_left", "mask": "iris",
                 "overlay": "story_swipe", "trending": "zoom_flash"}
TR_DUR = {n: t.duration for n, t in REGISTRY.items()}


def _group(tr: str) -> str | None:
    return next((g for g, names in TRANSITION_GROUPS.items() if tr in names), None)


def _describe_diff(a: dict, b: dict) -> list[str]:
    """Net changes from the editor's plan (a) to the delivered plan (b), in plain words."""
    out = []
    if a.get("look") != b.get("look"):
        out.append(f"look {a.get('look')} → {b.get('look')}")

    def avg(p):
        d = [(s["end"] - s["start"]) / s["speed"] for s in p["segments"]]
        return sum(d) / max(1, len(d))
    la, lb = avg(a), avg(b)
    if abs(la - lb) > 0.12:
        out.append(f"{'faster' if lb < la else 'slower'} pace: {la:.1f} s → {lb:.1f} s per shot "
                   f"({len(a['segments'])} → {len(b['segments'])} shots)")
    sa = sum(1 for s in a["segments"][1:] if s["transition_in"] != "cut")
    sb = sum(1 for s in b["segments"][1:] if s["transition_in"] != "cut")
    ta = sorted(s["transition_in"] for s in a["segments"][1:] if s["transition_in"] != "cut")
    tb = sorted(s["transition_in"] for s in b["segments"][1:] if s["transition_in"] != "cut")
    if sa != sb:
        out.append(f"stylised transitions {sa} → {sb}")
    elif ta != tb:
        out.append("swapped transition types: " + ", ".join(sorted(set(tb) - set(ta))))
    ga, gb = float(a.get("grain", 0)), float(b.get("grain", 0))
    if abs(ga - gb) > 0.05:
        out.append(f"film grain {ga:.2f} → {gb:.2f}")
    if bool(a.get("letterbox")) != bool(b.get("letterbox")):
        out.append(("added" if b.get("letterbox") else "removed") + " letterbox")
    ma = sum(1 for s in a["segments"] if s["speed"] < 0.95)
    mb = sum(1 for s in b["segments"] if s["speed"] < 0.95)
    if ma != mb:
        out.append(f"slow-motion shots {ma} → {mb}")
    return out


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", (s or "").lower())
            if w not in {"the", "and", "with", "for", "reel", "make", "video", "some", "that", "this", "from"}}


class Learner:
    def __init__(self, path):
        self.store = Store(path)
        self.model = RewardModel(self.store.get("reward", {}))
        self._rng = random.Random()

    # -- settings ------------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(self.store.get("enabled", True))

    def set_enabled(self, on: bool):
        self.store.put("enabled", bool(on))

    def _save(self):
        self.store.put("reward", self.model.state())

    def reset(self):
        self.store.reset()
        self.model = RewardModel()

    # -- what it has learned -------------------------------------------------------------
    def _wf(self, key: str, cat: str) -> float:
        return self.model.w.get(key, 0.0) + self.model.w.get(f"{cat}×{key}", 0.0)

    def _liked_edits(self, cat: str | None = None):
        rows = [e for e in self.store.edits(limit=500) if (e.get("rating") or 0) >= 4 or e.get("downloaded")]
        return [e for e in rows if cat is None or e.get("category") == cat]

    def profile(self, brief: str) -> dict:
        cat = category(brief)
        conf = self.model.confidence
        params: dict = {}
        bullets: list[str] = []
        if self.model.n == 0:
            return {"category": cat, "confidence": 0.0, "bullets": [], "params": {}, "looks": []}
        # pace: learned weights + the average shot length of edits you liked
        pace = self._wf("pace_short", cat) + 0.7 * self._wf("cuts_per_10s", cat) - self._wf("avg_len", cat)
        liked = self._liked_edits(cat) or self._liked_edits()
        lens = [e["summary"].get("avg_len") for e in liked if e.get("summary", {}).get("avg_len")]
        if lens and len(lens) >= 2:
            params["avg_len"] = round(float(np.median(lens)), 2)
        if pace > 0.15:
            params["avg_len"] = round(min(params.get("avg_len", 1.3), 1.3), 2)
            bullets.append(f"Faster cutting: about {params['avg_len']:.1f} s per shot")
        elif pace < -0.15:
            params["avg_len"] = round(max(params.get("avg_len", 2.8), 2.4), 2)
            bullets.append(f"Slower, breathing shots: about {params['avg_len']:.1f} s per shot")
        elif "avg_len" in params:
            bullets.append(f"Shots of about {params['avg_len']:.1f} s (like your favourite edits)")
        # transitions
        styl = self._wf("styl_frac", cat)
        params["styl_max"] = round(float(np.clip(0.3 + 0.2 * math.tanh(2 * styl), 0.05, 0.5)), 2)
        if styl < -0.15:
            bullets.append("Mostly clean cuts; stylised transitions only rarely")
        elif styl > 0.15:
            bullets.append("You enjoy stylised transitions (up to about a third of the edits)")
        avoid = [g for g in TRANSITION_GROUPS if self._wf(f"t_{g}", cat) < -0.12]
        prefer = sorted((g for g in TRANSITION_GROUPS if self._wf(f"t_{g}", cat) > 0.12),
                        key=lambda g: -self._wf(f"t_{g}", cat))
        if avoid:
            params["avoid_transitions"] = [n for g in avoid for n in TRANSITION_GROUPS[g]]
            bullets.append("Avoid " + ", ".join(g.replace("_", " ") for g in avoid) + " transitions")
        if prefer:
            params["prefer_transitions"] = [GROUP_DEFAULT[g] for g in prefer]
            bullets.append("Favourite transitions: " + ", ".join(g.replace("_", " ") for g in prefer))
        # looks
        hist: dict[str, list] = {}
        for e in self.store.edits(limit=500):
            if e.get("category") == cat and e.get("rating"):
                hist.setdefault(e["summary"].get("look"), []).append(e["rating"])
        ranked = []
        for lk in look_names():
            s = self._wf(f"lf_{look_family(lk)}", cat)
            if lk in hist:
                r = hist[lk]
                s += 0.25 * (sum(r) + 3 * 2) / (len(r) + 2) - 0.75  # smoothed mean rating, centred
            ranked.append((s, lk))
        ranked.sort(reverse=True)
        looks = [lk for s, lk in ranked[:3] if s > 0.05]
        if looks:
            params["looks"] = looks
            bullets.append(("Looks you like" if cat == "general" else f"Looks you like for {cat} reels") + ": "
                           + ", ".join(looks))
        disliked_fams = [f for f in FAMILIES if self._wf(f"lf_{f}", cat) < -0.2]
        if disliked_fams:
            bullets.append("Not a fan of " + ", ".join(disliked_fams) + " looks")
        # texture & extras
        for key, pos, neg in (("grain", "Film grain", "No film grain"), ("letterbox", "Cinematic letterbox bars",
                                                                          "No letterbox bars"),
                              ("slowmo_frac", "Some slow-motion moments", "Avoid slow motion"),
                              ("has_title", "An opening title", "No titles"),
                              ("m_handheld", "Handheld energy", "Steady camera, no handheld shake"),
                              ("speech_frac", "Keep the original speech/sound", "Music only, little speech")):
            v = self._wf(key, cat)
            if v > 0.2:
                params[key] = 1
                bullets.append(pos)
            elif v < -0.2:
                params[key] = -1
                bullets.append(neg)
        return {"category": cat, "confidence": round(conf, 2), "bullets": bullets, "params": params, "looks": looks}

    def exemplars(self, brief: str, k: int = 2) -> list[dict]:
        cat, toks = category(brief), _tokens(brief)
        scored = []
        for e in self._liked_edits():
            et = _tokens(e.get("brief", ""))
            sim = (len(toks & et) / max(1, len(toks | et))) * 2 + (1.0 if e.get("category") == cat else 0)
            scored.append((sim + (e.get("rating") or 4) / 5, e))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:k]]

    def prompt_block(self, brief: str) -> str:
        if not self.enabled:
            return ""
        p = self.profile(brief)
        ex = self.exemplars(brief)
        if not p["bullets"] and not ex:
            return ""
        c = self.store.counts()
        L = [f"# Your house style (learned from {c['edits']} of your edits and {c['rated'] + c['feedback']} pieces of "
             "feedback - follow it unless the brief asks otherwise)"]
        L += [f"- {b}" for b in p["bullets"]]
        if ex:
            L.append("# Edits you liked (match their feel)")
            for e in ex:
                s = e.get("summary", {})
                trs = ", ".join(f"{k}×{v}" for k, v in sorted(s.get("transitions", {}).items(), key=lambda kv: -kv[1]))
                stars = f"{e['rating']:.0f}★" if e.get("rating") else "downloaded"
                L.append(f"- \"{(e.get('brief') or '')[:80]}\" → {s.get('shots')} shots, ~{s.get('avg_len')} s each, "
                         f"look {s.get('look')}, transitions {trs or 'cuts'}"
                         + (f", title \"{s['title']}\"" if s.get("title") else "")
                         + (f", {s['slowmo']} slow-mo" if s.get("slowmo") else "") + f" ({stars})")
        return "\n".join(L)

    # -- policy improvement: reward-guided variant search -------------------------------------
    def improve(self, plan: dict, analysis: dict, options: dict, brief: str, level: str, polish_fn, sanitize_fn,
                music=None) -> dict:
        if not self.enabled:
            return plan
        prof = self.profile(brief)
        params, cat = prof["params"], prof["category"]
        conf = self.model.confidence
        target = float(options.get("target_duration", 30))
        shots = {s["id"]: s for s in analysis.get("all_shots", [])}
        shots.update({s["id"]: s for s in analysis.get("shots", [])})
        clips = {c["id"]: c for c in analysis["clips"]}

        def finish(p):
            p = polish_fn(copy.deepcopy(p), analysis, options, level, brief, params)
            return sanitize_fn(p, analysis, music, options)

        def score(p):
            f = plan_features(p, analysis, options, brief)
            dur = p.get("estimated_duration") or target
            prior = (-2.0 * max(0.0, f.get("styl_frac", 0) - 0.45) - 1.5 * abs(dur / target - 1)
                     + 0.4 * f.get("hook_q", 0.5) + 0.4 * f.get("mean_q", 0.5))
            return self.model.score(f) + prior, f

        base = finish(plan)
        base_s, base_f = score(base)
        lam = 0.15 + 0.85 * (1 - conf)

        def total(p):
            s, f = score(p)
            dist = sum(abs(f.get(k, 0) - base_f.get(k, 0)) for k in base_f if "×" not in k)
            return s - lam * 0.15 * dist

        def bounds(s):
            sh = shots.get(s.get("shot_id"))
            return (sh["start"], sh["end"]) if sh else (0.0, clips[s["clip_id"]]["duration"])

        # ---- mutation operators (each returns (description, new_plan))
        def m_pace(p, factor):
            q = copy.deepcopy(p)
            for s in q["segments"]:
                if s.get("keep_audio"):
                    continue
                lo, hi = bounds(s)
                out = (s["end"] - s["start"]) / s["speed"] * factor
                s["end"] = round(max(s["start"] + 0.3, min(hi, s["start"] + out * s["speed"])), 3)
            return (f"{'faster' if factor < 1 else 'slower'} pace (shots ×{factor:g})", q)

        prefer = params.get("prefer_transitions") or PREFERRED_BY_CATEGORY.get(cat, PREFERRED_BY_CATEGORY["general"])
        avoid = set(params.get("avoid_transitions", []))
        prefer = [t for t in prefer if t not in avoid] or ["crossfade"]

        def m_trans(p, mode):
            q = copy.deepcopy(p)
            segs = q["segments"]
            n = len(segs)
            if n < 3:
                return None
            if mode == "swap":
                changed = False
                for i, s in enumerate(segs[1:], 1):
                    if s["transition_in"] in avoid:
                        s["transition_in"] = prefer[i % len(prefer)]
                        s["transition_duration"] = TR_DUR.get(s["transition_in"], 0.4)
                        changed = True
                return ("swapped transitions you dislike", q) if changed else None
            budget = {"restrained": 0.12, "richer": min(0.35, params.get("styl_max", 0.3))}[mode]
            k = max(1, round(budget * (n - 1)))
            idx = {int(round((j + 1) * (n - 1) / (k + 1))) for j in range(k)}
            for i, s in enumerate(segs[1:], 1):
                if i in idx:
                    if s["transition_in"] == "cut" or s["transition_in"] in avoid:
                        s["transition_in"] = prefer[len(idx) % len(prefer)]
                        s["transition_duration"] = TR_DUR.get(s["transition_in"], 0.4)
                elif mode == "restrained":
                    s["transition_in"] = "cut"
            return (f"{'fewer' if mode == 'restrained' else 'more'} stylised transitions", q)

        def m_look(p, lk):
            if lk == p.get("look"):
                return None
            q = copy.deepcopy(p)
            old_tex, tex = look_texture(p.get("look", "clean")), look_texture(lk)
            q["look"] = lk
            for key in ("halation", "bloom"):
                q[key] = tex[key]
            if abs(q.get("grain", 0) - old_tex["grain"]) < 0.02:
                q["grain"] = tex["grain"]
            return (f"look {lk}", q)

        def m_grain(p, on):
            q = copy.deepcopy(p)
            q["grain"] = 0.3 if on else 0.0
            return (("added" if on else "removed") + " film grain", q)

        def m_letterbox(p, on):
            if bool(p.get("letterbox")) == on:
                return None
            q = copy.deepcopy(p)
            q["letterbox"] = on
            return (("added" if on else "removed") + " letterbox", q)

        def m_slowmo(p):
            q = copy.deepcopy(p)
            cands = [s for s in q["segments"][1:] if not s.get("keep_audio") and s["speed"] >= 0.95 and
                     (shots.get(s.get("shot_id")) or {}).get("motion", 0) > 0.45]
            for s in cands[:2]:
                out = (s["end"] - s["start"]) / s["speed"]
                s["speed"] = 0.5
                s["end"] = round(s["start"] + out * 0.5, 3)
            return ("slow-motion on high-motion shots", q) if cands else None

        locks = set(options.get("_locks") or [])  # dimensions your instructions fixed - never searched
        dims = [
            [lambda p: m_pace(p, 0.8), lambda p: m_pace(p, 1.25)],
            [] if "transitions" in locks else
            [lambda p: m_trans(p, "restrained"), lambda p: m_trans(p, "richer"), lambda p: m_trans(p, "swap")],
            [] if "look" in locks else
            [(lambda lk: (lambda p: m_look(p, lk)))(lk) for lk in (params.get("looks") or [])[:3]],
            ([] if "grain" in locks else [lambda p: m_grain(p, True), lambda p: m_grain(p, False)]) +
            ([] if "letterbox" in locks else [lambda p: m_letterbox(p, True), lambda p: m_letterbox(p, False)]),
            [] if "slowmo" in locks else [m_slowmo],
        ]
        best, best_t, applied = base, total(base), []
        tried = [(best_t, [], base)]
        for _ in range(2):  # coordinate ascent, two passes
            improved = False
            for dim in dims:
                for op in dim:
                    r = op(best)
                    if not r:
                        continue
                    desc, cand = r
                    try:
                        cand = finish(cand)
                    except Exception:
                        continue
                    t = total(cand)
                    tried.append((t, applied + [desc], cand))
                    if t > best_t + 1e-6:
                        best, best_t, applied, improved = cand, t, applied + [desc], True
            if not improved:
                break
        explored = None
        if self.model.n > 0:
            # exploration keeps it learning: sometimes try a near-best variant, or a look family you
            # haven't been shown yet for this kind of reel (so it can discover you like it even more)
            if self._rng.random() < 0.03 + 0.2 * (1 - conf):
                near = [x for x in tried if x[2] is not best and best_t - x[0] < 0.12]
                if near:
                    best_t, _, best = self._rng.choice(near)
                    explored = "a near-best variant"
            if explored is None and "look" not in locks and self._rng.random() < 0.12:
                seen = {look_family(e["summary"].get("look") or "clean") for e in self.store.edits(limit=500)
                        if e.get("category") == cat}
                fresh = [lk for lk in look_names() if look_family(lk) not in seen and
                         self._wf(f"lf_{look_family(lk)}", cat) > -0.2]
                if fresh:
                    r = m_look(best, self._rng.choice(fresh))
                    if r:
                        cand = finish(r[1])
                        t = total(cand)
                        if best_t - t < 0.35:
                            best, best_t, explored = cand, t, f"a look you haven't seen yet ({cand['look']})"
        best["learning"] = {
            "locked": sorted(locks),
            "applied": _describe_diff(base, best), "predicted": round(best_t, 3), "baseline": round(base_s, 3),
            "explored": explored, "confidence": round(conf, 2), "candidates": len(tried),
            "house_style": prof["bullets"], "category": cat,
        }
        return best

    # -- feedback ------------------------------------------------------------------------
    def record_edit(self, job_id, version, plan, analysis, options, brief, trigger) -> int | None:
        if not self.enabled:
            return None
        f = plan_features(plan, analysis, options, brief)
        ed = plan.get("editor") or {}
        return self.store.add_edit(job_id, version, brief, category(brief), ed.get("label"), ed.get("model"), trigger,
                                   f, summarize(plan), plan.get("learning") or {})

    def rate(self, job_id, version, stars: float) -> dict:
        e = self.store.edit(job_id, version)
        if not e or not self.enabled:
            return {"ok": False}
        stars = float(max(1, min(5, stars)))
        self.store.set_rating(e["id"], stars)
        self.model.update_rating(e["features"], stars)
        pairs = 0
        for o in self.store.edits(job_id):
            if o["id"] == e["id"]:
                continue
            if o.get("rating"):
                d = stars - o["rating"]
                if abs(d) >= 1:
                    w, l = (e, o) if d > 0 else (o, e)
                    self.model.update_pair(w["features"], l["features"], min(1.0, abs(d) / 2))
                    self.store.add_pair(w["id"], l["id"], min(1.0, abs(d) / 2), "rating")
                    pairs += 1
            elif stars >= 4 and o["version"] < version:
                self.model.update_pair(e["features"], o["features"], 0.3)
                self.store.add_pair(e["id"], o["id"], 0.3, "revised-and-rated")
                pairs += 1
        self._save()
        return {"ok": True, "pairs": pairs}

    def downloaded(self, job_id, version) -> None:
        e = self.store.edit(job_id, version)
        if not e or not self.enabled:
            return
        first = not e.get("downloaded")
        self.store.set_downloaded(e["id"])
        if first:
            for o in self.store.edits(job_id):
                if o["id"] != e["id"] and (o.get("rating") or 0) <= (e.get("rating") or 5):
                    self.model.update_pair(e["features"], o["features"], 0.5)
                    self.store.add_pair(e["id"], o["id"], 0.5, "download")
            self._save()

    def revision(self, job_id, version, text, brief) -> dict:
        deltas = parse(text)
        if not self.enabled:
            return deltas
        cat = category(brief)
        self.store.add_feedback(job_id, version, cat, text, deltas)
        if deltas:
            nd = dict(deltas)
            if cat != "general":
                nd.update({f"{cat}×{k}": v for k, v in deltas.items() if k in CONTEXT_CROSSED})
            self.model.nudge(nd, 0.6)
            self._save()
        return deltas

    def look_choice(self, job_id, version, old_look, new_look, brief):
        if not self.enabled or not new_look or new_look == old_look:
            return
        cat = category(brief)
        fo, fn = look_family(old_look or "clean"), look_family(new_look)
        if fo != fn:
            d = {f"lf_{fn}": 1.0, f"lf_{fo}": -1.0}
            if cat != "general":
                d.update({f"{cat}×lf_{fn}": 1.0, f"{cat}×lf_{fo}": -1.0})
            self.model.nudge(d, 0.4)
        self.store.add_feedback(job_id, version, cat, f"[look lab] {old_look} → {new_look}", {f"lf_{fn}": 1})
        self._save()

    # -- transparency --------------------------------------------------------------------
    def stats(self, brief: str = "") -> dict:
        c = self.store.counts()
        rated = [e for e in reversed(self.store.edits(limit=1000)) if e.get("rating")]
        first = [e["rating"] for e in rated[:10]]
        last = [e["rating"] for e in rated[-10:]]
        prof = self.profile(brief)
        return {
            "enabled": self.enabled, **c, "confidence": round(self.model.confidence, 2), "updates": self.model.n,
            "avg_first": round(float(np.mean(first)), 2) if first else None,
            "avg_last": round(float(np.mean(last)), 2) if len(rated) > 10 else None,
            "ratings": [e["rating"] for e in rated][-40:],
            "learned": [{"feature": k, "label": label(k), "weight": round(v, 3)} for k, v in self.model.top(14)],
            "house_style": prof["bullets"],
        }

    def export(self) -> str:
        """JSONL of the edits you liked (brief, category, features, summary) - a fine-tuning dataset."""
        lines = []
        for e in self._liked_edits():
            lines.append(json.dumps({"brief": e["brief"], "category": e["category"], "rating": e.get("rating"),
                                     "downloaded": bool(e.get("downloaded")), "job_id": e["job_id"],
                                     "version": e["version"], "features": e["features"], "summary": e["summary"],
                                     "created": time.strftime("%Y-%m-%d", time.localtime(e["created"]))}))
        return "\n".join(lines) + ("\n" if lines else "")
