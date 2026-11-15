"""Does the learner actually learn? A simulated user with a hidden taste rates every reel.

The 'AI editor' is simulated by random plans (random look, pace and transitions - like an LLM that
doesn't know you). Each round: plan -> learner improves it -> user rates it (from their hidden taste
+ noise) and sometimes types a revision request -> learner updates. We track the hidden utility of
what gets delivered vs. what the editor alone would have delivered.

Run:  python tests/simulate_learning.py  [rounds]
"""
from __future__ import annotations

import os
import random
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("REELSMITH_WORKSPACE", tempfile.mkdtemp(prefix="rs-sim-"))
os.environ["REELSMITH_NO_TRANSCRIBE"] = "1"

from app.grading import look_family, look_names  # noqa: E402
from app.learning import get_learner  # noqa: E402
from app.learning.features import plan_features  # noqa: E402
from app.planner import _finalize, heuristic_raw  # noqa: E402

BRIEF = "Travel reel of my trip to Lisbon"
TRANSITIONS = ["cut", "cut", "cut", "crossfade", "whip_right", "flash", "glitch", "zoom_punch", "light_leak"]


def hidden_utility(plan, analysis, options):
    f = plan_features(plan, analysis, options, BRIEF)
    fam = look_family(plan.get("look"))
    u = 2.0 * f["pace_short"] - 1.0 * f["avg_len"]
    u += {"warm": 1.2, "film": 0.9, "cinematic": 0.3, "cool": -1.0, "mono": -0.8, "neon": -0.6}.get(fam, 0.0)
    u -= 2.5 * f["t_glitch"] + 1.5 * f["t_flash"]
    u += 0.8 if f["grain"] >= 0.2 else -0.3
    return u


def user_feedback(plan, analysis, options):
    f = plan_features(plan, analysis, options, BRIEF)
    fam = look_family(plan.get("look"))
    wants = []
    if f["avg_len"] > 0.4:
        wants.append("faster")
    if fam in ("cool", "mono", "neon"):
        wants.append("warmer")
    if f["t_glitch"] > 0:
        wants.append("no glitch")
    if f["t_flash"] > 0:
        wants.append("no flash")
    if f["grain"] < 0.2:
        wants.append("more grain")
    return ", ".join(wants)


def random_plan(analysis, options, rng):
    raw = heuristic_raw(analysis, {"beat_interval": 0.5, "suggested_start": 0, "duration": 120, "beats": [],
                                   "downbeats": []} if False else None, options, rng.choice(
        ["fast hype", "slow calm", "cinematic", ""]) + " " + BRIEF, seed=rng.randint(1, 10 ** 6))
    raw["look"] = rng.choice(look_names())
    raw["grain"] = rng.choice([0.0, 0.0, 0.15, 0.35])
    for s in raw["segments"][1:]:
        s["transition_in"] = rng.choice(TRANSITIONS)
    return raw


def build_analysis():
    from pathlib import Path
    from app.analyze import analyze_clip, score_shots
    from app.video_io import probe
    clips, shots = [], []
    tmp = Path(tempfile.mkdtemp())
    for i, p in enumerate(sys.argv[2:] or ["/tmp/rs_test/land.mp4", "/tmp/rs_test/track.mp4"]):
        c, s = analyze_clip(probe(p), f"c{i}", None, tmp)
        clips.append(c)
        shots += s
    score_shots(shots)
    return {"clips": clips, "all_shots": shots, "shots": shots, "sheets": [], "transcripts": {}}


def run(rounds=40, seed=7, analysis=None, verbose=True):
    rng = random.Random(seed)
    analysis = analysis or build_analysis()
    options = {"target_duration": 12, "beat_sync": False, "auto_pick": True}
    learner = get_learner()
    learner.reset()
    learner._rng.seed(seed)
    hist = []
    for r in range(rounds):
        raw = random_plan(analysis, options, rng)
        ed = {"provider": "sim", "label": "simulated editor", "model": None}
        # what the editor alone would deliver (learning off)
        learner.set_enabled(False)
        alone = _finalize(raw, analysis, None, options, BRIEF, "full", ed)
        learner.set_enabled(True)
        delivered = _finalize(raw, analysis, None, options, BRIEF, "full", ed)
        u_alone, u_del = hidden_utility(alone, analysis, options), hidden_utility(delivered, analysis, options)
        stars = max(1, min(5, round(3 + 0.9 * u_del + rng.gauss(0, 0.6))))
        job = f"sim{r}"
        learner.record_edit(job, 1, delivered, analysis, options, BRIEF, "new")
        learner.rate(job, 1, stars)
        if r % 3 == 0:
            text = user_feedback(delivered, analysis, options)
            if text:
                learner.revision(job, 1, text, BRIEF)
        hist.append((u_alone, u_del, stars))
        if verbose:
            lr = delivered.get("learning", {})
            print(f"round {r + 1:2d}: editor alone {u_alone:+.2f} → delivered {u_del:+.2f}  {stars}★  "
                  f"look={delivered['look']:<14} conf={lr.get('confidence', 0):.2f}  "
                  f"applied={', '.join(lr.get('applied', [])) or '-'}")
    return hist, learner


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    hist, learner = run(n)
    k = max(5, n // 4)
    a0 = sum(h[0] for h in hist) / len(hist)
    first = sum(h[1] for h in hist[:k]) / k
    last = sum(h[1] for h in hist[-k:]) / k
    s_first = sum(h[2] for h in hist[:k]) / k
    s_last = sum(h[2] for h in hist[-k:]) / k
    print(f"\nhidden utility: editor alone avg {a0:+.2f} | delivered first {k}: {first:+.2f} → last {k}: {last:+.2f}")
    print(f"stars: first {k}: {s_first:.2f} → last {k}: {s_last:.2f}")
    print("learned house style:", learner.profile(BRIEF)["bullets"])
