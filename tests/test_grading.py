"""The colour presets must be visibly strong, distinct from each other, scale with strength, and keep skin sane."""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grading_metrics as M  # noqa: E402


def test_every_preset_is_visible_and_distinct():
    from app.grading import LOOK_PARAMS
    looks = [n for n in LOOK_PARAMS if n != "none"]
    base = M.lab(M.PTS)
    labs = {n: M.lab(M.graded(n, 1.0)) for n in looks}
    strength = {n: float(M.de2000(base, labs[n]).mean()) for n in looks}
    weakest = min(strength, key=strength.get)
    assert strength[weakest] >= 4.5, f"{weakest} barely changes the image (ΔE {strength[weakest]:.1f})"
    assert np.median(list(strength.values())) >= 9.0      # a CapCut-strength library, not 'barely visible'
    close = min((float(M.de2000(labs[a], labs[b]).mean()), a, b) for a, b in itertools.combinations(looks, 2))
    assert close[0] >= 4.5, f"{close[1]} and {close[2]} look the same (ΔE {close[0]:.1f})"


def test_strength_scales_the_look():
    base = M.lab(M.PTS)
    d = [float(M.de2000(base, M.lab(M.graded("teal_orange", k))).mean()) for k in (0.0, 0.5, 1.0, 1.5)]
    assert d[0] < 0.5 and d[0] < d[1] < d[2] < d[3]


def test_original_is_identity_and_skin_stays_skin():
    from app.grading import LOOK_CATEGORY, rgb_to_hsv
    out = M.graded("none", 1.0)
    assert np.abs(out - M.PTS).max() < 1e-3
    skin = M.PTS[:2]  # ColorChecker dark & light skin
    for n, cat in LOOK_CATEGORY.items():
        if cat in ("Life", "Film", "Warm", "Soft") and n != "none":
            h = rgb_to_hsv(np.clip(M.graded(n, 1.0)[:2], 0, 1))[:, 0]
            assert all(5 <= x <= 50 for x in h), (n, h, rgb_to_hsv(skin)[:, 0])
