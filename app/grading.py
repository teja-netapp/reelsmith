"""Color engine: per-shot balance -> creative look, baked into a 3D LUT that FFmpeg applies (lut3d).

Pipeline per shot (what a colorist does):
  1. Balance   - per-shot white balance + exposure, measured during analysis (conservative, so a
                 golden-hour shot stays golden but mismatched shots stop jumping between casts).
  2. Look      - filmic grade: white balance, exposure, highlight shoulder, contrast S-curve,
                 fade/black point, lift/gamma/gain (CDL), split toning, saturation + vibrance,
                 hue-selective tweaks (greens->teal, deeper skies...) with skin-tone protection,
                 monochrome mixes.
  3. Texture   - halation, bloom, grain, vignette (applied per frame in effects.FinishingPass).

Steps 1+2 are evaluated on a 33x33x33 grid and written as a .cube file per (look, shot balance).
Your own .cube LUTs (from any grading app) dropped into assets/luts/ show up as looks "lut:<name>".
"""
from __future__ import annotations

import functools
import math
import os
from pathlib import Path

import numpy as np

from .config import ROOT

LUT_DIR = ROOT / "assets" / "luts"
CUBE_SIZE = 33
LUMA = np.array([0.2126, 0.7152, 0.0722])

# ----------------------------------------------------------------------------------------------
# Look library (CapCut-style presets, grouped by category)
# hue angles in degrees (0 red, 30 orange/skin, 60 yellow, 120 green, 180 cyan, 210 sky, 240 blue,
# 300 magenta). tints are (hue, amount). hue_adj bands: (center, width, hue_shift, sat_mult, lum_shift).
# curves: {"rgb"|"r"|"g"|"b": [(x, y), ...]} monotone tone curves (0..1). mix: 3x3 channel mixer (rows
# sum to 1 keep neutrals neutral).
# Every preset is measured (tests/test_grading.py): visibly strong (CIEDE2000 vs the original) and
# distinct from every other preset.
# ----------------------------------------------------------------------------------------------
DEFAULTS = dict(temp=0.0, tint=0.0, exposure=0.0, contrast=1.0, pivot=0.45, toe=0.0, shoulder=0.0,
                black=0.0, white=1.0, sat=1.0, vibrance=0.0, lift=(0, 0, 0), gamma=(1, 1, 1), gain=(1, 1, 1),
                shadow_tint=(0, 0.0), highlight_tint=(0, 0.0), hue_adj=(), skin_protect=0.7, tint_skin_protect=0.6,
                mono=False, mono_mix=(0.30, 0.59, 0.11), mono_tone=(0, 0.0), curves=None, mix=None,
                halation=0.0, bloom=0.0, grain=0.0, vignette=0.0)

# name: (label, category, description, params)
_LOOKS: dict[str, tuple[str, str, str, dict]] = {
    'none': ('Original', 'Life', 'no grade: original colours',
        dict()),
    # --- Life ------------------------------------------------------------------------------------
    'clean': ('Natural', 'Life', 'natural polish: richer contrast and colour, true skin',
        dict(contrast=1.16, vibrance=0.4, sat=1.04, temp=0.03, shoulder=0.2, curves={'rgb': [(0, 0), (0.25,
             0.215), (0.75, 0.795), (1, 1)]})),
    'fresh': ('Fresh', 'Life', 'crisp, bright and cool-clean: vivid blues and greens',
        dict(exposure=0.24, temp=-0.3, contrast=1.14, vibrance=0.6, sat=1.1, shoulder=0.4, hue_adj=((210, 35,
             -4, 1.45, 0.0), (120, 40, 6, 1.4, 0.05)))),
    'sunny': ('Sunny', 'Life', 'warm daylight: bright, golden and happy',
        dict(exposure=0.26, temp=0.42, tint=-0.06, contrast=1.04, sat=1.12, vibrance=0.25, shoulder=0.45,
             highlight_tint=(55, 0.14), black=0.02)),
    'food': ('Food', 'Life', 'appetising: warm, rich reds, oranges and creamy highlights',
        dict(temp=0.3, tint=0.06, contrast=1.28, sat=1.28, vibrance=0.25, shoulder=0.25, exposure=-0.04,
             toe=0.2, hue_adj=((15, 25, -2, 1.4, -0.02), (40, 20, -4, 1.3, 0.02), (120, 35, -6, 1.15, 0.0)))),
    'portrait': ('Portrait', 'Life', 'flattering skin: soft contrast, warm glow, gentle pink highlights',
        dict(temp=0.26, tint=0.12, exposure=0.12, contrast=0.86, black=0.06, sat=0.9, shoulder=0.45,
             highlight_tint=(10, 0.1), hue_adj=((28, 18, -3, 0.88, 0.07),), bloom=0.15, skin_protect=0.0)),
    # --- Vibrant ---------------------------------------------------------------------------------
    'vibrant': ('Vivid', 'Vibrant', 'bright, colourful, social-first',
        dict(contrast=1.14, sat=1.55, vibrance=0.9, shoulder=0.35, exposure=0.12, temp=0.08)),
    'pop': ('Pop', 'Vibrant', 'maximum punch: deep blacks, saturated everything',
        dict(contrast=1.5, toe=0.6, sat=1.6, vibrance=0.5, shoulder=0.2, exposure=-0.06, temp=-0.04,
             curves={'rgb': [(0, 0), (0.2, 0.12), (0.8, 0.88), (1, 1)]})),
    'tropical': ('Tropical', 'Vibrant', 'turquoise water, lush greens, warm skin',
        dict(contrast=1.14, sat=1.22, vibrance=0.4, temp=0.08, exposure=0.06, hue_adj=((205, 30, -10, 1.2, 0.03),
             (180, 25, -6, 1.45, 0.06), (120, 40, -16, 1.5, 0.04), (60, 25, -8, 1.25, 0.0)))),
    'candy': ('Candy', 'Vibrant', 'pink-magenta sweetness, lifted and bright',
        dict(tint_skin_protect=0.0, tint=0.42, exposure=0.12, black=0.06, contrast=1.0, sat=1.25, highlight_tint=(330, 0.16),
             shadow_tint=(280, 0.1), shoulder=0.3)),
    # --- Warm ------------------------------------------------------------------------------------
    'golden_hour': ('Golden Hour', 'Warm', 'sunset warmth, glowing orange highlights',
        dict(temp=0.72, exposure=0.08, contrast=1.08, sat=1.16, vibrance=0.2, shoulder=0.35, highlight_tint=(34,
             0.2), shadow_tint=(15, 0.08), bloom=0.3)),
    'desert_heat': ('Desert', 'Warm', 'sun-baked: orange earth, dusty teal skies',
        dict(temp=0.6, tint=0.08, contrast=1.24, sat=1.0, highlight_tint=(38, 0.12), shadow_tint=(25, 0.1),
             hue_adj=((210, 40, -28, 0.5, 0.03), (120, 40, -45, 0.55, 0.0)), bloom=0.15)),
    'autumn': ('Autumn', 'Warm', 'fall colours: greens turned gold, rich reds',
        dict(temp=0.34, contrast=1.16, sat=1.18, hue_adj=((120, 45, -50, 1.1, -0.02), (80, 30, -28, 1.3, 0.0),
             (10, 25, 0, 1.3, -0.03)), shadow_tint=(20, 0.08))),
    # --- Film ------------------------------------------------------------------------------------
    'kodak_2383': ('Kodak 2383', 'Film', 'print film: dense contrast, cyan shadows, warm highlights, halation',
        dict(temp=0.18, contrast=1.55, pivot=0.42, toe=0.6, shoulder=0.35, sat=0.9, shadow_tint=(195, 0.24),
             highlight_tint=(40, 0.16), hue_adj=((120, 35, 30, 0.75, -0.1), (0, 25, 6, 1.12, -0.06)),
             halation=0.35, grain=0.28)),
    'portra': ('Portra', 'Film', 'Portra 400: warm creamy skin, pastel greens, soft contrast',
        dict(temp=0.3, contrast=0.88, black=0.07, white=0.96, sat=0.84, shoulder=0.45, highlight_tint=(45, 0.1),
             shadow_tint=(200, 0.06), hue_adj=((120, 40, -22, 0.55, 0.06), (210, 35, -14, 0.7, 0.05), (28, 18,
             0, 1.08, 0.03)), grain=0.22)),
    'fuji_eterna': ('Eterna', 'Film', 'Fuji Eterna cinema stock: low contrast, muted, green-cyan',
        dict(contrast=0.8, black=0.07, sat=0.58, temp=-0.12, tint=-0.28, shoulder=0.4, shadow_tint=(170, 0.14),
             hue_adj=((120, 40, 16, 0.75, 0.0), (30, 25, -3, 0.88, 0.0)), grain=0.2)),
    'warm_film': ('Warm Film', 'Film', 'faded warm negative: soft blacks, amber cast, halation',
        dict(temp=0.58, contrast=1.06, black=0.07, white=0.95, sat=0.85, shoulder=0.4, gamma=(1.0, 1.0, 0.9),
             highlight_tint=(35, 0.14), shadow_tint=(15, 0.08), halation=0.3, grain=0.28)),
    'cinestill': ('CineStill 800T', 'Film', 'tungsten night film: cool cast, red glow around lights',
        dict(temp=-0.7, tint=0.08, contrast=1.18, sat=1.08, highlight_tint=(15, 0.14), shadow_tint=(195, 0.18),
             hue_adj=((0, 30, 0, 1.35, 0.0),), halation=0.7, grain=0.3)),
    # --- Movies ----------------------------------------------------------------------------------
    'teal_orange': ('Teal & Orange', 'Movies', 'blockbuster: teal shadows and greens, warm skin and highlights',
        dict(temp=0.1, contrast=1.24, sat=1.1, vibrance=0.1, shoulder=0.3, shadow_tint=(188, 0.32),
             highlight_tint=(30, 0.2), hue_adj=((120, 45, 60, 0.7, -0.05), (210, 30, -16, 1.25, -0.05), (30, 18,
             2, 1.15, 0.0)))),
    'cinematic_cool': ('Blue Hour', 'Movies', 'cool cinematic night / blue hour',
        dict(temp=-0.55, tint=-0.04, contrast=1.3, sat=0.85, toe=0.35, exposure=-0.06, shadow_tint=(195, 0.24),
             highlight_tint=(190, 0.06))),
    'matrix': ('Matrix', 'Movies', 'digital green cast, cold and hard',
        dict(tint_skin_protect=0.0, tint=-0.7, temp=-0.3, contrast=1.3, sat=0.68, toe=0.3, mix=((0.8, 0.2, 0.0), (0.05, 0.95, 0.0),
             (0.0, 0.25, 0.75)), shadow_tint=(150, 0.14))),
    'storybook': ('Storybook', 'Movies', 'whimsical pastel: warm yellows and pinks, lifted, symmetrical-film feel',
        dict(temp=0.3, tint=0.16, black=0.09, contrast=0.9, sat=1.1, highlight_tint=(55, 0.12),
             shadow_tint=(340, 0.1), hue_adj=((0, 30, 8, 1.2, 0.05), (60, 25, -5, 1.25, 0.05), (120, 40, -24,
             0.65, 0.04)))),
    'thriller': ('Thriller', 'Movies', 'crime-drama: sickly yellow-green highs, teal shadows, desaturated',
        dict(tint_skin_protect=0.0, sat=0.58, contrast=1.32, toe=0.3, temp=0.22, tint=-0.25, shadow_tint=(175, 0.16),
             highlight_tint=(68, 0.2), exposure=-0.06)),
    # --- Retro -----------------------------------------------------------------------------------
    'vintage': ('70s', 'Retro', 'faded 70s print: lifted blacks, teal shadows, yellow highs, grain',
        dict(black=0.15, white=0.92, contrast=0.9, sat=0.65, temp=0.3, tint=-0.1, gamma=(1.0, 1.02, 0.86),
             shadow_tint=(175, 0.18), highlight_tint=(55, 0.22), grain=0.4, halation=0.2, vignette=0.35)),
    'polaroid': ('Polaroid', 'Retro', 'instant film: cyan-green shadows, creamy warm highlights, faded',
        dict(black=0.1, white=0.95, contrast=0.86, sat=0.8, exposure=0.1, temp=-0.08, curves={'r': [(0, 0.03),
             (0.5, 0.5), (1, 0.98)], 'g': [(0, 0.09), (0.5, 0.52), (1, 0.97)], 'b': [(0, 0.2), (0.5, 0.5), (1,
             0.86)]}, shadow_tint=(170, 0.14), highlight_tint=(40, 0.06), vignette=0.25)),
    'retro_80s': ('80s', 'Retro', 'synthwave video: magenta highs, cyan shadows, soft and glowy',
        dict(tint_skin_protect=0.0, black=0.08, contrast=1.05, sat=1.2, shadow_tint=(190, 0.24), highlight_tint=(315, 0.22), tint=0.16,
             bloom=0.3, grain=0.15)),
    'cross_process': ('Cross Process', 'Retro', 'cross-processed slide: blue shadows, yellow-green highs, punchy',
        dict(contrast=1.25, sat=1.3, curves={'r': [(0, 0), (0.25, 0.18), (0.75, 0.85), (1, 1)], 'g': [(0, 0.02),
             (0.5, 0.55), (1, 1)], 'b': [(0, 0.2), (0.5, 0.45), (1, 0.7)]})),
    'sepia': ('Sepia', 'Retro', 'warm sepia monochrome, aged print',
        dict(mono=True, mono_tone=(35, 0.09), black=0.05, white=0.95, contrast=1.1, grain=0.3, vignette=0.4)),
    # --- Night -----------------------------------------------------------------------------------
    'neon_night': ('Neon', 'Night', 'night-city neon: indigo shadows, magenta and cyan pop, glow',
        dict(tint_skin_protect=0.0, contrast=1.22, sat=1.35, vibrance=0.2, temp=-0.2, shadow_tint=(255, 0.24), highlight_tint=(320,
             0.06), hue_adj=((300, 40, 0, 1.5, 0.06), (190, 30, 0, 1.45, 0.04)), bloom=0.4)),
    'cyberpunk': ('Cyberpunk', 'Night', 'teal-magenta split, electric and saturated',
        dict(tint_skin_protect=0.0, contrast=1.26, sat=1.32, shadow_tint=(185, 0.3), highlight_tint=(305, 0.24), tint=0.18,
             hue_adj=((30, 20, -10, 0.9, 0.0),), bloom=0.3, skin_protect=0.3)),
    'midnight': ('Midnight', 'Night', 'deep blue night, dark and cold',
        dict(tint_skin_protect=0.0, exposure=-0.5, temp=-0.85, tint=0.05, contrast=1.25, sat=0.7, toe=0.45, shadow_tint=(235, 0.2),
             vignette=0.45)),
    # --- Mood ------------------------------------------------------------------------------------
    'moody': ('Moody', 'Mood', 'dark, desaturated, cold shadows',
        dict(exposure=-0.38, contrast=1.36, sat=0.52, temp=-0.2, black=0.02, toe=0.35, shadow_tint=(205, 0.18),
             hue_adj=((120, 40, 25, 0.45, -0.14), (210, 40, 0, 0.65, -0.12)), vignette=0.5, grain=0.2)),
    'emerald_moody': ('Emerald', 'Mood', 'dark forest: deep emerald greens, moody',
        dict(exposure=-0.22, contrast=1.24, sat=0.95, temp=-0.1, tint=-0.18, shadow_tint=(160, 0.2),
             hue_adj=((120, 45, 40, 1.2, -0.16), (60, 25, 35, 0.85, -0.1)), vignette=0.4, grain=0.15)),
    'bleach_bypass': ('Bleach Bypass', 'Mood', 'desaturated, harsh silvery contrast',
        dict(sat=0.3, contrast=1.62, toe=0.4, shoulder=0.1, temp=-0.08, grain=0.2)),
    'cool_blue': ('Nordic', 'Mood', 'clean, cold, desaturated Scandinavian blues',
        dict(temp=-0.7, sat=0.7, contrast=1.05, exposure=0.08, black=0.04, shadow_tint=(210, 0.16),
             hue_adj=((30, 20, 0, 0.8, 0.0),))),
    # --- Soft ------------------------------------------------------------------------------------
    'high_key': ('Airy', 'Soft', 'bright and airy: lifted, soft highlights, gentle colour',
        dict(exposure=0.62, contrast=0.8, black=0.05, sat=0.74, vibrance=0.1, temp=0.04, shoulder=0.6,
             highlight_tint=(200, 0.04), bloom=0.25)),
    'matte_pastel': ('Pastel', 'Soft', 'matte pastel: lifted blacks, soft pink and mint tones',
        dict(exposure=0.14, black=0.16, white=0.94, contrast=0.82, sat=0.66, highlight_tint=(330, 0.11),
             shadow_tint=(165, 0.12))),
    'dreamy': ('Dreamy', 'Soft', 'soft glow, lavender highlights, lifted and hazy',
        dict(exposure=0.16, black=0.1, contrast=0.82, sat=0.92, tint=0.16, highlight_tint=(285, 0.14),
             shadow_tint=(240, 0.09), bloom=0.55, halation=0.15)),
    # --- B&W -------------------------------------------------------------------------------------
    'noir': ('Noir', 'B&W', 'high-contrast black & white',
        dict(mono=True, mono_mix=(0.35, 0.55, 0.1), contrast=1.45, toe=0.35, shoulder=0.2, vignette=0.45,
             grain=0.35)),
    'mono_soft': ('Soft B&W', 'B&W', 'gentle black & white, lifted and silky',
        dict(mono=True, mono_mix=(0.3, 0.59, 0.11), contrast=0.85, black=0.08, white=0.96, exposure=0.1,
             grain=0.15)),
    'silver': ('Silver', 'B&W', 'bright, cool-toned silver monochrome',
        dict(mono=True, mono_mix=(0.25, 0.6, 0.15), mono_tone=(210, 0.06), exposure=0.2, contrast=1.15,
             shoulder=0.4)),
}
LOOK_PARAMS: dict[str, dict] = {k: v[3] for k, v in _LOOKS.items()}
LOOK_LABELS: dict[str, str] = {k: v[0] for k, v in _LOOKS.items()}
LOOK_CATEGORY: dict[str, str] = {k: v[1] for k, v in _LOOKS.items()}
LOOK_DESCRIPTIONS_BUILTIN: dict[str, str] = {k: v[2] for k, v in _LOOKS.items()}
LOOK_CATEGORIES = ["Life", "Vibrant", "Warm", "Film", "Movies", "Retro", "Night", "Mood", "Soft", "B&W"]

# families help the learner generalize ("you like warm looks for travel")
LOOK_FAMILY = {
    "none": "clean", "clean": "clean", "portrait": "clean", "fresh": "bright", "sunny": "warm", "food": "warm",
    "vibrant": "vibrant", "pop": "vibrant", "tropical": "vibrant", "candy": "vibrant",
    "golden_hour": "warm", "desert_heat": "warm", "autumn": "warm",
    "kodak_2383": "film", "portra": "film", "fuji_eterna": "film", "warm_film": "film", "cinestill": "film",
    "teal_orange": "cinematic", "cinematic_cool": "cool", "matrix": "cinematic", "storybook": "cinematic",
    "thriller": "cinematic",
    "vintage": "film", "polaroid": "film", "retro_80s": "vibrant", "cross_process": "vibrant", "sepia": "mono",
    "neon_night": "neon", "cyberpunk": "neon", "midnight": "cool",
    "moody": "dark", "emerald_moody": "dark", "bleach_bypass": "dark", "cool_blue": "cool",
    "high_key": "bright", "matte_pastel": "bright", "dreamy": "bright",
    "noir": "mono", "mono_soft": "mono", "silver": "mono",
}
FAMILIES = sorted(set(LOOK_FAMILY.values()) | {"custom"})


# ----------------------------------------------------------------------------------------------
# User LUTs
# ----------------------------------------------------------------------------------------------
def user_luts() -> dict[str, Path]:
    if not LUT_DIR.exists():
        return {}
    out = {}
    for p in sorted(LUT_DIR.glob("*.cube")):
        name = "lut:" + "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in p.stem)[:40]
        out[name] = p
    return out


def look_names() -> list[str]:
    return list(LOOK_PARAMS) + list(user_luts())


def look_descriptions() -> dict[str, str]:
    d = dict(LOOK_DESCRIPTIONS_BUILTIN)
    for name, p in user_luts().items():
        d[name] = f"your LUT '{p.stem}'"
    return d


def look_label(name: str) -> str:
    if name.startswith("lut:"):
        return name[4:].replace("_", " ")
    return LOOK_LABELS.get(name, name.replace("_", " ").title())


def look_category(name: str) -> str:
    return "Your LUTs" if name.startswith("lut:") else LOOK_CATEGORY.get(name, "Life")


def look_family(name: str) -> str:
    return LOOK_FAMILY.get(name, "custom")


def look_texture(name: str) -> dict:
    """Default halation / bloom / grain / vignette for a look."""
    p = {**DEFAULTS, **LOOK_PARAMS.get(name, {})}
    return {k: p[k] for k in ("halation", "bloom", "grain", "vignette")}


@functools.lru_cache(maxsize=16)
def _load_cube_cached(path: str, mtime: float) -> np.ndarray:
    return load_cube(path)


def load_cube(path: str | os.PathLike) -> np.ndarray:
    """Parse a .cube 3D LUT -> array [b][g][r] -> RGB in 0..1."""
    size, lo, hi, rows = None, np.zeros(3), np.ones(3), []
    with open(path, "r", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            up = ln.upper()
            if up.startswith("LUT_3D_SIZE"):
                size = int(ln.split()[1])
            elif up.startswith("LUT_1D_SIZE"):
                raise ValueError("1D LUTs are not supported - export a 3D .cube")
            elif up.startswith("DOMAIN_MIN"):
                lo = np.array([float(x) for x in ln.split()[1:4]])
            elif up.startswith("DOMAIN_MAX"):
                hi = np.array([float(x) for x in ln.split()[1:4]])
            elif up[0].isdigit() or up[0] in "-.":
                parts = ln.split()
                if len(parts) >= 3:
                    rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not size or len(rows) != size ** 3:
        raise ValueError(f"bad .cube file {path}")
    arr = (np.array(rows, np.float64) - lo) / np.maximum(hi - lo, 1e-9)
    return np.clip(arr, 0, 1).reshape(size, size, size, 3)


def write_cube(path: str | os.PathLike, cube: np.ndarray, title: str = "Reelsmith"):
    n = cube.shape[0]
    flat = np.clip(cube.reshape(-1, 3), 0, 1)
    with open(path, "w") as f:
        f.write(f'TITLE "{title}"\nLUT_3D_SIZE {n}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n')
        np.savetxt(f, flat, fmt="%.6f")


# ----------------------------------------------------------------------------------------------
# Color math (vectorised over N x 3 arrays, RGB in 0..1)
# ----------------------------------------------------------------------------------------------
def _hue_rgb(h_deg: float) -> np.ndarray:
    """Luma-neutral chroma direction for a hue (adding it shifts color without changing brightness)."""
    h = (h_deg % 360) / 60.0
    c = np.clip(np.array([abs(h - 3) - 1, 2 - abs(h - 2), 2 - abs(h - 4)]), 0, 1)
    return c - float(c @ LUMA)


def rgb_to_hsv(rgb):
    mx, mn = rgb.max(1), rgb.min(1)
    d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-9
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    i = m & (mx == r)
    h[i] = ((g[i] - b[i]) / d[i]) % 6
    i = m & (mx == g) & ~(mx == r)
    h[i] = (b[i] - r[i]) / d[i] + 2
    i = m & (mx == b) & ~(mx == r) & ~(mx == g)
    h[i] = (r[i] - g[i]) / d[i] + 4
    s = np.where(mx > 1e-9, d / np.maximum(mx, 1e-9), 0)
    return np.stack([h * 60.0, s, mx], 1)


def hsv_to_rgb(hsv):
    h, s, v = (hsv[:, 0] % 360) / 60.0, np.clip(hsv[:, 1], 0, 1), hsv[:, 2]
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    out = np.empty((len(h), 3))
    for k, (a, b, c) in enumerate([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]):
        m = i == k
        out[m, 0], out[m, 1], out[m, 2] = a[m], b[m], c[m]
    return out


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _scurve(x, c, pivot):
    if abs(c - 1.0) < 1e-4:
        return x
    x = np.clip(x, 0, 1)
    lo = pivot * np.power(np.maximum(x, 0) / pivot, c)
    hi = 1 - (1 - pivot) * np.power(np.maximum(1 - x, 0) / (1 - pivot), c)
    return np.where(x < pivot, lo, hi)


def _gamut_fit(L: np.ndarray, chroma: np.ndarray) -> np.ndarray:
    """L + chroma, with chroma scaled down (never hue-shifted) just enough to stay inside 0..1:
    strong saturation without the flat, posterised patches that hard clipping makes."""
    Lc = np.clip(L, 0, 1)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        up = np.where(chroma > 1e-9, (1 - Lc) / chroma, np.inf)
        dn = np.where(chroma < -1e-9, (0 - Lc) / chroma, np.inf)
    t = np.minimum(np.min(up, 1), np.min(dn, 1))
    t = np.clip(t, 0, 1)
    # soft knee: start easing at 85% of the boundary
    soft = np.where(t < 1, t, 1.0)
    return Lc + chroma * soft[:, None]


def _curve(x: np.ndarray, pts) -> np.ndarray:
    """Monotone cubic (Fritsch-Carlson) tone curve through control points [(x, y), ...]."""
    P = np.array(sorted(pts), float)
    xs, ys = P[:, 0], P[:, 1]
    if len(xs) == 2:
        return np.interp(np.clip(x, 0, 1), xs, ys)
    h = np.diff(xs)
    d = np.diff(ys) / h
    m = np.zeros(len(xs))
    m[0], m[-1] = d[0], d[-1]
    for i in range(1, len(xs) - 1):
        m[i] = 0.0 if d[i - 1] * d[i] <= 0 else 3 * (h[i - 1] + h[i]) / ((2 * h[i] + h[i - 1]) / d[i - 1] +
                                                                           (h[i] + 2 * h[i - 1]) / d[i])
    xc = np.clip(x, 0, 1)
    k = np.clip(np.searchsorted(xs, xc) - 1, 0, len(xs) - 2)
    t = (xc - xs[k]) / h[k]
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * ys[k] + (t3 - 2 * t2 + t) * h[k] * m[k] + (-2 * t3 + 3 * t2) * ys[k + 1]
            + (t3 - t2) * h[k] * m[k + 1])


def _shoulder(x: np.ndarray, s: float) -> np.ndarray:
    """Soft-clip above a knee: y = k + a(1 - e^{-(x-k)/a}), with a solved so the brightest value maps to 1.
    Identity when nothing exceeds 1, so it only ever recovers blown highlights."""
    xm = float(x.max())
    if xm <= 1.0 + 1e-6:
        return x
    k = 1.0 - 0.5 * max(0.2, min(1.0, s if s > 0 else 0.3))
    lo, hi = 1e-4, 50.0
    for _ in range(60):  # solve k + a(1 - exp(-(xm-k)/a)) = 1
        a = (lo + hi) / 2
        if k + a * (1 - math.exp(-(xm - k) / a)) > 1:
            hi = a
        else:
            lo = a
    a = (lo + hi) / 2
    return np.where(x > k, k + a * (1 - np.exp(-(x - k) / a)), x)


def balance(rgb: np.ndarray, correction: dict | None) -> np.ndarray:
    """Per-shot white balance (linear-light gains) + exposure."""
    if not correction:
        return rgb
    lin = np.power(np.clip(rgb, 0, 1), 2.2)
    g = np.asarray(correction.get("wb", (1, 1, 1)), np.float64)
    lin = lin * g * (2.0 ** float(correction.get("ev", 0.0)))
    return np.power(np.clip(lin, 0, None), 1 / 2.2)


def apply_look_params(rgb: np.ndarray, p: dict) -> np.ndarray:
    p = {**DEFAULTS, **p}
    lin = np.power(np.clip(rgb, 0, 1), 2.2)
    # white balance & exposure in linear light
    t, ti = p["temp"], p["tint"]   # temp +1 = very warm (~+3000 K feel), tint +1 = magenta, -1 = green
    wb = np.array([1 + 0.32 * t + 0.06 * ti, 1 - 0.22 * ti, 1 - 0.38 * t + 0.06 * ti])
    lin = lin * (wb / float(wb @ LUMA)) * (2.0 ** p["exposure"])
    x = np.power(np.clip(lin, 0, None), 1 / 2.2)
    if p["mix"] is not None:  # channel mixer (rows sum to 1 keep neutrals neutral)
        x = x @ np.asarray(p["mix"], float).T
    # highlight shoulder: values pushed past white roll off smoothly instead of clipping (slope 1 at
    # the knee, so mid-tones are untouched); then contrast S-curve, toe and fade
    x = _shoulder(x, p["shoulder"])
    x = _scurve(np.clip(x, 0, 1), p["contrast"], p["pivot"])
    if p["toe"] > 0:
        b = 0.05 * p["toe"]
        x = np.clip((x - b) / (1 - b), 0, 1)
    if p["curves"]:  # tone curves: master first, then per channel (how most app filters are built)
        cv = p["curves"]
        if "rgb" in cv:
            x = _curve(x, cv["rgb"])
        for i, ch in enumerate("rgb"):
            if ch in cv:
                x[:, i] = _curve(x[:, i], cv[ch])
    x = p["black"] + (p["white"] - p["black"]) * x
    # CDL-style lift / gamma / gain per channel
    lift, gam, gain = np.array(p["lift"], float), np.array(p["gamma"], float), np.array(p["gain"], float)
    x = gain * (x + lift * (1 - x))
    x = np.power(np.clip(x, 0, None), 1 / np.maximum(gam, 1e-3))
    # split toning (skin keeps most of its natural hue unless the look is meant to be stylised)
    L = x @ LUMA
    keep = 1.0
    if p["tint_skin_protect"] and (p["shadow_tint"][1] or p["highlight_tint"][1]):
        hsv0 = rgb_to_hsv(np.clip(x, 0, 1))
        skin0 = np.exp(-0.5 * (((hsv0[:, 0] - 22 + 180) % 360 - 180) / 16) ** 2) * _smooth(0.12, 0.35, hsv0[:, 1])
        keep = (1 - p["tint_skin_protect"] * skin0)[:, None]
    for (hue, amt), w in ((p["shadow_tint"], _smooth(0.55, 0.0, L)), (p["highlight_tint"], _smooth(0.45, 1.0, L))):
        if amt:
            x = x + amt * w[:, None] * _hue_rgb(hue)[None, :] * keep
    # saturation + vibrance (protects already-saturated colors)
    L = x @ LUMA
    chroma = x - L[:, None]
    s_cur = np.clip(x.max(1) - x.min(1), 0, 1)
    fac = p["sat"] * (1 + p["vibrance"] * (1 - s_cur))
    x = _gamut_fit(L, chroma * fac[:, None])
    # hue-selective adjustments with skin protection
    if p["hue_adj"]:
        hsv = rgb_to_hsv(np.clip(x, 0, 1))
        h = hsv[:, 0]
        skin = np.exp(-0.5 * (((h - 25 + 180) % 360 - 180) / 14) ** 2) * _smooth(0.1, 0.35, hsv[:, 1])
        for center, width, dh, sm, dl in p["hue_adj"]:
            dist = (h - center + 180) % 360 - 180
            w = np.exp(-0.5 * (dist / width) ** 2) * _smooth(0.02, 0.15, hsv[:, 1])
            if not (10 <= center <= 45):
                w = w * (1 - p["skin_protect"] * skin)
            hsv[:, 0] = hsv[:, 0] + w * dh
            hsv[:, 1] = hsv[:, 1] * (1 + w * (sm - 1))
            hsv[:, 2] = hsv[:, 2] * (1 + w * dl)
        x = hsv_to_rgb(hsv)
    if p["mono"]:
        m = np.clip(x, 0, 1) @ np.array(p["mono_mix"], float)
        x = np.repeat(m[:, None], 3, 1)
        hue, amt = p["mono_tone"]
        if amt:
            x = x + amt * (_smooth(0.0, 0.5, m) * _smooth(1.0, 0.6, m))[:, None] * _hue_rgb(hue)[None, :] * 2
    return np.clip(x, 0, 1)


def _grid(n=CUBE_SIZE):
    v = np.linspace(0, 1, n)
    b, g, r = np.meshgrid(v, v, v, indexing="ij")
    return np.stack([r, g, b], -1).reshape(-1, 3)


def sample_cube(cube: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Trilinear lookup of N x 3 RGB points (0..1) in a [b][g][r] cube."""
    n = cube.shape[0]
    f = np.clip(rgb, 0, 1) * (n - 1)
    i0 = np.clip(np.floor(f).astype(int), 0, n - 2)
    t = f - i0
    out = np.zeros_like(rgb, dtype=np.float64)
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                w = ((t[:, 0] if dr else 1 - t[:, 0]) * (t[:, 1] if dg else 1 - t[:, 1]) *
                     (t[:, 2] if db else 1 - t[:, 2]))
                out += w[:, None] * cube[i0[:, 2] + db, i0[:, 1] + dg, i0[:, 0] + dr]
    return out


def build_cube(look: str, intensity: float = 1.0, correction: dict | None = None, n: int = CUBE_SIZE) -> np.ndarray:
    """Balance (full strength) followed by the look (blended by intensity), as an n^3 cube."""
    grid = _grid(n)
    base = balance(grid, correction)
    luts = user_luts()
    if look in luts:
        p = luts[look]
        graded = sample_cube(_load_cube_cached(str(p), p.stat().st_mtime), base)
    elif look in LOOK_PARAMS and LOOK_PARAMS[look]:
        graded = apply_look_params(base, LOOK_PARAMS[look])
    else:
        graded = base
    k = float(np.clip(intensity, 0, 1.5))
    out = np.clip(base + k * (graded - base), 0, 1)
    return out.reshape(n, n, n, 3)


def cube_file(look: str, intensity: float, correction: dict | None, out_dir: Path) -> str:
    """Write (or reuse) the cube for this look+balance; returns the file NAME inside out_dir."""
    corr = None
    if correction:
        corr = {"wb": [round(float(v), 3) for v in correction.get("wb", (1, 1, 1))],
                "ev": round(float(correction.get("ev", 0.0)), 3)}
    key = f"{look}|{round(float(intensity), 3)}|{corr}"
    import hashlib

    name = "grade_" + hashlib.sha1(key.encode()).hexdigest()[:12] + ".cube"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (out_dir / name).exists():
        write_cube(out_dir / name, build_cube(look, intensity, corr), title=look)
    return name


def apply_to_image(img_bgr: np.ndarray, look: str, intensity: float = 1.0, correction: dict | None = None,
                   n: int = 17) -> np.ndarray:
    """Grade a (small) BGR uint8 image in Python - used for Look Lab previews and tests."""
    cube = build_cube(look, intensity, correction, n=n)
    h, w = img_bgr.shape[:2]
    rgb = img_bgr[..., ::-1].reshape(-1, 3).astype(np.float64) / 255.0
    out = sample_cube(cube, rgb).reshape(h, w, 3)[..., ::-1]
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------------------------
# Per-shot balance measurement (called from analysis on small frames)
# ----------------------------------------------------------------------------------------------
def frame_color_stats(small_bgr: np.ndarray):
    """(cast, median luma, neutral fraction) for one frame.

    cast = chromaticity of near-neutral mid-tone pixels (shades-of-grey p=4, linear light), or None
    when too few neutral pixels exist (a frame full of grass or sky says nothing about the light).
    """
    rgb = small_bgr[..., ::-1].reshape(-1, 3).astype(np.float64) / 255.0
    L = rgb @ LUMA
    mx = rgb.max(1)
    chroma_ratio = (mx - rgb.min(1)) / np.maximum(mx, 1e-6)
    m = (L > 0.1) & (L < 0.92) & (chroma_ratio < 0.22)
    frac = float(m.mean())
    cast = None
    if frac >= 0.04:
        lin = np.power(rgb[m], 2.2)
        il = np.power(np.mean(np.power(lin, 4), 0) + 1e-12, 0.25)
        cast = il / il.sum()
    return cast, float(np.median(L)), frac


def shot_color(casts: list, lumas: list, fracs: list) -> dict:
    cs = [c for c in casts if c is not None]
    return {"cast": [round(float(v), 4) for v in np.median(np.array(cs), 0)] if cs else None,
            "luma": round(float(np.median(lumas)), 3) if lumas else None,
            "neutral": round(float(np.mean(fracs)), 3) if fracs else 0.0}


def match_shots(colors: dict) -> dict:
    """Shot matching: pull every shot's cast and brightness toward the reel's median, so cuts don't jump
    between color temperatures - while keeping the reel's overall mood (a sunset reel stays warm)."""
    casts = [np.array(c["cast"]) for c in colors.values() if c and c.get("cast")]
    lumas = [c["luma"] for c in colors.values() if c and c.get("luma")]
    ref_cast = np.median(np.array(casts), 0) if casts else None
    ref_luma = float(np.median(lumas)) if lumas else None
    lift = 0.0
    if ref_luma and ref_luma < 0.22:  # the whole reel is underexposed: lift it a little
        lift = min(0.3, math.log2(0.3 / ref_luma) * 0.5)
    out = {}
    for sid, c in colors.items():
        if not c:
            continue
        wb = np.ones(3)
        if ref_cast is not None and c.get("cast"):
            wb = (ref_cast / np.maximum(np.array(c["cast"]), 1e-6)) ** 0.7
            wb = np.clip(wb / float(wb @ LUMA), 0.9, 1.1)
        ev = lift
        if ref_luma and c.get("luma") and c["luma"] > 0.02:
            ev += float(np.clip(math.log2(ref_luma / c["luma"]) * 0.5, -0.4, 0.4))
        ev = float(np.clip(ev, -0.6, 0.6))
        out[sid] = {"wb": [round(float(g), 3) for g in wb], "ev": round(ev, 3)}
    return out
