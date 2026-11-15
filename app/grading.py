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
# Look library
# hue angles in degrees (0 red, 30 orange/skin, 60 yellow, 120 green, 180 cyan, 210 sky, 240 blue,
# 300 magenta). tints are (hue, amount). hue_adj bands: (center, width, hue_shift, sat_mult, lum_shift)
# ----------------------------------------------------------------------------------------------
DEFAULTS = dict(temp=0.0, tint=0.0, exposure=0.0, contrast=1.0, pivot=0.45, toe=0.0, shoulder=0.0,
                black=0.0, white=1.0, sat=1.0, vibrance=0.0, lift=(0, 0, 0), gamma=(1, 1, 1), gain=(1, 1, 1),
                shadow_tint=(0, 0.0), highlight_tint=(0, 0.0), hue_adj=(), skin_protect=0.7,
                mono=False, mono_mix=(0.30, 0.59, 0.11), mono_tone=(0, 0.0),
                halation=0.0, bloom=0.0, grain=0.0, vignette=0.0)

LOOK_PARAMS: dict[str, dict] = {
    "none": dict(),
    "clean": dict(contrast=1.08, vibrance=0.15, sat=1.02, shoulder=0.15),
    "vibrant": dict(contrast=1.08, sat=1.12, vibrance=0.35, shoulder=0.2),
    "teal_orange": dict(temp=0.05, contrast=1.16, sat=1.05, vibrance=0.1, shoulder=0.25,
                        shadow_tint=(195, 0.10), highlight_tint=(32, 0.08),
                        hue_adj=((120, 40, 28, 0.8, -0.05), (210, 30, -6, 1.1, -0.04), (30, 18, 2, 1.06, 0.0))),
    "kodak_2383": dict(contrast=1.3, pivot=0.42, toe=0.4, shoulder=0.3, sat=1.03,
                       shadow_tint=(190, 0.07), highlight_tint=(38, 0.07),
                       hue_adj=((120, 35, 16, 0.85, -0.06), (0, 25, 6, 1.05, -0.04)),
                       halation=0.3, grain=0.28),
    "warm_film": dict(temp=0.15, contrast=1.05, black=0.035, white=0.97, sat=0.92, shoulder=0.35,
                      gamma=(1.0, 1.0, 0.97), highlight_tint=(40, 0.05), shadow_tint=(200, 0.04),
                      halation=0.25, grain=0.25),
    "fuji_eterna": dict(contrast=0.92, black=0.03, sat=0.82, temp=-0.03, tint=-0.05, shoulder=0.3,
                        hue_adj=((120, 40, -10, 0.85, 0.0), (30, 25, -3, 0.92, 0.0)), grain=0.2),
    "golden_hour": dict(temp=0.35, exposure=0.05, contrast=1.06, sat=1.12, vibrance=0.2,
                        highlight_tint=(38, 0.10), shadow_tint=(20, 0.04), bloom=0.25),
    "desert_heat": dict(temp=0.3, tint=0.05, contrast=1.12, sat=1.05, highlight_tint=(40, 0.08),
                        hue_adj=((210, 40, -15, 0.7, 0.0),), bloom=0.15),
    "high_key": dict(exposure=0.35, contrast=0.92, black=0.02, sat=0.9, vibrance=0.1, temp=0.02, shoulder=0.45,
                     highlight_tint=(200, 0.02), bloom=0.2),
    "matte_pastel": dict(exposure=0.1, black=0.1, white=0.95, contrast=0.88, sat=0.8,
                         highlight_tint=(330, 0.04), shadow_tint=(210, 0.05)),
    "moody": dict(exposure=-0.2, contrast=1.2, sat=0.72, temp=-0.08, black=0.02, shadow_tint=(205, 0.08),
                  hue_adj=((120, 40, 20, 0.6, -0.1), (210, 40, 0, 0.8, -0.08)), vignette=0.5, grain=0.2),
    "emerald_moody": dict(exposure=-0.15, contrast=1.15, sat=0.9, temp=-0.05, shadow_tint=(170, 0.08),
                          hue_adj=((120, 45, 30, 0.95, -0.1), (60, 25, 25, 0.8, -0.05)), vignette=0.4, grain=0.15),
    "cinematic_cool": dict(temp=-0.2, tint=0.05, contrast=1.18, sat=0.9, toe=0.2,
                           shadow_tint=(215, 0.10), highlight_tint=(190, 0.03)),
    "cool_blue": dict(temp=-0.25, sat=0.85, contrast=1.06, shadow_tint=(210, 0.06)),
    "neon_night": dict(contrast=1.15, sat=1.2, vibrance=0.2, shadow_tint=(250, 0.10), highlight_tint=(320, 0.025),
                       hue_adj=((300, 40, 0, 1.25, 0.05), (190, 30, 0, 1.2, 0.03)), bloom=0.35),
    "cross_process": dict(contrast=1.2, sat=1.25, shadow_tint=(240, 0.12), highlight_tint=(70, 0.12),
                          gamma=(1.05, 0.97, 1.1)),
    "bleach_bypass": dict(sat=0.45, contrast=1.35, toe=0.25, shoulder=0.1, temp=-0.03, grain=0.2),
    "vintage": dict(black=0.08, white=0.93, contrast=0.95, sat=0.78, temp=0.1, gamma=(1.0, 1.02, 0.95),
                    shadow_tint=(175, 0.08), highlight_tint=(45, 0.10), grain=0.4, halation=0.2, vignette=0.35),
    "noir": dict(mono=True, mono_mix=(0.35, 0.55, 0.10), contrast=1.38, toe=0.3, shoulder=0.2,
                 vignette=0.45, grain=0.35),
    "sepia": dict(mono=True, mono_tone=(35, 0.07), black=0.04, contrast=1.1, grain=0.3, vignette=0.4),
}

LOOK_DESCRIPTIONS_BUILTIN = {
    "none": "no grade (balance only)",
    "clean": "natural Rec.709 polish, gentle contrast",
    "vibrant": "bright, colorful, social-first",
    "teal_orange": "blockbuster: teal shadows & greens, warm skin & highlights",
    "kodak_2383": "Kodak print film: rich contrast, warm highlights, cyan-leaning greens, halation",
    "warm_film": "soft warm film negative, faded blacks, halation",
    "fuji_eterna": "Fuji Eterna cinema stock: soft, low contrast, muted, slightly green",
    "golden_hour": "sunset warmth, glowing highlights",
    "desert_heat": "hot, dry, sun-baked with muted teal skies",
    "high_key": "bright & airy, lifted, soft highlights",
    "matte_pastel": "matte pastel, lifted blacks, soft pink/blue tones",
    "moody": "dark, desaturated, cold shadows",
    "emerald_moody": "dark forest/emerald greens, moody",
    "cinematic_cool": "cool blockbuster night/blue hour",
    "cool_blue": "clean cold nordic tones",
    "neon_night": "night-city neon: indigo shadows, magenta & cyan pop, glow",
    "cross_process": "cross-processed: blue shadows, yellow-green highlights, punchy",
    "bleach_bypass": "bleach bypass: desaturated, harsh silvery contrast",
    "vintage": "faded 70s print: lifted blacks, teal shadows, yellow highs, grain",
    "noir": "high-contrast black & white",
    "sepia": "warm sepia monochrome",
}

# families help the learner generalize ("you like warm looks for travel")
LOOK_FAMILY = {
    "none": "clean", "clean": "clean", "vibrant": "vibrant", "teal_orange": "cinematic", "kodak_2383": "film",
    "warm_film": "film", "fuji_eterna": "film", "golden_hour": "warm", "desert_heat": "warm", "high_key": "bright",
    "matte_pastel": "bright", "moody": "dark", "emerald_moody": "dark", "cinematic_cool": "cool", "cool_blue": "cool",
    "neon_night": "neon", "cross_process": "vibrant", "bleach_bypass": "dark", "vintage": "film", "noir": "mono",
    "sepia": "mono",
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
    t, ti = p["temp"], p["tint"]
    lin = lin * np.array([1 + 0.12 * t, 1 - 0.08 * ti, 1 - 0.12 * t]) * (2.0 ** p["exposure"])
    x = np.power(np.clip(lin, 0, None), 1 / 2.2)
    # highlight shoulder: values pushed past white roll off smoothly instead of clipping (slope 1 at
    # the knee, so mid-tones are untouched); then contrast S-curve, toe and fade
    x = _shoulder(x, p["shoulder"])
    x = _scurve(np.clip(x, 0, 1), p["contrast"], p["pivot"])
    if p["toe"] > 0:
        b = 0.05 * p["toe"]
        x = np.clip((x - b) / (1 - b), 0, 1)
    x = p["black"] + (p["white"] - p["black"]) * x
    # CDL-style lift / gamma / gain per channel
    lift, gam, gain = np.array(p["lift"], float), np.array(p["gamma"], float), np.array(p["gain"], float)
    x = gain * (x + lift * (1 - x))
    x = np.power(np.clip(x, 0, None), 1 / np.maximum(gam, 1e-3))
    # split toning
    L = x @ LUMA
    for (hue, amt), w in ((p["shadow_tint"], _smooth(0.55, 0.0, L)), (p["highlight_tint"], _smooth(0.45, 1.0, L))):
        if amt:
            x = x + amt * w[:, None] * _hue_rgb(hue)[None, :]
    # saturation + vibrance (protects already-saturated colors)
    L = x @ LUMA
    chroma = x - L[:, None]
    s_cur = np.clip(x.max(1) - x.min(1), 0, 1)
    fac = p["sat"] * (1 + p["vibrance"] * (1 - s_cur))
    x = L[:, None] + chroma * fac[:, None]
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
    k = float(np.clip(intensity, 0, 1.2))
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
