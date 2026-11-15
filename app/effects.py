"""Cinematic looks (color grades), film grain, vignette, letterbox, on-screen text and captions."""
from __future__ import annotations

import functools
import math
import os
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import FONTS_DIR

# ----------------------------------------------------------------------------------------------
# Fonts
# ----------------------------------------------------------------------------------------------
_MAC_SUP = "/System/Library/Fonts/Supplemental"
_LINUX_G = "/usr/share/fonts/truetype/google-fonts"
_DEJAVU = "/usr/share/fonts/truetype/dejavu"

FONT_CANDIDATES = {
    "caption": [FONTS_DIR / "Poppins-ExtraBold.ttf", FONTS_DIR / "Poppins-Bold.ttf", f"{_MAC_SUP}/Arial Black.ttf",
                f"{_MAC_SUP}/Arial Bold.ttf", f"{_LINUX_G}/Poppins-Bold.ttf", f"{_DEJAVU}/DejaVuSans-Bold.ttf"],
    "title": [FONTS_DIR / "BebasNeue-Regular.ttf", f"{_MAC_SUP}/DIN Condensed Bold.ttf",
              "/System/Library/Fonts/Avenir Next Condensed.ttc", f"{_MAC_SUP}/Futura.ttc",
              f"{_LINUX_G}/Poppins-Medium.ttf", f"{_DEJAVU}/DejaVuSans.ttf"],
    "bold": [FONTS_DIR / "Anton-Regular.ttf", f"{_MAC_SUP}/Impact.ttf", FONTS_DIR / "Poppins-ExtraBold.ttf",
             f"{_MAC_SUP}/Arial Black.ttf", f"{_LINUX_G}/Poppins-Bold.ttf", f"{_DEJAVU}/DejaVuSans-Bold.ttf"],
    "light": [FONTS_DIR / "Poppins-Light.ttf", "/System/Library/Fonts/Avenir Next.ttc",
              "/System/Library/Fonts/Helvetica.ttc", f"{_LINUX_G}/Poppins-Light.ttf", f"{_DEJAVU}/DejaVuSans.ttf"],
}


@functools.lru_cache(maxsize=64)
def load_font(role: str, size: int) -> ImageFont.FreeTypeFont:
    for cand in FONT_CANDIDATES.get(role, FONT_CANDIDATES["caption"]):
        p = str(cand)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # old Pillow
        return ImageFont.load_default()


# ----------------------------------------------------------------------------------------------
# Color grades ("looks")
# Channel tuples are (R, G, B) offsets applied in shadows / mids / highlights.
# ----------------------------------------------------------------------------------------------
LOOKS: dict[str, dict] = {
    "none": {},
    "clean": dict(contrast=1.05, sat=1.06),
    "teal_orange": dict(contrast=1.12, sat=1.12, shadows=(-0.05, 0.012, 0.075), highs=(0.065, 0.012, -0.06)),
    "warm_film": dict(contrast=1.06, sat=0.9, lift=0.05, gain=0.96, mids=(0.035, 0.012, -0.04)),
    "golden_hour": dict(contrast=1.05, sat=1.15, mids=(0.06, 0.025, -0.07), highs=(0.03, 0.012, -0.03)),
    "moody": dict(contrast=1.2, sat=0.72, gamma=1.12, shadows=(-0.02, 0.015, 0.05)),
    "noir": dict(contrast=1.35, sat=0.0, gamma=1.05),
    "vintage": dict(contrast=0.95, sat=0.78, lift=0.08, gain=0.93, shadows=(0.0, 0.03, 0.05),
                    highs=(0.04, 0.02, -0.04)),
    "vibrant": dict(contrast=1.08, sat=1.35, mids=(0.01, 0.0, -0.01)),
    "cool_blue": dict(contrast=1.06, sat=0.85, mids=(-0.03, 0.0, 0.05), shadows=(-0.01, 0.0, 0.03)),
    "neon_night": dict(contrast=1.15, sat=1.3, shadows=(0.05, -0.03, 0.10), highs=(0.04, -0.02, 0.02)),
}

LOOK_DESCRIPTIONS = {
    "none": "no grade",
    "clean": "natural, slightly punchier",
    "teal_orange": "blockbuster: teal shadows, warm skin/highlights",
    "warm_film": "soft faded film stock, warm mids",
    "golden_hour": "sunset warmth, rich saturation",
    "moody": "dark, desaturated, cold shadows",
    "noir": "high-contrast black & white",
    "vintage": "faded 70s/80s print, lifted blacks",
    "vibrant": "bright, saturated, social-first",
    "cool_blue": "clean cold nordic tones",
    "neon_night": "magenta/blue night-city punch",
}


def _solve_k(c: float) -> float:
    lo, hi = 1e-4, 30.0
    for _ in range(60):
        k = (lo + hi) / 2
        slope = 0.5 * k / math.tanh(k / 2)
        if slope < c:
            lo = k
        else:
            hi = k
    return (lo + hi) / 2


def _contrast(y: np.ndarray, c: float) -> np.ndarray:
    if abs(c - 1.0) < 1e-3:
        return y
    if c < 1.0:
        return 0.5 + (y - 0.5) * c
    k = _solve_k(c)
    return 0.5 + 0.5 * np.tanh(k * (y - 0.5)) / math.tanh(k * 0.5)


def build_grade(look: str, intensity: float = 1.0):
    """Return (bgr 3-channel LUT for cv2.LUT, saturation multiplier)."""
    p = LOOKS.get(look, {})
    intensity = float(np.clip(intensity, 0.0, 1.5))
    x = np.linspace(0, 1, 256)
    chans = []
    for ch in range(3):  # R, G, B
        y = p.get("lift", 0.0) + (p.get("gain", 1.0) - p.get("lift", 0.0)) * x
        y = np.clip(y, 0, 1) ** p.get("gamma", 1.0)
        y = _contrast(y, p.get("contrast", 1.0))
        ws = np.clip(1 - y / 0.6, 0, 1) ** 2
        wh = np.clip((y - 0.4) / 0.6, 0, 1) ** 2
        wm = np.clip(1 - np.abs(y - 0.5) / 0.5, 0, 1)
        y = (y + p.get("shadows", (0, 0, 0))[ch] * ws + p.get("mids", (0, 0, 0))[ch] * wm
             + p.get("highs", (0, 0, 0))[ch] * wh)
        y = np.clip(y, 0, 1)
        y = x + intensity * (y - x)
        chans.append(np.clip(np.round(y * 255), 0, 255).astype(np.uint8))
    r, g, b = chans
    lut = np.stack([b, g, r], axis=1).reshape(256, 1, 3)
    sat = 1.0 + intensity * (p.get("sat", 1.0) - 1.0)
    return lut, sat


class FinishingPass:
    """Per-frame global finishing: grade -> vignette -> grain -> letterbox."""

    def __init__(self, W: int, H: int, look="clean", intensity=1.0, grain=0.0, vignette=0.0,
                 letterbox=False, seed=7):
        self.W, self.H = W, H
        self.lut, self.sat = build_grade(look, intensity)
        self.identity = look == "none" or intensity <= 0.01
        self.rng = np.random.default_rng(seed)
        self.grain = float(np.clip(grain, 0, 1))
        self.vig = None
        if vignette > 0.01:
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 * 0.85 + ((yy - H / 2) / (H / 2)) ** 2 * 0.65)
            t = np.clip((d - 0.45) / 0.8, 0, 1)
            m = 1.0 - float(np.clip(vignette, 0, 1)) * 0.75 * (t * t * (3 - 2 * t))
            self.vig = cv2.merge([np.round(m * 255).astype(np.uint8)] * 3)
        self.grain_bank = []
        if self.grain > 0.01:
            for _ in range(6):
                n = self.rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32)
                n = cv2.GaussianBlur(n, (0, 0), 0.7)
                n = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)
                n /= n.std() + 1e-6
                u = np.clip(128 + n * 40, 0, 255).astype(np.uint8)
                self.grain_bank.append(cv2.merge([u, u, u]))
        self.bar = int(round(H * 0.085)) if letterbox else 0

    @property
    def safe_top(self):
        return self.bar

    @property
    def safe_bottom(self):
        return self.H - self.bar

    def apply(self, frame: np.ndarray, k: int = 0) -> np.ndarray:
        out = frame
        if not self.identity:
            out = cv2.LUT(out, self.lut)
            if abs(self.sat - 1.0) > 0.01:
                gray = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                out = cv2.addWeighted(out, self.sat, gray, 1.0 - self.sat, 0)
        if self.vig is not None:
            out = cv2.multiply(out, self.vig, scale=1 / 255.0)
        if self.grain_bank:
            s = self.grain * 0.42
            g = self.grain_bank[(k * 7 + int(self.rng.integers(0, 5))) % len(self.grain_bank)]
            out = cv2.addWeighted(out, 1.0, g, s, -128.0 * s)
        if self.bar:
            if out is frame:
                out = frame.copy()
            out[: self.bar] = 0
            out[self.H - self.bar:] = 0
        return out


def fade(frame: np.ndarray, f: float) -> np.ndarray:
    if f >= 0.999:
        return frame
    return cv2.convertScaleAbs(frame, alpha=max(0.0, f))


# ----------------------------------------------------------------------------------------------
# Compositing helpers
# ----------------------------------------------------------------------------------------------
def alpha_blit(frame: np.ndarray, img_bgra: np.ndarray, cx: float, cy: float, alpha: float = 1.0,
               scale: float = 1.0):
    """Alpha-composite img (BGRA) centred at (cx, cy) onto frame in place."""
    if alpha <= 0.003 or img_bgra is None:
        return
    img = img_bgra
    if abs(scale - 1.0) > 0.005:
        nw, nh = max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    H, W = frame.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x0 + w), min(H, y0 + h)
    if x1c <= x0c or y1c <= y0c:
        return
    sub = img[sy0: sy0 + (y1c - y0c), sx0: sx0 + (x1c - x0c)]
    a = sub[..., 3:4].astype(np.float32) * (alpha / 255.0)
    roi = frame[y0c:y1c, x0c:x1c].astype(np.float32)
    frame[y0c:y1c, x0c:x1c] = (roi * (1.0 - a) + sub[..., :3].astype(np.float32) * a).astype(np.uint8)


def _smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def _ease_out_back(x):
    x = min(1.0, max(0.0, x))
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


# ----------------------------------------------------------------------------------------------
# Text rendering
# ----------------------------------------------------------------------------------------------
TEXT_STYLES = {
    "cinematic_title": dict(font="title", size=0.12, tracking=0.18, upper=True, shadow=True, stroke=0.0,
                            anim="title", line=1.05),
    "bold": dict(font="bold", size=0.095, tracking=0.0, upper=True, shadow=True, stroke=0.07, anim="pop",
                 line=1.05),
    "subtitle": dict(font="light", size=0.05, tracking=0.04, upper=False, shadow=True, stroke=0.0,
                     anim="fade_up", line=1.25),
    "kicker": dict(font="caption", size=0.034, tracking=0.32, upper=True, shadow=True, stroke=0.0,
                   anim="fade", line=1.3),
}
TEXT_POSITIONS = {"top": 0.2, "center": 0.43, "bottom": 0.64}


def _line_width(font, text, tracking_px):
    if tracking_px <= 0.01:
        return font.getlength(text)
    return sum(font.getlength(c) for c in text) + tracking_px * max(0, len(text) - 1)


def _wrap(font, text, max_w, tracking_px):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if _line_width(font, cand, tracking_px) <= max_w or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def render_text(text: str, style: str, W: int, color=(255, 255, 255)) -> np.ndarray:
    st = TEXT_STYLES.get(style, TEXT_STYLES["bold"])
    size = max(12, int(st["size"] * W))
    font = load_font(st["font"], size)
    txt = text.upper() if st["upper"] else text
    tracking_px = st["tracking"] * size
    lines = _wrap(font, txt, W * 0.86, tracking_px)
    widths = [_line_width(font, ln, tracking_px) for ln in lines]
    asc, desc = font.getmetrics()
    lh = int((asc + desc) * st["line"])
    stroke = int(round(st["stroke"] * size))
    pad = int(size * 0.5) + stroke
    cw = int(max(widths)) + 2 * pad
    chh = lh * len(lines) + 2 * pad
    img = Image.new("RGBA", (cw, chh), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, (ln, lw) in enumerate(zip(lines, widths)):
        x = pad + (max(widths) - lw) / 2
        y = pad + i * lh
        if tracking_px <= 0.01:
            d.text((x, y), ln, font=font, fill=color + (255,), stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        else:
            for c in ln:
                d.text((x, y), c, font=font, fill=color + (255,), stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
                x += font.getlength(c) + tracking_px
    arr = np.array(img)
    if st["shadow"]:
        arr = _add_shadow(arr, size)
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)


def _add_shadow(rgba: np.ndarray, size: int, strength=0.55) -> np.ndarray:
    a = rgba[..., 3].astype(np.float32) / 255.0
    sh = cv2.GaussianBlur(a, (0, 0), max(1.0, size * 0.07))
    off = max(1, int(size * 0.035))
    sh = np.roll(sh, off, axis=0)
    sh_a = np.clip(sh * strength, 0, 1)
    out_a = a + sh_a * (1 - a)
    rgb = rgba[..., :3].astype(np.float32)
    out_rgb = np.where(out_a[..., None] > 1e-4, rgb * (a / np.maximum(out_a, 1e-4))[..., None], 0)
    out = np.dstack([np.clip(out_rgb, 0, 255), np.clip(out_a * 255, 0, 255)]).astype(np.uint8)
    return out


class TextOverlay:
    """An animated text card attached to a range of output frames."""

    def __init__(self, text: str, style: str, position: str, start: float, end: float, W: int, H: int,
                 safe_top: int = 0, safe_bottom: int | None = None):
        self.style = style if style in TEXT_STYLES else "bold"
        self.img = render_text(text, self.style, W)
        self.start, self.end = start, end
        self.W, self.H = W, H
        y = TEXT_POSITIONS.get(position, TEXT_POSITIONS["center"]) * H
        hh = self.img.shape[0] / 2
        lo = safe_top + hh
        hi = (safe_bottom if safe_bottom is not None else H) - hh
        self.cy = float(np.clip(y, lo, max(lo, hi)))
        self.anim = TEXT_STYLES[self.style]["anim"]

    def draw(self, frame: np.ndarray, t: float):
        if t < self.start or t > self.end:
            return
        lt, rt = t - self.start, self.end - t
        dur = max(0.01, self.end - self.start)
        alpha, scale, dy = 1.0, 1.0, 0.0
        if self.anim == "title":
            alpha = _smooth(lt / 0.7) * _smooth(rt / 0.5)
            scale = 0.96 + 0.06 * (lt / dur)
        elif self.anim == "pop":
            alpha = _smooth(lt / 0.08) * _smooth(rt / 0.15)
            scale = 0.55 + 0.45 * _ease_out_back(lt / 0.22)
        elif self.anim == "fade_up":
            alpha = _smooth(lt / 0.45) * _smooth(rt / 0.35)
            dy = (1 - _smooth(lt / 0.6)) * 0.02 * self.H
        else:
            alpha = _smooth(lt / 0.4) * _smooth(rt / 0.4)
        alpha_blit(frame, self.img, self.W / 2, self.cy + dy, alpha, scale)


# ----------------------------------------------------------------------------------------------
# Captions (word-by-word)
# ----------------------------------------------------------------------------------------------
CAPTION_STYLES = ("pop_yellow", "clean_white", "boxed")


def build_chunks(words, max_words=3, max_dur=1.6, gap=0.45):
    chunks, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or w["start"] - cur[-1]["end"] > gap
                    or w["end"] - cur[0]["start"] > max_dur):
            chunks.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.!?,;:]$", w["word"]):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    out = []
    for i, c in enumerate(chunks):
        nxt = chunks[i + 1][0]["start"] if i + 1 < len(chunks) else 1e9
        out.append({"start": c[0]["start"], "end": min(nxt, c[-1]["end"] + 0.35), "words": c})
    return out


class CaptionTrack:
    def __init__(self, words: list[dict], style: str, W: int, H: int, y_frac: float = 0.70):
        self.style = style if style in CAPTION_STYLES else "pop_yellow"
        self.W, self.H = W, H
        self.cy = H * y_frac
        clean = [dict(w, word=w["word"].strip()) for w in words if w.get("word", "").strip()]
        mw = 3 if self.style == "pop_yellow" else 5
        self.chunks = build_chunks(clean, max_words=mw, max_dur=1.6 if mw == 3 else 2.4)
        self._cache: dict = {}
        self._ci = 0

    def _render(self, ci: int, active: int) -> np.ndarray:
        key = (ci, active if self.style == "pop_yellow" else -1)
        if key in self._cache:
            return self._cache[key]
        words = [w["word"] for w in self.chunks[ci]["words"]]
        if self.style == "pop_yellow":
            words = [re.sub(r"[.,;:]$", "", w).upper() for w in words]
            size = int(self.W * 0.078)
            font = load_font("caption", size)
            stroke = max(2, int(size * 0.13))
        else:
            words = [re.sub(r"[,;:]$", "", w) for w in words]
            size = int(self.W * 0.056)
            font = load_font("caption", size)
            stroke = 0
        space = font.getlength(" ")
        max_w = self.W * 0.84
        # lay out words in lines
        lines, cur, cur_w = [], [], 0.0
        for i, w in enumerate(words):
            ww = font.getlength(w)
            if cur and cur_w + space + ww > max_w:
                lines.append(cur)
                cur, cur_w = [], 0.0
            cur_w += (space if cur else 0) + ww
            cur.append((i, w, ww))
        if cur:
            lines.append(cur)
        asc, desc = font.getmetrics()
        lh = int((asc + desc) * 1.12)
        pad = int(size * 0.55) + stroke
        line_ws = [sum(x[2] for x in ln) + space * (len(ln) - 1) for ln in lines]
        cw, chh = int(max(line_ws)) + 2 * pad, lh * len(lines) + 2 * pad
        img = Image.new("RGBA", (cw, chh), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        if self.style == "boxed":
            d.rounded_rectangle([pad * 0.35, pad * 0.35, cw - pad * 0.35, chh - pad * 0.35],
                                radius=int(size * 0.35), fill=(0, 0, 0, 175))
        for li, ln in enumerate(lines):
            x = pad + (max(line_ws) - line_ws[li]) / 2
            y = pad + li * lh
            for i, w, ww in ln:
                color = (255, 225, 60, 255) if (self.style == "pop_yellow" and i == active) else (255, 255, 255, 255)
                d.text((x, y), w, font=font, fill=color, stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
                x += ww + space
        arr = np.array(img)
        if self.style != "boxed":
            arr = _add_shadow(arr, size, 0.6)
        out = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        self._cache[key] = out
        return out

    def draw(self, frame: np.ndarray, t: float):
        if not self.chunks:
            return
        # chunks are sorted; walk pointer
        if self._ci >= len(self.chunks) or t < self.chunks[self._ci]["start"]:
            self._ci = 0
        while self._ci < len(self.chunks) - 1 and t >= self.chunks[self._ci]["end"]:
            self._ci += 1
        c = self.chunks[self._ci]
        if not (c["start"] <= t < c["end"]):
            return
        active = 0
        for i, w in enumerate(c["words"]):
            if t >= w["start"]:
                active = i
        img = self._render(self._ci, active)
        lt = t - c["start"]
        if self.style == "pop_yellow":
            scale = 0.82 + 0.18 * _ease_out_back(lt / 0.14)
            alpha = _smooth(lt / 0.05)
        else:
            scale = 1.0
            alpha = _smooth(lt / 0.12) * _smooth((c["end"] - t) / 0.1)
        alpha_blit(frame, img, self.W / 2, self.cy, alpha, scale)
