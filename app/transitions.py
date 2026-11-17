"""Transition library - CapCut-style categories, implemented with OpenCV/NumPy.

Every transition is f(a, b, p) -> frame, where a = outgoing frame, b = incoming frame (BGR uint8,
same size) and p = progress 0..1. Categories:

  basic       mix, black/white fades, flash, pull in/out, soft wipes, barn door
  camera      zoom switch, whip pans, shake, spin, 3D flip, 3D cube, speed sweep
  light       light leak, lens flare, film burn, glow flash, heatwave flash, flash blast
  blur        dreamy dissolve, focus blur, motion blur, chromatic blur, zoom blur, spin blur
  glitch      glitch, RGB split, static, color glitch, scanline glitch, pixelate
  distortion  ripple, wave, fisheye, twist, stretch, liquid
  slide       push (4 directions), slide-over (4), diagonal, screen swap, split push
  mask        circle, heart, star, diamond, blinds, clock wipe, line wipe
  overlay     paper tear, film strip, scrapbook cutout, story swipe, ribs collage
  trending    zoom flash, spin zoom, glitch whip, shake punch
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class T:
    name: str
    category: str
    duration: float
    energy: int          # 1 subtle .. 3 high-energy
    description: str


_REG: list[T] = [
    T("cut", "basic", 0.0, 1, "hard cut (default for most beat cuts)"),
    # basic
    T("crossfade", "basic", 0.5, 1, "Mix: soft dissolve"),
    T("dip_to_black", "basic", 0.6, 1, "Black Fade: fade through black (time jump, chapter)"),
    T("dip_to_white", "basic", 0.5, 1, "White Fade: fade through white (dreamy, memories)"),
    T("flash", "basic", 0.25, 2, "White Flash: quick white flash cut on a beat"),
    T("pull_in", "basic", 0.45, 2, "Pull In: zoom into the next shot"),
    T("pull_out", "basic", 0.45, 2, "Pull Out: zoom out into the next shot"),
    T("wipe_left", "basic", 0.45, 1, "soft wipe to the left"),
    T("wipe_right", "basic", 0.45, 1, "soft wipe to the right"),
    T("wipe_up", "basic", 0.45, 1, "soft wipe upwards"),
    T("wipe_down", "basic", 0.45, 1, "soft wipe downwards"),
    T("wipe_diagonal", "basic", 0.5, 1, "soft diagonal wipe"),
    T("barn_door", "basic", 0.5, 1, "opens from the centre outwards"),
    # camera
    T("zoom_punch", "camera", 0.35, 3, "Zoom Switch: zoom-blur punch into the next shot"),
    T("whip_left", "camera", 0.3, 2, "Whip Pan left with motion blur"),
    T("whip_right", "camera", 0.3, 2, "Whip Pan right with motion blur"),
    T("whip_up", "camera", 0.3, 2, "vertical whip up"),
    T("whip_down", "camera", 0.3, 2, "vertical whip down"),
    T("shake", "camera", 0.35, 3, "camera shake hit"),
    T("spin", "camera", 0.4, 3, "Rotate: spin with motion blur"),
    T("flip_3d", "camera", 0.55, 2, "3D Flip: card flips around"),
    T("cube_3d", "camera", 0.6, 2, "3D Cube: rotates to the next face"),
    T("speed_sweep", "camera", 0.35, 3, "speed-ramped camera sweep with zoom"),
    # light
    T("light_leak", "light", 0.8, 1, "warm film light leak dissolve"),
    T("lens_flare", "light", 0.6, 2, "anamorphic lens flare sweep"),
    T("film_burn", "light", 0.8, 2, "film burn: orange burn-out into the next shot"),
    T("glow_flash", "light", 0.45, 2, "glowing bloom flash"),
    T("heatwave_flash", "light", 0.5, 3, "Heatwave Flash: warm flash with heat shimmer"),
    T("flash_blast", "light", 0.4, 3, "Flash Blast: radial white burst"),
    # blur
    T("blur_dissolve", "blur", 0.5, 1, "dreamy out-of-focus dissolve"),
    T("focus_blur", "blur", 0.5, 1, "Focus Blur: rack focus out and in"),
    T("motion_blur", "blur", 0.4, 2, "directional motion blur dissolve"),
    T("chromatic_blur", "blur", 0.45, 2, "Chromatic Blur: blur with colour fringing"),
    T("zoom_blur", "blur", 0.4, 2, "radial zoom blur dissolve"),
    T("spin_blur", "blur", 0.45, 2, "rotational blur"),
    # glitch
    T("glitch", "glitch", 0.25, 3, "RGB split + slice glitch"),
    T("rgb_split", "glitch", 0.3, 3, "strong RGB channel split"),
    T("static", "glitch", 0.3, 3, "TV static interference"),
    T("color_glitch", "glitch", 0.3, 3, "Color Glitch: blocky colour corruption"),
    T("scanline_glitch", "glitch", 0.3, 3, "scanline jitter"),
    T("pixelate", "glitch", 0.35, 2, "mosaic pixelate in and out"),
    # distortion
    T("ripple", "distortion", 0.5, 2, "water ripple from the centre"),
    T("wave", "distortion", 0.5, 2, "horizontal wave warp"),
    T("fisheye", "distortion", 0.45, 2, "fisheye bulge warp"),
    T("twist", "distortion", 0.5, 3, "swirl twist"),
    T("stretch", "distortion", 0.35, 3, "horizontal stretch smear"),
    T("liquid", "distortion", 0.55, 2, "liquid morph"),
    # slide
    T("push_left", "slide", 0.4, 2, "next shot pushes the current one left"),
    T("push_right", "slide", 0.4, 2, "push right"),
    T("push_up", "slide", 0.4, 2, "push up"),
    T("push_down", "slide", 0.4, 2, "push down"),
    T("slide_left", "slide", 0.4, 2, "next shot slides in over from the right"),
    T("slide_right", "slide", 0.4, 2, "next shot slides in over from the left"),
    T("slide_up", "slide", 0.4, 2, "next shot slides up over the current one (story-style)"),
    T("slide_down", "slide", 0.4, 2, "next shot slides down over"),
    T("slide_diagonal", "slide", 0.45, 2, "next shot slides in diagonally"),
    T("screen_swap", "slide", 0.55, 2, "Screen Swap: cards swap places"),
    T("split_push", "slide", 0.45, 2, "split-screen push: halves slide apart"),
    # mask
    T("iris", "mask", 0.5, 1, "circle reveal"),
    T("heart", "mask", 0.55, 2, "heart-shaped reveal"),
    T("star", "mask", 0.55, 2, "star-shaped reveal"),
    T("diamond", "mask", 0.5, 2, "diamond reveal"),
    T("blinds", "mask", 0.5, 2, "venetian blinds reveal"),
    T("clock_wipe", "mask", 0.55, 2, "clock-hand radial wipe"),
    T("line_wipe", "mask", 0.55, 2, "staggered diagonal lines reveal"),
    # overlay / social
    T("paper_tear", "overlay", 0.6, 2, "torn-paper reveal"),
    T("film_strip", "overlay", 0.6, 2, "film frames roll to the next shot"),
    T("scrapbook", "overlay", 0.65, 2, "scrapbook cutout flies in and fills the frame"),
    T("story_swipe", "overlay", 0.45, 2, "UI story swipe with rounded card"),
    T("ribs_collage", "overlay", 0.6, 2, "Ribs Collage: vertical strips drop in"),
    # trending combos
    T("zoom_flash", "trending", 0.4, 3, "zoom punch + white flash (viral beat drop)"),
    T("spin_zoom", "trending", 0.45, 3, "spin + zoom blur"),
    T("glitch_whip", "trending", 0.35, 3, "whip pan with RGB glitch"),
    T("shake_punch", "trending", 0.35, 3, "shake hit + zoom punch"),
]

REGISTRY: dict[str, T] = {t.name: t for t in _REG}
CATEGORIES = ["basic", "camera", "light", "blur", "glitch", "distortion", "slide", "mask", "overlay", "trending"]
CATEGORY_LABELS = {"basic": "Basic", "camera": "Camera", "light": "Light effects", "blur": "Blur", "glitch": "Glitch",
                   "distortion": "Distortion", "slide": "Slide", "mask": "Mask", "overlay": "Overlay & social",
                   "trending": "Trending"}
# backwards-compatible view: name -> (default seconds, description)
TRANSITIONS = {t.name: (t.duration, f"[{t.category}] {t.description}") for t in _REG}


def by_category() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    for t in _REG:
        if t.name != "cut":
            out[t.category].append(t.name)
    return out


def category_of(name: str) -> str | None:
    t = REGISTRY.get(name)
    return t.category if t and name != "cut" else None


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def smooth(p):
    p = min(1.0, max(0.0, p))
    return p * p * (3 - 2 * p)


def ease_in_out_cubic(p):
    p = min(1.0, max(0.0, p))
    return 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def ease_out(p, k=3):
    p = min(1.0, max(0.0, p))
    return 1 - (1 - p) ** k


def bell(p, w=0.5):
    """Pulse peaking at p=0.5 and exactly 0 at both ends (no pop when the transition starts/ends).
    Smaller w = narrower pulse."""
    p = min(1.0, max(0.0, p))
    return math.sin(math.pi * p) ** max(1.0, 2.0 / w)


def _mix(a, b, m):
    """Blend with a single-channel uint8 mask (255 -> b)."""
    m3 = cv2.merge([m, m, m])
    return cv2.add(cv2.multiply(b, m3, scale=1 / 255.0), cv2.multiply(a, cv2.bitwise_not(m3), scale=1 / 255.0))


def _fade(a, b, t):
    t = min(1.0, max(0.0, t))
    return cv2.addWeighted(a, 1 - t, b, t, 0)


def _affine(img, M, W, H, border=cv2.BORDER_REFLECT, value=(0, 0, 0)):
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=border, borderValue=value)


def _zoom(img, z, angle=0.0, border=cv2.BORDER_REFLECT, cx=None, cy=None):
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2 if cx is None else cx, H / 2 if cy is None else cy), angle, z)
    return _affine(img, M, W, H, border)


def _shift(img, dx, dy, border=cv2.BORDER_CONSTANT):
    H, W = img.shape[:2]
    return _affine(img, np.float32([[1, 0, dx], [0, 1, dy]]), W, H, border)


def _small(img, f=4):
    H, W = img.shape[:2]
    return cv2.resize(img, (max(2, W // f), max(2, H // f)), interpolation=cv2.INTER_AREA)


def _up(img, W, H):
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)


def _blur(img, r):
    """Fast strong blur via downscale/upscale (r ~ 0..1)."""
    if r < 0.03:
        return img
    H, W = img.shape[:2]
    s = 1.0 / (1 + 14 * r)
    sm = cv2.resize(img, (max(2, int(W * s)), max(2, int(H * s))), interpolation=cv2.INTER_AREA)
    return cv2.resize(sm, (W, H), interpolation=cv2.INTER_LINEAR)


def _mask_from_small(m_small, W, H, feather=2.0):
    m = cv2.resize(m_small, (W, H), interpolation=cv2.INTER_LINEAR)
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    return m


def _add_light(img, layer_f32, k=1.0):
    """Additive light (layer in 0..255 float, any size)."""
    H, W = img.shape[:2]
    layer_f32 = np.asarray(layer_f32, np.float32)
    if layer_f32.ndim == 3 and layer_f32.shape[2] == 1:
        layer_f32 = layer_f32[..., 0]
    if layer_f32.shape[:2] != (H, W):
        layer_f32 = cv2.resize(layer_f32, (W, H), interpolation=cv2.INTER_LINEAR)
    if layer_f32.ndim == 2:
        layer_f32 = cv2.merge([layer_f32] * 3)
    return cv2.add(img, np.clip(layer_f32 * k, 0, 255).astype(np.uint8))


def _card(img, scale, cx, cy, W, H, angle=0.0, radius=0.0, shadow=True, border=0, canvas=None):
    """Draw img as a scaled/rotated card (rounded corners, optional white border & drop shadow)."""
    canvas = np.zeros((H, W, 3), np.uint8) if canvas is None else canvas.copy()
    src = img
    if border > 0:
        src = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    sh, sw = src.shape[:2]
    M = cv2.getRotationMatrix2D((sw / 2, sh / 2), angle, scale)
    M[0, 2] += cx - sw / 2
    M[1, 2] += cy - sh / 2
    mask = np.full((sh, sw), 255, np.uint8)
    if radius > 0:
        r = int(radius * min(sw, sh))
        mask[:] = 0
        cv2.rectangle(mask, (r, 0), (sw - r, sh), 255, -1)
        cv2.rectangle(mask, (0, r), (sw, sh - r), 255, -1)
        for x, y in ((r, r), (sw - r, r), (r, sh - r), (sw - r, sh - r)):
            cv2.circle(mask, (x, y), r, 255, -1, cv2.LINE_AA)
    warped = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    wm = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if shadow:
        sm = cv2.GaussianBlur(np.roll(wm, int(H * 0.012), axis=0), (0, 0), max(2, W * 0.02))
        canvas = cv2.multiply(canvas, cv2.merge([255 - (sm * 0.6).astype(np.uint8)] * 3), scale=1 / 255.0)
    return _mix(canvas, warped, wm)


class TransitionFX:
    def __init__(self, W: int, H: int, seed: int = 11):
        self.W, self.H = W, H
        self.white = np.full((H, W, 3), 255, np.uint8)
        self.black = np.zeros((H, W, 3), np.uint8)
        self.rng = np.random.default_rng(seed)
        self._cache: dict = {}

    def apply(self, kind: str, a: np.ndarray, b: np.ndarray, p: float) -> np.ndarray:
        fn = getattr(self, "_" + kind, None) or self._crossfade
        return fn(a, b, float(min(1.0, max(0.0, p))))

    # -- cached grids / noise -----------------------------------------------------------------
    def _grid(self):
        if "grid" not in self._cache:
            xs, ys = np.meshgrid(np.arange(self.W, dtype=np.float32), np.arange(self.H, dtype=np.float32))
            cx, cy = self.W / 2, self.H / 2
            dx, dy = xs - cx, ys - cy
            r = np.sqrt(dx * dx + dy * dy)
            self._cache["grid"] = (xs, ys, dx, dy, r, float(math.hypot(cx, cy)))
        return self._cache["grid"]

    def _noise(self, key, scale=8, seed=0):
        k = ("noise", key, scale)
        if k not in self._cache:
            rng = np.random.default_rng(seed)
            n = rng.random((max(2, self.H // (scale * 4)), max(2, self.W // (scale * 4)))).astype(np.float32)
            n = cv2.resize(n, (self.W // 4, self.H // 4), interpolation=cv2.INTER_CUBIC)
            n = cv2.GaussianBlur(n, (0, 0), 2)
            n = (n - n.min()) / (n.max() - n.min() + 1e-6)
            self._cache[k] = n
        return self._cache[k]

    def _remap(self, img, mx, my):
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # ------------------------------------------------------------------------------------------
    # BASIC
    # ------------------------------------------------------------------------------------------
    def _cut(self, a, b, p):
        return a if p < 0.5 else b

    def _crossfade(self, a, b, p):
        return _fade(a, b, smooth(p))

    def _dip_to_black(self, a, b, p):
        if p < 0.5:
            return cv2.convertScaleAbs(a, alpha=1 - smooth(p / 0.5))
        return cv2.convertScaleAbs(b, alpha=smooth((p - 0.5) / 0.5))

    def _dip_to_white(self, a, b, p):
        if p < 0.5:
            return _fade(a, self.white, smooth(p / 0.5))
        return _fade(self.white, b, smooth((p - 0.5) / 0.5))

    def _flash(self, a, b, p):
        base = a if p < 0.5 else b
        i = bell(p, 0.8) * 0.92
        return cv2.addWeighted(base, 1 - i, self.white, i, 0)

    def _pull_in(self, a, b, p):
        e = ease_in_out_cubic(p)
        za = 1 + 0.45 * e
        zb = 0.7 + 0.3 * e
        return _fade(_zoom(a, za), _zoom(b, zb), smooth((p - 0.3) / 0.4))

    def _pull_out(self, a, b, p):
        e = ease_in_out_cubic(p)
        za = 1 - 0.3 * e
        zb = 1.45 - 0.45 * e
        return _fade(_zoom(a, za), _zoom(b, zb), smooth((p - 0.3) / 0.4))

    def _wipe(self, a, b, p, direction):
        W, H = self.W, self.H
        e = ease_in_out_cubic(p)
        sw, sh = W // 4, H // 4
        if direction in ("left", "right"):
            ramp = np.linspace(0, 1, sw, dtype=np.float32)[None, :].repeat(sh, 0)
            if direction == "left":
                ramp = 1 - ramp
        elif direction in ("up", "down"):
            ramp = np.linspace(0, 1, sh, dtype=np.float32)[:, None].repeat(sw, 1)
            if direction == "up":
                ramp = 1 - ramp
        else:  # diagonal
            xs, ys = np.meshgrid(np.linspace(0, 1, sw), np.linspace(0, 1, sh))
            ramp = ((xs + ys) / 2).astype(np.float32)
        f = 0.08
        m = np.clip((e * (1 + f) - ramp) / f, 0, 1)
        return _mix(a, b, _mask_from_small((m * 255).astype(np.uint8), W, H, 1.0))

    def _wipe_left(self, a, b, p):
        return self._wipe(a, b, p, "left")

    def _wipe_right(self, a, b, p):
        return self._wipe(a, b, p, "right")

    def _wipe_up(self, a, b, p):
        return self._wipe(a, b, p, "up")

    def _wipe_down(self, a, b, p):
        return self._wipe(a, b, p, "down")

    def _wipe_diagonal(self, a, b, p):
        return self._wipe(a, b, p, "diag")

    def _barn_door(self, a, b, p):
        W, H = self.W, self.H
        e = ease_in_out_cubic(p)
        sw, sh = W // 4, H // 4
        ramp = np.abs(np.linspace(-1, 1, sw, dtype=np.float32))[None, :].repeat(sh, 0)
        m = np.clip((e * 1.1 - ramp) / 0.1, 0, 1)
        return _mix(a, b, _mask_from_small((m * 255).astype(np.uint8), W, H, 1.0))

    # ------------------------------------------------------------------------------------------
    # CAMERA
    # ------------------------------------------------------------------------------------------
    def _zblur(self, img, z, strength, n=4, angle=0.0):
        W, H = self.W, self.H
        if strength < 0.01 and abs(angle) < 0.01:
            return _zoom(img, z)
        sw, sh = W // 2, H // 2
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        acc = np.zeros((sh, sw, 3), np.float32)
        for i in range(n):
            k = i / (n - 1)
            M = cv2.getRotationMatrix2D((sw / 2, sh / 2), angle * k, z * (1 + strength * k))
            acc += cv2.warpAffine(small, M, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return cv2.resize((acc / n).astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)

    def _zoom_punch(self, a, b, p):
        if p < 0.5:
            q = p / 0.5
            return self._zblur(a, 1 + 0.6 * q * q, 0.28 * q)
        q = (p - 0.5) / 0.5
        return self._zblur(b, 1.6 - 0.6 * ease_out(q), 0.28 * (1 - q))

    def _whip(self, a, b, p, direction):
        e = ease_in_out_cubic(p)
        W, H = self.W, self.H
        out = np.empty_like(a)
        if direction in ("left", "right"):
            s = int(round(e * W))
            if direction == "left":
                out[:, : W - s] = a[:, s:]
                out[:, W - s:] = b[:, :s]
            else:
                out[:, s:] = a[:, : W - s]
                out[:, :s] = b[:, W - s:]
            k = int(math.sin(math.pi * p) * 0.2 * W)
            return cv2.blur(out, (k, 1)) if k > 2 else out
        s = int(round(e * H))
        if direction == "up":
            out[: H - s] = a[s:]
            out[H - s:] = b[:s]
        else:
            out[s:] = a[: H - s]
            out[:s] = b[H - s:]
        k = int(math.sin(math.pi * p) * 0.2 * H)
        return cv2.blur(out, (1, k)) if k > 2 else out

    def _whip_left(self, a, b, p):
        return self._whip(a, b, p, "left")

    def _whip_right(self, a, b, p):
        return self._whip(a, b, p, "right")

    def _whip_up(self, a, b, p):
        return self._whip(a, b, p, "up")

    def _whip_down(self, a, b, p):
        return self._whip(a, b, p, "down")

    def _shake(self, a, b, p):
        W = self.W
        amp = math.sin(math.pi * p) * 0.05 * W
        rng = np.random.default_rng(int(p * 1000) + 7)
        dx, dy = rng.uniform(-amp, amp), rng.uniform(-amp, amp)
        rot = rng.uniform(-3, 3) * math.sin(math.pi * p)
        base = (a if p < 0.5 else b)
        img = _zoom(base, 1.0 + 0.14 * math.sin(math.pi * p), rot)
        img = _shift(img, dx, dy, cv2.BORDER_REFLECT)
        k = int(abs(dx) * 0.6)
        return cv2.blur(img, (k, 1)) if k > 2 else img

    def _spin(self, a, b, p):
        if p < 0.5:
            q = ease_in_out_cubic(p / 0.5)
            return self._spin_img(a, 120 * q, 1 + 0.3 * q, q)
        q = 1 - ease_in_out_cubic((p - 0.5) / 0.5)
        return self._spin_img(b, -120 * q, 1 + 0.3 * q, q)

    def _spin_img(self, img, angle, z, blur):
        W, H = self.W, self.H
        sw, sh = W // 2, H // 2
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        n = 5
        acc = np.zeros((sh, sw, 3), np.float32)
        span = 18 * blur
        for i in range(n):
            M = cv2.getRotationMatrix2D((sw / 2, sh / 2), angle - span * i / (n - 1), z)
            acc += cv2.warpAffine(small, M, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return cv2.resize((acc / n).astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)

    def _persp(self, img, angle_deg, axis="y", canvas=None, shade=True):
        """Render img rotated in 3D around the vertical (y) axis through its centre."""
        W, H = self.W, self.H
        th = math.radians(angle_deg)
        f = 1.8 * W
        pts = []
        for x, y in ((-W / 2, -H / 2), (W / 2, -H / 2), (W / 2, H / 2), (-W / 2, H / 2)):
            xr, zr = x * math.cos(th), x * math.sin(th)
            s = f / (f + zr)
            pts.append((W / 2 + xr * s, H / 2 + y * s))
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        M = cv2.getPerspectiveTransform(src, np.float32(pts))
        canvas = self.black.copy() if canvas is None else canvas
        warped = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        m = cv2.warpPerspective(np.full((H // 2, W // 2), 255, np.uint8),
                                M @ np.diag([2.0, 2.0, 1.0]).astype(np.float64), (W, H))
        if shade:
            warped = cv2.convertScaleAbs(warped, alpha=0.55 + 0.45 * abs(math.cos(th)))
        return _mix(canvas, warped, m)

    def _flip_3d(self, a, b, p):
        e = ease_in_out_cubic(p)
        if e < 0.5:
            return self._persp(a, 90 * (e / 0.5))
        return self._persp(b, -90 * (1 - (e - 0.5) / 0.5))

    def _cube_3d(self, a, b, p):
        W, H = self.W, self.H
        phi = math.radians(90 * ease_in_out_cubic(p))
        f, D = 1.5 * W, 1.5 * W + W / 2
        canvas = self.black.copy()
        faces = []
        for img, a0 in ((a, 0.0), (b, -math.pi / 2)):
            ang = a0 + phi
            nz = math.cos(ang)  # face normal z (towards camera)
            if nz <= 0.01:
                continue
            pts = []
            for u, v in ((-W / 2, -H / 2), (W / 2, -H / 2), (W / 2, H / 2), (-W / 2, H / 2)):
                # face point before rotation: front face (u, v, W/2); rotate around Y by ang
                x0, z0 = u, W / 2
                x = x0 * math.cos(ang) + z0 * math.sin(ang)
                z = -x0 * math.sin(ang) + z0 * math.cos(ang)
                d = D - z
                pts.append((W / 2 + x * f / d, H / 2 + v * f / d))
            faces.append((nz, img, pts))
        for nz, img, pts in sorted(faces, key=lambda t: t[0]):
            M = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), np.float32(pts))
            warped = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR)
            m = cv2.warpPerspective(np.full((H, W), 255, np.uint8), M, (W, H))
            canvas = _mix(canvas, cv2.convertScaleAbs(warped, alpha=0.5 + 0.5 * nz), m)
        return canvas

    def _speed_sweep(self, a, b, p):
        out = self._whip(a, b, p, "left")
        return _zoom(out, 1 + 0.25 * math.sin(math.pi * p))

    # ------------------------------------------------------------------------------------------
    # LIGHT
    # ------------------------------------------------------------------------------------------
    def _leak_frame(self, p):
        sw, sh = 90, 160
        if "leak" not in self._cache:
            r = self.rng
            self._cache["leak"] = [(r.uniform(-0.3, 0.3), r.uniform(0.1, 0.9), r.uniform(0.45, 0.8), c)
                                   for c in [(40, 140, 255), (30, 60, 255), (120, 210, 255)]]
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        xx /= sw
        yy /= sh
        acc = np.zeros((sh, sw, 3), np.float32)
        for i, (x0, y0, rad, col) in enumerate(self._cache["leak"]):
            cx = x0 + p * (1.3 if i % 2 == 0 else -0.6) + (0.0 if i % 2 == 0 else 0.9)
            g = np.exp(-(((xx - cx) / rad) ** 2 + ((yy - y0) / (rad * 1.6)) ** 2) * 2.2)[..., None]
            acc += g * np.array(col, np.float32)[None, None, :]
        return np.clip(acc, 0, 255)

    def _light_leak(self, a, b, p):
        base = _fade(a, b, smooth(p))
        return _add_light(base, self._leak_frame(p), math.sin(math.pi * p) ** 1.4)

    def _lens_flare(self, a, b, p):
        W, H = self.W, self.H
        base = _fade(a, b, smooth((p - 0.3) / 0.4))
        sw, sh = W // 4, H // 4
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        fx, fy = (-0.2 + 1.4 * p) * sw, 0.38 * sh
        streak = np.exp(-((yy - fy) / (sh * 0.012)) ** 2) * np.exp(-((xx - fx) / (sw * 0.9)) ** 2)
        core = np.exp(-(((xx - fx) ** 2 + (yy - fy) ** 2) / (2 * (sw * 0.06) ** 2)))
        layer = streak[..., None] * np.array([255, 200, 120], np.float32) + core[..., None] * np.array([230, 245, 255],
                                                                                                         np.float32)
        cx, cy = sw / 2, sh / 2
        for k, (sz, col) in enumerate(((0.05, (120, 255, 160)), (0.09, (255, 140, 200)), (0.035, (120, 200, 255)))):
            gx, gy = cx - (fx - cx) * (0.5 + 0.4 * k), cy - (fy - cy) * (0.5 + 0.4 * k)
            ring = np.exp(-(((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * (sw * sz) ** 2)))
            layer += ring[..., None] * np.array(col, np.float32) * 0.35
        return _add_light(base, layer, 0.4 + 0.9 * bell(p, 0.9))

    def _film_burn(self, a, b, p):
        n = self._noise("burn", 6, seed=3)
        sh, sw = n.shape
        grad = np.linspace(0, 1, sw, dtype=np.float32)[None, :].repeat(sh, 0)
        field = 0.65 * grad + 0.35 * n
        burn = np.clip((p * 1.7 - field) / 0.3, 0, 1)
        edge = np.clip(burn * (1 - burn) * 4, 0, 1)
        base = _fade(a, b, smooth((p - 0.4) / 0.25))
        hot = np.clip(burn * bell(p, 0.7), 0, 1)
        layer = (edge[..., None] * np.array([30, 120, 255], np.float32) * 1.1 +
                 hot[..., None] * np.array([150, 220, 255], np.float32))
        return _add_light(base, layer, 1.0)

    def _glow(self, img, k):
        W, H = self.W, self.H
        sm = cv2.GaussianBlur(cv2.resize(img, (W // 8, H // 8), interpolation=cv2.INTER_AREA), (0, 0), 3)
        return cv2.addWeighted(img, 1.0, _up(sm, W, H), k, 0)

    def _glow_flash(self, a, b, p):
        base = _fade(a, b, smooth((p - 0.4) / 0.2))
        i = bell(p, 0.9)
        out = self._glow(base, 1.6 * i)
        return cv2.addWeighted(out, 1 - 0.35 * i, self.white, 0.35 * i, 0)

    def _heatwave_flash(self, a, b, p):
        xs, ys, *_ = self._grid()
        i = bell(p, 0.9)
        amp = i * 0.012 * self.W
        my = ys + amp * np.sin(xs / (self.W * 0.04) + p * 20).astype(np.float32)
        mx = xs + amp * 0.6 * np.sin(ys / (self.H * 0.03) + p * 14).astype(np.float32)
        base = self._remap(_fade(a, b, smooth((p - 0.4) / 0.2)), mx, my)
        warm = np.zeros_like(base)
        warm[:] = (40, 150, 255)
        out = cv2.addWeighted(base, 1.0, warm, 0.55 * i, 0)
        return cv2.addWeighted(out, 1 - 0.3 * i, self.white, 0.3 * i, 0)

    def _flash_blast(self, a, b, p):
        W, H = self.W, self.H
        base = a if p < 0.42 else b
        sw, sh = W // 4, H // 4
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        d = np.sqrt((xx - sw / 2) ** 2 + (yy - sh / 2) ** 2) / math.hypot(sw / 2, sh / 2)
        r = ease_out(p, 2) * 1.3
        ring = np.exp(-((d - r) / 0.08) ** 2) * (1 - p)
        core = np.exp(-(d / max(0.05, 0.6 * bell(p, 0.6))) ** 2) * bell(p, 0.6)
        layer = (ring + core)[..., None] * 255.0
        out = _add_light(base, layer, 1.0)
        return cv2.addWeighted(out, 1 - 0.5 * bell(p, 0.35), self.white, 0.5 * bell(p, 0.35), 0)

    # ------------------------------------------------------------------------------------------
    # BLUR
    # ------------------------------------------------------------------------------------------
    def _blur_dissolve(self, a, b, p):
        r = math.sin(math.pi * p)
        out = _fade(_blur(a, r), _blur(b, r), smooth(p))
        return cv2.convertScaleAbs(out, alpha=1 + 0.12 * r)

    def _focus_blur(self, a, b, p):
        if p < 0.5:
            img = _blur(a, smooth(p / 0.5) * 0.9)
        else:
            img = _blur(b, (1 - smooth((p - 0.5) / 0.5)) * 0.9)
        return img

    def _motion_blur(self, a, b, p):
        k = int(math.sin(math.pi * p) * 0.18 * self.W)
        out = _fade(a, b, smooth(p))
        return cv2.blur(out, (k, 1)) if k > 2 else out

    def _chromatic_blur(self, a, b, p):
        r = math.sin(math.pi * p)
        img = _blur(_fade(a, b, smooth(p)), 0.4 * r)
        s = int(r * 0.03 * self.W)
        if s:
            bc, gc, rc = cv2.split(img)
            img = cv2.merge([_shift(bc, -s, 0, cv2.BORDER_REFLECT), gc, _shift(rc, s, 0, cv2.BORDER_REFLECT)])
        return img

    def _zoom_blur(self, a, b, p):
        r = math.sin(math.pi * p)
        za = self._zblur(a, 1 + 0.2 * r, 0.25 * r)
        zb = self._zblur(b, 1 + 0.2 * r, 0.25 * r)
        return _fade(za, zb, smooth(p))

    def _spin_blur(self, a, b, p):
        if p < 0.5:
            q = smooth(p / 0.5)
            return self._spin_img(a, 30 * q, 1 + 0.1 * q, q)
        q = 1 - smooth((p - 0.5) / 0.5)
        return self._spin_img(b, -30 * q, 1 + 0.1 * q, q)

    # ------------------------------------------------------------------------------------------
    # GLITCH
    # ------------------------------------------------------------------------------------------
    def _glitch(self, a, b, p):
        base = (a if p < 0.5 else b).copy()
        amt = math.sin(math.pi * p)
        W, H = self.W, self.H
        rng = np.random.default_rng(int(p * 997) + 3)
        shift = int(amt * 0.035 * W)
        if shift:
            bch, gch, rch = cv2.split(base)
            base = cv2.merge([np.roll(bch, -shift, axis=1), gch, np.roll(rch, shift, axis=1)])
        for _ in range(int(3 + 6 * amt)):
            y = int(rng.integers(0, H - 10))
            h = int(rng.integers(6, max(8, int(H * 0.06))))
            dx = int(rng.integers(-int(0.12 * W * amt) - 1, int(0.12 * W * amt) + 2))
            base[y: y + h] = np.roll(base[y: y + h], dx, axis=1)
        if amt > 0.5:
            base[::3] = (base[::3] * 0.75).astype(np.uint8)
        return base

    def _rgb_split(self, a, b, p):
        base = a if p < 0.5 else b
        amt = math.sin(math.pi * p)
        s = int(amt * 0.07 * self.W)
        bc, gc, rc = cv2.split(base)
        out = cv2.merge([_shift(bc, -s, int(s * 0.3), cv2.BORDER_REFLECT), gc, _shift(rc, s, -int(s * 0.3),
                                                                                    cv2.BORDER_REFLECT)])
        return _zoom(out, 1 + 0.04 * amt)

    def _static(self, a, b, p):
        base = a if p < 0.5 else b
        i = bell(p, 0.8)
        rng = np.random.default_rng(int(p * 1000) + 1)
        n = rng.integers(0, 255, (self.H // 2, self.W // 2), dtype=np.uint8)
        n = cv2.cvtColor(cv2.resize(n, (self.W, self.H), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        roll = np.roll(base, int(i * self.H * 0.15), axis=0)
        return cv2.addWeighted(roll, 1 - 0.85 * i, n, 0.85 * i, 0)

    def _color_glitch(self, a, b, p):
        base = (a if p < 0.5 else b).copy()
        W, H = self.W, self.H
        amt = math.sin(math.pi * p)
        rng = np.random.default_rng(int(p * 333) + 9)
        for _ in range(int(18 * amt)):
            bw, bh = int(rng.integers(W // 12, W // 3)), int(rng.integers(H // 40, H // 8))
            x, y = int(rng.integers(0, W - bw)), int(rng.integers(0, H - bh))
            dx = int(rng.integers(-W // 6, W // 6) * amt)
            sx = int(np.clip(x + dx, 0, W - bw))
            block = base[y: y + bh, sx: sx + bw].copy()
            mode = rng.integers(0, 3)
            if mode == 0:
                block = block[..., [2, 0, 1]]
            elif mode == 1:
                block = 255 - block
            else:
                block = cv2.convertScaleAbs(block, alpha=1.6, beta=20)
            base[y: y + bh, x: x + bw] = block
        return base

    def _scanline_glitch(self, a, b, p):
        base = (a if p < 0.5 else b)
        amt = math.sin(math.pi * p)
        rng = np.random.default_rng(int(p * 541) + 2)
        offs = (rng.normal(0, 1, self.H // 4) * amt * 0.05 * self.W).astype(np.float32)
        offs = np.repeat(offs, 4)[: self.H]
        if len(offs) < self.H:
            offs = np.pad(offs, (0, self.H - len(offs)))
        xs, ys, *_ = self._grid()
        out = self._remap(base, xs + offs[:, None], ys)
        out[::2] = (out[::2] * (1 - 0.3 * amt)).astype(np.uint8)
        return out

    def _pixelate(self, a, b, p):
        base = a if p < 0.5 else b
        amt = math.sin(math.pi * p)
        blk = max(1, int(2 + amt * 0.06 * self.W))
        if blk <= 2:
            return base
        sm = cv2.resize(base, (max(2, self.W // blk), max(2, self.H // blk)), interpolation=cv2.INTER_AREA)
        return cv2.resize(sm, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

    # ------------------------------------------------------------------------------------------
    # DISTORTION
    # ------------------------------------------------------------------------------------------
    def _ripple(self, a, b, p):
        xs, ys, dx, dy, r, rmax = self._grid()
        amp = bell(p, 0.9) * 0.025 * self.W
        off = amp * np.sin(r / (self.W * 0.05) - p * 25)
        rn = np.maximum(r, 1)
        mx, my = xs + dx / rn * off, ys + dy / rn * off
        return self._remap(_fade(a, b, smooth((p - 0.35) / 0.3)), mx.astype(np.float32), my.astype(np.float32))

    def _wave(self, a, b, p):
        xs, ys, *_ = self._grid()
        amp = bell(p, 0.9) * 0.06 * self.W
        mx = (xs + amp * np.sin(ys / (self.H * 0.05) + p * 12)).astype(np.float32)
        return self._remap(_fade(a, b, smooth((p - 0.35) / 0.3)), mx, ys)

    def _fisheye(self, a, b, p):
        xs, ys, dx, dy, r, rmax = self._grid()
        k = bell(p, 0.9) * 0.8
        rn = r / rmax
        scale = 1 / (1 + k * (1 - rn) ** 2)
        mx, my = (self.W / 2 + dx * scale).astype(np.float32), (self.H / 2 + dy * scale).astype(np.float32)
        return self._remap(_fade(a, b, smooth((p - 0.4) / 0.2)), mx, my)

    def _twist(self, a, b, p):
        xs, ys, dx, dy, r, rmax = self._grid()
        ang = bell(p, 0.9) * 3.0 * np.clip(1 - r / (rmax * 0.9), 0, 1) ** 2
        c, s = np.cos(ang), np.sin(ang)
        mx = (self.W / 2 + dx * c - dy * s).astype(np.float32)
        my = (self.H / 2 + dx * s + dy * c).astype(np.float32)
        return self._remap(_fade(a, b, smooth((p - 0.4) / 0.2)), mx, my)

    def _stretch(self, a, b, p):
        W, H = self.W, self.H
        if p < 0.5:
            sx, img = 1 + 2.5 * smooth(p / 0.5), a
        else:
            sx, img = 1 + 2.5 * (1 - smooth((p - 0.5) / 0.5)), b
        M = np.float32([[sx, 0, W / 2 * (1 - sx)], [0, 1, 0]])
        out = _affine(img, M, W, H)
        k = int((sx - 1) * 0.05 * W)
        return cv2.blur(out, (k, 1)) if k > 2 else out

    def _liquid(self, a, b, p):
        xs, ys, *_ = self._grid()
        n1, n2 = self._noise("liq1", 3, 5), self._noise("liq2", 3, 6)
        amp = bell(p, 0.9) * 0.08 * self.W
        ox = cv2.resize(((n1 - 0.5) * 2 * amp).astype(np.float32), (self.W, self.H))
        oy = cv2.resize(((n2 - 0.5) * 2 * amp).astype(np.float32), (self.W, self.H))
        return self._remap(_fade(a, b, smooth((p - 0.35) / 0.3)), xs + ox, ys + oy)

    # ------------------------------------------------------------------------------------------
    # SLIDE
    # ------------------------------------------------------------------------------------------
    def _push(self, a, b, p, dx, dy):
        e = ease_in_out_cubic(p)
        W, H = self.W, self.H
        oa = _shift(a, -dx * e * W, -dy * e * H)
        ob = _shift(b, dx * (1 - e) * W, dy * (1 - e) * H)
        m = _shift(np.full((H, W), 255, np.uint8), dx * (1 - e) * W, dy * (1 - e) * H)
        return _mix(oa, ob, m)

    def _push_left(self, a, b, p):
        return self._push(a, b, p, 1, 0)

    def _push_right(self, a, b, p):
        return self._push(a, b, p, -1, 0)

    def _push_up(self, a, b, p):
        return self._push(a, b, p, 0, 1)

    def _push_down(self, a, b, p):
        return self._push(a, b, p, 0, -1)

    def _slide_over(self, a, b, p, dx, dy):
        e = ease_in_out_cubic(p)
        W, H = self.W, self.H
        under = cv2.convertScaleAbs(_shift(a, -dx * 0.3 * e * W, -dy * 0.3 * e * H, cv2.BORDER_REFLECT),
                                    alpha=1 - 0.45 * e)
        ob = _shift(b, dx * (1 - e) * W, dy * (1 - e) * H)
        m = _shift(np.full((H, W), 255, np.uint8), dx * (1 - e) * W, dy * (1 - e) * H)
        return _mix(under, ob, m)

    def _slide_left(self, a, b, p):
        return self._slide_over(a, b, p, 1, 0)

    def _slide_right(self, a, b, p):
        return self._slide_over(a, b, p, -1, 0)

    def _slide_up(self, a, b, p):
        return self._slide_over(a, b, p, 0, 1)

    def _slide_down(self, a, b, p):
        return self._slide_over(a, b, p, 0, -1)

    def _slide_diagonal(self, a, b, p):
        return self._slide_over(a, b, p, 1, 1)

    def _screen_swap(self, a, b, p):
        W, H = self.W, self.H
        e = ease_in_out_cubic(p)
        if e < 0.01 or e > 0.99:
            return a if e < 0.5 else b
        k = math.sin(math.pi * e)
        s = 1 - 0.32 * k
        bg = cv2.convertScaleAbs(_blur(a if e < 0.5 else b, 0.6), alpha=1 - 0.5 * k)
        ca = (W / 2 - e * W * 0.9, H / 2)
        cb = (W / 2 + (1 - e) * W * 0.9, H / 2)
        out = _card(a, s, ca[0], ca[1], W, H, radius=0.04 * k, canvas=bg)
        return _card(b, s, cb[0], cb[1], W, H, radius=0.04 * k, canvas=out)

    def _split_push(self, a, b, p):
        W, H = self.W, self.H
        e = ease_in_out_cubic(p)
        out = _zoom(b, 1.12 - 0.12 * e)
        h2 = H // 2
        top = _shift(a[:h2], -e * W, 0)
        bot = _shift(a[h2:], e * W, 0)
        mt = _shift(np.full((h2, W), 255, np.uint8), -e * W, 0)
        mb = _shift(np.full((H - h2, W), 255, np.uint8), e * W, 0)
        out[:h2] = _mix(out[:h2], top, mt)
        out[h2:] = _mix(out[h2:], bot, mb)
        return out

    # ------------------------------------------------------------------------------------------
    # MASK
    # ------------------------------------------------------------------------------------------
    def _shape_reveal(self, a, b, p, poly_fn):
        W, H = self.W, self.H
        sw, sh = W // 4, H // 4
        m = np.zeros((sh, sw), np.uint8)
        e = ease_in_out_cubic(p)
        if e > 0.001:
            pts = poly_fn(sw / 2, sh / 2, e * math.hypot(sw, sh) * 1.6)
            cv2.fillPoly(m, [np.int32(pts)], 255, cv2.LINE_AA)
        m = _mask_from_small(m, W, H, 1.5)
        if e > 0.8:  # guarantee the shape ends fully revealed whatever its geometry
            m = np.maximum(m, np.uint8(255 * smooth((e - 0.8) / 0.2)))
        return _mix(a, b, m)

    def _iris(self, a, b, p):
        return self._shape_reveal(a, b, p, lambda cx, cy, s: [
            (cx + s * 0.5 * math.cos(t), cy + s * 0.5 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 64)])

    def _heart(self, a, b, p):
        def poly(cx, cy, s):
            ts = np.linspace(0, 2 * math.pi, 90)
            x = 16 * np.sin(ts) ** 3
            y = -(13 * np.cos(ts) - 5 * np.cos(2 * ts) - 2 * np.cos(3 * ts) - np.cos(4 * ts))
            k = s / 32
            return list(zip(cx + x * k, cy + y * k + s * 0.05))
        return self._shape_reveal(a, b, p, poly)

    def _star(self, a, b, p):
        def poly(cx, cy, s):
            pts = []
            rot = p * 0.8
            for i in range(10):
                r = s * (0.5 if i % 2 == 0 else 0.22)
                t = -math.pi / 2 + i * math.pi / 5 + rot
                pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
            return pts
        return self._shape_reveal(a, b, p, poly)

    def _diamond(self, a, b, p):
        return self._shape_reveal(a, b, p, lambda cx, cy, s: [(cx, cy - s * 0.5), (cx + s * 0.5, cy),
                                                             (cx, cy + s * 0.5), (cx - s * 0.5, cy)])

    def _blinds(self, a, b, p):
        W, H = self.W, self.H
        n = 8
        e = ease_in_out_cubic(p)
        sh = H // 4
        y = (np.arange(sh, dtype=np.float32) / sh * n) % 1.0
        row = np.clip((e * 1.1 - y) / 0.1, 0, 1)
        m = (row[:, None].repeat(W // 4, 1) * 255).astype(np.uint8)
        return _mix(a, b, _mask_from_small(m, W, H, 0.8))

    def _clock_wipe(self, a, b, p):
        W, H = self.W, self.H
        sw, sh = W // 4, H // 4
        m = np.zeros((sh, sw), np.uint8)
        ang = 360 * ease_in_out_cubic(p)
        if ang > 0.5:
            cv2.ellipse(m, (sw // 2, sh // 2), (sw * 2, sh * 2), -90, 0, ang, 255, -1, cv2.LINE_AA)
        return _mix(a, b, _mask_from_small(m, W, H, 1.2))

    def _line_wipe(self, a, b, p):
        W, H = self.W, self.H
        sw, sh = W // 4, H // 4
        n = 7
        xs, ys = np.meshgrid(np.linspace(0, 1, sw), np.linspace(0, 1, sh))
        u = (xs * 0.6 + ys * 0.4)
        k = np.floor(u * n)
        s = u * n - k
        t = np.clip((p * (n + 3) - k) / 3, 0, 1)
        m = (np.clip((t - s) / 0.08 + 0.5, 0, 1) * 255).astype(np.uint8)
        return _mix(a, b, _mask_from_small(m, W, H, 0.8))

    # ------------------------------------------------------------------------------------------
    # OVERLAY / SOCIAL
    # ------------------------------------------------------------------------------------------
    def _paper_tear(self, a, b, p):
        W, H = self.W, self.H
        if "tear" not in self._cache:
            rng = np.random.default_rng(4)
            steps = rng.normal(0, 1, H).cumsum()
            steps -= np.linspace(steps[0], steps[-1], H)
            jag = rng.normal(0, 0.35, H)
            self._cache["tear"] = ((steps / (np.abs(steps).max() + 1e-6)) * 0.06 + jag * 0.006).astype(np.float32)
        e = ease_in_out_cubic(p)
        bx = ((-0.15 + 1.3 * e) * W + self._cache["tear"] * W)[:, None]
        xs = np.arange(W, dtype=np.float32)[None, :]
        reveal = xs < bx
        out = np.where(reveal[..., None], b, a)
        edge = (np.abs(xs - bx) < max(2, W * 0.008))
        out[edge] = (235, 238, 240)
        shadow = (xs > bx) & (xs < bx + W * 0.03)
        out[shadow] = (out[shadow] * 0.7).astype(np.uint8)
        return out.astype(np.uint8)

    def _film_strip(self, a, b, p):
        W, H = self.W, self.H
        e = ease_in_out_cubic(p)
        canvas = np.full((H, W, 3), 18, np.uint8)
        fw, fh = int(W * 0.8), int(H * 0.8)
        pitch = fh + int(H * 0.06)
        fa, fb = cv2.resize(a, (fw, fh)), cv2.resize(b, (fw, fh))
        y0 = int((H - fh) / 2 - e * pitch)
        x0 = (W - fw) // 2
        for img, y in ((fa, y0), (fb, y0 + pitch)):
            ya, yb = max(0, y), min(H, y + fh)
            if yb > ya:
                canvas[ya:yb, x0:x0 + fw] = img[ya - y: yb - y]
        hole_w, hole_h = int(W * 0.035), int(H * 0.018)
        step = int(H * 0.05)
        off = int(-e * pitch) % step
        for y in range(off - step, H + step, step):
            for x in (int(W * 0.03), W - int(W * 0.03) - hole_w):
                cv2.rectangle(canvas, (x, y), (x + hole_w, y + hole_h), (225, 225, 225), -1)
        k = int(bell(p, 0.8) * 0.06 * H)
        out = cv2.blur(canvas, (1, k)) if k > 2 else canvas
        if p < 0.18:
            out = _fade(a, out, smooth(p / 0.18))
        elif p > 0.82:
            out = _fade(out, b, smooth((p - 0.82) / 0.18))
        return out

    def _scrapbook(self, a, b, p):
        W, H = self.W, self.H
        if p < 0.72:
            q = ease_out(p / 0.72, 3)
            bg = cv2.convertScaleAbs(a, alpha=1 - 0.4 * q)
            return _card(b, 0.35 + 0.45 * q, W / 2, H * (1.2 - 0.7 * q), W, H, angle=-10 * (1 - q) + 4,
                         border=int(W * 0.025), canvas=bg)
        q = smooth((p - 0.72) / 0.28)
        card = _card(b, 0.8 + 0.25 * q, W / 2, H / 2, W, H, angle=4 * (1 - q), border=int(W * 0.025 * (1 - q)),
                     canvas=cv2.convertScaleAbs(a, alpha=0.6))
        return _fade(card, b, q)

    def _story_swipe(self, a, b, p):
        W, H = self.W, self.H
        e = ease_in_out_cubic(p)
        bg = _card(a, 1 - 0.08 * e, W / 2 - 0.15 * e * W, H / 2, W, H, radius=0.05 * e, shadow=False,
                   canvas=self.black)
        bg = cv2.convertScaleAbs(bg, alpha=1 - 0.4 * e)
        out = _card(b, 1.0, W / 2 + (1 - e) * W, H / 2, W, H, radius=0.05 * (1 - e), canvas=bg)
        return out

    def _ribs_collage(self, a, b, p):
        W, H = self.W, self.H
        n = 6
        out = a.copy()
        sw = W // n
        for i in range(n):
            q = ease_out(np.clip((p - i * 0.07) / 0.6, 0, 1), 3)
            off = int((1 - q) * -H)
            x0, x1 = i * sw, W if i == n - 1 else (i + 1) * sw
            strip = b[:, x0:x1]
            ya, yb = max(0, off), min(H, off + H)
            if yb > ya:
                out[ya:yb, x0:x1] = strip[ya - off: yb - off]
        return out

    # ------------------------------------------------------------------------------------------
    # TRENDING
    # ------------------------------------------------------------------------------------------
    def _zoom_flash(self, a, b, p):
        out = self._zoom_punch(a, b, p)
        i = bell(p, 0.5) * 0.85
        return cv2.addWeighted(out, 1 - i, self.white, i, 0)

    def _spin_zoom(self, a, b, p):
        if p < 0.5:
            q = ease_in_out_cubic(p / 0.5)
            return self._spin_img(a, 90 * q, 1 + 0.5 * q, q)
        q = 1 - ease_in_out_cubic((p - 0.5) / 0.5)
        return self._spin_img(b, -90 * q, 1 + 0.5 * q, q)

    def _glitch_whip(self, a, b, p):
        out = self._whip(a, b, p, "left")
        return self._rgb_split(out, out, p)

    def _shake_punch(self, a, b, p):
        out = self._zoom_punch(a, b, p)
        return self._shake(out, out, p)
