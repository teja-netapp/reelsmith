"""Cinematic transitions between two BGR frames. Each takes (a=outgoing, b=incoming, p in [0,1])."""
from __future__ import annotations

import math

import cv2
import numpy as np

TRANSITIONS = {
    # name: (default duration seconds, description for the planner)
    "cut": (0.0, "hard cut (default, use for most beat cuts)"),
    "crossfade": (0.5, "soft dissolve, calm/emotional moments"),
    "dip_to_black": (0.6, "fade through black, chapter change or time jump"),
    "flash": (0.25, "white flash cut, energetic drop / impact"),
    "whip_left": (0.3, "fast whip-pan with motion blur, content flies left"),
    "whip_right": (0.3, "fast whip-pan with motion blur, content flies right"),
    "whip_up": (0.3, "vertical whip with motion blur"),
    "zoom_punch": (0.35, "zoom-blur punch into the next shot, hype moments"),
    "slide_up": (0.4, "next shot slides up over the current one, story-style"),
    "light_leak": (0.8, "warm film light-leak burn dissolve, nostalgic/travel"),
    "glitch": (0.25, "RGB-split digital glitch, tech/urban/edgy"),
    "blur_dissolve": (0.5, "dreamy defocus dissolve"),
    "iris": (0.5, "circular reveal from the centre"),
}


def smooth(p):
    p = min(1.0, max(0.0, p))
    return p * p * (3 - 2 * p)


def ease_in_out_cubic(p):
    p = min(1.0, max(0.0, p))
    return 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


class TransitionFX:
    def __init__(self, W: int, H: int, seed: int = 11):
        self.W, self.H = W, H
        self.white = np.full((H, W, 3), 255, np.uint8)
        self.rng = np.random.default_rng(seed)
        self._leak_params = None

    # ------------------------------------------------------------------------------------------
    def apply(self, kind: str, a: np.ndarray, b: np.ndarray, p: float) -> np.ndarray:
        fn = getattr(self, "_" + kind, None)
        if fn is None:
            fn = self._crossfade
        return fn(a, b, float(min(1.0, max(0.0, p))))

    # ------------------------------------------------------------------------------------------
    def _cut(self, a, b, p):
        return a if p < 0.5 else b

    def _crossfade(self, a, b, p):
        e = smooth(p)
        return cv2.addWeighted(a, 1 - e, b, e, 0)

    def _dip_to_black(self, a, b, p):
        if p < 0.5:
            return cv2.convertScaleAbs(a, alpha=1 - smooth(p / 0.5))
        return cv2.convertScaleAbs(b, alpha=smooth((p - 0.5) / 0.5))

    def _flash(self, a, b, p):
        base = a if p < 0.5 else b
        i = math.exp(-((p - 0.5) / 0.2) ** 2) * 0.9
        return cv2.addWeighted(base, 1 - i, self.white, i, 0)

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
            if k > 2:
                out = cv2.blur(out, (k, 1))
        else:
            s = int(round(e * H))
            out[: H - s] = a[s:]
            out[H - s:] = b[:s]
            k = int(math.sin(math.pi * p) * 0.2 * H)
            if k > 2:
                out = cv2.blur(out, (1, k))
        return out

    def _whip_left(self, a, b, p):
        return self._whip(a, b, p, "left")

    def _whip_right(self, a, b, p):
        return self._whip(a, b, p, "right")

    def _whip_up(self, a, b, p):
        return self._whip(a, b, p, "up")

    def _zoom_blur(self, img, z, strength, n=4):
        W, H = self.W, self.H
        if strength < 0.01:
            M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z)
            return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        sw, sh = W // 2, H // 2
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        acc = np.zeros((sh, sw, 3), np.float32)
        for i in range(n):
            zz = z * (1 + strength * i / (n - 1))
            M = cv2.getRotationMatrix2D((sw / 2, sh / 2), 0, zz)
            acc += cv2.warpAffine(small, M, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        acc /= n
        return cv2.resize(acc.astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)

    def _zoom_punch(self, a, b, p):
        if p < 0.5:
            q = p / 0.5
            return self._zoom_blur(a, 1 + 0.6 * q * q, 0.28 * q)
        q = (p - 0.5) / 0.5
        e = 1 - (1 - q) ** 3
        return self._zoom_blur(b, 1.6 - 0.6 * e, 0.28 * (1 - q))

    def _slide_up(self, a, b, p):
        e = ease_in_out_cubic(p)
        H = self.H
        off = int(round(e * H))
        out = np.empty_like(a)
        up = int(off * 0.35)
        dark = cv2.convertScaleAbs(a, alpha=1 - 0.45 * e)
        out[: H - up] = dark[up:]
        out[H - up:] = 0
        if off > 0:
            out[H - off:] = b[:off]
        return out

    def _leak_frame(self, p):
        W, H = self.W, self.H
        sw, sh = 90, 160
        if self._leak_params is None:
            r = self.rng
            self._leak_params = [
                (r.uniform(-0.3, 0.3), r.uniform(0.1, 0.9), r.uniform(0.45, 0.8), c)
                for c in [(40, 140, 255), (30, 60, 255), (120, 210, 255)]  # BGR warm tones
            ]
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        xx /= sw
        yy /= sh
        acc = np.zeros((sh, sw, 3), np.float32)
        for i, (x0, y0, rad, col) in enumerate(self._leak_params):
            # blobs drift across the frame in alternating directions as the transition progresses
            cx = x0 + p * (1.3 if i % 2 == 0 else -0.6) + (0.0 if i % 2 == 0 else 0.9)
            d2 = ((xx - cx) / rad) ** 2 + ((yy - y0) / (rad * 1.6)) ** 2
            g = np.exp(-d2 * 2.2)[..., None]
            acc += g * np.array(col, np.float32)[None, None, :]
        acc = np.clip(acc, 0, 255)
        return cv2.resize(acc, (W, H), interpolation=cv2.INTER_CUBIC)

    def _light_leak(self, a, b, p):
        base = cv2.addWeighted(a, 1 - smooth(p), b, smooth(p), 0)
        inten = math.sin(math.pi * p) ** 1.4
        leak = (self._leak_frame(p) * inten).astype(np.uint8)
        return cv2.add(base, leak)

    def _glitch(self, a, b, p):
        base = (a if p < 0.5 else b).copy()
        amt = math.sin(math.pi * p)
        W, H = self.W, self.H
        rng = np.random.default_rng(int(p * 997) + 3)
        shift = int(amt * 0.035 * W)
        if shift:
            bch, gch, rch = cv2.split(base)
            rch = np.roll(rch, shift, axis=1)
            bch = np.roll(bch, -shift, axis=1)
            base = cv2.merge([bch, gch, rch])
        for _ in range(int(3 + 6 * amt)):
            y = int(rng.integers(0, H - 10))
            h = int(rng.integers(6, max(8, int(H * 0.06))))
            dx = int(rng.integers(-int(0.12 * W * amt) - 1, int(0.12 * W * amt) + 2))
            base[y: y + h] = np.roll(base[y: y + h], dx, axis=1)
        if amt > 0.5:
            base[::3] = (base[::3] * 0.75).astype(np.uint8)
        return base

    def _blur_dissolve(self, a, b, p):
        e = smooth(p)
        r = math.sin(math.pi * p)
        W, H = self.W, self.H
        if r > 0.05:
            s = 1.0 / (1 + 14 * r)
            sz = (max(2, int(W * s)), max(2, int(H * s)))
            a = cv2.resize(cv2.resize(a, sz, interpolation=cv2.INTER_AREA), (W, H), interpolation=cv2.INTER_LINEAR)
            b = cv2.resize(cv2.resize(b, sz, interpolation=cv2.INTER_AREA), (W, H), interpolation=cv2.INTER_LINEAR)
        out = cv2.addWeighted(a, 1 - e, b, e, 0)
        return cv2.convertScaleAbs(out, alpha=1 + 0.12 * r)

    def _iris(self, a, b, p):
        W, H = self.W, self.H
        e = ease_in_out_cubic(p)
        rad = e * math.hypot(W, H) / 2 * 1.02
        mask = np.zeros((H // 4, W // 4), np.uint8)
        cv2.circle(mask, (W // 8, H // 8), int(rad / 4), 255, -1, lineType=cv2.LINE_AA)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.GaussianBlur(mask, (0, 0), 3)
        m3 = cv2.merge([mask, mask, mask])
        inv = cv2.bitwise_not(m3)
        return cv2.add(cv2.multiply(b, m3, scale=1 / 255.0), cv2.multiply(a, inv, scale=1 / 255.0))
