"""Clip effects: applied over a whole shot (CapCut 'Effects'), several of them synced to the music beat."""
from __future__ import annotations

import math

import cv2
import numpy as np

CLIP_EFFECTS = {
    "none": "no effect",
    "zoom_pulse": "zoom pulse on every beat (beat-synced)",
    "shake": "camera shake hit on every beat (beat-synced)",
    "flash_beat": "white flash on every beat (beat-synced)",
    "rgb_split": "RGB split glitch, stronger on beats",
    "vhs": "VHS tape: scanlines, colour bleed, tracking noise",
    "film_dust": "old film: dust, scratches and flicker",
    "glow": "dreamy soft glow",
    "light_leak": "drifting warm light leak over the shot",
}
BEAT_EFFECTS = {"zoom_pulse", "shake", "flash_beat", "rgb_split"}


class ClipFX:
    def __init__(self, W: int, H: int, seed: int = 5):
        self.W, self.H = W, H
        self.rng = np.random.default_rng(seed)
        self.white = np.full((H, W, 3), 255, np.uint8)
        self._cache: dict = {}

    def apply(self, effect: str, frame: np.ndarray, t: float, since_beat: float, beat_idx: int) -> np.ndarray:
        """t = seconds into the shot; since_beat = seconds since the last beat (large if none)."""
        fn = getattr(self, "_" + (effect or "none"), None)
        if fn is None or effect == "none":
            return frame
        return fn(frame, t, since_beat, beat_idx)

    # -- beat-synced ------------------------------------------------------------------------
    def _zoom_pulse(self, f, t, sb, bi):
        k = math.exp(-sb * 9.0)
        if k < 0.01:
            return f
        M = cv2.getRotationMatrix2D((self.W / 2, self.H / 2), 0, 1 + 0.08 * k)
        return cv2.warpAffine(f, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def _shake(self, f, t, sb, bi):
        k = math.exp(-sb * 11.0)
        if k < 0.02:
            return f
        r = np.random.default_rng(bi * 31 + int(t * 60))
        dx, dy = r.uniform(-1, 1, 2) * 0.03 * self.W * k
        M = cv2.getRotationMatrix2D((self.W / 2, self.H / 2), float(r.uniform(-2, 2) * k), 1.05)
        M[0, 2] += dx
        M[1, 2] += dy
        return cv2.warpAffine(f, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def _flash_beat(self, f, t, sb, bi):
        k = 0.55 * math.exp(-sb * 14.0)
        return f if k < 0.01 else cv2.addWeighted(f, 1 - k, self.white, k, 0)

    def _rgb_split(self, f, t, sb, bi):
        s = int((0.006 + 0.03 * math.exp(-sb * 10.0)) * self.W)
        if s < 1:
            return f
        b, g, r = cv2.split(f)
        return cv2.merge([np.roll(b, -s, axis=1), g, np.roll(r, s, axis=1)])

    # -- looks over time ----------------------------------------------------------------------
    def _vhs(self, f, t, sb, bi):
        W, H = self.W, self.H
        ycc = cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycc)
        sw = max(2, W // 6)
        cr = cv2.resize(cv2.resize(cr, (sw, H)), (W, H))  # chroma bleed
        cb = cv2.resize(cv2.resize(cb, (sw, H)), (W, H))
        cr = np.roll(cr, max(1, W // 120), axis=1)
        out = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)
        out = cv2.addWeighted(out, 0.9, cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR), 0.1, 0)
        out[::3] = (out[::3] * 0.82).astype(np.uint8)  # scanlines
        r = np.random.default_rng(int(t * 30) + 3)
        n = r.integers(-14, 14, (H // 2, W // 2, 1), dtype=np.int16)
        n = cv2.resize(n.astype(np.float32), (W, H))[..., None]
        out = np.clip(out.astype(np.float32) + n, 0, 255).astype(np.uint8)
        y0 = int((t * 0.35 % 1.0) * H)  # rolling tracking band
        band = slice(y0, min(H, y0 + max(4, H // 40)))
        out[band] = np.roll(out[band], int(r.integers(-W // 30, W // 30 + 1)), axis=1)
        out[band] = cv2.addWeighted(out[band], 0.7, np.full_like(out[band], 230), 0.3, 0)
        return out

    def _film_dust(self, f, t, sb, bi):
        W, H = self.W, self.H
        r = np.random.default_rng(int(t * 24) + 11)
        out = cv2.convertScaleAbs(f, alpha=float(1 + r.uniform(-0.04, 0.04)))  # flicker
        for _ in range(int(r.integers(3, 12))):
            x, y = int(r.integers(0, W)), int(r.integers(0, H))
            cv2.circle(out, (x, y), int(r.integers(1, max(2, W // 300) + 2)),
                       (25, 25, 25) if r.random() < 0.6 else (220, 220, 220), -1, cv2.LINE_AA)
        if r.random() < 0.5:
            x = int(r.integers(0, W))
            cv2.line(out, (x, 0), (x + int(r.integers(-W // 50, W // 50)), H), (200, 200, 200), 1, cv2.LINE_AA)
        return out

    def _glow(self, f, t, sb, bi):
        W, H = self.W, self.H
        sm = cv2.GaussianBlur(cv2.resize(f, (W // 8, H // 8), interpolation=cv2.INTER_AREA), (0, 0), 2.5)
        glow = cv2.resize(sm, (W, H), interpolation=cv2.INTER_LINEAR)
        return cv2.addWeighted(f, 0.85, glow, 0.4, 4)

    def _light_leak(self, f, t, sb, bi):
        W, H = self.W, self.H
        sw, sh = 60, 107
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        xx /= sw
        yy /= sh
        cx = 0.5 + 0.55 * math.sin(t * 0.7)
        cy = 0.3 + 0.2 * math.cos(t * 0.5)
        g = np.exp(-(((xx - cx) / 0.45) ** 2 + ((yy - cy) / 0.7) ** 2) * 2)[..., None]
        layer = g * np.array([40, 130, 255], np.float32) * (0.45 + 0.15 * math.sin(t * 2.1))
        layer = cv2.resize(layer, (W, H), interpolation=cv2.INTER_CUBIC)
        return cv2.add(f, np.clip(layer, 0, 255).astype(np.uint8))
