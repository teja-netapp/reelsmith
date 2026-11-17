"""Tiny looping preview clips for the transition / effect pickers (made on demand, cached).

With a job they use that reel's own shots (the two shots either side of the cut), so you see exactly
what the transition will look like on your footage; without one, two synthetic frames.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .clipfx import BEAT_EFFECTS, CLIP_EFFECTS, ClipFX
from .transitions import REGISTRY, TransitionFX
from .video_io import FrameWriter

PW, PH, FPS = 180, 320, 30


def _fit(img: np.ndarray | None, W: int = PW, H: int = PH) -> np.ndarray | None:
    """Centre-crop to 9:16 and resize."""
    if img is None:
        return None
    h, w = img.shape[:2]
    if w / h > W / H:
        cw = int(h * W / H)
        x0 = (w - cw) // 2
        img = img[:, x0:x0 + cw]
    else:
        ch = int(w * H / W)
        y0 = (h - ch) // 2
        img = img[y0:y0 + ch]
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


def synthetic(which: str, W: int = PW, H: int = PH) -> np.ndarray:
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    if which == "a":  # warm sunset with a sun
        t = y / H
        img = np.stack([60 + 40 * t, 110 + 60 * t, 230 - 40 * t], -1)
        d = np.hypot(x - W * 0.62, y - H * 0.38)
        img += (np.clip(1 - d / (W * 0.45), 0, 1) ** 2)[..., None] * np.array([60, 120, 25])
        img[y > H * 0.72] = (40, 55, 70)
    else:  # cool city blues with stripes
        t = x / W
        img = np.stack([200 - 40 * t, 120 + 30 * t, 50 + 20 * t], -1)
        img[(((x + y * 0.5) // 22) % 2 == 0) & (y > H * 0.55)] *= 0.6
    out = np.clip(img, 0, 255).astype(np.uint8)
    cv2.putText(out, which.upper(), (W // 2 - 16, H // 2 + 16), cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 2,
                cv2.LINE_AA)
    return out


def _write(frames, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    w = FrameWriter(str(tmp), PW, PH, FPS, crf=26, preset="veryfast")
    try:
        for f in frames:
            w.write(f)
    finally:
        w.close()
    tmp.replace(out)


def transition_clip(name: str, a: np.ndarray | None, b: np.ndarray | None, out: Path) -> Path:
    """A holds, the transition plays (at its real length), B holds - loops nicely."""
    if name not in REGISTRY:
        raise KeyError(name)
    if out.exists():
        return out
    A, B = _fit(a) if a is not None else synthetic("a"), _fit(b) if b is not None else synthetic("b")
    fx = TransitionFX(PW, PH)
    m = max(8, int(round(max(REGISTRY[name].duration, 0.3) * FPS)))
    frames = [A] * 12 + [fx.apply(name, A, B, (k + 0.5) / m) for k in range(m)] + [B] * 16
    if name == "cut":
        frames = [A] * 20 + [B] * 20
    _write(frames, out)
    return out


def effect_clip(name: str, img: np.ndarray | None, out: Path) -> Path:
    """2 s of the shot with the effect, beats every 0.5 s, a slow push so motion effects read."""
    if name not in CLIP_EFFECTS:
        raise KeyError(name)
    if out.exists():
        return out
    base = _fit(img) if img is not None else synthetic("a")
    big = cv2.resize(base, (int(PW * 1.12), int(PH * 1.12)))
    cfx = ClipFX(PW, PH)
    frames = []
    n = 60
    for k in range(n):
        t = k / FPS
        z = 1.0 + 0.1 * k / n
        bw, bh = int(PW * z), int(PH * z)
        x0, y0 = (big.shape[1] - bw) // 2, (big.shape[0] - bh) // 2
        f = cv2.resize(big[y0:y0 + bh, x0:x0 + bw], (PW, PH))
        since = (t % 0.5) if name in BEAT_EFFECTS else 99.0
        frames.append(cfx.apply(name, f, t, since, int(t / 0.5)))
    _write(frames, out)
    return out
