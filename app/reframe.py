"""Smart 9:16 reframing: find the subject over time and produce a smooth virtual-camera path."""
from __future__ import annotations

import math
import os

import cv2
import numpy as np

from .video_io import FrameReader, MediaInfo


class FaceDetector:
    """OpenCV YuNet face detector (tiny ONNX model). Falls back to 'unavailable' gracefully."""

    def __init__(self, model_path: str | os.PathLike | None):
        self.det = None
        if model_path and os.path.exists(model_path) and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.det = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), 0.7, 0.3, 50)
            except Exception:
                self.det = None

    @property
    def available(self) -> bool:
        return self.det is not None

    def detect(self, img: np.ndarray):
        """Return list of (cx, cy, w, h, score) normalised to [0,1]."""
        if self.det is None:
            return []
        h, w = img.shape[:2]
        try:
            self.det.setInputSize((w, h))
            _, faces = self.det.detect(img)
        except Exception:
            return []
        if faces is None:
            return []
        out = []
        for f in faces:
            x, y, fw, fh, score = float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1])
            if fw * fh < (w * h) * 0.0008:
                continue
            out.append(((x + fw / 2) / w, (y + fh / 2) / h, fw / w, fh / h, score))
        return out


def saliency_map(img: np.ndarray, prev_gray: np.ndarray | None = None):
    """Spectral-residual saliency blended with (camera-motion-compensated) motion energy."""
    small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    F = np.fft.fft2(g)
    la = np.log(np.abs(F) + 1e-6)
    ph = np.angle(F)
    sr = la - cv2.blur(la, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(sr + 1j * ph))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 2.5)
    sal /= sal.max() + 1e-8
    if prev_gray is not None:
        mot = np.abs(g - prev_gray)
        mot = np.maximum(mot - np.median(mot) * 1.5, 0)
        mot = cv2.GaussianBlur(mot, (0, 0), 2.5)
        if mot.max() > 4:
            sal = 0.55 * sal + 0.45 * (mot / mot.max())
    # gentle centre bias (subjects are usually framed near the centre)
    xs = np.linspace(-1, 1, 64)
    bias = np.exp(-(xs[None, :] ** 2) * 0.8) * np.exp(-(xs[:, None] ** 2) * 0.5)
    return sal * (0.6 + 0.4 * bias), g


def _centroid(m: np.ndarray):
    thr = m.mean() + m.std()
    w = np.where(m > thr, m - thr, 0)
    s = w.sum()
    if s < 1e-6:
        return 0.5, 0.5, 0.0
    xs = (np.arange(m.shape[1]) + 0.5) / m.shape[1]
    ys = (np.arange(m.shape[0]) + 0.5) / m.shape[0]
    cx = float((w.sum(0) * xs).sum() / s)
    cy = float((w.sum(1) * ys).sum() / s)
    conf = float(min(1.0, (m.max() - m.mean()) * 2))
    return cx, cy, conf


class CropPath:
    """Normalised centre (cx, cy) of the 9:16 window as a function of source time (relative)."""

    def __init__(self, ts, cxs, cys):
        self.ts = np.asarray(ts, np.float64)
        self.cxs = np.asarray(cxs, np.float64)
        self.cys = np.asarray(cys, np.float64)

    def __call__(self, t: float):
        if len(self.ts) == 1:
            return float(self.cxs[0]), float(self.cys[0])
        return float(np.interp(t, self.ts, self.cxs)), float(np.interp(t, self.ts, self.cys))

    @classmethod
    def static(cls, cx=0.5, cy=0.5):
        return cls([0.0], [cx], [cy])


def _gauss_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if len(x) < 3 or sigma < 0.5:
        return x.copy()
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")


def _camera_path(target: np.ndarray, crop_n: float, sample_fps: float) -> np.ndarray:
    sm = _gauss_smooth(target, 0.6 * sample_fps)
    if sm.max() - sm.min() < crop_n * 0.3:
        return np.full_like(sm, float(np.median(target)))
    return sm


def compute_crop_path(info: MediaInfo, src_in: float, src_dur: float, crop_wn: float, crop_hn: float,
                      face_det: FaceDetector | None, hint_x: float | None = None, sample_fps: float = 6.0) -> CropPath:
    """Track the subject inside [src_in, src_in+src_dur] and return a smoothed window-centre path.

    crop_wn / crop_hn: window size normalised by source width / height (at zoom 1).
    """
    hx = 0.5 if hint_x is None else float(np.clip(hint_x, 0, 1))
    need_x = crop_wn < 0.97
    need_y = crop_hn < 0.97
    if not (need_x or need_y):
        return CropPath.static(0.5, 0.5)
    sw = 320
    sh = max(2, int(round(sw * info.height / max(1, info.width) / 2)) * 2)
    n = max(2, int(math.ceil(src_dur * sample_fps)) + 1)
    reader = FrameReader(info.path, src_in, src_dur, sw, sh, speed=1.0, fps=sample_fps, src_fps=info.fps,
                         sar_filter=info.sar_filter, blend_slowmo=False, scale_flags="area")
    tx, ty, conf = [], [], []
    prev = None
    try:
        for _ in range(n):
            fr = reader.read()
            if fr is None:
                break
            faces = face_det.detect(fr) if (face_det and face_det.available) else []
            if faces:
                faces.sort(key=lambda f: f[2] * f[3], reverse=True)
                big = faces[0]
                group = [f for f in faces if f[2] * f[3] > big[2] * big[3] * 0.35]
                xs = [f[0] for f in group]
                span = (max(xs) - min(xs)) + big[2]
                if span < crop_wn * 0.9:
                    cx = (max(xs) + min(xs)) / 2
                else:
                    cx = big[0]
                cy = big[1] + big[3] * 0.35  # keep some headroom: put eyes in the upper third
                tx.append(cx)
                ty.append(cy)
                conf.append(1.0)
                _, prev = saliency_map(fr, prev)
            else:
                m, prev = saliency_map(fr, prev)
                cx, cy, c = _centroid(m)
                # trust confident saliency/motion; fall back towards the planner's hint
                # (Claude looked at the thumbnail) when the frame has no clear subject
                wgt = float(np.clip(0.25 + 0.7 * c, 0.0, 0.9))
                tx.append(wgt * cx + (1 - wgt) * hx)
                ty.append(wgt * cy + (1 - wgt) * 0.45)
                conf.append(c * 0.5)
    finally:
        reader.close()
    if not tx:
        return CropPath.static(hx, 0.5)
    tx, ty = np.array(tx), np.array(ty)
    # median filter to kill detector flicker
    if len(tx) >= 5:
        pad_x = np.pad(tx, 2, mode="edge")
        pad_y = np.pad(ty, 2, mode="edge")
        tx = np.array([np.median(pad_x[i: i + 5]) for i in range(len(tx))])
        ty = np.array([np.median(pad_y[i: i + 5]) for i in range(len(ty))])
    # Offline, so use zero-phase smoothing (no lag). If the subject fits in a locked-off frame for the
    # whole shot, keep the camera static - steadier and more cinematic than constant micro-pans.
    px = _camera_path(tx, crop_wn, sample_fps)
    py = _camera_path(ty, crop_hn, sample_fps)
    px = np.clip(px, crop_wn / 2, 1 - crop_wn / 2) if need_x else np.full_like(px, 0.5)
    py = np.clip(py, crop_hn / 2, 1 - crop_hn / 2) if need_y else np.full_like(py, 0.5)
    ts = np.arange(len(px)) / sample_fps
    return CropPath(ts, px, py)


def motion_transform(kind: str, p: float, t: float, seed: int = 0):
    """Return (zoom, dx, dy) for a camera-move effect. dx/dy are fractions of the window size."""
    def ease(x):
        return 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, x)))

    if kind == "push_in":
        return 1.0 + 0.13 * ease(p), 0.0, 0.0
    if kind == "pull_out":
        return 1.13 - 0.13 * ease(p), 0.0, 0.0
    if kind == "punch_in":
        q = min(1.0, p / 0.18)
        return 1.0 + 0.16 * (1 - (1 - q) ** 4), 0.0, 0.0
    if kind == "drift_left":
        return 1.1, 0.04 - 0.08 * ease(p), 0.0
    if kind == "drift_right":
        return 1.1, -0.04 + 0.08 * ease(p), 0.0
    if kind == "rise":
        return 1.1, 0.0, 0.035 - 0.07 * ease(p)
    if kind == "ken_burns":
        return 1.02 + 0.12 * ease(p), (-0.03 + 0.06 * ease(p)) * (1 if seed % 2 else -1), -0.015 * ease(p)
    if kind == "handheld":
        r = seed * 1.37
        dx = 0.010 * (math.sin(t * 1.9 + r) + 0.5 * math.sin(t * 4.3 + 2 * r))
        dy = 0.008 * (math.sin(t * 1.4 + 3 * r) + 0.5 * math.sin(t * 3.7 + r))
        return 1.06, dx, dy
    return 1.0, 0.0, 0.0


MOTIONS = {
    "none": "locked-off, no camera move",
    "push_in": "slow cinematic push-in (dolly in)",
    "pull_out": "slow pull-out / reveal",
    "punch_in": "quick punch-in zoom at the start (emphasis on a beat)",
    "drift_left": "slow lateral drift left",
    "drift_right": "slow lateral drift right",
    "rise": "slow upward tilt/rise",
    "ken_burns": "slow zoom + pan (photos, landscapes)",
    "handheld": "subtle organic handheld shake",
}
