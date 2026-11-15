"""Footage analysis: one decoding pass per clip -> scene cuts, shots, quality scores, thumbnails."""
from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .effects import load_font
from .reframe import FaceDetector
from .video_io import MediaInfo, sample_frames

MAX_SHOT = 6.0       # scenes longer than this are split into windows the planner can choose from
TARGET_WINDOW = 4.0
MIN_SHOT = 0.5


def _detect_cuts(times, scores, min_gap=0.6, ratio_thr=3.0, min_score=14.0):
    """Adaptive content-based cut detection (similar to PySceneDetect's AdaptiveDetector)."""
    s = np.asarray(scores, np.float32)
    cut_idx: list[int] = []
    w = 3
    for i in range(1, len(s)):
        lo, hi = max(1, i - w), min(len(s), i + w + 1)
        neigh = np.concatenate([s[lo:i], s[i + 1: hi]])
        avg = float(neigh.mean()) if len(neigh) else 0.0
        ratio = s[i] / (avg + 1.0)
        if (s[i] >= min_score and ratio >= ratio_thr) or (s[i] >= 45 and ratio >= 1.8):
            if not cut_idx or times[i] - times[cut_idx[-1]] >= min_gap:
                cut_idx.append(i)
            elif s[i] > s[cut_idx[-1]]:  # keep the stronger of two close candidates
                cut_idx[-1] = i
    return [times[i] for i in cut_idx]


def analyze_clip(info: MediaInfo, clip_id: str, face_det: FaceDetector | None, thumbs_dir: Path,
                 progress=None) -> tuple[dict, list[dict]]:
    dur = max(0.1, info.duration)
    every = 0.1 if dur <= 300 else (0.2 if dur <= 1200 else 0.5)
    thumb_every = 0.5 if dur <= 600 else (1.0 if dur <= 1800 else 2.0)
    face_every = 0.5
    times, scores, sharp, bright, sat, motion = [], [], [], [], [], []
    face_samples = []  # (t, has_face, cx)
    thumbs: dict[float, bytes] = {}
    prev_hsv = prev_gray = None
    next_thumb = next_face = 0.0
    for t, fr in sample_frames(info.path, info, every, width=320):
        h = max(2, int(fr.shape[0] * 160 / fr.shape[1]))
        small = cv2.resize(fr, (160, h), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.int16)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16)
        if prev_hsv is None:
            sc, mo = 0.0, 0.0
        else:
            sc = float(np.abs(hsv - prev_hsv).mean())
            mo = float(np.abs(gray - prev_gray).mean())
        prev_hsv, prev_gray = hsv, gray
        times.append(round(t, 3))
        scores.append(sc)
        motion.append(mo)
        g320 = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        sharp.append(float(cv2.Laplacian(g320, cv2.CV_32F).var()))
        bright.append(float(gray.mean()) / 255.0)
        sat.append(float(hsv[..., 1].mean()) / 255.0)
        if t >= next_thumb:
            ok, jpg = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                thumbs[round(t, 3)] = jpg.tobytes()
            next_thumb = t + thumb_every
        if face_det is not None and face_det.available and t >= next_face:
            faces = face_det.detect(fr)
            if faces:
                big = max(faces, key=lambda f: f[2] * f[3])
                face_samples.append((t, 1, big[0]))
            else:
                face_samples.append((t, 0, None))
            next_face = t + face_every
        if progress and len(times) % 25 == 0:
            progress(min(1.0, t / dur))
    if not times:
        raise RuntimeError(f"Could not decode any frames from {os.path.basename(info.path)}")

    # --- scenes ----------------------------------------------------------------------------
    cuts = _detect_cuts(times, scores)
    bounds = [0.0] + [c for c in cuts if 0.3 < c < dur - 0.3] + [dur]
    scenes = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] - bounds[i] >= 0.25]

    # --- shots (scenes split into pickable windows) -------------------------------------
    raw = []
    for si, (a, b) in enumerate(scenes):
        L = b - a
        if L < MIN_SHOT:
            continue
        k = 1 if L <= MAX_SHOT else int(math.ceil(L / TARGET_WINDOW))
        for j in range(k):
            raw.append((si, a + L * j / k, a + L * (j + 1) / k))

    t_arr = np.asarray(times)
    shots = []
    thumb_ts = np.array(sorted(thumbs.keys())) if thumbs else np.array([0.0])
    for n, (si, a, b) in enumerate(raw):
        idx = np.where((t_arr >= a) & (t_arr < b))[0]
        if len(idx) == 0:
            idx = np.array([int(np.argmin(np.abs(t_arr - (a + b) / 2)))])
        # skip the first sample after a cut for motion (it contains the cut itself)
        midx = idx[1:] if len(idx) > 2 else idx
        fs = [f for f in face_samples if a <= f[0] < b]
        face_frac = (sum(f[1] for f in fs) / len(fs)) if fs else 0.0
        fxs = [f[2] for f in fs if f[1]]
        sid = f"{clip_id}s{n:02d}"
        mid = (a + b) / 2
        tt = float(thumb_ts[int(np.argmin(np.abs(thumb_ts - mid)))])
        thumb_path = thumbs_dir / f"{sid}.jpg"
        if thumbs:
            thumb_path.write_bytes(thumbs[tt])
        shots.append({
            "id": sid,
            "clip_id": clip_id,
            "scene": si,
            "start": round(a, 3),
            "end": round(b, 3),
            "duration": round(b - a, 3),
            "sharp_raw": float(np.median([sharp[i] for i in idx])),
            "bright": float(np.mean([bright[i] for i in idx])),
            "sat": float(np.mean([sat[i] for i in idx])),
            "motion_raw": float(np.mean([motion[i] for i in midx])),
            "face_frac": round(face_frac, 2),
            "focus_x": round(float(np.mean(fxs)), 3) if fxs else None,
            "thumb": str(thumb_path),
        })
    if progress:
        progress(1.0)
    w, h = info.width, info.height
    orient = "portrait" if h > w * 1.05 else ("landscape" if w > h * 1.05 else "square")
    clip = {
        "id": clip_id,
        "name": os.path.basename(info.path),
        "path": info.path,
        "duration": round(info.duration, 3),
        "width": w,
        "height": h,
        "fps": round(info.fps, 3),
        "has_audio": info.has_audio,
        "orientation": orient,
        "num_scenes": len(scenes),
        "info": info.to_dict(),
    }
    return clip, shots


def score_shots(shots: list[dict]):
    """Normalise metrics across all footage and compute a 0..1 'usability/beauty' score."""
    if not shots:
        return
    ls = np.log1p([s["sharp_raw"] for s in shots])
    p10, p90 = np.percentile(ls, 10), np.percentile(ls, 90)
    mo = np.array([s["motion_raw"] for s in shots])
    m90 = max(1.0, float(np.percentile(mo, 90)))
    for s, l, m in zip(shots, ls, mo):
        sharp = float(np.clip((l - p10) / (p90 - p10 + 1e-6), 0, 1)) if p90 > p10 else 0.7
        expo = 1.0 - min(1.0, abs(s["bright"] - 0.48) / 0.48) ** 2
        satn = float(np.clip(s["sat"] / 0.45, 0, 1))
        mn = float(np.clip(m / m90, 0, 1.5))
        mscore = mn / 0.5 if mn < 0.5 else max(0.0, 1.0 - (mn - 0.5) * 0.6)  # some motion good, chaos bad
        dur_ok = min(1.0, s["duration"] / 1.5)
        sc = 0.34 * sharp + 0.24 * expo + 0.14 * satn + 0.14 * mscore + 0.08 * s["face_frac"] + 0.06 * dur_ok
        s["sharpness"] = round(sharp, 2)
        s["motion"] = round(min(1.0, mn), 2)
        s["score"] = round(float(np.clip(sc, 0, 1)), 2)


def select_shots(shots: list[dict], max_shots: int, clip_order: list[str]) -> list[dict]:
    """Downselect for the planner: best window of every scene first, then fill by score."""
    if len(shots) <= max_shots:
        return shots
    by_scene: dict = {}
    for s in shots:
        by_scene.setdefault((s["clip_id"], s["scene"]), []).append(s)
    firsts = [max(v, key=lambda s: s["score"]) for v in by_scene.values()]
    firsts.sort(key=lambda s: s["score"], reverse=True)
    chosen = {s["id"] for s in firsts[:max_shots]}
    rest = sorted((s for s in shots if s["id"] not in chosen), key=lambda s: s["score"], reverse=True)
    for s in rest:
        if len(chosen) >= max_shots:
            break
        chosen.add(s["id"])
    order = {c: i for i, c in enumerate(clip_order)}
    return sorted((s for s in shots if s["id"] in chosen), key=lambda s: (order.get(s["clip_id"], 0), s["start"]))


def make_contact_sheets(shots: list[dict], out_dir: Path, per_sheet: int = 12, cols: int = 4,
                        cell: int = 300) -> list[str]:
    """Tile shot thumbnails into labelled sheets so Claude can 'see' the footage cheaply."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    font = load_font("caption", 24)
    small = load_font("light", 19)
    label_h = 40
    for si in range(0, len(shots), per_sheet):
        group = shots[si: si + per_sheet]
        rows = int(math.ceil(len(group) / cols))
        sheet = Image.new("RGB", (cols * (cell + 8) + 8, rows * (cell + label_h + 8) + 8), (18, 18, 18))
        d = ImageDraw.Draw(sheet)
        for i, s in enumerate(group):
            r, c = divmod(i, cols)
            x0 = 8 + c * (cell + 8)
            y0 = 8 + r * (cell + label_h + 8)
            try:
                im = Image.open(s["thumb"]).convert("RGB")
                im.thumbnail((cell, cell))
            except Exception:
                im = Image.new("RGB", (cell, cell // 2), (40, 40, 40))
            sheet.paste(im, (x0 + (cell - im.width) // 2, y0 + (cell - im.height) // 2))
            d.rectangle([x0, y0 + cell, x0 + cell, y0 + cell + label_h], fill=(0, 0, 0))
            d.text((x0 + 8, y0 + cell + 6), s["id"], font=font, fill=(255, 220, 70))
            d.text((x0 + cell - 92, y0 + cell + 10), f"{s['duration']:.1f}s", font=small, fill=(230, 230, 230))
        p = out_dir / f"sheet_{si // per_sheet:02d}.jpg"
        sheet.save(p, quality=84)
        paths.append(str(p))
    return paths
