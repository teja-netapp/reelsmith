"""Renderer: streams every segment through reframe/camera-move, blends transitions, applies the
finishing pass (grade, vignette, grain, letterbox), text and captions, then muxes the soundtrack."""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .audio import mix_audio
from .effects import CaptionTrack, FinishingPass, TextOverlay, fade
from .reframe import CropPath, FaceDetector, compute_crop_path, motion_transform
from .transitions import TransitionFX
from .video_io import FrameReader, FrameWriter, MediaInfo, mux


@dataclass
class RenderSettings:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    crf: int = 18
    preset: str = "medium"
    interp: int = cv2.INTER_CUBIC

    @classmethod
    def preset_named(cls, name: str) -> "RenderSettings":
        if name == "preview":
            return cls(width=540, height=960, crf=23, preset="veryfast", interp=cv2.INTER_LINEAR)
        if name == "draft":
            return cls(width=720, height=1280, crf=21, preset="veryfast", interp=cv2.INTER_LINEAR)
        return cls()


def _segment_frames(seg: dict, idx: int, st: RenderSettings, face_det: FaceDetector | None, smart: bool, log):
    """Generator yielding exactly seg['n_frames'] reframed BGR frames of size (W,H)."""
    W, H, fps = st.width, st.height, st.fps
    info = MediaInfo.from_dict(seg["info"])
    n = seg["n_frames"]
    speed = seg["speed"]
    src_dur = n / fps * speed
    dw, dh = info.width, info.height
    aspect = W / H
    if dw / dh > aspect:
        ch, cw = float(dh), dh * aspect
    else:
        cw, ch = float(dw), dw / aspect
    hint = seg.get("focus_x")
    if smart:
        try:
            path = compute_crop_path(info, seg["src_in"], src_dur, cw / dw, ch / dh, face_det, hint)
        except Exception as e:  # never fail a render because tracking failed
            log(f"reframe tracking failed on segment {idx + 1}: {e}")
            path = CropPath.static(hint if hint is not None else 0.5, 0.5)
    else:
        path = CropPath.static(0.5, 0.5)

    wins = []
    for k in range(n):
        p = k / max(1, n - 1)
        t_src = k / fps * speed
        cxn, cyn = path(t_src)
        z, dx, dy = motion_transform(seg["motion"], p, k / fps, seed=idx)
        w, h = cw / z, ch / z
        cx = cxn * dw + dx * w
        cy = cyn * dh + dy * h
        x0 = min(max(cx - w / 2, 0.0), dw - w)
        y0 = min(max(cy - h / 2, 0.0), dh - h)
        wins.append((x0, y0, w, h))
    bx0 = max(0, int(math.floor(min(w[0] for w in wins))) - 2)
    by0 = max(0, int(math.floor(min(w[1] for w in wins))) - 2)
    bx1 = min(dw, int(math.ceil(max(w[0] + w[2] for w in wins))) + 2)
    by1 = min(dh, int(math.ceil(max(w[1] + w[3] for w in wins))) + 2)
    bw, bh = (bx1 - bx0) // 2 * 2, (by1 - by0) // 2 * 2
    w_min = min(w[2] for w in wins)
    f = min(1.0, W / w_min)  # pre-scale in ffmpeg so the tightest window is ~1:1 with the output
    dec_w = max(2, int(round(bw * f / 2)) * 2)
    dec_h = max(2, int(round(bh * f / 2)) * 2)
    fx, fy = dec_w / bw, dec_h / bh
    reader = FrameReader(info.path, seg["src_in"], src_dur, dec_w, dec_h, crop=(bw, bh, bx0, by0), speed=speed,
                         fps=fps, src_fps=info.fps, sar_filter=info.sar_filter)
    last = None
    try:
        for k in range(n):
            fr = reader.read()
            if fr is None:
                if last is None:
                    err = reader.errors()
                    if err:
                        log(f"decoder: {err.strip()[:300]}")
                    fr = np.zeros((dec_h, dec_w, 3), np.uint8)
                else:
                    fr = last
            last = fr
            x0, y0, w, h = wins[k]
            # window in decoded-frame pixels -> full output frame
            x0d, y0d, wd, hd = (x0 - bx0) * fx, (y0 - by0) * fy, w * fx, h * fy
            sx, sy = W / wd, H / hd
            M = np.array([[sx, 0, -x0d * sx], [0, sy, -y0d * sy]], np.float32)
            yield cv2.warpAffine(fr, M, (W, H), flags=st.interp, borderMode=cv2.BORDER_REPLICATE)
    finally:
        reader.close()


def render(timeline: dict, plan: dict, music: dict | None, out_path: str, work_dir: str,
           settings: RenderSettings, face_det: FaceDetector | None = None, smart_reframe: bool = True,
           progress=None, log=print, metadata: dict | None = None) -> dict:
    W, H, fps = settings.width, settings.height, settings.fps
    segs = timeline["segments"]
    N = timeline["total_frames"]
    finish = FinishingPass(W, H, plan.get("look", "clean"), plan.get("look_intensity", 0.9),
                           plan.get("grain", 0.0), plan.get("vignette", 0.0), plan.get("letterbox", False))
    fx = TransitionFX(W, H)
    texts = [TextOverlay(t["content"], t.get("style", "bold"), t.get("position", "center"), t["start"], t["end"],
                         W, H, finish.safe_top + int(H * 0.04), finish.safe_bottom - int(H * 0.04))
             for t in timeline.get("texts", [])]
    if timeline.get("texts") and timeline.get("caption_words"):
        # keep text overlays out of the caption zone
        for ov in texts:
            if ov.cy > H * 0.58:
                ov.cy = H * 0.22 + finish.safe_top
    captions = None
    if timeline.get("caption_words"):
        y = 0.70 if not plan.get("letterbox") else 0.68
        captions = CaptionTrack(timeline["caption_words"], plan.get("caption_style", "pop_yellow"), W, H, y)

    os.makedirs(work_dir, exist_ok=True)
    vid_tmp = os.path.join(work_dir, "video_only.mp4")
    writer = FrameWriter(vid_tmp, W, H, fps, settings.crf, settings.preset)
    fade_in_f = int(0.35 * fps) if plan.get("fade_in", True) else 0
    fade_out_f = int(0.6 * fps) if plan.get("fade_out", True) else 0
    t0 = time.time()
    out_k = 0

    def emit(frame):
        nonlocal out_k
        fr = finish.apply(frame, out_k)
        if fr is frame:
            fr = frame.copy()
        t = out_k / fps
        for ov in texts:
            ov.draw(fr, t)
        if captions:
            captions.draw(fr, t)
        if fade_in_f and out_k < fade_in_f:
            fr = fade(fr, (out_k + 1) / (fade_in_f + 1))
        if fade_out_f and out_k >= N - fade_out_f:
            fr = fade(fr, (N - out_k) / (fade_out_f + 1))
        writer.write(fr)
        out_k += 1
        if progress and (out_k % 10 == 0 or out_k == N):
            el = time.time() - t0
            progress(out_k / max(1, N), f"Rendering frame {out_k}/{N} ({out_k / max(el, 1e-3):.1f} fps)")

    gens: dict[int, object] = {}

    def gen(i):
        if i not in gens:
            gens[i] = _segment_frames(segs[i], i, settings, face_det, smart_reframe, log)
        return gens[i]

    try:
        for i, seg in enumerate(segs):
            g = gen(i)
            solo = seg["n_frames"] - seg["m_in"] - seg["m_out"]
            for _ in range(solo):
                emit(next(g))
            if seg["m_out"]:
                g2 = gen(i + 1)
                kind = segs[i + 1]["transition_in"]
                m = seg["m_out"]
                for k in range(m):
                    a, b = next(g), next(g2)
                    emit(fx.apply(kind, a, b, (k + 0.5) / m))
            g.close()
            gens.pop(i, None)
    finally:
        for g in gens.values():
            try:
                g.close()
            except Exception:
                pass
        writer.close()

    wav = os.path.join(work_dir, "mix.wav")
    has_audio = mix_audio(timeline, plan, music, wav)
    mux(vid_tmp, wav if has_audio else None, out_path, metadata)
    elapsed = time.time() - t0
    log(f"Rendered {N} frames ({N / fps:.1f}s) in {elapsed:.1f}s")
    return {"frames": N, "duration": N / fps, "render_seconds": round(elapsed, 1), "audio": has_audio}
