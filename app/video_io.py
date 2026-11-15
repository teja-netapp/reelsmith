"""Thin FFmpeg wrappers: probing, frame-accurate decoding to numpy, encoding from numpy, audio decode."""
from __future__ import annotations

import functools
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


class FFmpegError(RuntimeError):
    pass


def check_ffmpeg() -> bool:
    try:
        subprocess.run([FFMPEG, "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def video_encoder() -> str:
    """Prefer libx264 (best quality/size); fall back to Apple VideoToolbox or mpeg4."""
    try:
        out = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    except Exception:
        return "libx264"
    for enc in ("libx264", "h264_videotoolbox", "mpeg4"):
        if f" {enc} " in out:
            return enc
    return "libx264"


@dataclass
class MediaInfo:
    path: str
    duration: float
    width: int = 0
    height: int = 0
    fps: float = 30.0
    has_video: bool = False
    has_audio: bool = False
    sar_filter: str | None = None  # filter to square pixels if needed

    def to_dict(self):
        return {
            "path": self.path,
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "has_video": self.has_video,
            "has_audio": self.has_audio,
            "sar_filter": self.sar_filter,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


def _ratio(s: str | None, default: float = 0.0) -> float:
    if not s:
        return default
    try:
        if "/" in s:
            n, d = s.split("/")
            return float(n) / float(d) if float(d) else default
        return float(s)
    except Exception:
        return default


def probe(path: str) -> MediaInfo:
    cmd = [FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise FFmpegError(f"ffprobe failed for {path}: {res.stderr.strip()[:400]}")
    info = json.loads(res.stdout or "{}")
    streams = info.get("streams", [])
    v = next(
        (s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")),
        None,
    )
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    dur = _ratio(info.get("format", {}).get("duration"), 0.0)
    if not dur:
        dur = _ratio((v or a or {}).get("duration"), 0.0)
    mi = MediaInfo(path=path, duration=dur, has_video=v is not None, has_audio=a is not None)
    if v:
        w, h = int(v.get("width", 0)), int(v.get("height", 0))
        rot = 0
        tags = v.get("tags", {}) or {}
        if "rotate" in tags:
            rot = int(float(tags["rotate"]))
        for sd in v.get("side_data_list", []) or []:
            if "rotation" in sd:
                rot = int(float(sd["rotation"]))
        sar = _ratio((v.get("sample_aspect_ratio") or "1:1").replace(":", "/"), 1.0) or 1.0
        if abs(sar - 1.0) > 0.01:
            w = int(round(w * sar / 2) * 2)
            mi.sar_filter = "scale=trunc(iw*sar/2)*2:ih,setsar=1"
        if abs(rot) % 180 == 90:
            w, h = h, w
        mi.width, mi.height = w, h
        fps = _ratio(v.get("avg_frame_rate"), 0.0) or _ratio(v.get("r_frame_rate"), 30.0)
        mi.fps = fps if 1 < fps < 241 else 30.0
    return mi


class FrameReader:
    """Decode a time range of a video into BGR numpy frames at a fixed output fps.

    Filters (in order): square pixels -> crop (display coords) -> scale -> speed -> fps/blend.
    """

    def __init__(
        self,
        path: str,
        start: float,
        src_duration: float,
        out_w: int,
        out_h: int,
        crop: tuple[int, int, int, int] | None = None,
        speed: float = 1.0,
        fps: float = 30.0,
        src_fps: float = 30.0,
        sar_filter: str | None = None,
        blend_slowmo: bool = True,
        scale_flags: str = "bicubic",
    ):
        self.w, self.h = int(out_w), int(out_h)
        vf: list[str] = []
        if sar_filter:
            vf.append(sar_filter)
        if crop:
            cw, ch, cx, cy = crop
            vf.append(f"crop={int(cw)}:{int(ch)}:{int(cx)}:{int(cy)}")
        vf.append(f"scale={self.w}:{self.h}:flags={scale_flags}")
        if abs(speed - 1.0) > 1e-3:
            vf.append(f"setpts=(PTS-STARTPTS)/{speed:.5f}")
        else:
            vf.append("setpts=PTS-STARTPTS")
        if blend_slowmo and src_fps * speed < fps * 0.95:
            vf.append(f"framerate=fps={fps}")  # frame blending for smoother slow motion
        else:
            vf.append(f"fps={fps}")
        cmd = [
            FFMPEG, "-nostdin", "-v", "error",
            "-ss", f"{max(0.0, start):.3f}", "-t", f"{src_duration + 0.25:.3f}",
            "-i", path, "-an", "-sn", "-dn",
            "-vf", ",".join(vf),
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ]
        self._err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self._err, bufsize=self.w * self.h * 3 * 2)
        self.nbytes = self.w * self.h * 3

    def read(self) -> np.ndarray | None:
        buf = self.proc.stdout.read(self.nbytes)
        if not buf or len(buf) < self.nbytes:
            return None
        return np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)

    def errors(self) -> str:
        try:
            self._err.seek(0)
            return self._err.read().decode("utf-8", "ignore")[-800:]
        except Exception:
            return ""

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait(timeout=5)
        except Exception:
            pass
        try:
            self.proc.stdout.close()
            self._err.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class FrameWriter:
    def __init__(self, path: str, w: int, h: int, fps: float, crf: int = 18, preset: str = "medium"):
        enc = video_encoder()
        cmd = [
            FFMPEG, "-y", "-nostdin", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", enc,
        ]
        if enc == "libx264":
            cmd += ["-preset", preset, "-crf", str(crf), "-profile:v", "high", "-tune", "film"]
        elif enc == "h264_videotoolbox":
            cmd += ["-b:v", "12M", "-profile:v", "high"]
        else:
            cmd += ["-q:v", "3"]
        cmd += ["-pix_fmt", "yuv420p", "-r", str(fps), "-movflags", "+faststart", path]
        self._err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._err)
        self.shape = (h, w, 3)

    def write(self, frame: np.ndarray):
        if frame.shape != self.shape:
            raise ValueError(f"frame shape {frame.shape} != {self.shape}")
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).data)

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        rc = self.proc.wait()
        if rc != 0:
            self._err.seek(0)
            raise FFmpegError("encoder failed: " + self._err.read().decode("utf-8", "ignore")[-800:])
        self._err.close()


def atempo_chain(speed: float) -> list[str]:
    out = []
    s = float(speed)
    while s > 2.0:
        out.append("atempo=2.0")
        s /= 2.0
    while s < 0.5:
        out.append("atempo=0.5")
        s /= 0.5
    if abs(s - 1.0) > 1e-3:
        out.append(f"atempo={s:.5f}")
    return out


def decode_audio(path: str, start: float = 0.0, duration: float | None = None, sr: int = 48000,
                 channels: int = 2, speed: float = 1.0) -> np.ndarray:
    """Return float32 array of shape (n, channels). Empty array if the file has no audio."""
    cmd = [FFMPEG, "-nostdin", "-v", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", path, "-vn", "-sn"]
    af = atempo_chain(speed)
    if af:
        cmd += ["-af", ",".join(af)]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "-ac", str(channels), "-ar", str(sr), "-"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0 or not res.stdout:
        return np.zeros((0, channels), np.float32)
    a = np.frombuffer(res.stdout, np.float32)
    n = len(a) // channels
    return a[: n * channels].reshape(n, channels).copy()


def sample_frames(path: str, info: MediaInfo, every: float, width: int = 320):
    """Yield (time, frame) at a fixed interval over the whole clip with one decoding pass."""
    w = int(width) // 2 * 2
    h = max(2, int(round(w * info.height / max(1, info.width) / 2)) * 2)
    vf = []
    if info.sar_filter:
        vf.append(info.sar_filter)
    vf += [f"fps={1.0 / every:.5f}", f"scale={w}:{h}:flags=area", "format=bgr24"]
    cmd = [FFMPEG, "-nostdin", "-v", "error", "-i", path, "-an", "-sn", "-vf", ",".join(vf),
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 3 * 4)
    nbytes = w * h * 3
    k = 0
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if not buf or len(buf) < nbytes:
                break
            yield k * every, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            k += 1
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def grab_frame(path: str, info: MediaInfo, t: float, width: int = 480) -> np.ndarray | None:
    w = int(width) // 2 * 2
    h = max(2, int(round(w * info.height / max(1, info.width) / 2)) * 2)
    vf = ([info.sar_filter] if info.sar_filter else []) + [f"scale={w}:{h}", "format=bgr24"]
    cmd = [FFMPEG, "-nostdin", "-v", "error", "-ss", f"{max(0, t):.3f}", "-i", path, "-frames:v", "1",
           "-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    res = subprocess.run(cmd, capture_output=True)
    if len(res.stdout) < w * h * 3:
        return None
    return np.frombuffer(res.stdout[: w * h * 3], np.uint8).reshape(h, w, 3).copy()


def mux(video_path: str, audio_path: str | None, out_path: str, metadata: dict | None = None):
    cmd = [FFMPEG, "-y", "-nostdin", "-v", "error", "-i", video_path]
    if audio_path:
        cmd += ["-i", audio_path, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest"]
    else:
        cmd += ["-c:v", "copy"]
    for k, v in (metadata or {}).items():  # e.g. title / comment: which editor + model made it
        cmd += ["-metadata", f"{k}={v}"]
    cmd += ["-movflags", "+faststart+use_metadata_tags", out_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise FFmpegError("mux failed: " + res.stderr[-600:])
