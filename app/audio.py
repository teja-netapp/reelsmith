"""Music analysis (tempo, beats, energy, best section), speech transcription, and final audio mix."""
from __future__ import annotations

import hashlib
import json
import os
import wave

import numpy as np

from .config import WHISPER_MODEL, WORKSPACE
from .video_io import decode_audio

CACHE = WORKSPACE / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


def _key(path: str, extra: str = "") -> str:
    st = os.stat(path)
    return hashlib.sha1(f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime}|{extra}".encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------
# Music
# ----------------------------------------------------------------------------------------------
def analyze_music(path: str, target_duration: float = 30.0) -> dict:
    import librosa  # heavy import, keep local

    sr = 22050
    y = decode_audio(path, sr=sr, channels=1)[:, 0] if os.path.exists(path) else np.zeros(0)
    if len(y) < sr:
        raise RuntimeError("Music track is empty or too short")
    dur = len(y) / sr
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time", tightness=100)
    tempo = float(np.atleast_1d(tempo)[0])
    beats = [round(float(b), 3) for b in np.atleast_1d(beats)]
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    o_times = librosa.times_like(onset, sr=sr)
    # downbeat phase: pick the 4-beat phase with the most onset energy
    downbeats = beats[::4]
    if len(beats) >= 8:
        strengths = []
        for ph in range(4):
            bs = beats[ph::4]
            strengths.append(float(np.sum(np.interp(bs, o_times, onset))))
        ph = int(np.argmax(strengths))
        downbeats = beats[ph::4]
    # energy curve (RMS per 0.5 s)
    hop = int(sr * 0.5)
    rms = np.array([np.sqrt(np.mean(y[i: i + hop] ** 2) + 1e-12) for i in range(0, len(y) - hop + 1, hop)])
    rms_n = (rms / (rms.max() + 1e-9)).round(3)
    # best section of target length: max mean energy, starting on a downbeat
    win = max(1, int(target_duration / 0.5))
    best_start, best_val = 0.0, -1.0
    cands = [d for d in downbeats if d + target_duration <= dur + 0.5] or [0.0]
    for d in cands:
        i0 = int(d / 0.5)
        val = float(np.mean(rms_n[i0: i0 + win])) if i0 < len(rms_n) else 0
        # slight preference for earlier starts so the song's arc is kept
        val -= 0.0005 * d
        if val > best_val:
            best_val, best_start = val, d
    return {
        "path": path,
        "name": os.path.basename(path),
        "duration": round(dur, 3),
        "tempo": round(tempo, 1),
        "beat_interval": round(60.0 / tempo, 3) if tempo > 0 else 0.5,
        "beats": beats,
        "downbeats": [round(float(d), 3) for d in downbeats],
        "energy_half_seconds": rms_n.tolist(),
        "suggested_start": round(best_start, 3),
    }


# ----------------------------------------------------------------------------------------------
# Transcription
# ----------------------------------------------------------------------------------------------
_whisper = None


def whisper_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("faster_whisper") is not None


def transcribe(path: str, model_size: str = WHISPER_MODEL) -> dict | None:
    """Return {"words": [{start,end,word}], "segments": [{start,end,text}], "language": str} or None."""
    global _whisper
    cache = CACHE / f"tr_{_key(path, model_size)}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return None
    audio = decode_audio(path, sr=16000, channels=1)
    if len(audio) < 16000 * 0.5 or float(np.abs(audio).max()) < 1e-3:
        res = {"words": [], "segments": [], "language": None}
        cache.write_text(json.dumps(res))
        return res
    if _whisper is None or getattr(_whisper, "_size", None) != model_size:
        _whisper = WhisperModel(model_size, device="cpu", compute_type="int8")
        _whisper._size = model_size
    segs, info = _whisper.transcribe(audio[:, 0], word_timestamps=True, vad_filter=True, beam_size=1)
    words, segments = [], []
    for s in segs:
        segments.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()})
        for w in (s.words or []):
            if w.word.strip():
                words.append({"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word.strip()})
    res = {"words": words, "segments": segments, "language": getattr(info, "language", None)}
    cache.write_text(json.dumps(res))
    return res


# ----------------------------------------------------------------------------------------------
# Final mix
# ----------------------------------------------------------------------------------------------
def _fade(n, sr, fin, fout):
    g = np.ones(n, np.float32)
    a = min(n, int(fin * sr))
    b = min(n, int(fout * sr))
    if a > 0:
        g[:a] = np.linspace(0, 1, a, dtype=np.float32) ** 1.5
    if b > 0:
        g[n - b:] *= np.linspace(1, 0, b, dtype=np.float32) ** 1.5
    return g


def mix_audio(timeline: dict, plan: dict, music: dict | None, out_wav: str, sr: int = 48000) -> bool:
    """Assemble the soundtrack: original-audio segments (with crossfades) + music (ducked)."""
    fps = timeline["fps"]
    total = timeline["total_frames"] / fps
    N = int(round(total * sr))
    if N <= 0:
        return False
    voice = np.zeros((N, 2), np.float32)
    have_voice = False
    for seg in timeline["segments"]:
        if not seg.get("keep_audio") or not seg.get("has_audio"):
            continue
        src_len = seg["n_frames"] / fps * seg["speed"]
        a = decode_audio(seg["path"], start=seg["src_in"], duration=src_len, sr=sr, channels=2, speed=seg["speed"])
        if len(a) == 0:
            continue
        s0 = int(round(seg["start_frame"] / fps * sr))
        n = min(len(a), int(round(seg["n_frames"] / fps * sr)), N - s0)
        if n <= 0:
            continue
        fin = max(0.02, seg["m_in"] / fps)
        fout = max(0.04, seg["m_out"] / fps)
        g = _fade(n, sr, fin, fout)[:, None]
        voice[s0: s0 + n] += a[:n] * g * float(plan.get("original_audio_volume", 1.0))
        have_voice = True

    mus = np.zeros((N, 2), np.float32)
    have_music = False
    if music and music.get("path"):
        m = decode_audio(music["path"], start=float(plan.get("music_start", 0.0)), duration=total + 0.5,
                         sr=sr, channels=2)
        if len(m):
            n = min(len(m), N)
            fin = 0.25 if plan.get("music_start", 0) > 0.1 else 0.02
            fout = min(2.0, total * 0.15)
            mus[:n] = m[:n] * _fade(n, sr, fin, fout)[:, None]
            have_music = True

    if not (have_voice or have_music):
        return False

    mv = float(plan.get("music_volume", 0.9))
    if have_voice and have_music:
        # sidechain-style ducking driven by the voice envelope
        hop = sr // 100
        mono = np.abs(voice).mean(1)
        env = np.array([mono[i: i + hop].mean() for i in range(0, N, hop)])
        thr = max(0.01, np.percentile(env[env > 1e-4], 60) * 0.5) if np.any(env > 1e-4) else 0.02
        tgt = np.clip(env / thr, 0, 1)
        sm = np.zeros_like(tgt)
        v = 0.0
        for i, x in enumerate(tgt):
            k = 0.35 if x > v else 0.04  # fast attack, slow release (per 10 ms)
            v += (x - v) * k
            sm[i] = v
        duck = 1.0 - 0.7 * sm
        gain = np.repeat(duck, hop)[:N].astype(np.float32)
        if len(gain) < N:
            gain = np.pad(gain, (0, N - len(gain)), constant_values=gain[-1] if len(gain) else 1.0)
        mus *= gain[:, None]
    mix = voice + mus * mv
    # fade the whole mix with the picture fade-out
    if plan.get("fade_out", True):
        mix *= _fade(N, sr, 0.0, min(0.8, total * 0.1))[:, None]
    peak = float(np.abs(mix).max())
    if peak > 0.98:
        mix *= 0.98 / peak
    elif 0 < peak < 0.5:
        mix *= 0.85 / peak
    pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return True
