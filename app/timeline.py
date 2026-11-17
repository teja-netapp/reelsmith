"""Turn a validated plan into a frame-exact timeline (with beat-snapped cuts, transitions, captions)."""
from __future__ import annotations


def _clamp_transitions(segs, min_solo=3):
    segs[0]["m_in"] = 0
    for i in range(1, len(segs)):
        a, b = segs[i - 1], segs[i]
        if b["m_in"] <= 0:
            continue
        lim = min(a["n_frames"] - a["m_in"] - min_solo, b["n_frames"] // 2)
        b["m_in"] = max(0, min(b["m_in"], lim))
        if b["m_in"] < 2:
            b["m_in"] = 0
            b["transition_dropped"] = b["transition_in"]  # shots too short for it -> reported
            b["transition_in"] = "cut"


def _layout(segs):
    F = 0
    for i, s in enumerate(segs):
        s["start_frame"] = F
        s["m_out"] = segs[i + 1]["m_in"] if i + 1 < len(segs) else 0
        F += s["n_frames"] - s["m_out"]
    return F


def _snap_to_beats(segs, music, music_start, fps):
    beats = music.get("beats") or []
    downs = set(int(round((d - music_start) * fps)) for d in music.get("downbeats", []))
    bfs = sorted(set(int(round((b - music_start) * fps)) for b in beats if b - music_start > 0.15))
    if len(bfs) < 2:
        return 0
    bi = max(4, int(round(music.get("beat_interval", 0.5) * fps)))
    F = 0
    snapped = 0
    for i, s in enumerate(segs):
        last = i == len(segs) - 1
        m_next = 0 if last else segs[i + 1]["m_in"]
        offset = m_next / 2.0
        target = F + s["n_frames"] - offset
        min_n = s["m_in"] + m_next + max(6, int(0.3 * fps))
        best, best_cost = None, None
        for bf in bfs:
            if abs(bf - target) > 1.6 * bi:
                continue
            new_n = int(round(bf - F + offset))
            if new_n < min_n:
                continue
            need = new_n / fps * s["speed"]
            src_in = s["src_in"]
            if src_in + need > s["clip_dur"]:
                src_in = s["clip_dur"] - need
                if src_in < 0 or s["src_in"] - src_in > 1.0:
                    continue
            cost = abs(bf - target) - (0.2 * bi if bf in downs else 0)
            if best_cost is None or cost < best_cost:
                best, best_cost = (new_n, src_in), cost
        if best:
            old_n, old_in = s["n_frames"], s["src_in"]
            s["n_frames"], s["src_in"] = best
            s["beat_delta_s"] = round((s["n_frames"] - old_n) / fps, 3)   # + longer, - shorter on screen
            s["beat_src_shift_s"] = round(s["src_in"] - old_in, 3)        # in-point moved to make room
            snapped += 1
        F += s["n_frames"] - m_next
    return snapped


def build_timeline(plan: dict, analysis: dict, music: dict | None, options: dict, fps: int = 30) -> dict:
    clips = {c["id"]: c for c in analysis["clips"]}
    shot_idx = {s["id"]: s for s in analysis.get("all_shots", [])}
    shot_idx.update({s["id"]: s for s in analysis.get("shots", [])})
    segs = []
    for s in plan["segments"]:
        c = clips[s["clip_id"]]
        n = max(8, int(round((s["end"] - s["start"]) / s["speed"] * fps)))
        tr = s["transition_in"]
        m = 0 if tr == "cut" else max(2, int(round(s["transition_duration"] * fps)))
        segs.append({
            "clip_id": c["id"], "path": c["path"], "info": c["info"], "clip_dur": c["duration"],
            "has_audio": c["has_audio"], "src_in": float(s["start"]), "speed": float(s["speed"]),
            "n_frames": n, "m_in": m, "transition_in": tr, "motion": s["motion"],
            "focus_x": s.get("focus_x", 0.5), "keep_audio": s.get("keep_audio", False), "text": s.get("text"),
            "shot_id": s.get("shot_id"),
            "color": (shot_idx.get(s.get("shot_id")) or {}).get("color"),
            "effect": s.get("effect", "none"),
        })
    _clamp_transitions(segs)
    snapped = 0
    if options.get("beat_sync") and music:
        snapped = _snap_to_beats(segs, music, float(plan.get("music_start", 0.0)), fps)
        _clamp_transitions(segs)
    total = _layout(segs)

    # captions: map transcript words from source time to output time
    words = []
    trs = analysis.get("transcripts", {})
    if options.get("captions"):
        for s in segs:
            tr = trs.get(s["clip_id"])
            if not (s["keep_audio"] and tr and tr.get("words")):
                continue
            a = s["src_in"]
            b = a + s["n_frames"] / fps * s["speed"]
            t0 = s["start_frame"] / fps
            vis0 = (s["start_frame"] + s["m_in"] / 2) / fps
            vis1 = (s["start_frame"] + s["n_frames"] - s["m_out"] / 2) / fps
            for w in tr["words"]:
                if w["start"] >= a - 0.02 and w["end"] <= b + 0.15:
                    ws = t0 + (w["start"] - a) / s["speed"]
                    we = t0 + (w["end"] - a) / s["speed"]
                    if we > vis0 and ws < vis1:
                        words.append({"start": max(ws, vis0), "end": min(we, vis1), "word": w["word"]})
                        s["caption_words"] = s.get("caption_words", 0) + 1
        words.sort(key=lambda w: w["start"])

    # on-screen text
    texts = []
    for s in segs:
        t = s.get("text")
        if not t:
            continue
        st = (s["start_frame"] + s["m_in"] / 2) / fps + 0.08
        en = (s["start_frame"] + s["n_frames"] - s["m_out"] / 2) / fps
        if t.get("style") == "cinematic_title":
            en = max(en, min(st + 2.2, total / fps - 0.2))
        texts.append({**t, "start": round(st, 3), "end": round(max(st + 0.6, en), 3)})

    return {
        "fps": fps,
        "segments": segs,
        "total_frames": total,
        "duration": round(total / fps, 3),
        "beats_snapped": snapped,
        "caption_words": words,
        "texts": texts,
    }
