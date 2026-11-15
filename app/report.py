"""'How this reel was made': a per-version edit report (JSON for the UI, Markdown for download).

It answers: which editor/model made it, what it was asked, what it decided (shot by shot), and
everything that was changed automatically afterwards (validation fixes, assistant-editor rules,
beat sync) - so nothing about the video is a mystery.
"""
from __future__ import annotations

import os
import time

from . import __version__
from .craft import fmt_t
from .grading import look_descriptions, look_family

TRIGGER_LABELS = {"new": "New reel", "refine": "Revision", "plan": "Manual plan edit", "look": "Look Lab"}


def _color_desc(c):
    """Human description of a shot-matching correction, e.g. 'warmer 4%, +0.2 EV'."""
    if not c:
        return None
    r, g, b = c.get("wb", (1, 1, 1))
    parts = []
    warm = (r - b) / 2
    if abs(warm) >= 0.015:
        parts.append(f"{'warmer' if warm > 0 else 'cooler'} {abs(warm) * 100:.0f}%")
    if abs(g - 1) >= 0.015:
        parts.append(f"{'less' if g < 1 else 'more'} green {abs(g - 1) * 100:.0f}%")
    if abs(c.get("ev", 0)) >= 0.05:
        parts.append(f"{c['ev']:+.1f} EV")
    return ", ".join(parts) or "matched (no change needed)"


def _sound(seg, clip, music):
    if seg.get("keep_audio") and clip.get("has_audio"):
        return "original sound + music (ducked)" if music else "original sound"
    if seg.get("keep_audio") and not clip.get("has_audio"):
        return "music (clip has no sound)" if music else "silent (clip has no sound)"
    return "music only" if music else "silent"


def _reframe(seg, clip, smart):
    w, h = clip.get("width", 1), clip.get("height", 1)
    if w / max(1, h) <= 9 / 16 * 1.03:
        return "full frame (already vertical)"
    if not smart:
        return "centre crop to 9:16"
    return f"subject tracking to 9:16 (subject ≈ {float(seg.get('focus_x', 0.5)):.0%} from left)"


def build_report(job, tl: dict, plan: dict, settings, render_res: dict, trigger: dict, timings: dict,
                 output_path: str) -> dict:
    fps = tl["fps"]
    analysis = job.analysis or {}
    clips = {c["id"]: c for c in analysis.get("clips", [])}
    shots = {s["id"]: s for s in analysis.get("all_shots", [])}
    shots.update({s["id"]: s for s in analysis.get("shots", [])})
    music = job.music_info
    opts = job.options
    smart = opts.get("smart_reframe", True)

    segs = []
    beat_moves = []
    dropped_tr = []
    for i, s in enumerate(tl["segments"]):
        c = clips.get(s["clip_id"], {})
        src_out = s["src_in"] + s["n_frames"] / fps * s["speed"]
        tr = "start" if i == 0 else s["transition_in"]
        tr_len = s["m_in"] / fps if i and s["transition_in"] != "cut" else 0.0
        if s.get("transition_dropped") and s["transition_dropped"] != "cut":
            dropped_tr.append(f"{c.get('name', s['clip_id'])} @ {fmt_t(s['src_in'])}: {s['transition_dropped']} → cut "
                              f"(shots too short for the transition)")
        if "beat_delta_s" in s:
            beat_moves.append(s["beat_delta_s"])
        sh = shots.get(s.get("shot_id")) or {}
        t = s.get("text") or None
        segs.append({
            "n": i + 1,
            "clip": c.get("name", s["clip_id"]),
            "shot_id": s.get("shot_id"),
            "what": (sh.get("vlm") or {}).get("description"),
            "source_in": round(s["src_in"], 2),
            "source_out": round(src_out, 2),
            "source": f"{fmt_t(s['src_in'])}–{fmt_t(src_out)}",
            "starts_at": round(s["start_frame"] / fps, 2),
            "on_screen_s": round(s["n_frames"] / fps, 2),
            "speed": s["speed"],
            "camera": s["motion"],
            "transition_in": tr,
            "transition_s": round(tr_len, 2),
            "reframe": _reframe(s, c, smart),
            "sound": _sound(s, c, music),
            "text": {"content": t["content"], "style": t.get("style")} if t else None,
            "captions_words": s.get("caption_words", 0),
            "beat_adjust_s": s.get("beat_delta_s"),
            "color_match": _color_desc(s.get("color_correction")),
            "quality_score": sh.get("score"),
        })

    beat_summary = None
    if opts.get("beat_sync") and music:
        n_cuts = len(tl["segments"])
        if beat_moves:
            mx = max(abs(b) for b in beat_moves)
            beat_summary = (f"{len(beat_moves)} of {n_cuts} cuts moved onto the beat of \"{music.get('name')}\" "
                            f"({music.get('tempo')} BPM); largest move {mx:.2f} s")
        else:
            beat_summary = "Beat sync was on, but no cut needed (or could be) moved"

    ed = dict(plan.get("editor") or {"label": plan.get("planner"), "provider": "unknown"})
    size = os.path.getsize(output_path) if os.path.exists(output_path) else None
    return {
        "reelsmith_version": __version__,
        "version": job.version,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trigger": {"type": trigger.get("type"), "label": TRIGGER_LABELS.get(trigger.get("type"), trigger.get("type")),
                    "feedback": trigger.get("feedback")},
        "editor": ed,
        "brief": job.description,
        "settings": {
            "target_duration_s": opts.get("target_duration"), "auto_pick": opts.get("auto_pick"),
            "beat_sync": opts.get("beat_sync"), "captions": opts.get("captions"),
            "smart_reframe": smart, "quality": opts.get("quality"), "editor_requested": opts.get("planner", "auto"),
        },
        "footage": [{"name": c["name"], "duration_s": c["duration"], "resolution": f"{c['width']}x{c['height']}",
                     "fps": c["fps"], "orientation": c["orientation"], "scenes": c.get("num_scenes"),
                     "speech_words": len((analysis.get("transcripts", {}).get(c["id"]) or {}).get("words", []))}
                    for c in analysis.get("clips", [])],
        "concept": plan.get("reasoning"),
        "title": plan.get("title"),
        "pace": plan.get("pace"),
        "finishing": {
            "look": plan.get("look"), "look_description": look_descriptions().get(plan.get("look"), ""),
            "look_family": look_family(plan.get("look")), "halation": plan.get("halation"), "bloom": plan.get("bloom"),
            "auto_color": opts.get("auto_color", True),
            "look_intensity": plan.get("look_intensity"), "grain": plan.get("grain"), "vignette": plan.get("vignette"),
            "letterbox": plan.get("letterbox"), "fade_in": plan.get("fade_in"), "fade_out": plan.get("fade_out"),
            "captions": {"style": plan.get("caption_style"), "words": len(tl.get("caption_words", []))}
            if tl.get("caption_words") else None,
            "texts": [{"content": t["content"], "style": t.get("style"), "from_s": t["start"], "to_s": t["end"]}
                      for t in tl.get("texts", [])],
        },
        "audio": {
            "music": {"name": music.get("name"), "tempo_bpm": music.get("tempo"),
                      "starts_at_s": plan.get("music_start"), "volume": plan.get("music_volume")} if music else None,
            "original_sound_shots": sum(1 for s in segs if s["sound"].startswith("original")),
            "original_volume": plan.get("original_audio_volume"),
            "ducking": bool(music) and any(s["sound"].startswith("original") for s in segs),
        },
        "segments": segs,
        "changes": {
            "validation": plan.get("warnings", []),
            "assistant_editor": plan.get("craft_changes", []),
            "assistant_editor_level": ed.get("craft_level"),
            "timeline": dropped_tr,
            "beat_sync": beat_summary,
        },
        "render": {
            "resolution": f"{settings.width}x{settings.height}", "fps": settings.fps, "quality": opts.get("quality"),
            "codec": "H.264 + AAC", "crf": settings.crf, "preset": settings.preset,
            "frames": render_res.get("frames"), "duration_s": round(render_res.get("duration", 0), 2),
            "render_seconds": render_res.get("render_seconds"), "file_mb": round(size / 1e6, 1) if size else None,
        },
        "learning": plan.get("learning"),
        "timings_s": timings,
        "files": {"video": os.path.basename(output_path)},
    }


def mp4_metadata(plan: dict, tl: dict, version: int) -> dict:
    """Tags written into the MP4 itself (title/comment/description), so the file records which editor and
    model made it even after it leaves the app. Check with: ffprobe -show_entries format_tags reel.mp4"""
    ed = plan.get("editor") or {}
    comment = (f"Made with Reelsmith {__version__} (v{version}). Editor: {ed.get('label') or plan.get('planner')}; "
               f"model: {ed.get('model') or 'none'}. Look: {plan.get('look')}. {len(tl['segments'])} shots, "
               f"{tl['duration']:.1f} s.")
    return {"title": plan.get("title") or "Reel", "comment": comment, "description": comment}


# ----------------------------------------------------------------------------------------------
def _fmt_editor(ed: dict) -> list[str]:
    L = [f"- **Editor:** {ed.get('label', '?')}"]
    if ed.get("model"):
        L.append(f"- **Model:** `{ed['model']}`" + (f" (version `{ed['model_version']}`)" if ed.get("model_version") else "")
                 + (f", {ed['model_source']}" if ed.get("model_source") else "")
                 + (f", output: {ed['output_format']}" if ed.get("output_format") else ""))
    if ed.get("shot_log_model"):
        L.append(f"- **Shot logging model:** `{ed['shot_log_model']}`: {ed.get('shots_shown', '?')} shots "
                 f"({ed.get('shots_logged', 0)} newly logged, {ed.get('shots_cached', 0)} from cache"
                 + (f", {ed['shots_failed']} failed" if ed.get("shots_failed") else "") + ")")
    u = ed.get("usage") or {}
    if u.get("input_tokens") or u.get("output_tokens"):
        L.append(f"- **Tokens:** {u.get('input_tokens', '?')} in / {u.get('output_tokens', '?')} out"
                 + (f" / {u['thinking_tokens']} thinking" if u.get("thinking_tokens") else ""))
    if ed.get("images_sent"):
        L.append(f"- **Contact sheets sent:** {ed['images_sent']}")
    if ed.get("seconds") is not None:
        L.append(f"- **Planning time:** {ed['seconds']} s")
    L.append(f"- **Requested:** {ed.get('requested', 'auto')}")
    for a in ed.get("attempts", []) or []:
        L.append(f"- Tried **{a.get('label')}** first, which failed: {a.get('error')}")
    for s in ed.get("skipped", []) or []:
        L.append(f"- Skipped {s.get('provider')}: {s.get('reason')}")
    if ed.get("feedback"):
        L.append(f"- **Revision request:** “{ed['feedback']}”")
    if ed.get("based_on"):
        L.append(f"- **Based on:** {ed['based_on']}")
    return L


def report_markdown(r: dict) -> str:
    f, a, rd = r["finishing"], r["audio"], r["render"]
    L = [f"# {r.get('title') or 'Reel'}: how it was made (v{r['version']})", "",
         f"_{r['trigger']['label']} · {r['created']} · Reelsmith {r['reelsmith_version']}_", "",
         "## Editor", *_fmt_editor(r["editor"]), "",
         "## Brief", r.get("brief") or "(none)", ""]
    s = r["settings"]
    L += ["## Settings",
          f"- Target {s['target_duration_s']} s · auto-pick {'on' if s['auto_pick'] else 'off'} · beat sync "
          f"{'on' if s['beat_sync'] else 'off'} · captions {'on' if s['captions'] else 'off'} · smart reframe "
          f"{'on' if s['smart_reframe'] else 'off'} · quality {s['quality']}", ""]
    L += ["## Footage"] + [f"- {c['name']}: {c['duration_s']:.1f} s, {c['resolution']} {c['orientation']}, "
                           f"{c['fps']:.0f} fps, {c['scenes']} scenes" + (f", {c['speech_words']} spoken words" if c['speech_words'] else "")
                           for c in r["footage"]] + [""]
    if r.get("concept"):
        L += ["## Editor's concept", r["concept"], ""]
    L += ["## Look & sound",
          f"- Look: **{f['look']}** ({f['look_description']}), intensity {f['look_intensity']:.2f}",
          f"- Shot matching {'on' if f.get('auto_color', True) else 'off'} · halation {f.get('halation') or 0:.2f} · bloom {f.get('bloom') or 0:.2f}",
          f"- Grain {f['grain']:.2f} · vignette {f['vignette']:.2f} · letterbox {'yes' if f['letterbox'] else 'no'} · "
          f"fade in {'yes' if f['fade_in'] else 'no'} · fade out {'yes' if f['fade_out'] else 'no'}"]
    if f.get("captions"):
        L.append(f"- Captions: {f['captions']['style']}, {f['captions']['words']} words")
    for t in f.get("texts", []):
        L.append(f"- Text “{t['content']}” ({t['style']}) {t['from_s']:.1f}–{t['to_s']:.1f} s")
    if a.get("music"):
        m = a["music"]
        L.append(f"- Music: {m['name']} ({m['tempo_bpm']} BPM) from {m['starts_at_s']:.1f} s, volume {m['volume']:.2f}"
                 + (" · ducked under original sound" if a["ducking"] else ""))
    L.append(f"- Shots with their original sound: {a['original_sound_shots']}")
    L += ["", "## Edit decision list", "",
          "| # | Source | On screen | Speed | Camera | Transition in | Reframe | Sound | Extras |",
          "|---|---|---|---|---|---|---|---|---|"]
    for g in r["segments"]:
        extras = []
        if g["text"]:
            extras.append(f"text “{g['text']['content']}”")
        if g["captions_words"]:
            extras.append(f"{g['captions_words']} caption words")
        if g["beat_adjust_s"]:
            extras.append(f"beat {g['beat_adjust_s']:+.2f} s")
        if g.get("color_match") and g["color_match"] != "matched (no change needed)":
            extras.append(f"color: {g['color_match']}")
        if g.get("what"):
            extras.append(f"_{g['what']}_")
        tr = g["transition_in"] + (f" ({g['transition_s']:.2f} s)" if g["transition_s"] else "")
        L.append(f"| {g['n']} | {g['clip']} {g['source']} | {g['starts_at']:.1f} s +{g['on_screen_s']:.1f} s | "
                 f"{g['speed']:g}× | {g['camera']} | {tr} | {g['reframe']} | {g['sound']} | {'; '.join(extras)} |")
    ch = r["changes"]
    L += ["", "## Changes made automatically after the editor's plan"]
    if not (ch["validation"] or ch["assistant_editor"] or ch["timeline"] or ch["beat_sync"]):
        L.append("None: the editor's plan was used as-is.")
    if ch["validation"]:
        L += ["", "**Validation** (the plan asked for something impossible):"] + [f"- {w}" for w in ch["validation"]]
    if ch["assistant_editor"]:
        L += ["", f"**Assistant editor** (rule level: {ch.get('assistant_editor_level')}):"] + \
             [f"- [{c['rule']}] {c['change']}, because {c['why']}" for c in ch["assistant_editor"]]
    if ch["timeline"]:
        L += ["", "**Timeline:**"] + [f"- {t}" for t in ch["timeline"]]
    if ch["beat_sync"]:
        L += ["", f"**Beat sync:** {ch['beat_sync']}"]
    lr = r.get("learning") or {}
    if lr and not lr.get("error"):
        L += ["", "## What it learned from you and applied", f"- Taste model confidence: {lr.get('confidence', 0):.0%} "
              f"({lr.get('candidates', 0)} variants scored)"]
        L += [f"- House style: {b}" for b in lr.get("house_style", [])]
        L += [f"- Applied: {a}" for a in lr.get("applied", [])] or ["- Applied: nothing (the editor's plan already fit)"]
        if lr.get("explored"):
            L.append(f"- Exploring: tried {lr['explored']} to keep learning what you like")
    L += ["", "## Render",
          f"- {rd['resolution']} @ {rd['fps']} fps, {rd['codec']}, CRF {rd['crf']} ({rd['preset']}), quality {rd['quality']}",
          f"- {rd['frames']} frames, {rd['duration_s']} s, rendered in {rd['render_seconds']} s"
          + (f", {rd['file_mb']} MB" if rd.get("file_mb") else ""),
          "- Time: " + ", ".join(f"{k} {v} s" for k, v in (r.get("timings_s") or {}).items()), ""]
    return "\n".join(L)
