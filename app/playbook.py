"""The editing playbook shared by every AI editor (Claude, Gemini, local Ollama models).

One set of editing principles, one plan schema and one way of describing the footage, so each
provider is held to the same craft rules. Small local models get a simplified "compact" schema
(see providers/ollama.py), but the same principles.
"""
from __future__ import annotations

import base64
import json

from .effects import CAPTION_STYLES, LOOK_DESCRIPTIONS, LOOKS, TEXT_POSITIONS, TEXT_STYLES
from .reframe import MOTIONS
from .transitions import TRANSITIONS

ROLE = ("You are Reelsmith, a senior short-form video editor and colorist who cuts scroll-stopping Instagram "
        "Reels. You receive a creative brief, a log of the user's footage and optional music analysis, and you "
        "return a complete edit decision list.")

FORMAT_NOTE = ("Output format: vertical 9:16, 1080x1920, 30 fps. The renderer reframes horizontal footage "
               "automatically with subject tracking.")

PRINCIPLES = f"""Editing principles:
- Hook: the first 1-1.5 s must be the most striking image or moment. No slow, empty openings.
- Respect the target duration (within ±10%).
- Pick the best part of each shot; avoid low-quality (blurry/dark/shaky) shots. Never reuse the same moment twice.
- Pacing: energetic briefs ~0.5-1.5 s per segment; cinematic/calm ~2-4 s. Vary the rhythm, build to a peak, and \
end on a strong, resolved image.
- Flow: order shots so they read as a story (establishing wide -> medium -> close-up details), match movement \
direction across cuts, and follow any order the brief asks for ("open on...", "end with...").
- Speed: 1.0 normal. 0.5 slow motion suits high-motion shots and high-fps sources. 1.5-2 for time-lapse energy. \
Speed ramp = one continuous moment split into consecutive segments with different speeds joined by a cut.
- Transitions: use 'cut' for most edits; stylised transitions land harder when rare (at most ~1 in 4 edits) and \
must match the mood: {json.dumps({k: v[1] for k, v in TRANSITIONS.items()})}
- Camera motion per segment: {json.dumps(MOTIONS)}. Avoid identical motion on consecutive segments.
- Looks: {json.dumps(LOOK_DESCRIPTIONS)}. Grain suits film/vintage looks; letterbox suits 'cinematic' briefs.
- Text: optional, max ~6 words, only when it adds value or the brief asks for it. cinematic_title for an opening \
title / place / theme; bold for punchy hooks; subtitle for a quiet line; kicker for small labels.
- Audio: when music is present it plays under everything. Keep the original sound on segments where it matters \
(speech, laughter, ambient impact) and keep their speed at 1.0; music ducks automatically under speech. Without \
music, all original sound is kept.
- Captions (if enabled) are generated from speech on segments that keep their sound: cut on word boundaries, never \
mid-word.
- Beat sync (if enabled): cuts snap to the nearest beat automatically, so make segment lengths roughly whole \
multiples of the beat interval."""

FULL_FIELD_GUIDE = """Plan fields: start/end are absolute seconds in the source clip; stay inside the chosen shot's \
range where possible (a shot never crosses a scene cut). focus_x is the main subject's horizontal position in the \
SOURCE frame (0 = left edge, 1 = right edge), judged from the thumbnail. transition_in is how a segment enters (the \
first segment's is ignored). keep_audio plays the shot's original sound. music_start is where in the track the reel \
begins: choose a downbeat in an energetic section."""

SYSTEM_PROMPT_FULL = "\n\n".join([ROLE, FORMAT_NOTE, FULL_FIELD_GUIDE, PRINCIPLES])

# ----------------------------------------------------------------------------------------------
# Full plan schema (Claude tool input / Gemini response schema)
# ----------------------------------------------------------------------------------------------
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string",
                      "description": "2-4 sentences: the creative concept, the hook, structure and pacing."},
        "title": {"type": "string", "description": "Short working title for the reel."},
        "look": {"type": "string", "enum": list(LOOKS)},
        "look_intensity": {"type": "number", "minimum": 0, "maximum": 1.2},
        "grain": {"type": "number", "minimum": 0, "maximum": 1, "description": "film grain amount, 0 = none"},
        "vignette": {"type": "number", "minimum": 0, "maximum": 1},
        "letterbox": {"type": "boolean", "description": "cinematic black bars top and bottom"},
        "fade_in": {"type": "boolean"},
        "fade_out": {"type": "boolean"},
        "music_start": {"type": "number", "description": "seconds into the music track where the reel starts"},
        "music_volume": {"type": "number", "minimum": 0, "maximum": 1},
        "original_audio_volume": {"type": "number", "minimum": 0, "maximum": 1.5},
        "caption_style": {"type": "string", "enum": list(CAPTION_STYLES)},
        "segments": {
            "type": "array",
            "minItems": 1,
            "maxItems": 60,
            "items": {
                "type": "object",
                "properties": {
                    "shot_id": {"type": "string"},
                    "start": {"type": "number", "description": "absolute seconds in the source clip"},
                    "end": {"type": "number", "description": "absolute seconds in the source clip"},
                    "speed": {"type": "number", "minimum": 0.25, "maximum": 4},
                    "motion": {"type": "string", "enum": list(MOTIONS)},
                    "transition_in": {"type": "string", "enum": list(TRANSITIONS)},
                    "transition_duration": {"type": "number", "minimum": 0, "maximum": 1.2},
                    "focus_x": {"type": "number", "minimum": 0, "maximum": 1,
                                "description": "horizontal position of the main subject in the source frame"},
                    "keep_audio": {"type": "boolean", "description": "play this shot's original sound"},
                    "text": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "style": {"type": "string", "enum": list(TEXT_STYLES)},
                            "position": {"type": "string", "enum": list(TEXT_POSITIONS)},
                        },
                        "required": ["content"],
                    },
                },
                "required": ["shot_id", "start", "end", "speed", "motion", "transition_in"],
            },
        },
    },
    "required": ["reasoning", "title", "look", "segments"],
}


# ----------------------------------------------------------------------------------------------
# Footage description
# ----------------------------------------------------------------------------------------------
def words_in(tr, a, b):
    if not tr:
        return []
    return [w for w in tr.get("words", []) if w["start"] >= a - 0.05 and w["end"] <= b + 0.05]


def settings_lines(options: dict, music: dict | None) -> list[str]:
    return [
        f"- Target duration: {float(options.get('target_duration', 30)):.0f} s (±10%)",
        f"- Auto-pick best moments: {'yes - choose freely among shots' if options.get('auto_pick', True) else 'NO - use every clip, in the given order, trimming only the weak parts'}",
        f"- Beat sync: {'on' if options.get('beat_sync') and music else 'off'}",
        f"- Captions: {'on' if options.get('captions') else 'off'}",
        f"- Smart reframe: {'on' if options.get('smart_reframe', True) else 'off (centre crop)'}",
    ]


def clip_lines(analysis: dict) -> list[str]:
    tr_all = analysis.get("transcripts", {})
    out = []
    for c in analysis["clips"]:
        tr = tr_all.get(c["id"])
        nw = len(tr["words"]) if tr else 0
        out.append(f"Clip {c['id']} \"{c['name']}\": {c['duration']:.1f}s, {c['width']}x{c['height']} "
                   f"{c['orientation']}, {c['fps']:.0f}fps, {'audio' if c['has_audio'] else 'no audio'}"
                   f"{f', speech: {nw} words' if nw else ''}, {c['num_scenes']} scenes")
    return out


def transcript_lines(analysis: dict, limit_chars: int = 5000) -> list[str]:
    out = []
    for cid, tr in analysis.get("transcripts", {}).items():
        if tr and tr.get("segments"):
            if not out:
                out.append("# Transcript sentence timings (absolute clip seconds)")
            txt = " ".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in tr["segments"])
            out.append(f"{cid}: {txt[:limit_chars]}")
    return out


def music_lines(music: dict | None, energy: bool = True) -> list[str]:
    if not music:
        return ["# Music", "none: segments keep their original audio"]
    bi = music["beat_interval"]
    near = [d for d in music["downbeats"] if abs(d - music["suggested_start"]) < 20][:12]
    out = [f"# Music \"{music['name']}\"",
           f"tempo {music['tempo']} BPM, beat every {bi:.3f}s, 4-beat bar {4 * bi:.2f}s, duration {music['duration']:.0f}s",
           f"suggested music_start: {music['suggested_start']:.2f}s (most energetic window of the target length)",
           f"downbeats near it: {', '.join(f'{d:.2f}' for d in near)}"]
    if energy:
        e2 = music["energy_half_seconds"][::4]
        out.append("energy per 2 s from 0 s: " + " ".join(f"{e:.2f}" for e in e2[:150]))
    return out


def build_context_text(analysis: dict, music: dict | None, options: dict, description: str) -> str:
    """Context for big vision models (Claude / Gemini), who also receive the contact sheets."""
    tr_all = analysis.get("transcripts", {})
    L = ["# Creative brief", description.strip() or "(no brief: make a polished, engaging reel)", "",
         "# Settings", *settings_lines(options, music), "", "# Footage", *clip_lines(analysis), "",
         "# Shots (id | start-end s | dur | score | motion | faces | speech)"]
    for s in analysis["shots"]:
        ws = words_in(tr_all.get(s["clip_id"]), s["start"], s["end"])
        sp = ""
        if ws:
            sp = ' | "' + " ".join(w["word"] for w in ws[:20]) + (" ..." if len(ws) > 20 else "") + '"'
        L.append(f"{s['id']} | {s['start']:.2f}-{s['end']:.2f} | {s['duration']:.1f}s | {s['score']:.2f} | "
                 f"{s['motion']:.2f} | {s['face_frac']:.1f}{sp}")
    tl = transcript_lines(analysis)
    if tl:
        L += [""] + tl
    L += [""] + music_lines(music)
    n = len(analysis.get("sheets", []))
    L += ["", "# Contact sheets", f"The {n} attached image(s) show each shot's middle frame labelled with its id.",
          "", "Plan the reel and return the complete edit plan."]
    return "\n".join(L)


def refine_suffix(prev_plan: dict, feedback: str) -> str:
    prev = {k: v for k, v in prev_plan.items() if k not in ("planner", "warnings", "estimated_duration")}
    return ("\n\n# Current edit (already rendered)\n" + json.dumps(prev, indent=1)[:30000] +
            "\n\n# Revision request from the user\n" + feedback.strip() +
            "\n\nApply the revision. Keep what works, change what the user asked, and return the COMPLETE updated plan.")


def sheet_images_b64(analysis: dict, limit: int = 20) -> list[str]:
    out = []
    for p in analysis.get("sheets", [])[:limit]:
        with open(p, "rb") as f:
            out.append(base64.standard_b64encode(f.read()).decode())
    return out
