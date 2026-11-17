"""Turn revision requests ("make it faster, warmer, no glitch") into structured preference signals."""
from __future__ import annotations

import re

# (pattern, feature deltas, already_negated). Deltas point in the direction you asked for.
RULES = [
    (r"\b(faster|quicker|punch(y|ier)|more energy|more energetic|snappier|tighter|shorter (shots|clips|cuts)|"
     r"quick cuts|fast cuts|speed it up)\b", {"pace_short": 1, "avg_len": -1, "cuts_per_10s": 1}, False),
    (r"\b(slower|calmer|longer (shots|clips|cuts)|let it breathe|less rushed|more relaxed|slow it down)\b",
     {"pace_short": -1, "avg_len": 1, "cuts_per_10s": -1}, False),
    (r"\b(warm(er)?|golden|cozy|cosy|sunny)\b", {"lf_warm": 1, "lf_film": 0.4, "lf_cool": -1}, False),
    (r"\b(cool(er)?|cold(er)?|blu(e|er|ish))\b", {"lf_cool": 1, "lf_warm": -1}, False),
    (r"\b(moody|moodier|dark(er)?|dramatic|gloomy)\b", {"lf_dark": 1, "lf_bright": -1}, False),
    (r"\b(bright(er)?|light(er)?|airy|happier|cheerful)\b(?!\s+(leaks?|flares?))", {"lf_bright": 1, "lf_dark": -1},
     False),
    (r"\b(more colou?rs?|colou?rful|vibrant|saturated|pop(py)?)\b", {"lf_vibrant": 1, "lf_mono": -1}, False),
    (r"\b(muted|desaturated|less colou?r|subtle colou?rs?)\b", {"lf_vibrant": -1, "look_intensity": -0.5}, False),
    (r"\b(black and white|b&w|b/w|monochrome|grayscale|greyscale)\b", {"lf_mono": 1}, False),
    (r"\b(film(ic)?|analog(ue)?|vintage|retro|nostalgic)\b", {"lf_film": 1, "grain": 0.5}, False),
    (r"\b(cinematic|movie|blockbuster)\b", {"lf_cinematic": 1, "letterbox": 0.4}, False),
    (r"\bneon|cyberpunk\b", {"lf_neon": 1}, False),
    (r"\b(no|less|without|remove( the)?) grain\b|\bcleaner image\b", {"grain": -1}, True),
    (r"\bmore grain\b|\bgrainy\b", {"grain": 1}, False),
    (r"\b(no|less|fewer|without) (fancy )?(transitions|effects)\b|\bclean(er)? cuts?\b|\bsimple cuts?\b|"
     r"\bjust cuts\b|\bhard cuts\b", {"styl_frac": -1}, True),
    (r"\bmore transitions\b|\bmore effects\b", {"styl_frac": 1}, False),
    (r"\bsmooth(er)? transitions\b|\bsoft(er)? transitions\b", {"t_basic": 1, "t_blur": 0.5, "t_glitch": -0.5}, False),
    (r"\b(no|less|fewer|without) glitch(es|y)?\b", {"t_glitch": -1}, True),
    (r"\bglitch(y)?\b", {"t_glitch": 1}, False),
    (r"\b(no|less|fewer|without) flash(es)?\b", {"t_light": -1}, True),
    (r"\b(no|less|fewer|without) (whips?( pans?)?|spins?|3d|shakes?)\b", {"t_camera": -1}, True),
    (r"\b(whips?|spins?|3d (flip|cube)|camera moves?)\b", {"t_camera": 1}, False),
    (r"\b(no|less|fewer|without) zoom(s|ing)?\b", {"t_camera": -0.6, "m_push": -0.5}, True),
    (r"\b(no|less|fewer|without) (light leaks?|flares?|film burns?)\b", {"t_light": -1}, True),
    (r"\b(light leaks?|lens flares?|film burns?|flares?)\b", {"t_light": 1}, False),
    (r"\b(slides?|swipes?|push(es)?)\b", {"t_slide": 1}, False),
    (r"\b(blur(ry)? transitions?|focus blur|dreamy transitions?)\b", {"t_blur": 1}, False),
    (r"\b(shapes?|heart|star|masks?|iris)\b", {"t_mask": 1}, False),
    (r"\b(warps?|ripples?|distort(ion)?|liquid|twist)\b", {"t_distortion": 1}, False),
    (r"\b(paper tear|film strip|scrapbook|collage|story swipe)\b", {"t_overlay": 1}, False),
    (r"\b(trend(y|ing)|viral|tiktok|capcut)\b", {"t_trending": 1, "styl_frac": 0.4}, False),
    (r"\b(beat|on the beat|pulse|bounce)\b", {"e_beat": 1}, False),
    (r"\b(vhs|camcorder|retro tape)\b", {"e_retro": 1}, False),
    (r"\bmore slow[- ]?mo(tion)?\b|\badd slow[- ]?mo(tion)?\b|\bslow[- ]?mo(tion)?\b", {"slowmo_frac": 1}, False),
    (r"\b(no|less|without) slow[- ]?mo(tion)?\b", {"slowmo_frac": -1}, True),
    (r"\b(no|remove|without|drop)( the)? (text|title|titles)\b", {"has_title": -1, "text_frac": -1}, True),
    (r"\badd (a |some )?(title|text)\b|\bmore text\b", {"has_title": 1}, False),
    (r"\b(letterbox|cinematic bars|black bars|widescreen)\b", {"letterbox": 1}, False),
    (r"\b(no|remove|without)( the)? (bars|letterbox)\b", {"letterbox": -1}, True),
    (r"\b(less|no) (shake|shaky|handheld)\b|\bsteadier\b|\bmore stable\b", {"m_handheld": -1}, True),
    (r"\b(more|smoother) (camera )?(movement|motion|moves)\b", {"m_drift": 0.6, "m_push": 0.6, "m_static": -0.6}, False),
    (r"\b(stronger|better|more exciting) (hook|opening|start|intro)\b|\bopen with\b", {"hook_q": 1}, False),
    (r"\b(keep|more) (the )?(talking|speech|voice|dialogue|audio)\b", {"speech_frac": 1}, False),
    (r"\b(no|less|remove)( the)? (talking|speech|voice)\b|\bjust music\b|\bmusic only\b", {"speech_frac": -1}, True),
    (r"\bcaptions?\b|\bsubtitles?\b", {"captions": 1}, False),
    (r"\b(no|remove|without) (captions?|subtitles?)\b", {"captions": -1}, True),
    (r"\bvignette\b", {"vignette": 0.6}, False),
    (r"\b(glow|dreamy|soft glow|bloom)\b", {"bloom": 0.6, "halation": 0.4}, False),
]
# a negator only applies to words right after it ("less warm", "not too dark") - never across a comma/"and"
NEGATORS = re.compile(r"\b(less|no|not|without|fewer|too much|stop|don't|dont|remove)\s+((?!and\b)\w+\s+){0,2}$")


def parse(text: str) -> dict[str, float]:
    """Feature deltas implied by a revision request. Handles simple negation ("less warm")."""
    t = (text or "").lower()
    out: dict[str, float] = {}
    for pat, deltas, negated in RULES:
        for m in re.finditer(pat, t):
            sign = 1.0
            if not negated and NEGATORS.search(t[max(0, m.start() - 24): m.start()]):
                sign = -1.0
            for k, v in deltas.items():
                out[k] = max(-1.5, min(1.5, out.get(k, 0.0) + sign * v))
    # "no glitch" also matched the generic positive rules? cancel double counts toward zero
    return {k: round(v, 3) for k, v in out.items() if abs(v) > 1e-6}
