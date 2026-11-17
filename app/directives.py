"""Hard rules from what you typed.

AI editors (small local ones especially) and the automatic passes (assistant editor, learner,
shot matching) treat the brief as inspiration. Some sentences are not inspiration, they are
instructions: "don't touch the colour grading", "no text", "only cuts", "keep the clips in order".

This module finds those instructions in the brief and every revision (later ones win), and:
  * tells the AI editor about them as non-negotiable rules,
  * locks those dimensions so the learner / assistant editor don't change them,
  * enforces them on the final plan - whatever the AI wrote - and reports every fix,
  * switches the renderer to a true raw-colour path when grading is off (no LUT at all).
"""
from __future__ import annotations

import re

from .clipfx import CLIP_EFFECTS
from .transitions import REGISTRY, by_category

_CAT = by_category()
_ALL_TR = {n for n in REGISTRY if n != "cut"}
# clip effects that recolour a whole shot (a "grade" in disguise) - off in raw-colour mode unless asked for by name
COLOR_FX = {"vhs", "glow"}

NEG = re.compile(r"\b(don'?t|do not|dont|never|no|not|without|w/o|skip|avoid|stop|remove|drop|leave out|lose|"
                 r"get rid of|nothing|zero|ditch|kill|turn off|disable|exclude|forget|nix)\b")
NEG_FALSE = re.compile(r"^\s+(like|love|care|mind|know|think|want this|matter|bad|too|so)\b")  # "don't like..."
POS = re.compile(r"\b(add|use|make|give|apply|put|include|with|want|need|keep|try|more|some|lots? of|plenty of|"
                 r"turn on|enable|grade|do)\b")
ONLY = re.compile(r"\b(just|only|simply|purely|nothing but)\b")

# term -> (family, payload). Order matters: longer / more specific phrases first (spans are consumed).
_s = lambda *names: frozenset(names)  # noqa: E731
TERMS: list[tuple[str, str, frozenset]] = [
    # --- colour -------------------------------------------------------------------------------
    (r"colou?r[- ]?(grad(e|es|ed|ing)|correct(ion|ions|ing|ed)?|chang(e|es|ing)|edits?|adjust(ment|ments)?|"
     r"filters?|tweaks?|work|treatment|enhancements?|effects?)", "grading", _s()),
    (r"\b(grad(e|es|ed|ing)|luts?|filters?|tint(s|ing)?|looks?)\b(?! (like|at|good|great|better|nice))",
     "grading_weak", _s()),
    (r"\b(touch\w*|chang\w*|alter\w*|mess\w* with|edit\w*|modif\w*|adjust\w*|correct\w*|tweak\w*)\b[^,;.]{0,24}?"
     r"\bcolou?rs?\b", "grading_verb", _s()),
    # --- transitions ------------------------------------------------------------------------
    (r"\b(transition (effects?|fx)|transitions?)\b", "transitions", frozenset(_ALL_TR)),
    (r"\b(light (effects?|transitions?))\b", "tr", frozenset(_CAT["light"])),
    (r"\blight[- ]?leaks?\b", "tr+fx", _s("light_leak")),
    (r"\b(lens )?flares?\b", "tr", _s("lens_flare")),
    (r"\b(film )?burns?\b", "tr", _s("film_burn")),
    (r"\bglitch(es|y|ing)?\b", "tr+fx", frozenset(_CAT["glitch"]) | {"glitch_whip", "rgb_split"}),
    (r"\b(rgb|chromatic) split\b", "tr+fx", _s("rgb_split", "chromatic_blur")),
    (r"\bflash(es|y|ing)?\b", "tr+fx", _s("flash", "glow_flash", "heatwave_flash", "flash_blast", "zoom_flash",
                                         "dip_to_white", "flash_beat")),
    (r"\bwhip(s|[- ]?pans?)?\b", "tr", frozenset(n for n in _ALL_TR if "whip" in n)),
    (r"\b(spin(s|ning)?|rotat\w*)\b", "tr", _s("spin", "spin_blur", "spin_zoom")),
    (r"\b(3d|flips?|cubes?)\b", "tr", _s("flip_3d", "cube_3d")),
    (r"\b(shak(e|es|y|ing)|camera shake)\b", "tr+fx+mo", _s("shake", "shake_punch", "handheld")),
    (r"\bzoom[- ]?pulses?\b|\bbeat (zooms?|pulses?)\b|\bpulse on the beat\b", "fx", _s("zoom_pulse")),
    (r"\bshake on (the|every) beat\b|\bbeat shakes?\b", "fx", _s("shake")),
    (r"\bflash(es)? on (the|every) beat\b|\bbeat flash(es)?\b", "fx", _s("flash_beat")),
    (r"\b(punch|push)[- ]?in( zooms?)?\b|\bzoom[- ]?(ins?|outs?)\b", "mo", _s("push_in", "pull_out", "punch_in")),
    (r"\bdip[- ]?to[- ]?black\b|\bfade (through|to) black\b", "tr", _s("dip_to_black")),
    (r"\bdip[- ]?to[- ]?white\b|\bwhite (flash|dip)\b", "tr", _s("dip_to_white")),
    (r"\bzoom(s|ing)?\b", "tr+fx+mo", _s("zoom_punch", "zoom_blur", "zoom_flash", "spin_zoom", "pull_in", "pull_out",
                                         "zoom_pulse", "push_in", "punch_in")),
    (r"\bblur(s|ry)?\b", "tr", frozenset(_CAT["blur"])),
    (r"\b(cross[- ]?fades?|dissolves?|fades?(?![- ]?(in|out|from|to)\b)|mix(es)?)\b", "tr",
     _s("crossfade", "dip_to_black", "dip_to_white", "blur_dissolve")),
    (r"\bwipes?\b", "tr", frozenset(n for n in _ALL_TR if "wipe" in n) | {"barn_door"}),
    (r"\b(slides?|swipes?|push(es)?(?![- ]?ins?)|screen swaps?|split[- ]screens?)\b", "tr",
     frozenset(_CAT["slide"]) | {"story_swipe"}),
    (r"\bhearts?\b", "tr", _s("heart")),
    (r"\bstars?\b", "tr", _s("star")),
    (r"\b(iris|circles?)\b", "tr", _s("iris")),
    (r"\b(masks?|shapes?|shape transitions?)\b", "tr", frozenset(_CAT["mask"])),
    (r"\b(warps?|distort\w*|ripples?|waves?|liquid|fisheye|twist\w*|stretch\w*)\b", "tr", frozenset(_CAT["distortion"])),
    (r"\b(paper[- ]?tears?|film[- ]?strips?|scrapbook|collage|story swipes?)\b", "tr", frozenset(_CAT["overlay"])),
    # --- clip effects -----------------------------------------------------------------------
    (r"\b(vhs|camcorder)\b", "fx", _s("vhs")),
    (r"\b(dust|scratches)\b", "fx", _s("film_dust")),
    (r"\b(pulse|pulses|bounce|bounces|beat effects?)\b", "fx", _s("zoom_pulse")),
    (r"\b(effects?|fx|filters? and effects)\b", "effects", frozenset(CLIP_EFFECTS) - {"none"}),
    # --- finishing --------------------------------------------------------------------------
    (r"\b(letter[- ]?box(ing|ed)?|(black|cinematic) bars|bars|widescreen|cinemascope)\b", "letterbox", _s()),
    (r"\b(film )?grain\b|\bnoise\b", "grain", _s()),
    (r"\bvignett\w*\b", "vignette", _s()),
    (r"\b(bloom|halation|glow(s|ing|y)?)\b", "glow", _s("glow")),
    # --- text / captions --------------------------------------------------------------------
    (r"\b(captions?|subtitles?|subs)\b", "captions", _s()),
    (r"\b((on[- ]?screen )?text|titles?|words on screen|overlays?|typography)\b", "text", _s()),
    # --- time -------------------------------------------------------------------------------
    (r"\bslow[- ]?(mo|motion|down)\b|\bslowed( down)?\b", "slowmo", _s()),
    (r"\b(speed[- ]?(ramps?|ramping|changes?|ups?)|speed(ing)? up|fast[- ]?forward|time[- ]?lapses?|ramps?)\b",
     "speed", _s()),
    (r"\b(camera (moves?|movements?|motion)|ken burns|pans?|drift\w*|push[- ]?ins?|punch[- ]?ins?|movement)\b", "mo",
     _s("push_in", "pull_out", "punch_in", "drift_left", "drift_right", "rise", "ken_burns", "handheld")),
    # --- audio ------------------------------------------------------------------------------
    (r"\b(background music|music|songs?|soundtrack|bgm)\b", "music", _s()),
    (r"\b(original|natural|clip|video|source|camera) (audio|sound)\b|\b(talking|speech|voices?|dialogue)\b",
     "orig_audio", _s()),
    (r"\bfade[- ]?(in|out)s?\b|\bfade (from|to) black\b|\bfades\b(?= (in|at|on)\b)", "fades", _s()),
]
TERMS_RE = [(re.compile(p), fam, pay) for p, fam, pay in TERMS]
BARE_OK = re.compile(r"^(light[- ]?leaks?|(lens )?flares?|film burns?|glitch\w*|(rgb|chromatic) split|whip\w*( pans?)?|"
                     r"cross[- ]?fades?|dissolves?|wipes?|zoom[- ]?pulses?|beat \w+|vhs|camcorder|letter[- ]?box\w*|"
                     r"cinematic bars|captions?|subtitles?|(film )?grain|vignett\w*|slow[- ]?(mo|motion)|dip[- ]?to[- ]?\w+|"
                     r"transitions?|transition effects?|3d|paper[- ]?tears?|film[- ]?strips?|scrapbook|"
                     r"(punch|push)[- ]?in( zooms?)?|zoom[- ]?(ins?|outs?)|shake on .*|flash(es)? on .*)$")

# whole-phrase rules: (regex, key, value, label)
EXPLICIT = [
    (r"\b(raw|original|natural|true|real|untouched|ungraded) (footage|colou?rs?|look|video|clips?|image)\b|"
     r"\bas (it was )?shot\b|\bungraded\b|\bcolou?rs? (as is|untouched|alone|natural|original|the same)\b|"
     r"\bleave\b[^,;.]{0,20}\bcolou?rs?\b|\bkeep\b[^,;.]{0,16}\bcolou?rs?\b(?! (pop|vibrant|bright|warm|cool))",
     "grading", False, "Keep the original colours (no grading)"),
    (r"\b(colou?rs?|grading|colou?r grade|tones?) (are|is|look|looks|seem) (fine|good|great|perfect|nice|ok|okay|right)\b",
     "grading", False, "Keep the original colours (no grading)"),
    (r"\b(just|only|straight|hard|simple|plain|clean) cuts?\b|\bcuts? only\b|\bjump cuts? only\b",
     "transitions", "none", "Cuts only (no transitions)"),
    (r"\b(normal|real|original|natural|actual) speed\b|\breal[- ]?time\b|\b(don'?t|do not|no) (change|alter|mess with)"
     r"[^,;.]{0,10}speed\b", "speed", False, "Keep everything at normal speed"),
    (r"\b(keep|in|same|original|chronological|recorded|upload(ed)?)\b[^,;.]{0,18}\border\b(?! to\b)|"
     r"\bchronologic\w*\b|\b(don'?t|do not|no) (re-?order|shuffl\w*|rearrang\w*)\b|\bas (they are|uploaded|recorded)\b",
     "keep_order", True, "Keep the clips in their original order"),
    (r"\b(use|include|keep|show) (all|every|each)( of)?( the| my)? (clips?|videos?|footage)\b",
     "use_all", True, "Use every clip"),
    (r"\b(static|steady|locked[- ]?off|still) (shots?|camera|frames?)\b|\bkeep (it|the camera|shots) (static|steady|still)\b",
     "motion", False, "No camera moves"),
    (r"\b(original|natural|clip|source) (audio|sound) only\b|\bonly (the )?(original|natural|clip|source) "
     r"(audio|sound)\b", "music", False, "Original audio only (no music)"),
    (r"\bmute( the)? (original|clip|video|camera|source)?\s*(audio|sound)\b", "orig_audio", False,
     "Mute the original audio"),
    (r"\bnothing (else|fancy)\b|\bkeep it raw\b|\bminimal (edit|editing)\b|\bdon'?t (do|add|change) anything else\b",
     "only", "minimal",
     "Minimal edit (nothing extra)"),
]
EXPLICIT_RE = [(re.compile(p), k, v, lbl) for p, k, v, lbl in EXPLICIT]

DURATION_RE = re.compile(
    r"(?:\b(make it|keep it|under|max(?:imum)?|at most|no (?:longer|more) than|exactly|around|about|approx\w*|~)\s*"
    r"(\d{1,3})\s*-?\s*(?:s|secs?|seconds?)\b)|(?:\b(\d{1,3})\s*-?\s*(?:s|secs?|seconds?)\s+(long|reel|video|edit|"
    r"total|max|or less|or under|tops|version|cut)\b)")
TITLE_RE = re.compile(r"\b(?:title|caption it|call it|text|heading)\s*(?:[:=]|of|saying|that says|reading|should say|"
                      r"says)?\s*[\"“'‘]([^\"”'’]{1,60})[\"”'’]", re.I)

_ONLY_LEAD = re.compile(r"^(just|only|simply|purely|nothing but)\s+((add|use|do|apply|put|want|need|include|with|"
                        r"some|the|a|few|nice|cool|smooth|simple|basic|clean|give it|make it|keep)\s+){0,3}$")
_LIST_JOIN = re.compile(r"^\s*(,|and|or|&|\+|/)?\s*(the |some |a few |a )?$")
DECOR = ("grading", "texture", "letterbox", "text", "effects", "speed", "motion")   # what "just/only X" turns off


def _clauses(text: str) -> list[str]:
    t = (text or "").lower().replace("’", "'")
    t = re.sub(r"\bblack and white\b", "black&white", t)
    parts = re.split(r"[.;!?\n]+|,|\bbut\b|\bhowever\b|\bthen\b|\balso\b|\bplus\b", t)
    return [p.strip() for p in parts if p and p.strip()]


def _cues(clause: str):
    """Ordered (pos, kind) cues inside a clause: 'neg', 'pos', 'only'."""
    cues = []
    negs = []
    for m in NEG.finditer(clause):
        if NEG_FALSE.match(clause[m.end():]):
            continue
        negs.append(m)
        cues.append((m.start(), m.end(), "neg"))
    for m in POS.finditer(clause):
        if any(0 <= m.start() - n.end() <= 12 or n.start() <= m.start() < n.end() for n in negs):
            continue  # "don't add", "do not use" -> still a negation
        cues.append((m.start(), m.end(), "pos"))
    for m in ONLY.finditer(clause):
        cues.append((m.start(), m.end(), "only"))
    return sorted(cues)


class Directives:
    def __init__(self):
        self.v: dict = {}
        self.ban_tr: set[str] = set()
        self.want_tr: set[str] = set()
        self.want_groups: list[tuple[str, list[str]]] = []  # each kind you named, canonical name first
        self.only_tr: set[str] = set()
        self.ban_fx: set[str] = set()
        self.want_fx: set[str] = set()
        self.ban_mo: set[str] = set()
        self.only_scope: set[str] = set()
        self.positive: set[str] = set()
        self.rules: dict[str, dict] = {}   # key -> {"label", "source"}  (latest wins)
        self.looks: list[str] = []
        self._carry: str | None = None

    def _looks(self, text: str, source: str):
        """Explicit look requests: "use the golden hour look", "teal orange filter", "lut:my_lut"."""
        tl = (text or "").lower().replace("-", " ")
        for lk in self.looks:
            phrase = lk.replace("lut:", "").replace("_", " ")
            p = r"\s+(?:and\s+|&\s+|n\s+)?".join(re.escape(x) for x in phrase.split())
            if not re.search(r"\b" + p + r"\b", tl):
                continue
            if re.search(r"\b(no|not|without|don'?t|never)\b[^,.;]{0,20}\b" + p, tl):
                continue
            if re.search(r"\b(use|apply|with|try|give|make|grade|in|go)\b[^,.;]{0,30}\b" + p + r"\b", tl) or \
                    re.search(r"\b" + p + r"\s+(look|filter|grade|lut|preset|style)\b", tl):
                self.set("look", lk, f"Look: {phrase}", f"{source}: “{phrase}”")
                if self.v.get("grading") is False:
                    self.set("grading", True, "Colour grading on (you picked a look)", f"{source}: “{phrase}”")

    # -- building -------------------------------------------------------------------------
    def set(self, key, value, label, source):
        self.v[key] = value
        self.rules[key] = {"key": key, "value": value, "label": label, "source": source}

    def _tr(self, names, on, source, label_on, label_off):
        names = set(names)
        if on:
            self.want_tr |= names
            self.ban_tr -= names
            order = [n for n in REGISTRY if n in names]  # registry order: the classic variant comes first
            self.want_groups = [g for g in self.want_groups if set(g[1]) != set(order)] + [(label_on[4:], order)]
        else:
            self.ban_tr |= names
            self.want_tr -= names
            self.want_groups = [(lb, [n for n in g if n not in names]) for lb, g in self.want_groups]
            self.want_groups = [g for g in self.want_groups if g[1]]
        key = "tr:" + ",".join(sorted(names))[:60]
        self.rules[key] = {"key": key, "value": on, "label": label_on if on else label_off, "source": source}

    def _fx(self, names, on):
        names = set(names) & set(CLIP_EFFECTS)
        if on:
            self.want_fx |= names
            self.ban_fx -= names
        else:
            self.ban_fx |= names
            self.want_fx -= names

    def _mo(self, names, on):
        names = set(names) & {"push_in", "pull_out", "punch_in", "drift_left", "drift_right", "rise", "ken_burns",
                              "handheld"}
        if on:
            self.ban_mo -= names
        else:
            self.ban_mo |= names

    def add_text(self, text: str, source: str):
        self._carry = None
        for clause in _clauses(text):
            self._clause(clause, source)
        m = TITLE_RE.search((text or "").replace("’", "'"))
        if m:
            self.set("title", m.group(1).strip(), f"Title: “{m.group(1).strip()}”", source)
            self.positive.add("text")
        self._looks(text, source)
        for m in DURATION_RE.finditer((text or "").lower()):
            n = int(m.group(2) or m.group(3))
            if 5 <= n <= 180:
                self.set("duration", float(n), f"Length about {n} s", source)

    def _clause(self, c: str, source: str):
        quote = c if len(c) <= 90 else c[:87] + "…"
        src = f"{source}: “{quote}”"
        consumed: list[tuple[int, int]] = []
        for rx, key, val, label in EXPLICIT_RE:
            for m in rx.finditer(c):
                # "don't keep the colours" etc. - a negator right before an explicit phrase flips nothing; skip it
                pre = c[max(0, m.start() - 10): m.start()]
                if key == "grading" and re.search(r"\b(don'?t|not|no)\s*$", pre):
                    continue
                if key == "only" and source.startswith("Brief"):
                    self.only_scope.add("minimal")
                    self.rules["only"] = {"key": "only", "value": "minimal", "label": label, "source": src}
                else:
                    self.set(key, val, label, src)
                consumed.append((m.start(), m.end()))
        cues = _cues(c)
        hits = []
        for rx, fam, pay in TERMS_RE:
            for m in rx.finditer(c):
                if any(a < m.end() and m.start() < b for a, b in consumed):
                    continue
                consumed.append((m.start(), m.end()))
                hits.append((m.start(), m.end(), fam, pay, m.group(0)))
        hits.sort()
        last = None
        has_neg = any(k == "neg" for _, _, k in cues)
        carry = self._carry if (not cues and len(c.split()) <= 6) else None
        for st, en, fam, pay, word in hits:
            prev = [(cs, k) for (cs, ce, k) in cues if ce <= st + 1]
            kind = prev[-1][1] if prev else None
            # "just add the ...", "only use some ..." -> the 'only' governs what directly follows it
            if kind == "pos":
                for cs, k in reversed(prev):
                    if k == "neg":
                        break
                    if k == "only" and _ONLY_LEAD.match(c[cs:st]):
                        kind = "only"
                        break
            # list continuation: "only crossfades and whips", "no glitches or flashes"
            if last and _LIST_JOIN.match(c[last[0]:st]):
                kind = last[1]
            # imperative at the start of a clause: "colour grade it warm", "grade it like a movie"
            if kind is None and fam in ("grading", "grading_weak") and st <= 2 and re.match(r"\s+(it|this|the|them)\b", c[en:]):
                kind = "pos"
            # "cuts only", "transitions only": trailing only
            if kind is None and re.match(r"\s+only\b", c[en:]):
                kind = "only"
            if kind is None and fam in ("grading", "grading_verb") and re.search(r"\b(untouched|alone|as is)\b", c):
                kind = "neg"
            if kind is None and not has_neg:
                if carry:                                              # "add whip pans, hearts, stars"
                    kind = carry
                elif BARE_OK.match(word.strip()):                      # "..., light leaks, slow-mo"
                    kind = "pos"
                elif fam.startswith("tr") and re.search(r"\b(transitions?|effects?)\b", c[en:]):
                    kind = "pos"                                       # "heart and light leak transitions"
            if kind is None:
                continue  # a mention, not an instruction
            last = (en, kind)
            on = kind in ("pos", "only")
            if kind == "only" and not source.startswith("Brief"):
                kind = "pos"
                on = True
            if kind == "only":
                self.only_scope.add({"tr": "transitions", "tr+fx": "transitions", "tr+fx+mo": "transitions",
                                     "fx": "effects", "mo": "motion", "grading_weak": "grading",
                                     "grading_verb": "grading", "orig_audio": "audio", "glow": "texture",
                                     "grain": "texture", "vignette": "texture"}.get(fam, fam))
            if on:
                self.positive.add({"tr": "transitions", "tr+fx": "transitions", "tr+fx+mo": "transitions",
                                   "fx": "effects", "glow": "effects", "grading_weak": "grading",
                                   "grading_verb": "grading", "grain": "texture", "vignette": "texture"}.get(fam, fam))
            if "fx" in fam and fam.startswith("tr") and re.search(r"\btransitions?\b", c):
                fam = fam.replace("+fx", "")  # "light leak transitions" = the transition, not a shot overlay
            self._apply(fam, pay, on, kind, word, src)
        self._carry = last[1] if (last and last[1] == "pos") else None

    def _apply(self, fam, pay, on, kind, word, src):
        w = word.strip()
        if fam in ("grading", "grading_verb") or (fam == "grading_weak" and (not on or kind == "only" or
                                                                           re.search(r"grad|lut|filter", w))):
            if fam == "grading_weak" and on and re.fullmatch(r"looks?", w):
                return  # "make it look ..." is not a request to grade
            self.set("grading", on, "Colour grading on" if on else "Don't touch the colours (no grading)", src)
        elif fam == "transitions":
            if on:
                self.set("transitions", "any", "Use transitions", src)
            else:
                self.set("transitions", "none", "No transitions (cuts only)", src)
        elif fam.startswith("tr") or fam in ("fx", "mo"):
            names = set(pay)
            tr_names = names & _ALL_TR
            if fam.startswith("tr") and tr_names:
                if kind == "only":
                    self.only_tr |= tr_names
                self._tr(tr_names, on, src, f"Use {w}", f"No {w}")
                if on and self.v.get("transitions") == "none":
                    self.v.pop("transitions", None)
                    self.rules.pop("transitions", None)
            if "fx" in fam:
                self._fx(names, on)
                if fam == "fx":
                    self.rules["fx:" + w] = {"key": "fx:" + w, "value": on, "label": ("Use " if on else "No ") + w,
                                             "source": src}
            if "mo" in fam:
                self._mo(names, on)
                if fam == "mo":
                    self.rules["mo:" + w] = {"key": "mo:" + w, "value": on,
                                             "label": ("Camera moves: " + w) if on else f"No {w}", "source": src}
        elif fam == "effects":
            self.set("effects", on, "Use clip effects" if on else "No effects", src)
            if not on:
                self.set("texture", False, "No grain / glow / vignette", src)
        elif fam == "letterbox":
            self.set("letterbox", on, "Letterbox bars" if on else "No letterbox bars", src)
        elif fam == "grain":
            self.set("grain", on, "Film grain" if on else "No grain", src)
        elif fam == "vignette":
            self.set("vignette", on, "Vignette" if on else "No vignette", src)
        elif fam == "glow":
            self.set("glow", on, "Glow" if on else "No glow / bloom", src)
            self._fx({"glow"}, on)
        elif fam == "captions":
            self.set("captions", on, "Captions on" if on else "No captions", src)
        elif fam == "text":
            self.set("text", on, "On-screen text allowed" if on else "No on-screen text", src)
        elif fam == "slowmo":
            self.set("slowmo", on, "Add slow motion" if on else "No slow motion", src)
        elif fam == "speed":
            if not on:
                self.set("speed", False, "No speed changes", src)
        elif fam == "music":
            self.set("music", on, "Music on" if on else "No music", src)
        elif fam == "orig_audio":
            self.set("orig_audio", on, "Keep the original sound" if on else "Mute the original sound", src)
        elif fam == "fades":
            which = "in" if re.search(r"\bin\b", w) else "out" if re.search(r"out|to black", w) else None
            if which:
                self.set("fade_" + which, on, f"Fade {which}" if on else f"No fade {which}", src)
            else:
                self.set("fades", on, "Fade in/out" if on else "No fade in/out", src)

    def finalize(self, options: dict):
        if options.get("grading") is False:
            self.set("grading", False, "Colour grading switched off", "Color grading checkbox")
        if options.get("effects") is False:
            self.set("effects", False, "Effects switched off", "Effects checkbox")
        # "just add transitions" / "nothing else": everything decorative that wasn't asked for is off
        if self.only_scope & {"transitions", "effects", "minimal", "motion", "text", "captions", "grading", "texture"}:
            scope = self.only_scope | self.positive
            src = next((r["source"] for r in self.rules.values() if re.search(r"\b(just|only|nothing)\b", r["source"])),
                       (self.rules.get("only") or {}).get("source") or "your brief")
            names = {"grading": "colour grading", "texture": "grain / glow / vignette", "letterbox": "letterbox bars",
                     "text": "on-screen text", "effects": "clip effects", "speed": "speed changes",
                     "motion": "camera moves"}
            for k in DECOR:
                if k not in scope and k not in self.v:
                    self.set(k, False, f"Only what you asked for: no {names[k]}", src)
        if self.v.get("grading") is False:
            if "texture" not in self.v and "texture" not in self.positive:
                self.v["texture"] = False
            if "letterbox" not in self.v:
                self.v["letterbox"] = False
        return self

    # -- reading --------------------------------------------------------------------------
    def get(self, k, d=None):
        return self.v.get(k, d)

    @property
    def raw_color(self) -> bool:
        return self.v.get("grading") is False

    def locks(self) -> set[str]:
        """Dimensions the learner / assistant editor must not touch."""
        L = set()
        if self.raw_color or self.v.get("look"):
            L.add("look")
        if self.raw_color or self.v.get("texture") is False or "grain" in self.v:
            L.add("grain")
        if "letterbox" in self.v:
            L.add("letterbox")
        if self.v.get("transitions") in ("none", "any") or self.only_tr or self.want_tr:
            L.add("transitions")
        if "slowmo" in self.v or self.v.get("speed") is False:
            L.add("slowmo")
        if self.v.get("keep_order"):
            L.add("order")
        return L

    def to_public(self) -> dict:
        return {"rules": [{k: r[k] for k in ("key", "label", "source")} for r in self.rules.values()],
                "raw_color": self.raw_color,
                "banned_transitions": sorted(self.ban_tr), "wanted_transitions": sorted(self.want_tr),
                "only_transitions": sorted(self.only_tr), "banned_effects": sorted(self.ban_fx),
                "wanted_effects": sorted(self.want_fx), "banned_motion": sorted(self.ban_mo)}

    def summary_lines(self) -> list[str]:
        return [r["label"] for r in self.rules.values()]

    # -- prompt ---------------------------------------------------------------------------
    def prompt_block(self) -> str:
        if not self.rules and not self.ban_tr:
            return ""
        L = ["# HARD RULES - the user's explicit instructions. Obey them exactly; they override every other guideline."]
        v = self.v
        if self.raw_color:
            L.append("- NO colour grading: look \"none\", look_intensity 0, grain 0, vignette 0, no letterbox. "
                     "The footage keeps its original colours.")
        elif v.get("look"):
            L.append(f"- Use the look \"{v['look']}\".")
        if v.get("transitions") == "none":
            L.append("- Every transition_in must be \"cut\".")
        elif self.only_tr:
            L.append("- Only these transitions (or cut): " + ", ".join(sorted(self.only_tr)) + ".")
        elif self.want_tr:
            L.append("- Use these transitions: " + ", ".join(sorted(self.want_tr)[:12]) + ".")
        elif v.get("transitions") == "any":
            L.append("- Use stylised transitions between shots (not just cuts).")
        if self.ban_tr and v.get("transitions") != "none":
            L.append("- Never use: " + ", ".join(sorted(self.ban_tr)[:20]) + ".")
        if v.get("effects") is False:
            L.append("- Every effect must be \"none\".")
        elif self.ban_fx:
            L.append("- Never use effects: " + ", ".join(sorted(self.ban_fx)) + ".")
        if v.get("text") is False:
            L.append("- No on-screen text: text must be empty for every segment.")
        if v.get("title"):
            L.append(f"- The first shot gets the title \"{v['title']}\".")
        if v.get("slowmo") is False or v.get("speed") is False:
            L.append("- speed must be 1.0 for every segment.")
        if v.get("motion") is False:
            L.append("- motion must be \"none\" for every segment.")
        elif self.ban_mo:
            L.append("- Never use camera moves: " + ", ".join(sorted(self.ban_mo)) + ".")
        if v.get("keep_order"):
            L.append("- Keep the clips in their original order (by clip id, then time).")
        if v.get("music") is False:
            L.append("- No music: keep_audio true for every segment.")
        if v.get("letterbox") is False and not self.raw_color:
            L.append("- letterbox false.")
        if v.get("letterbox") is True:
            L.append("- letterbox true.")
        return "\n".join(L) if len(L) > 1 else ""


def collect(description: str, revisions: list[str] | None = None, options: dict | None = None,
            looks: list[str] | None = None) -> Directives:
    """Rules from the brief, then each revision in order (a later instruction overrides an earlier one)."""
    d = Directives()
    d.looks = [lk for lk in (looks or []) if lk not in ("none", "clean")]
    d.add_text(description or "", "Brief")
    for i, r in enumerate(revisions or [], 1):
        if r:
            d.add_text(r, f"Revision {i}")
    return d.finalize(options or {})


# ----------------------------------------------------------------------------------------------
# Enforcement
# ----------------------------------------------------------------------------------------------
def _default_for(names: set[str]) -> str:
    order = ["crossfade", "pull_in", "zoom_punch", "whip_left", "light_leak", "push_left", "focus_blur", "glitch",
             "iris", "ripple", "story_swipe", "zoom_flash"]
    for n in order:
        if n in names:
            return n
    return sorted(names)[0]


def enforce(plan: dict, d: Directives, analysis: dict, options: dict) -> list[dict]:
    """Make the plan obey the rules. Returns the changes made (also appended to plan["craft_changes"])."""
    from .grading import look_texture  # local: avoids an import cycle at module load
    changes: list[dict] = []
    segs = plan.get("segments") or []
    clips = {c["id"]: c for c in analysis["clips"]}
    shots = {s["id"]: s for s in analysis.get("all_shots", [])}
    shots.update({s["id"]: s for s in analysis.get("shots", [])})
    v = d.v

    def log(what, why):
        changes.append({"rule": "your instructions", "change": what, "why": why})

    def why(key):
        r = d.rules.get(key)
        return r["source"] if r else "your instructions"

    # ---- colour
    if d.raw_color:
        before = (plan.get("look"), plan.get("look_intensity"), plan.get("grain"), plan.get("vignette"),
                  plan.get("halation"), plan.get("bloom"))
        plan.update({"look": "none", "look_intensity": 0.0, "grading": False})
        for k in ("grain", "vignette", "halation", "bloom"):
            if v.get(k if k in ("grain", "vignette") else "glow") is not True:
                plan[k] = 0.0
        if before[0] not in (None, "none") or any(x for x in before[2:]):
            log(f"colour grading removed (look {before[0]} → none, grain/vignette/glow off, no shot matching)",
                why("grading"))
    else:
        plan["grading"] = True
        if v.get("look") and plan.get("look") != v["look"]:
            old = plan.get("look")
            plan["look"] = v["look"]
            tex = look_texture(v["look"])
            plan["halation"], plan["bloom"] = tex["halation"], tex["bloom"]
            log(f"look {old} → {v['look']}", why("look"))
        if v.get("texture") is False:
            if any(plan.get(k) for k in ("grain", "vignette", "halation", "bloom")):
                log("grain / vignette / glow off", why("texture"))
            for k in ("grain", "vignette", "halation", "bloom"):
                plan[k] = 0.0
    if v.get("grain") is False and plan.get("grain"):
        plan["grain"] = 0.0
        log("film grain off", why("grain"))
    if v.get("grain") is True and not plan.get("grain"):
        plan["grain"] = 0.3
        log("film grain on", why("grain"))
    if v.get("vignette") is False and plan.get("vignette"):
        plan["vignette"] = 0.0
        log("vignette off", why("vignette"))
    if v.get("glow") is False and (plan.get("halation") or plan.get("bloom")):
        plan["halation"] = plan["bloom"] = 0.0
        log("glow / bloom off", why("glow"))
    if "letterbox" in v and bool(plan.get("letterbox")) != bool(v["letterbox"]):
        plan["letterbox"] = bool(v["letterbox"])
        log(("added" if v["letterbox"] else "removed") + " letterbox bars", why("letterbox"))
    if v.get("fades") is False and (plan.get("fade_in", True) or plan.get("fade_out", True)):
        plan["fade_in"] = plan["fade_out"] = False
        log("no fade in / fade out", why("fades"))
    for k in ("fade_in", "fade_out"):
        if k in v and bool(plan.get(k, True)) != bool(v[k]):
            plan[k] = bool(v[k])
            log(("added " if v[k] else "no ") + k.replace("_", " "), why(k))

    # ---- order
    if v.get("keep_order") and len(segs) > 1:
        idx = {c["id"]: i for i, c in enumerate(analysis["clips"])}
        new = sorted(segs, key=lambda s: (idx.get(s["clip_id"], 99), s["start"]))
        if [id(s) for s in new] != [id(s) for s in segs]:
            first_tr = [s["transition_in"] for s in segs]  # keep the transition pattern by position
            for s, tr in zip(new, first_tr):
                s["transition_in"] = tr
            segs[:] = new
            log("put the shots back in their original order", why("keep_order"))

    # ---- transitions
    def set_tr(s, name):
        s["transition_in"] = name
        s["transition_duration"] = REGISTRY[name].duration if name in REGISTRY else 0.0

    if v.get("transitions") == "none":
        n = sum(1 for s in segs[1:] if s["transition_in"] != "cut")
        for s in segs[1:]:
            set_tr(s, "cut")
        if n:
            log(f"{n} transition(s) → cut", why("transitions"))
    else:
        allowed = d.only_tr or None
        pool = sorted(d.want_tr & (allowed or _ALL_TR)) or sorted(allowed or [])
        swapped = 0
        for i, s in enumerate(segs[1:], 1):
            t = s["transition_in"]
            if t == "cut":
                continue
            if t in d.ban_tr or (allowed and t not in allowed):
                set_tr(s, pool[i % len(pool)] if pool else "cut")
                swapped += 1
        if swapped:
            log(f"replaced {swapped} transition(s) you ruled out" + (f" with {', '.join(pool[:4])}" if pool else
                                                                     " with cuts"),
                "; ".join(r["source"] for r in d.rules.values() if r["key"].startswith("tr:"))[:200] or
                "your instructions")
        # every kind of transition you named appears at least once ("whip pans, hearts and light leaks")
        added = []
        used = set()
        if len(segs) > 1:
            for label, group in d.want_groups:
                group = [g for g in group if g not in d.ban_tr and (not allowed or g in allowed)]
                if not group or any(s["transition_in"] in group for s in segs[1:]):
                    continue
                wanted_all = d.want_tr
                slots = [k for k in range(1, len(segs)) if k not in used and segs[k]["transition_in"] == "cut"] or \
                        [k for k in range(1, len(segs)) if k not in used and segs[k]["transition_in"] not in wanted_all]
                if not slots:
                    continue
                k = slots[len(slots) // 2] if len(added) % 2 else slots[0]
                set_tr(segs[k], group[0])
                used.add(k)
                added.append(group[0])
        if added:
            log(f"added the transition(s) you asked for: {', '.join(added)}",
                next((r["source"] for r in d.rules.values() if r["key"].startswith("tr:") and r["value"]),
                     why("transitions")))
        # "add transitions" (no kind named): make sure the reel isn't all cuts
        if v.get("transitions") == "any" and not d.want_groups and len(segs) > 2:
            styl = [s for s in segs[1:] if s["transition_in"] != "cut"]
            need = max(1, round(0.3 * (len(segs) - 1)))
            if len(styl) < need:
                defaults = ["pull_in", "whip_left", "light_leak", "crossfade", "zoom_punch", "push_left"]
                cuts = [s for s in segs[1:] if s["transition_in"] == "cut"]
                step = max(1, len(cuts) // max(1, need - len(styl)))
                k = 0
                for s in cuts[::step][: need - len(styl)]:
                    set_tr(s, defaults[k % len(defaults)])
                    k += 1
                if k:
                    log(f"added {k} transition(s): {', '.join(defaults[:k])}", why("transitions"))
    if segs:
        set_tr(segs[0], "cut")

    # ---- clip effects
    n_fx = 0
    for s in segs:
        e = s.get("effect") or "none"
        if e == "none":
            continue
        if v.get("effects") is False or e in d.ban_fx or (d.raw_color and e in COLOR_FX and e not in d.want_fx):
            s["effect"] = "none"
            n_fx += 1
    if n_fx:
        log(f"removed {n_fx} clip effect(s)", why("effects") if v.get("effects") is False else
            ("raw colours: effects that recolour a shot are off" if d.raw_color else "your instructions"))
    if d.want_fx - d.ban_fx and v.get("effects") is not False and not any((s.get("effect") or "none") in d.want_fx
                                                                         for s in segs):
        fxl = sorted(d.want_fx - d.ban_fx)
        targets = [s for s in segs if not s.get("keep_audio")] or segs
        for j, s in enumerate(targets[::max(1, len(targets) // 3)][:3]):
            s["effect"] = fxl[j % len(fxl)]
        log(f"added effect(s): {', '.join(fxl)}", "you asked for them")

    # ---- camera moves
    n_mo = 0
    for s in segs:
        m = s.get("motion", "none")
        if m != "none" and (v.get("motion") is False or m in d.ban_mo):
            s["motion"] = "none"
            n_mo += 1
    if n_mo:
        log(f"removed {n_mo} camera move(s)", why("motion") if v.get("motion") is False else "your instructions")

    # ---- speed
    if v.get("slowmo") is False or v.get("speed") is False:
        n_sp = 0
        for s in segs:
            if (v.get("speed") is False and abs(s["speed"] - 1) > 1e-3) or (v.get("slowmo") is False and s["speed"] < 0.95):
                out = (s["end"] - s["start"]) / s["speed"]
                s["speed"] = 1.0
                cd = clips[s["clip_id"]]["duration"]
                s["end"] = round(min(cd, s["start"] + out), 3)
                if s["end"] - s["start"] < 0.3:
                    s["start"] = round(max(0.0, s["end"] - out), 3)
                n_sp += 1
        if n_sp:
            log(f"{n_sp} shot(s) back to normal speed", why("speed") if v.get("speed") is False else why("slowmo"))
    elif v.get("slowmo") is True and not any(s["speed"] < 0.95 for s in segs):
        cands = sorted([s for s in segs[1:] if not s.get("keep_audio")],
                       key=lambda s: -(shots.get(s.get("shot_id")) or {}).get("motion", 0))[:2]
        for s in cands:
            out = (s["end"] - s["start"]) / s["speed"]
            s["speed"] = 0.5
            s["end"] = round(s["start"] + out * 0.5, 3)
        if cands:
            log(f"slow motion on {len(cands)} shot(s)", why("slowmo"))

    # ---- text
    if v.get("text") is False:
        n = sum(1 for s in segs if s.get("text"))
        for s in segs:
            s["text"] = None
        if n:
            log(f"removed {n} text overlay(s)", why("text"))
    if v.get("title") and segs:
        cur = (segs[0].get("text") or {}).get("content")
        if cur != v["title"]:
            for s in segs[1:]:
                if s.get("text") and s["text"].get("style") == "cinematic_title":
                    s["text"] = None
            segs[0]["text"] = {"content": v["title"][:80], "style": "cinematic_title", "position": "center"}
            log(f"title “{v['title']}” on the first shot", why("title"))

    # ---- audio
    if v.get("music") is False:
        if plan.get("music_volume", 0.9) > 0:
            log("music muted", why("music"))
        plan["music_volume"] = 0.0
        for s in segs:
            s["keep_audio"] = True
    if v.get("orig_audio") is True:
        n = sum(1 for s in segs if not s.get("keep_audio"))
        for s in segs:
            s["keep_audio"] = True
        if n:
            log(f"kept the original sound on {n} more shot(s)", why("orig_audio"))
    elif v.get("orig_audio") is False and v.get("music") is not False:
        for s in segs:
            s["keep_audio"] = False
        log("original sound muted", why("orig_audio"))

    # ---- use every clip (report, can't invent footage)
    if v.get("use_all"):
        used = {s["clip_id"] for s in segs}
        missing = [clips[c]["name"] for c in clips if c not in used]
        if missing:
            plan.setdefault("warnings", []).append("You asked to use every clip, but these didn't fit the length: "
                                                   + ", ".join(missing))
    plan["segments"] = segs
    plan["directives"] = {**d.to_public(), "enforced": changes}
    plan.setdefault("craft_changes", [])
    plan["craft_changes"] = plan["craft_changes"] + changes
    return changes


def apply_options(options: dict, d: Directives) -> dict:
    """Options adjusted by the rules (captions, length, clip order, shot matching)."""
    o = dict(options)
    if "captions" in d.v:
        o["captions"] = bool(d.v["captions"])
    if d.v.get("duration"):
        o["target_duration"] = d.v["duration"]
    if d.v.get("use_all") or d.v.get("keep_order"):
        o["auto_pick"] = False if d.v.get("use_all") else o.get("auto_pick", True)
    if d.raw_color:
        o["auto_color"] = False
    o["_locks"] = sorted(d.locks())
    o["_keep_order"] = bool(d.v.get("keep_order"))
    return o
