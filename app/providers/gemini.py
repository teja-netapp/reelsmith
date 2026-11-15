"""Google Gemini editor (e.g. Gemini 4 Hydrogen) via the Gemini API (key from Google AI Studio).

Uses the same playbook and full plan schema as Claude; the contact sheets are sent as images and the
response is constrained to JSON with the plan schema.
"""
from __future__ import annotations

import copy
import json

from ..config import GEMINI_MODEL, gemini_key
from ..playbook import PLAN_SCHEMA, SYSTEM_PROMPT_FULL, build_context_text, refine_suffix, sheet_images_b64
from .base import Provider, ProviderError, http_json, parse_json_loose

BASE = "https://generativelanguage.googleapis.com/v1beta"
# auto-detect order when GEMINI_MODEL is not set (first one your key can use wins)
PREFERRED = ["gemini-4-hydrogen", "gemini-3.5-flash", "gemini-3-flash", "gemini-2.5-flash"]

_resolved_model: str | None = None


def _openapi_schema(schema: dict) -> dict:
    """Older 'responseSchema' accepts an OpenAPI subset - strip keywords it may reject."""
    s = copy.deepcopy(schema)

    def walk(node):
        if isinstance(node, dict):
            for k in ("minimum", "maximum", "minItems", "maxItems", "additionalProperties"):
                node.pop(k, None)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(s)
    return s


class GeminiProvider(Provider):
    name = "gemini"
    polish_level = "light"

    @property
    def label(self):
        return f"Gemini ({_resolved_model or GEMINI_MODEL or 'auto'})"

    def status(self):
        if not gemini_key():
            return {"available": False, "detail": "no GEMINI_API_KEY"}
        return {"available": True, "detail": GEMINI_MODEL or _resolved_model or "auto (prefers gemini-4-hydrogen)"}

    def _headers(self):
        return {"x-goog-api-key": gemini_key()}

    def model(self, log=print) -> str:
        global _resolved_model
        if GEMINI_MODEL:
            return GEMINI_MODEL
        if _resolved_model:
            return _resolved_model
        try:
            res = http_json(f"{BASE}/models?pageSize=1000", headers=self._headers(), timeout=20, retries=1)
            names = [m["name"].split("/", 1)[-1] for m in res.get("models", [])
                     if "generateContent" in m.get("supportedGenerationMethods", [])]
            pick = next((p for p in PREFERRED if p in names), None)
            if pick is None:
                pick = next((n for n in names if "hydrogen" in n), None) or \
                       next((n for n in sorted(names, reverse=True) if "flash" in n and "lite" not in n), None)
            _resolved_model = pick or PREFERRED[0]
        except Exception as e:
            log(f"Could not list Gemini models ({e}); trying {PREFERRED[0]}")
            _resolved_model = PREFERRED[0]
        return _resolved_model

    def _generate(self, text: str, analysis: dict, log) -> dict:
        model = self.model(log)
        parts = [{"inlineData": {"mimeType": "image/jpeg", "data": d}} for d in sheet_images_b64(analysis)]
        parts.append({"text": text})
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT_FULL}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": PLAN_SCHEMA,
                                 "maxOutputTokens": 24000},
        }
        url = f"{BASE}/models/{model}:generateContent"
        attempts = [
            body,
            # older API versions: OpenAPI-style responseSchema
            {**body, "generationConfig": {"responseMimeType": "application/json",
                                          "responseSchema": _openapi_schema(PLAN_SCHEMA), "maxOutputTokens": 24000}},
            # last resort: schema in the prompt, JSON mime only
            {**body, "contents": [{"role": "user", "parts": parts[:-1] + [{"text": text + "\n\nReturn JSON matching "
                                                                                   "this schema:\n" + json.dumps(PLAN_SCHEMA)}]}],
             "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": 24000}},
        ]
        last = None
        for i, b in enumerate(attempts):
            try:
                res = http_json(url, b, headers=self._headers(), timeout=300)
                break
            except ProviderError as e:
                last = e
                if getattr(e, "status", None) == 400 and i < len(attempts) - 1:
                    log(f"Gemini rejected request format #{i + 1}; retrying with a simpler one")
                    continue
                raise
        else:  # pragma: no cover
            raise last
        cands = res.get("candidates") or []
        if not cands:
            fb = res.get("promptFeedback", {})
            raise ProviderError(f"Gemini returned no candidates ({fb.get('blockReason', 'unknown reason')})")
        c = cands[0]
        out = "".join(p.get("text", "") for p in c.get("content", {}).get("parts", []) if not p.get("thought"))
        if not out.strip():
            raise ProviderError(f"Gemini returned an empty answer (finishReason={c.get('finishReason')})")
        usage = res.get("usageMetadata", {})
        if usage:
            log(f"Gemini {model}: {usage.get('promptTokenCount', '?')} in / "
                f"{usage.get('candidatesTokenCount', '?')} out tokens")
        return parse_json_loose(out)

    def plan(self, analysis, music, options, description, log, progress):
        progress(0.2, "Gemini is watching the footage and planning the edit")
        return self._generate(build_context_text(analysis, music, options, description), analysis, log)

    def refine(self, prev_plan, feedback, analysis, music, options, description, log, progress):
        progress(0.2, "Gemini is revising the edit")
        text = build_context_text(analysis, music, options, description) + refine_suffix(prev_plan, feedback)
        return self._generate(text, analysis, log)
