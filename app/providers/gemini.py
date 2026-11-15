"""Google Gemini editor (e.g. Gemini 4 Hydrogen) via the Gemini API (key from Google AI Studio).

Uses the same playbook and full plan schema as Claude; the contact sheets are sent as images and the
response is constrained to JSON with the plan schema.
"""
from __future__ import annotations

import copy
import json

from .. import config as C
from ..playbook import PLAN_SCHEMA, SYSTEM_PROMPT_FULL, build_context_text, refine_suffix, sheet_images_b64
from .base import Provider, ProviderError, http_json, parse_json_loose, rejected

BASE = "https://generativelanguage.googleapis.com/v1beta"
# auto-detect order when GEMINI_MODEL is not set (first one your key can use wins).
# Set any model yourself in Settings (e.g. a free-tier Flash / Flash-Lite / Gemma model).
PREFERRED = ["gemini-4-hydrogen", "gemini-3.5-flash", "gemini-3-flash", "gemini-2.5-flash", "gemini-3.1-flash-lite"]

_resolved_model: str | None = None
_model_info: dict[str, dict] = {}


def normalize_model(name: str | None) -> str:
    n = (name or "").strip()
    return n[len("models/"):] if n.startswith("models/") else n


def model_info(model: str, headers: dict) -> dict:
    """Model metadata (token limits, supported methods). Raises ProviderError(status=404) if unknown."""
    if model in _model_info:
        return _model_info[model]
    try:
        info = http_json(f"{BASE}/models/{model}", headers=headers, timeout=15, retries=0)
    except ProviderError as e:
        if getattr(e, "status", None) == 404:
            err = ProviderError(f"Gemini model '{model}' was not found for this API key. Check the name in Settings.")
            err.status = 404
            raise err
        return {}  # couldn't fetch details; carry on with safe defaults
    _model_info[model] = info
    return info


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
        return f"Gemini ({_resolved_model or C.GEMINI_MODEL or 'auto'})"

    def status(self):
        if not C.gemini_key():
            return {"available": False, "detail": "no GEMINI_API_KEY"}
        if rejected("gemini", C.gemini_key()):
            return {"available": False, "detail": "API key was rejected by Google - update it in Settings"}
        return {"available": True, "detail": C.GEMINI_MODEL or _resolved_model or "auto (prefers gemini-4-hydrogen)"}

    def _headers(self):
        return {"x-goog-api-key": C.gemini_key()}

    def model(self, log=print) -> str:
        global _resolved_model
        if C.GEMINI_MODEL:
            return normalize_model(C.GEMINI_MODEL)
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
        info = model_info(model, self._headers())          # unknown model -> clear error, next editor
        max_out = min(24000, int(info.get("outputTokenLimit") or 8192))
        images = [{"inlineData": {"mimeType": "image/jpeg", "data": d}} for d in sheet_images_b64(analysis)]
        schema_txt = "\n\nReturn JSON matching this schema:\n" + json.dumps(PLAN_SCHEMA)
        system = {"systemInstruction": {"parts": [{"text": SYSTEM_PROMPT_FULL}]}}
        user = lambda t: {"contents": [{"role": "user", "parts": images + [{"text": t}]}]}  # noqa: E731
        # Models differ in what they accept (e.g. Gemma: no system instruction, no JSON mode), so step
        # down from strict JSON-schema output to plain text until one is accepted.
        attempts = [
            ("JSON schema", {**system, **user(text), "generationConfig": {
                "responseMimeType": "application/json", "responseJsonSchema": PLAN_SCHEMA, "maxOutputTokens": max_out}}),
            ("OpenAPI schema", {**system, **user(text), "generationConfig": {
                "responseMimeType": "application/json", "responseSchema": _openapi_schema(PLAN_SCHEMA),
                "maxOutputTokens": max_out}}),
            ("schema in prompt", {**system, **user(text + schema_txt), "generationConfig": {
                "responseMimeType": "application/json", "maxOutputTokens": max_out}}),
            ("plain text", {**user(SYSTEM_PROMPT_FULL + "\n\n" + text + schema_txt +
                                   "\n\nReply with only the JSON object, no other text."),
                            "generationConfig": {"maxOutputTokens": max_out}}),
        ]
        url = f"{BASE}/models/{model}:generateContent"
        last = None
        used_format = None
        for i, (fmt, b) in enumerate(attempts):
            try:
                res = http_json(url, b, headers=self._headers(), timeout=300)
                used_format = fmt
                break
            except ProviderError as e:
                last = e
                if getattr(e, "status", None) == 400 and i < len(attempts) - 1:
                    log(f"{model} rejected the '{fmt}' request format; retrying with a simpler one")
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
        self.last_meta = {
            "model": model,
            "model_version": res.get("modelVersion"),
            "usage": {"input_tokens": usage.get("promptTokenCount"), "output_tokens": usage.get("candidatesTokenCount"),
                      "thinking_tokens": usage.get("thoughtsTokenCount")} if usage else None,
            "images_sent": len(images),
            "output_format": used_format,
            "model_source": "your choice (Settings)" if C.GEMINI_MODEL else "auto-detected",
        }
        return parse_json_loose(out)

    def plan(self, analysis, music, options, description, log, progress):
        progress(0.2, "Gemini is watching the footage and planning the edit")
        return self._generate(build_context_text(analysis, music, options, description), analysis, log)

    def refine(self, prev_plan, feedback, analysis, music, options, description, log, progress):
        progress(0.2, "Gemini is revising the edit")
        text = build_context_text(analysis, music, options, description) + refine_suffix(prev_plan, feedback)
        return self._generate(text, analysis, log)
