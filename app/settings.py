"""Settings panel backend: show (masked) keys, verify them against the real APIs, save to .env.

Keys are verified *before* they are saved: a key the provider rejects is never written. If the
provider can't be reached (offline), the key is saved with a warning so you can work offline.
Full keys never leave the server - the UI only ever sees a masked form like "AIza…Q2xM".
"""
from __future__ import annotations

import json
import re
import time

from . import config as C
from .providers import CHOICES
from .providers import gemini as G
from .providers.base import VERIFIED, ProviderError, http_json, key_fp

RECOMMENDED_OLLAMA = ["qwen3-vl:8b-instruct", "qwen3-vl:4b-instruct", "gemma4:12b"]
_KEY_OK = re.compile(r"^[A-Za-z0-9_\-.:+/=]{12,400}$")
_MODEL_OK = re.compile(r"^[A-Za-z0-9._:/\-]{1,120}$")
_HOST_OK = re.compile(r"^https?://[A-Za-z0-9.\-\[\]:]+(:\d{1,5})?/?$")


class SettingsError(ValueError):
    pass


def _api_message(e: Exception) -> str:
    """Pull the human message out of an API error body."""
    s = str(e)
    i = s.find("{")
    if i >= 0:
        try:
            body = json.loads(s[i:])
            err = body.get("error", body)
            if isinstance(err, dict):
                return str(err.get("message") or err.get("type") or s)[:300]
        except Exception:
            pass
    return s[:300]


# ----------------------------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------------------------
def _gemini_models_sorted(names):
    keep = [n for n in names if n.startswith(("gemini", "gemma")) and not any(
        x in n for x in ("embedding", "tts", "image", "audio", "live", "robotics", "aqa"))]
    pref = {p: i for i, p in enumerate(G.PREFERRED)}
    return sorted(set(keep), key=lambda n: (pref.get(n, 99), n))


def verify_gemini(key: str) -> dict:
    if not key:
        return {"ok": False, "auth": None, "detail": "No key entered"}
    try:
        res = http_json(f"{G.BASE}/models?pageSize=1000", headers={"x-goog-api-key": key}, timeout=15, retries=0)
    except ProviderError as e:
        code = getattr(e, "status", None)
        if code in (400, 401, 403):
            return {"ok": False, "auth": False, "detail": f"Google rejected this key: {_api_message(e)}"}
        if code == 429:
            return {"ok": True, "auth": True, "models": [], "detail": "Key accepted (rate-limited right now - try again soon)"}
        return {"ok": False, "auth": None, "detail": f"Couldn't reach Google to check the key: {_api_message(e)}"}
    names = [m["name"].split("/", 1)[-1] for m in res.get("models", [])
             if "generateContent" in m.get("supportedGenerationMethods", [])]
    models = _gemini_models_sorted(names)
    pick = C.GEMINI_MODEL or next((p for p in G.PREFERRED if p in names), None) or (models[0] if models else None)
    hyd = "gemini-4-hydrogen" in names
    return {"ok": True, "auth": True, "models": models, "will_use": pick, "hydrogen": hyd,
            "detail": f"Key works · {len(models)} Gemini models · will use {pick or 'auto'}"
                      + ("" if hyd else " (Gemini 4 Hydrogen isn't available to this key)")}


def verify_gemini_model(key: str, model: str) -> dict:
    """Check that a model name exists AND that this key's tier can actually call it (one tiny request)."""
    model = G.normalize_model(model)
    if not model:
        return {"ok": True, "detail": "Auto: the app picks the best model your key can use"}
    if not _MODEL_OK.match(model):
        return {"ok": False, "found": False, "detail": f"'{model}' isn't a valid model name"}
    if not key:
        return {"ok": None, "detail": "Save your API key first, then the model can be tested"}
    h = {"x-goog-api-key": key}
    try:
        info = http_json(f"{G.BASE}/models/{model}", headers=h, timeout=15, retries=0)
    except ProviderError as e:
        code = getattr(e, "status", None)
        if code == 404:
            return {"ok": False, "found": False, "model": model,
                    "detail": f"'{model}' wasn't found for this key. Check the exact name (it's case-sensitive), "
                              "e.g. gemini-3.1-flash-lite"}
        if code in (400, 401, 403):
            return {"ok": False, "auth": False, "detail": f"Google rejected the key: {_api_message(e)}"}
        return {"ok": None, "model": model, "detail": f"Couldn't reach Google to check the model: {_api_message(e)}"}
    methods = info.get("supportedGenerationMethods") or []
    if methods and "generateContent" not in methods:
        return {"ok": False, "found": True, "model": model,
                "detail": f"'{model}' can't write text/JSON (it supports: {', '.join(methods)})"}
    url = f"{G.BASE}/models/{model}:generateContent"
    ask = {"contents": [{"role": "user", "parts": [{"text": 'Reply with exactly this JSON: {"ok": true}'}]}]}
    json_mode = True
    t0 = time.time()
    try:
        try:
            http_json(url, {**ask, "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": 64}},
                      headers=h, timeout=60, retries=0)
        except ProviderError as e:
            if getattr(e, "status", None) != 400:
                raise
            json_mode = False  # e.g. Gemma models: no JSON mode -> the app falls back to plain text
            http_json(url, {**ask, "generationConfig": {"maxOutputTokens": 64}}, headers=h, timeout=60, retries=0)
    except ProviderError as e:
        code, raw = getattr(e, "status", None), str(e)
        if code == 429:
            if re.search(r"limit:\s*0\b", raw):
                return {"ok": False, "found": True, "quota": False, "model": model,
                        "detail": f"'{model}' has no free-tier quota on this key. Pick a Flash or Flash-Lite "
                                  "model, or enable billing in Google AI Studio"}
            return {"ok": True, "found": True, "model": model,
                    "detail": f"'{model}' exists and your key can use it (rate-limited right now: the free tier allows "
                              "a limited number of requests per minute and per day)"}
        if code in (401, 403):
            return {"ok": False, "auth": False, "detail": f"Google rejected the key: {_api_message(e)}"}
        if code == 400:
            return {"ok": False, "found": True, "model": model, "detail": f"'{model}' rejected a test request: {_api_message(e)}"}
        return {"ok": None, "model": model, "detail": f"Couldn't reach Google to test the model: {_api_message(e)}"}
    return {"ok": True, "found": True, "model": model, "json_mode": json_mode,
            "display_name": info.get("displayName"), "output_limit": info.get("outputTokenLimit"),
            "detail": f"'{model}' works with your key · replied in {time.time() - t0:.1f} s · "
                      + ("JSON output supported" if json_mode else "no JSON mode (the app will use its plain-text fallback)")}


def _anthropic_base() -> str:
    return (C._env("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")


def verify_claude(key: str) -> dict:
    if not key:
        return {"ok": False, "auth": None, "detail": "No key entered"}
    headers = {"anthropic-version": "2023-06-01"}
    if key.startswith("sk-ant-") or not C._env("ANTHROPIC_AUTH_TOKEN"):
        headers["x-api-key"] = key
    else:
        headers["Authorization"] = f"Bearer {key}"
    try:
        res = http_json(f"{_anthropic_base()}/v1/models?limit=100", headers=headers, timeout=15, retries=0)
    except ProviderError as e:
        code = getattr(e, "status", None)
        if code in (401, 403):
            return {"ok": False, "auth": False, "detail": f"Anthropic rejected this key: {_api_message(e)}"}
        return {"ok": False, "auth": None, "detail": f"Couldn't reach Anthropic to check the key: {_api_message(e)}"}
    models = [m.get("id") for m in res.get("data", []) if m.get("id")]
    have = C.CLAUDE_MODEL in models if models else True
    return {"ok": True, "auth": True, "models": models, "will_use": C.CLAUDE_MODEL,
            "detail": f"Key works · {len(models)} models · will use {C.CLAUDE_MODEL}"
                      + ("" if have else f" (warning: {C.CLAUDE_MODEL} isn't in your model list)")}


def _norm_host(h: str) -> str:
    h = (h or "").strip().rstrip("/")
    if h and not h.startswith("http"):
        h = "http://" + h
    return h.replace("0.0.0.0", "127.0.0.1")


def verify_ollama(host: str | None = None, model: str | None = None) -> dict:
    host = _norm_host(host or C.OLLAMA_HOST)
    model = model or C.OLLAMA_MODEL
    try:
        ver = http_json(f"{host}/api/version", timeout=3, retries=0).get("version")
        tags = http_json(f"{host}/api/tags", timeout=5, retries=0)
    except ProviderError as e:
        return {"ok": False, "auth": None, "host": host,
                "detail": f"Ollama isn't reachable at {host} ({_api_message(e)}). Open the Ollama app or run: "
                          "brew services start ollama"}
    installed = sorted(m.get("name", "") for m in tags.get("models", []))
    has = model in installed or f"{model}:latest" in installed
    return {"ok": has, "auth": True, "host": host, "version": ver, "installed": installed, "model": model,
            "detail": (f"Connected to Ollama {ver} · {model} is installed" if has else
                       f"Connected to Ollama {ver}, but {model} isn't downloaded yet. Run: ollama pull {model}")}


def verify(provider: str, key: str | None = None, host: str | None = None, model: str | None = None,
           remember: bool = True) -> dict:
    if provider == "gemini":
        k = key if key else C.gemini_key()
        if model is not None:  # "Test model": check this exact model with this key
            r = verify_gemini_model(k, model)
            return r
        r = verify_gemini(k)
    elif provider == "claude":
        k = key if key else (C._env("ANTHROPIC_API_KEY") or C._env("ANTHROPIC_AUTH_TOKEN"))
        r = verify_claude(k)
    elif provider == "ollama":
        k = ""
        r = verify_ollama(host, model)
    else:
        raise SettingsError(f"unknown provider {provider}")
    r["key_fp"] = key_fp(k)
    if remember:
        VERIFIED[provider] = r
        if provider == "gemini" and r.get("ok") and r.get("will_use") and not C.GEMINI_MODEL:
            G._resolved_model = r["will_use"]
    return {kk: vv for kk, vv in r.items() if kk != "key_fp"}


# ----------------------------------------------------------------------------------------------
# Read / save
# ----------------------------------------------------------------------------------------------
def _public_verify(name, secret=""):
    v = VERIFIED.get(name)
    if not v:
        return None
    if name != "ollama" and v.get("key_fp") != key_fp(secret):
        return None  # checked a different key than the one saved now
    return {k: v[k] for k in v if k != "key_fp"}


def get_settings() -> dict:
    gk = C.gemini_key()
    ck = C._env("ANTHROPIC_API_KEY") or C._env("ANTHROPIC_AUTH_TOKEN")
    return {
        "env_file": C.ENV_FILE.name,
        "default_editor": C.LLM_PROVIDER if C.LLM_PROVIDER in CHOICES else "auto",
        "gemini": {"configured": bool(gk), "masked": C.mask(gk), "model": C.GEMINI_MODEL,
                   "from_shell": bool({"GEMINI_API_KEY", "GOOGLE_API_KEY"} & C.SHELL_VARS),
                   "verified": _public_verify("gemini", gk), "get_key_url": "https://aistudio.google.com/apikey"},
        "claude": {"configured": bool(ck), "masked": C.mask(ck), "model": C.CLAUDE_MODEL,
                   "from_shell": bool({"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"} & C.SHELL_VARS),
                   "verified": _public_verify("claude", ck), "get_key_url": "https://console.anthropic.com/settings/keys"},
        "ollama": {"host": C.OLLAMA_HOST, "model": C.OLLAMA_MODEL, "plan_model": C.OLLAMA_PLAN_MODEL,
                   "plan_model_set": bool(C._env("OLLAMA_PLAN_MODEL")), "plan_depth": C.OLLAMA_PLAN_DEPTH,
                   "verified": _public_verify("ollama"), "recommended": RECOMMENDED_OLLAMA},
    }


def _clean_key(v) -> str:
    k = str(v or "").strip().strip('"').strip("'")
    if not _KEY_OK.match(k):
        raise SettingsError("That doesn't look like an API key (no spaces or line breaks, 12+ characters).")
    return k


def save_settings(body: dict) -> dict:
    """Apply a partial settings update. Returns {"settings", "results", "warnings"}."""
    updates: dict[str, str] = {}
    results: dict[str, dict] = {}
    warnings: list[str] = []

    for name, key_var, other_var, model_var, verify_fn in (
            ("gemini", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL", verify_gemini),
            ("claude", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_MODEL", verify_claude)):
        part = body.get(name)
        if not isinstance(part, dict):
            continue
        if part.get("remove"):
            updates[key_var] = ""
            if C._env(other_var):
                updates[other_var] = ""
            VERIFIED.pop(name, None)
            results[name] = {"ok": True, "detail": "Key removed"}
        elif part.get("key"):
            k = _clean_key(part["key"])
            r = verify_fn(k)
            r["key_fp"] = key_fp(k)
            VERIFIED[name] = r
            if r.get("auth") is False:
                raise SettingsError(r["detail"] + " - the key was NOT saved.")
            if r.get("auth") is None:
                warnings.append(r["detail"] + " - saved anyway; it will be checked on first use.")
            updates[key_var] = k
            if C._env(other_var) and name == "gemini":
                updates[other_var] = ""  # avoid an old GOOGLE_API_KEY shadowing confusion
            results[name] = {k2: v for k2, v in r.items() if k2 != "key_fp"}
        if "model" in part:
            m = str(part.get("model") or "").strip()
            if name == "gemini":
                m = G.normalize_model(m)   # accept "models/gemini-…" as copied from the API docs
            if m and not _MODEL_OK.match(m):
                raise SettingsError(f"Invalid model name: {m}")
            if name == "claude" and not m:
                m = "claude-sonnet-5-5"
            if name == "gemini" and m:
                # your own model choice: make sure it exists and this key's tier can call it
                key_for_test = updates.get(key_var) or C.gemini_key()
                mr = verify_gemini_model(key_for_test, m)
                results["gemini_model"] = mr
                if mr.get("found") is False:
                    raise SettingsError(mr["detail"] + " - the model was NOT saved.")
                if mr.get("ok") is False and mr.get("auth") is False:
                    raise SettingsError(mr["detail"])
                if mr.get("ok") is not True:
                    warnings.append(mr["detail"] + " - saved anyway.")
            updates[model_var] = m
        if name == "gemini" and (part.get("key") or "model" in part or part.get("remove")):
            G._resolved_model = None  # re-detect with the new key / preference

    o = body.get("ollama")
    if isinstance(o, dict):
        if "host" in o:
            h = _norm_host(str(o["host"]))
            if not _HOST_OK.match(h):
                raise SettingsError("Ollama host should look like http://127.0.0.1:11434")
            updates["OLLAMA_HOST"] = h
        if "plan_depth" in o:
            d = str(o["plan_depth"]).lower()
            if d not in ("best", "fast"):
                raise SettingsError("planning depth must be 'best' or 'fast'")
            updates["OLLAMA_PLAN_DEPTH"] = d
        for field, var in (("model", "OLLAMA_MODEL"), ("plan_model", "OLLAMA_PLAN_MODEL")):
            if field in o:
                m = str(o.get(field) or "").strip()
                if m and not _MODEL_OK.match(m):
                    raise SettingsError(f"Invalid model name: {m}")
                updates[var] = m

    if "default_editor" in body:
        d = str(body["default_editor"]).lower()
        if d not in CHOICES:
            raise SettingsError(f"default editor must be one of {', '.join(CHOICES)}")
        updates["LLM_PROVIDER"] = d

    if updates:
        C.update_env_file(updates)
    g = VERIFIED.get("gemini") or {}
    if isinstance(body.get("gemini"), dict) and g.get("ok") and g.get("key_fp") == key_fp(C.gemini_key()):
        G._resolved_model = C.GEMINI_MODEL or g.get("will_use")  # show the real model right away
    if isinstance(o, dict):
        results["ollama"] = verify("ollama")
    shadow = [k for k in updates if k in C.SHELL_VARS and updates[k]]
    if shadow:
        warnings.append(f"{', '.join(shadow)} is also set in your shell environment; the shell value will win the "
                        "next time you start the app. Remove it from ~/.zshrc to use the saved one.")
    return {"settings": get_settings(), "results": results, "warnings": warnings}
