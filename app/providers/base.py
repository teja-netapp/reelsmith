"""Shared helpers for AI editor providers."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request


class ProviderError(RuntimeError):
    pass


def http_json(url: str, payload: dict | None = None, headers: dict | None = None, timeout: float = 300,
              retries: int = 2, method: str | None = None) -> dict:
    """POST (or GET when payload is None) JSON with simple retry on 429/5xx."""
    data = None if payload is None else json.dumps(payload).encode()
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method or ("GET" if data is None else "POST"))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")[:800]
            last = ProviderError(f"HTTP {e.code}: {body}")
            last.status = e.code  # type: ignore[attr-defined]
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(8 * (attempt + 1))
                continue
            raise last
        except urllib.error.URLError as e:
            raise ProviderError(f"cannot reach {url.split('?')[0]}: {e.reason}")
    raise last  # pragma: no cover


def parse_json_loose(text: str) -> dict:
    """Parse model output that should be JSON but may be wrapped in prose or code fences."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        return json.loads(text[i: j + 1])
    raise ProviderError("model did not return JSON")


class Provider:
    """Interface: plan() and refine() return a *raw* plan dict in the full schema."""

    name = "base"
    label = "base"
    polish_level = "light"   # how much the craft pass may correct this provider's plans

    def status(self) -> dict:  # {"available": bool, "detail": str}
        raise NotImplementedError

    def plan(self, analysis, music, options, description, log, progress) -> dict:
        raise NotImplementedError

    def refine(self, prev_plan, feedback, analysis, music, options, description, log, progress) -> dict:
        raise NotImplementedError
