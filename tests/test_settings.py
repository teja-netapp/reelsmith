"""Settings panel: verify-then-save API keys, masking, .env handling, localhost protection."""
import os
import stat

import pytest
from fastapi.testclient import TestClient

GOOD_G = "AIzaSyTEST-good-key-1234567890abcd"
BAD_G = "AIzaSyTEST-bad-key-0000000000zzzz"
GOOD_C = "sk-ant-api03-TESTgoodkey-1234567890"


@pytest.fixture
def client(monkeypatch):
    from app import config as C
    from app import settings as S
    from app.main import app
    from app.providers.base import ProviderError

    def fake_http(url, payload=None, headers=None, **kw):
        if "generativelanguage" in url:
            if headers.get("x-goog-api-key") == GOOD_G:
                return {"models": [
                    {"name": "models/gemini-3.5-flash", "supportedGenerationMethods": ["generateContent"]},
                    {"name": "models/gemini-4-hydrogen", "supportedGenerationMethods": ["generateContent"]},
                    {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]}]}
            e = ProviderError('HTTP 400: {"error": {"code": 400, "message": "API key not valid. Please pass a valid '
                              'API key.", "status": "INVALID_ARGUMENT"}}')
            e.status = 400
            raise e
        if "anthropic" in url:
            if headers.get("x-api-key") == GOOD_C:
                return {"data": [{"id": "claude-sonnet-5-5"}, {"id": "claude-opus-5-5"}]}
            e = ProviderError('HTTP 401: {"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}')
            e.status = 401
            raise e
        raise ProviderError("cannot reach")

    monkeypatch.setattr(S, "http_json", fake_http)
    env = C.ENV_FILE
    env.write_text("# my notes stay here\nLLM_PROVIDER=auto\nWHISPER_MODEL=small\nGEMINI_API_KEY=\n")
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(k, "")
    C.reload()
    S.VERIFIED.clear()
    yield TestClient(app), C, env
    for k in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_MODEL", "LLM_PROVIDER"):
        os.environ.pop(k, None)
    C.reload()


def test_verify_without_saving(client):
    c, C, env = client
    r = c.post("/api/settings/verify", json={"provider": "gemini", "key": GOOD_G}).json()
    assert r["ok"] and r["hydrogen"] and r["will_use"] == "gemini-4-hydrogen"
    assert "text-embedding-004" not in r["models"]
    assert GOOD_G not in env.read_text()                       # verify never saves


def test_rejected_key_is_not_saved(client):
    c, C, env = client
    r = c.post("/api/settings", json={"gemini": {"key": BAD_G}})
    assert r.status_code == 400
    assert "API key not valid" in r.json()["detail"] and "NOT saved" in r.json()["detail"]
    assert BAD_G not in env.read_text() and not C.gemini_key()


def test_save_valid_key_applies_live_and_masks(client):
    c, C, env = client
    r = c.post("/api/settings", json={"gemini": {"key": GOOD_G, "model": ""}, "default_editor": "gemini"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["results"]["gemini"]["ok"]
    text = env.read_text()
    assert f"GEMINI_API_KEY={GOOD_G}" in text
    assert "# my notes stay here" in text and "WHISPER_MODEL=small" in text   # other settings preserved
    assert text.count("GEMINI_API_KEY=") == 1
    assert stat.S_IMODE(os.stat(env).st_mode) == 0o600                         # only you can read it
    # applied without restart
    assert C.gemini_key() == GOOD_G and C.LLM_PROVIDER == "gemini"
    cfg = c.get("/api/config").json()
    assert cfg["providers"]["gemini"]["available"] is True
    assert cfg["providers"]["gemini"]["detail"] == "gemini-4-hydrogen"   # the real model, not "auto"
    # never echoed back in full
    s = c.get("/api/settings").json()
    assert s["gemini"]["configured"] and s["gemini"]["masked"] == "AIza…abcd"
    assert GOOD_G not in c.get("/api/settings").text
    assert s["gemini"]["verified"]["ok"] is True


def test_claude_key_and_remove(client):
    c, C, env = client
    assert c.post("/api/settings", json={"claude": {"key": "sk-ant-wrong-key-12345678"}}).status_code == 400
    r = c.post("/api/settings", json={"claude": {"key": GOOD_C, "model": "claude-opus-5-5"}}).json()
    assert r["results"]["claude"]["ok"] and C.CLAUDE_MODEL == "claude-opus-5-5"
    assert c.get("/api/config").json()["providers"]["claude"]["available"] is True
    r = c.post("/api/settings", json={"claude": {"remove": True}}).json()
    assert not r["settings"]["claude"]["configured"]
    assert "ANTHROPIC_API_KEY=\n" in env.read_text() or env.read_text().rstrip().endswith("ANTHROPIC_API_KEY=")


def test_offline_key_saved_with_warning(client, monkeypatch):
    c, C, env = client
    from app import settings as S
    from app.providers.base import ProviderError

    def offline(*a, **k):
        raise ProviderError("cannot reach generativelanguage.googleapis.com: nodename nor servname provided")
    monkeypatch.setattr(S, "http_json", offline)
    r = c.post("/api/settings", json={"gemini": {"key": GOOD_G}}).json()
    assert r["warnings"] and "saved anyway" in r["warnings"][0]
    assert C.gemini_key() == GOOD_G


def test_input_validation(client):
    c, C, env = client
    assert c.post("/api/settings", json={"gemini": {"key": "has space in it 123456"}}).status_code == 400
    assert c.post("/api/settings", json={"gemini": {"key": "line\nbreak-123456789"}}).status_code == 400
    assert c.post("/api/settings", json={"ollama": {"host": "not a url"}}).status_code == 400
    assert c.post("/api/settings", json={"default_editor": "chatgpt"}).status_code == 400


def test_cross_site_and_rebinding_blocked(client):
    c, C, env = client
    r = c.post("/api/settings", json={"gemini": {"key": GOOD_G}}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403 and not C.gemini_key()
    r = c.post("/api/jobs", data={"paths": "/etc/hosts"}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    assert c.get("/api/settings", headers={"Host": "attacker.example"}).status_code == 403
    # the app's own page is allowed
    assert c.post("/api/settings", json={"default_editor": "auto"},
                  headers={"Origin": "http://127.0.0.1:8765"}).status_code == 200
