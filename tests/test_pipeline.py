"""End-to-end tests on synthetic footage. Run: pytest -q"""
import json
import subprocess
import time
import types

OPTS = {"target_duration": 8, "quality": "preview", "beat_sync": True, "captions": False}


def _duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout
    return float(out.strip())


def _streams(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height", "-of", "json",
                          path], capture_output=True, text=True).stdout
    return json.loads(out)["streams"]


def test_scene_detection(media, tmp_path):
    from app.analyze import analyze_clip
    from app.video_io import probe

    clip, shots = analyze_clip(probe(media["land"]), "c0", None, tmp_path)
    assert clip["num_scenes"] == 3
    assert [round(s["start"]) for s in shots] == [0, 3, 6]


def test_offline_rules_end_to_end(media):
    from app.pipeline import run_sync

    job = run_sync([media["land"], media["track"]], media["music"], 'energetic "GO"', {**OPTS, "planner": "heuristic"})
    assert job.status == "done", job.error
    assert job.plan["planner"].startswith("Built-in rules")
    st = _streams(job.output)
    v = next(s for s in st if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (540, 960)
    assert any(s["codec_type"] == "audio" for s in st)
    assert abs(_duration(job.output) - job.timeline_summary["duration"]) < 0.3


def test_local_ollama_editor(media, fake_ollama):
    from app.pipeline import run_sync

    job = run_sync([media["land"], media["track"]], media["music"], "fast hype reel", {**OPTS, "planner": "ollama"})
    assert job.status == "done", job.error
    p = job.plan
    assert p["planner"].startswith("Local AI")
    assert p["title"] == "Fake Local Edit"
    n_logged = sum(1 for r in fake_ollama if any(m.get("images") for m in r["messages"]))
    assert n_logged == len(job.analysis["shots"])          # every shot was logged by the vision model
    plan_req = [r for r in fake_ollama if not any(m.get("images") for m in r["messages"])][-1]
    assert "Editing principles" in plan_req["messages"][0]["content"]  # same playbook as the cloud editors
    assert plan_req["format"]["properties"]["segments"]["items"]["properties"]["shot"]["enum"]
    # craft pass fixed the sloppy plan
    trs = [s["transition_in"] for s in p["segments"][1:]]
    assert sum(t != "cut" for t in trs) <= max(1, round(0.3 * len(trs))) + 1
    motions = [s["motion"] for s in p["segments"]]
    assert all(a != b or a == "none" for a, b in zip(motions, motions[1:]))
    assert p["segments"][0]["text"]["content"] == "TEST TITLE"
    # the vision log's subject position drives reframing
    assert any(abs(s["focus_x"] - 0.3) < 1e-6 for s in p["segments"])

    # second run on the same footage reuses the cached shot log
    before = len(fake_ollama)
    from app.planner import refine_edit
    refine_edit(p, "make it slower", job.analysis, job.music_info, {**job.options}, job.description)
    logged_again = sum(1 for r in fake_ollama[before:] if any(m.get("images") for m in r["messages"]))
    assert logged_again == 0


def test_gemini_editor(media, monkeypatch):
    from app.analyze import analyze_clip, make_contact_sheets, score_shots
    from app.planner import plan_edit
    from app.providers import gemini
    from app.video_io import probe

    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    clip, shots = analyze_clip(probe(media["land"]), "c0", None, tmp)
    score_shots(shots)
    analysis = {"clips": [clip], "all_shots": shots, "shots": shots,
                "sheets": make_contact_sheets(shots, tmp / "sheets"), "transcripts": {}}
    calls = []

    def fake_http(url, payload=None, headers=None, **kw):
        calls.append((url, payload, headers))
        if url.endswith("/models?pageSize=1000"):
            return {"models": [{"name": "models/gemini-3.5-flash", "supportedGenerationMethods": ["generateContent"]},
                               {"name": "models/gemini-4-hydrogen", "supportedGenerationMethods": ["generateContent"]}]}
        plan = {"reasoning": "r", "title": "Gemini Cut", "look": "warm_film", "segments": [
            {"shot_id": "c0s01", "start": 3.2, "end": 5.0, "speed": 1, "motion": "push_in", "transition_in": "cut"},
            {"shot_id": "c0s00", "start": 0.5, "end": 2.5, "speed": 1, "motion": "drift_left",
             "transition_in": "light_leak"}]}
        return {"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True},
                                                       {"text": json.dumps(plan)}]}}],
                "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 300}}

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(gemini, "http_json", fake_http)
    monkeypatch.setattr(gemini, "_resolved_model", None)
    p = plan_edit(analysis, None, {"target_duration": 4, "planner": "gemini"}, "warm film", log=lambda m: None)
    assert p["title"] == "Gemini Cut" and p["planner"] == "Gemini (gemini-4-hydrogen)"
    gen = [c for c in calls if ":generateContent" in c[0]][0]
    assert "gemini-4-hydrogen" in gen[0]
    assert gen[1]["generationConfig"]["responseMimeType"] == "application/json"
    assert any("inlineData" in part for part in gen[1]["contents"][0]["parts"])
    assert gen[2]["x-goog-api-key"] == "test"


def test_claude_editor(media, monkeypatch):
    import anthropic
    from app.analyze import analyze_clip, score_shots
    from app.planner import plan_edit
    from app.video_io import probe

    import tempfile
    from pathlib import Path
    clip, shots = analyze_clip(probe(media["land"]), "c0", None, Path(tempfile.mkdtemp()))
    score_shots(shots)
    analysis = {"clips": [clip], "all_shots": shots, "shots": shots, "sheets": [], "transcripts": {}}

    class Msgs:
        def create(self, **kw):
            assert kw["tool_choice"]["name"] == "submit_edit_plan"
            inp = {"reasoning": "r", "title": "Claude Cut", "look": "noir", "segments": [
                {"shot_id": "c0s00", "start": 0, "end": 2, "speed": 1, "motion": "push_in", "transition_in": "cut"}]}
            return types.SimpleNamespace(content=[types.SimpleNamespace(type="tool_use", name="submit_edit_plan",
                                                                        input=inp)])

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: types.SimpleNamespace(messages=Msgs()))
    p = plan_edit(analysis, None, {"target_duration": 2, "planner": "claude"}, "noir", log=lambda m: None)
    assert p["title"] == "Claude Cut" and p["planner"].startswith("Claude")


def test_failed_editor_falls_back(media, monkeypatch):
    from app.analyze import analyze_clip, score_shots
    from app.planner import plan_edit
    from app.providers import gemini
    from app.providers.base import ProviderError
    from app.video_io import probe

    import tempfile
    from pathlib import Path
    clip, shots = analyze_clip(probe(media["land"]), "c0", None, Path(tempfile.mkdtemp()))
    score_shots(shots)
    analysis = {"clips": [clip], "all_shots": shots, "shots": shots, "sheets": [], "transcripts": {}}

    def boom(*a, **k):
        raise ProviderError("HTTP 403: API key invalid")

    monkeypatch.setenv("GEMINI_API_KEY", "bad")
    monkeypatch.setattr(gemini, "http_json", boom)
    logs = []
    p = plan_edit(analysis, None, {"target_duration": 5, "planner": "gemini"}, "x", log=logs.append)
    assert p["planner"].startswith("Built-in rules")
    assert any("could not plan" in m for m in logs)


def test_web_api(media, fake_ollama):
    from fastapi.testclient import TestClient
    from app.main import app

    c = TestClient(app)
    cfg = c.get("/api/config").json()
    assert cfg["providers"]["ollama"]["available"] is True
    with open(media["land"], "rb") as f, open(media["music"], "rb") as m:
        r = c.post("/api/jobs", files=[("clips", ("land.mp4", f, "video/mp4")), ("music", ("m.mp3", m, "audio/mpeg"))],
                   data={"description": "fast", "options": json.dumps({**OPTS, "planner": "ollama"})})
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    for _ in range(240):
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert j["status"] == "done", j["error"]
    assert c.get(j["output_url"]).status_code == 200
    r = c.post(f"/api/jobs/{jid}/refine", json={"feedback": "warmer"})
    assert r.status_code == 200
    for _ in range(240):
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error") and j["version"] == 2:
            break
        time.sleep(0.5)
    assert j["status"] == "done" and j["version"] == 2


def test_compact_roundtrip(media):
    from app.analyze import analyze_clip, score_shots
    from app.providers.ollama import compact_to_raw, full_to_compact
    from app.video_io import probe

    import tempfile
    from pathlib import Path
    clip, shots = analyze_clip(probe(media["land"]), "c0", None, Path(tempfile.mkdtemp()))
    score_shots(shots)
    analysis = {"clips": [clip], "all_shots": shots, "shots": shots}
    cp = {"concept": "c", "title": "t", "pace": "slow", "look": "moody", "grain": "medium", "letterbox": True,
          "caption_style": "boxed", "segments": [
              {"shot": "c0s02", "seconds": 2.0, "part": "end", "speed": 1, "camera": "pull_out",
               "transition": "crossfade", "sound": True, "text": "", "text_style": "bold"}]}
    raw = compact_to_raw(cp, analysis, None)
    s = raw["segments"][0]
    assert abs(s["end"] - shots[2]["end"]) < 1e-6 and abs(s["end"] - s["start"] - 2.0) < 1e-6
    back = full_to_compact(raw, analysis)
    assert back["segments"][0]["part"] == "end" and back["grain"] == "medium"
