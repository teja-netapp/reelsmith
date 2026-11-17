"""Your instructions are hard rules: parsing, enforcement, and a pixel-level proof that 'no grading' means no grading."""
import subprocess

import numpy as np
import pytest

OPTS = {"target_duration": 8, "quality": "preview", "beat_sync": False, "captions": False}


def _rules(text, revisions=(), options=None):
    from app.directives import collect
    from app.grading import look_names
    return collect(text, list(revisions), options or {}, look_names())


@pytest.mark.parametrize("text,raw", [
    ("Don't touch any color grading, just add the transition effects", True),
    ("keep the original colours please", True),
    ("no filters, no LUT", True),
    ("the colours are fine, don't change them", True),
    ("Raw footage with transitions and effects only", True),
    ("make it look cinematic and warm", False),
    ("I don't like this look, try something warmer", False),
    ("Keep the colors vibrant", False),
])
def test_detects_no_grading(text, raw):
    assert _rules(text).raw_color is raw


def test_parse_mixed_instructions():
    d = _rules("Goa trip! only cuts, no music, keep the clips in order, title 'Goa 24', 15 second reel")
    assert d.v["transitions"] == "none" and d.v["music"] is False and d.v["keep_order"] is True
    assert d.v["title"] == "Goa 24" and d.v["duration"] == 15
    assert not d.raw_color

    d = _rules("add whip pans and light leaks, no glitches or flashes")
    assert {"whip_left", "light_leak"} <= d.want_tr
    assert {"glitch", "flash", "flash_blast"} <= d.ban_tr

    d = _rules("I just want it fast with captions")  # 'just' here is not "only"
    assert d.v.get("captions") is True and not d.raw_color and "text" not in d.v


def test_later_instruction_wins_and_checkbox_is_master():
    assert not _rules("don't touch the colours", ["use the teal orange look"]).raw_color
    assert _rules("teal orange look", ["actually don't grade it"]).raw_color
    d = _rules("warm cinematic teal orange look", [], {"grading": False})
    assert d.raw_color and d.rules["grading"]["source"] == "Color grading checkbox"
    # a revision's "just add X" changes only X
    assert not _rules("fun trip", ["just add some whips"]).raw_color


def test_enforce_overrides_a_disobedient_plan(media, tmp_path):
    from app.analyze import analyze_clip, score_shots
    from app.directives import enforce
    from app.video_io import probe

    clip, shots = analyze_clip(probe(media["land"]), "c0", None, tmp_path)
    score_shots(shots)
    analysis = {"clips": [clip], "all_shots": shots, "shots": shots, "transcripts": {}}
    plan = {"look": "teal_orange", "look_intensity": 1.0, "grain": 0.4, "vignette": 0.5, "halation": 0.3, "bloom": 0.2,
            "letterbox": True, "segments": [
                {"shot_id": s["id"], "clip_id": "c0", "start": s["start"], "end": s["end"] - 1, "speed": 0.5 if i else 1,
                 "motion": "push_in", "transition_in": "glitch", "transition_duration": 0.3, "effect": "vhs",
                 "text": {"content": "HELLO", "style": "bold", "position": "center"}, "keep_audio": False}
                for i, s in enumerate(shots)]}
    d = _rules("Don't touch any color grading, just add the transition effects")
    changes = enforce(plan, d, analysis, {})
    assert plan["grading"] is False and plan["look"] == "none"
    assert plan["grain"] == plan["vignette"] == plan["halation"] == plan["bloom"] == 0 and not plan["letterbox"]
    segs = plan["segments"]
    assert all(s["effect"] == "none" and s["motion"] == "none" and s["speed"] == 1.0 and not s["text"] for s in segs)
    assert any(s["transition_in"] != "cut" for s in segs[1:])  # transitions were asked for - still there
    assert changes and all(c["rule"] == "your instructions" for c in changes)


def _frame_at(path, t, w=540, h=960):
    """Decode one frame the way a player does (file's own colour tags, accurate rounding)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1", "-vf",
                          f"scale={w}:{h}:flags=bicubic+accurate_rnd+full_chroma_int,format=rgb24", "-f", "rawvideo",
                          "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


@pytest.fixture(scope="module")
def green_clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    p = d / "green.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=0x4a7a3c:s=1280x720:r=30:d=6",
                    "-f", "lavfi", "-i", "color=c=0xd04030:s=80x80:r=30:d=6", "-filter_complex",
                    "[0:v][1:v]overlay=x='100+t*150':y=300:shortest=1,scale=out_color_matrix=bt709:out_range=tv:"
                    "flags=accurate_rnd+full_chroma_int,format=yuv420p[v]", "-map", "[v]", "-c:v", "libx264",
                    "-preset", "ultrafast", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                    "-color_range", "tv", str(p)], check=True)
    return str(p)


def test_raw_colour_proof_end_to_end(green_clip, media, fake_ollama):
    """The fake local AI always picks teal_orange + a title + push-ins. With the instruction, the output's
    pixels must match the source footage; without it, they must not."""
    from app.pipeline import run_sync

    brief = "Don't touch any color grading, just add the transition effects"
    job = run_sync([green_clip, media["track"]], None, brief, {**OPTS, "planner": "ollama"})
    assert job.status == "done", job.error
    p = job.plan
    assert p["grading"] is False and p["look"] == "none"
    plan_req = [r for r in fake_ollama if not any(m.get("images") for m in r["messages"])][-1]
    assert "HARD RULES" in plan_req["messages"][-1]["content"]           # the AI was told
    assert any(c["rule"] == "your instructions" for c in p["craft_changes"])  # ...and corrected when it ignored it
    assert not any(s.get("text") for s in p["segments"])
    rep = __import__("json").loads(job.report_path().read_text())
    assert rep["instructions"]["raw_color"] and rep["instructions"]["rules"]

    src_bg = np.median(_frame_at(green_clip, 1.0, 1280, 720).reshape(-1, 3), axis=0)   # like phone footage: BT.709
    segs = job.timeline_summary["segments"]
    seg = next(s for s in segs if s["shot_id"].startswith("c0"))
    t = (seg["start_frame"] + seg["frames"] / 2) / 30
    bg = np.median(_frame_at(job.output, t).reshape(-1, 3), axis=0)
    assert np.abs(bg - src_bg).max() <= 2, (bg, src_bg)                   # original colours, to codec precision

    graded = run_sync([green_clip, media["track"]], None, "fun reel", {**OPTS, "planner": "ollama"})
    assert graded.status == "done", graded.error
    assert graded.plan["grading"] is not False
    seg = next(s for s in graded.timeline_summary["segments"] if s["shot_id"].startswith("c0"))
    t = (seg["start_frame"] + seg["frames"] / 2) / 30
    bg2 = np.median(_frame_at(graded.output, t).reshape(-1, 3), axis=0)
    assert np.abs(bg2 - src_bg).max() > 8, (bg2, src_bg)                  # the look really changes the image


def test_directives_api(fake_ollama):
    from fastapi.testclient import TestClient

    from app.main import app
    c = TestClient(app)
    r = c.post("/api/directives", json={"text": "no text, keep the original colours, only cuts"},
               headers={"Origin": "http://127.0.0.1:8765"})
    assert r.status_code == 200
    j = r.json()
    assert j["raw_color"] is True
    assert any("cuts only" in x.lower() for x in j["text"])
