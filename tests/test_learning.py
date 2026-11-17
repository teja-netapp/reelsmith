"""Learning loop + color engine."""
import json
import subprocess
import time

import numpy as np


# ----------------------------------------------------------------------------------------------
# revision text -> preferences
# ----------------------------------------------------------------------------------------------
def test_parse_revision_requests():
    from app.learning.preferences import parse

    d = parse("Make it faster and warmer, no glitch transitions please")
    assert d["pace_short"] > 0 and d["avg_len"] < 0
    assert d["lf_warm"] > 0 and d["t_glitch"] < 0
    d = parse("less warm, not too dark, remove the title")
    assert d["lf_warm"] < 0 and d["lf_dark"] < 0 and d["has_title"] < 0
    d = parse("no slow mo, just cuts")
    assert d["slowmo_frac"] < 0 and d["styl_frac"] < 0
    assert parse("looks great") == {}


# ----------------------------------------------------------------------------------------------
# reward model learns a preference from pairs
# ----------------------------------------------------------------------------------------------
def test_reward_model_learns_from_pairs():
    from app.learning.reward import RewardModel

    m = RewardModel()
    fast, slow = {"pace_short": 0.8, "avg_len": 0.2}, {"pace_short": 0.1, "avg_len": 0.7}
    for _ in range(12):
        m.update_pair(fast, slow)
    assert m.score(fast) > m.score(slow) and m.w["pace_short"] > 0 and m.w["avg_len"] < 0
    assert 0.6 < m.confidence < 1


# ----------------------------------------------------------------------------------------------
# the whole loop improves edits for a simulated user with a hidden taste
# ----------------------------------------------------------------------------------------------
def test_learning_improves_edits_for_simulated_user(media):
    import simulate_learning as S
    from pathlib import Path
    from app.analyze import analyze_clip, score_shots
    from app.video_io import probe
    import tempfile

    clips, shots = [], []
    for i, p in enumerate([media["land"], media["track"]]):
        c, s = analyze_clip(probe(p), f"c{i}", None, Path(tempfile.mkdtemp()))
        clips.append(c)
        shots += s
    score_shots(shots)
    an = {"clips": clips, "all_shots": shots, "shots": shots, "sheets": [], "transcripts": {}}
    hist, learner = S.run(24, seed=7, analysis=an, verbose=False)
    alone = np.mean([h[0] for h in hist])
    first, last = np.mean([h[1] for h in hist[:8]]), np.mean([h[1] for h in hist[-8:]])
    assert last > alone + 0.5, (alone, last)
    assert last > first, (first, last)
    from app.grading import look_family
    prof = learner.profile(S.BRIEF)
    assert "faster" in " ".join(prof["bullets"]).lower()                       # hidden taste: fast cuts
    assert any(look_family(lk) in ("warm", "film") for lk in prof["looks"])    # warm / film looks
    assert learner.model.w.get("t_glitch", 0) <= 0                             # never learned to like glitch
    warm = max(learner._wf("lf_warm", "travel"), learner._wf("lf_film", "travel"))
    assert learner._wf("lf_cool", "travel") <= 0 < warm                       # prefers warm/film over cool looks
    learner.reset()


# ----------------------------------------------------------------------------------------------
# color engine
# ----------------------------------------------------------------------------------------------
def test_color_engine_identity_and_ffmpeg_match(tmp_path):
    from app import grading as G

    assert np.abs(G.build_cube("none").reshape(-1, 3) - G._grid(33)).max() < 1e-9
    img = (np.random.default_rng(1).random((40, 60, 3)) * 255).astype(np.uint8)
    import cv2
    cv2.imwrite(str(tmp_path / "in.png"), img)
    name = G.cube_file("kodak_2383", 1.0, {"wb": [1.04, 1.0, 0.96], "ev": 0.2}, tmp_path)
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", "in.png", "-vf", f"lut3d=file={name}:interp=tetrahedral",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, cwd=str(tmp_path)).stdout
    ff = np.frombuffer(out, np.uint8).reshape(40, 60, 3).astype(int)
    py = G.apply_to_image(img, "kodak_2383", 1.0, {"wb": [1.04, 1.0, 0.96], "ev": 0.2}, n=33).astype(int)
    assert np.abs(ff - py).mean() < 1.5
    # every built-in look produces a valid, non-identity LUT
    for lk in G.LOOK_PARAMS:
        c = G.build_cube(lk, n=9)
        assert c.min() >= 0 and c.max() <= 1
        if lk != "none":
            assert np.abs(c.reshape(-1, 3) - G._grid(9)).max() > 0.01, lk


def test_shot_matching_and_user_luts(tmp_path, monkeypatch):
    from app import grading as G

    warm, cool = {"cast": [0.38, 0.33, 0.29], "luma": 0.45}, {"cast": [0.30, 0.33, 0.37], "luma": 0.30}
    m = G.match_shots({"a": warm, "b": cool, "c": warm, "d": {"cast": None, "luma": 0.45}})
    assert m["b"]["wb"][0] > 1 > m["b"]["wb"][2]          # cool shot warmed toward the reel
    assert m["b"]["ev"] > 0                                # and brightened toward the others
    assert m["d"]["wb"] == [1.0, 1.0, 1.0]                 # no neutral pixels -> color untouched
    # your own .cube LUTs become looks
    monkeypatch.setattr(G, "LUT_DIR", tmp_path)
    G.write_cube(tmp_path / "My Teal.cube", G.build_cube("teal_orange", n=9))
    assert "lut:My_Teal" in G.look_names()
    assert np.abs(G.build_cube("lut:My_Teal", n=9) - G.build_cube("teal_orange", n=9)).max() < 0.02


def test_hdr_clip_is_tonemapped(tmp_path):
    from app.video_io import FrameReader, has_filter, probe

    p = tmp_path / "hlg.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=1", "-vf",
                    "format=yuv420p10le", "-c:v", "libx264", "-preset", "ultrafast", "-color_primaries", "bt2020",
                    "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc", str(p)], check=True)
    info = probe(str(p))
    assert info.hdr == "hlg"
    if has_filter("zscale"):
        assert info.hdr_filter and "tonemap" in info.hdr_filter
        r = FrameReader(str(p), 0.2, 0.5, 320, 180, pre_filters=info.pre_filters())
        assert r.read() is not None
        r.close()


# ----------------------------------------------------------------------------------------------
# API: ratings, Look Lab, downloads, learning panel
# ----------------------------------------------------------------------------------------------
def test_feedback_api_flow(media, fake_ollama):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.learning import get_learner

    get_learner().reset()
    c = TestClient(app)
    with open(media["land"], "rb") as f, open(media["track"], "rb") as t:
        r = c.post("/api/jobs", files=[("clips", ("land.mp4", f, "video/mp4")), ("clips", ("track.mp4", t, "video/mp4"))],
                   data={"description": "travel reel of my trip", "options": json.dumps(
                       {"target_duration": 8, "quality": "preview", "planner": "ollama"})})
    jid = r.json()["id"]

    def wait(v):
        for _ in range(240):
            j = c.get(f"/api/jobs/{jid}").json()
            if j["status"] in ("done", "error") and j["version"] == v:
                return j
            time.sleep(0.5)
        raise AssertionError("timeout")

    j = wait(1)
    assert j["status"] == "done", j["error"]
    rep = j["report"]
    assert rep["editor"]["plan_depth"].startswith("story outline")    # 2-pass local editing
    assert rep["learning"] and "confidence" in rep["learning"]
    assert any(s.get("color_match") for s in rep["segments"])          # shot matching reported
    # rate v1 low, ask for a revision, rate v2 high -> pairwise learning
    assert c.post(f"/api/jobs/{jid}/rate", json={"version": 1, "stars": 2}).json()["ok"]
    c.post(f"/api/jobs/{jid}/refine", json={"feedback": "faster and warmer, no glitch"})
    j = wait(2)
    r = c.post(f"/api/jobs/{jid}/rate", json={"version": 2, "stars": 5}).json()
    assert r["ok"] and r["pairs"] >= 1
    stats = c.get("/api/learning").json()
    assert stats["rated"] == 2 and stats["feedback"] >= 1 and stats["updates"] >= 3
    assert any("glitch" in b.lower() for b in stats["house_style"])
    # Look Lab: previews + re-render with a picked look
    looks = c.get(f"/api/jobs/{jid}/looks").json()
    assert len(looks["looks"]) >= 20
    img = c.get(f"/api/jobs/{jid}/look_preview", params={"look": "golden_hour"})
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    c.post(f"/api/jobs/{jid}/look", json={"look": "golden_hour"})
    j = wait(3)
    assert j["report"]["trigger"]["type"] == "look" and j["report"]["finishing"]["look"] == "golden_hour"
    # downloading counts as a positive signal
    c.get(f"/api/jobs/{jid}/output", params={"v": 3, "download": 1})
    assert c.get("/api/learning").json()["downloaded"] == 1
    # the next reel's prompt carries the learned house style
    block = get_learner().prompt_block("travel reel of the beach")
    assert "house style" in block.lower()
    exp = c.get("/api/learning/export")
    assert exp.status_code == 200 and exp.text.strip()
    assert c.post("/api/learning", json={"enabled": False}).json()["enabled"] is False
    c.post("/api/learning", json={"enabled": True})
    assert c.post("/api/learning/reset").json()["edits"] == 0
