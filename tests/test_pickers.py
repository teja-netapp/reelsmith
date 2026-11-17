"""Transition / effect pickers and Look Lab endpoints."""
import json
import subprocess

H = {"Origin": "http://127.0.0.1:8765"}


def _dur(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                         capture_output=True, text=True).stdout
    return float(out.strip())


def test_catalog_and_previews(media, tmp_path):
    from fastapi.testclient import TestClient

    from app.main import app
    from app.pipeline import run_sync
    c = TestClient(app)
    cat = c.get("/api/transitions").json()
    names = [x["name"] for g in cat["categories"] for x in g["items"]]
    assert len(names) >= 70 and names[0] == "cut" and len(cat["categories"]) == 11
    assert {"whip_left", "heart", "paper_tear", "flip_3d", "glitch", "light_leak"} <= set(names)
    assert {e["name"] for e in cat["effects"]} >= {"zoom_pulse", "vhs", "rgb_split"}

    r = c.get("/api/transition_preview", params={"name": "flip_3d"})
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    p = tmp_path / "demo.mp4"
    p.write_bytes(r.content)
    assert 1.0 < _dur(str(p)) < 3.0
    assert c.get("/api/transition_preview", params={"name": "nope"}).status_code == 404

    job = run_sync([media["land"], media["track"]], None, "quick test",
                   {"target_duration": 6, "quality": "preview", "planner": "heuristic"})
    from app.main import runner
    runner.jobs[job.id] = job
    segs = job.plan["segments"]
    a, b = segs[0]["shot_id"], segs[1]["shot_id"]
    r = c.get("/api/transition_preview", params={"name": "whip_left", "job": job.id, "a": a, "b": b})
    assert r.status_code == 200 and (job.dir / "previews" / f"tr_whip_left_{a}_{b}.mp4").exists()
    r = c.get("/api/effect_preview", params={"name": "zoom_pulse", "job": job.id, "shot": a})
    assert r.status_code == 200 and len(r.content) > 1000

    looks = c.get(f"/api/jobs/{job.id}/looks").json()
    assert "Film" in looks["categories"] and all(x["label"] and x["category"] for x in looks["looks"])
    assert not any(x["name"] == "none" for x in looks["looks"])
    r1 = c.get(f"/api/jobs/{job.id}/look_preview", params={"look": "teal_orange", "strength": 0.5})
    r2 = c.get(f"/api/jobs/{job.id}/look_preview", params={"look": "teal_orange", "strength": 1.5})
    assert r1.status_code == r2.status_code == 200 and r1.content != r2.content

    # manual per-cut pick: re-render keeps it and reports it
    plan = {k: v for k, v in job.plan.items() if k not in ("editor", "craft_changes", "warnings")}
    plan["segments"][1]["transition_in"] = "heart"
    plan["segments"][1]["transition_duration"] = 0.6
    plan["segments"][0]["effect"] = "rgb_split"
    from app.pipeline import _run
    _run(job, "plan", plan)
    assert job.status == "done", job.error
    assert job.plan["segments"][1]["transition_in"] == "heart" and job.plan["segments"][0]["effect"] == "rgb_split"
    rep = json.loads(job.report_path().read_text())
    assert rep["segments"][0]["effect"] == "rgb_split"
