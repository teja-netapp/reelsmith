"""A tiny stand-in for the Ollama HTTP API so the local-AI path can be tested without a model.

It answers /api/tags and /api/chat. Shot-logging requests (with images) get a deterministic log;
planning requests get a deliberately sloppy plan (too many fancy transitions, repeated motion,
a duplicated moment) so the craft pass has something to fix.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "qwen3-vl:8b-instruct"


class _Handler(BaseHTTPRequestHandler):
    requests_log: list = []

    def log_message(self, *a):  # silence
        pass

    def _send(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/api/tags"):
            return self._send({"models": [{"name": MODEL}]})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        _Handler.requests_log.append(body)
        if not self.path.startswith("/api/chat"):
            return self._send({"error": "not found"}, 404)
        has_images = any(m.get("images") for m in body.get("messages", []))
        if has_images:
            n = len(_Handler.requests_log)
            content = {"description": f"test frame {n}", "subject": "yellow box", "subject_x": 0.3,
                       "shot_type": "medium", "mood": "energetic", "quality": 7, "hook": 3 + n % 7,
                       "best_part": "end", "camera": "pan", "action": "box moves right"}
        elif "sections" in body["format"]["properties"]:  # story outline pass
            ids = body["format"]["properties"]["sections"]["items"]["properties"]["shots"]["items"]["enum"]
            content = {"concept": "fake story", "title": "Fake Story", "sections": [
                {"name": "hook", "shots": ids[:1], "feeling": "bold"},
                {"name": "build", "shots": ids[1:3], "feeling": "rising"},
                {"name": "resolve", "shots": ids[-1:], "feeling": "calm"}]}
        else:
            ids = body["format"]["properties"]["segments"]["items"]["properties"]["shot"]["enum"]
            trs = ["cut", "glitch", "glitch", "flash", "whip_left", "light_leak", "zoom_punch"]
            segs = []
            for i in range(7):
                sid = ids[i % len(ids)] if i != 3 else ids[0]  # segment 3 repeats the first moment
                segs.append({"shot": sid, "seconds": 1.6, "part": "start" if i == 3 else "middle",
                             "speed": 1, "camera": "push_in", "transition": trs[i], "sound": False,
                             "text": "TEST TITLE" if i == 0 else "", "text_style": "cinematic_title"})
            content = {"concept": "fake plan", "title": "Fake Local Edit", "pace": "fast", "look": "teal_orange",
                       "grain": "light", "letterbox": False, "caption_style": "pop_yellow", "segments": segs}
        self._send({"model": body.get("model"), "message": {"role": "assistant", "content": json.dumps(content)},
                    "done": True})


def start(port: int):
    _Handler.requests_log = []
    srv = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, _Handler.requests_log
