"""SQLite store for everything the app learns from you (workspace/learning/reelsmith.db, git-ignored)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS edits (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL, version INTEGER NOT NULL, created REAL,
  brief TEXT, category TEXT, editor TEXT, model TEXT, trigger TEXT,
  features TEXT, summary TEXT, learning TEXT,
  rating REAL, downloaded INTEGER DEFAULT 0,
  UNIQUE(job_id, version));
CREATE TABLE IF NOT EXISTS pairs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, winner INTEGER, loser INTEGER, weight REAL, source TEXT, created REAL);
CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, version INTEGER, category TEXT, text TEXT, deltas TEXT,
  created REAL);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
"""


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()

    # -- generic -------------------------------------------------------------------------
    def q(self, sql, args=()):
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def x(self, sql, args=()):
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur.lastrowid

    def get(self, k, default=None):
        r = self.q("SELECT v FROM kv WHERE k=?", (k,))
        return json.loads(r[0]["v"]) if r else default

    def put(self, k, v):
        self.x("INSERT INTO kv(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, json.dumps(v)))

    # -- edits ---------------------------------------------------------------------------
    def add_edit(self, job_id, version, brief, cat, editor, model, trigger, features, summary, learning) -> int:
        self.x("""INSERT INTO edits(job_id, version, created, brief, category, editor, model, trigger, features,
                  summary, learning) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                  ON CONFLICT(job_id, version) DO UPDATE SET features=excluded.features, summary=excluded.summary,
                  learning=excluded.learning, editor=excluded.editor, model=excluded.model""",
               (job_id, version, time.time(), brief, cat, editor, model, trigger, json.dumps(features),
                json.dumps(summary), json.dumps(learning)))
        return self.edit(job_id, version)["id"]

    def edit(self, job_id, version):
        r = self.q("SELECT * FROM edits WHERE job_id=? AND version=?", (job_id, version))
        return self._decode(r[0]) if r else None

    def edits(self, job_id=None, limit=1000):
        if job_id:
            rows = self.q("SELECT * FROM edits WHERE job_id=? ORDER BY version", (job_id,))
        else:
            rows = self.q("SELECT * FROM edits ORDER BY created DESC LIMIT ?", (limit,))
        return [self._decode(r) for r in rows]

    @staticmethod
    def _decode(r):
        for k in ("features", "summary", "learning"):
            r[k] = json.loads(r[k]) if r.get(k) else {}
        return r

    def set_rating(self, edit_id, stars):
        self.x("UPDATE edits SET rating=? WHERE id=?", (stars, edit_id))

    def set_downloaded(self, edit_id):
        self.x("UPDATE edits SET downloaded=downloaded+1 WHERE id=?", (edit_id,))

    def add_pair(self, winner, loser, weight, source):
        self.x("INSERT INTO pairs(winner, loser, weight, source, created) VALUES(?,?,?,?,?)",
               (winner, loser, weight, source, time.time()))

    def add_feedback(self, job_id, version, cat, text, deltas):
        self.x("INSERT INTO feedback(job_id, version, category, text, deltas, created) VALUES(?,?,?,?,?,?)",
               (job_id, version, cat, text, json.dumps(deltas), time.time()))

    def counts(self):
        r = self.q("""SELECT (SELECT COUNT(*) FROM edits) AS edits,
                             (SELECT COUNT(*) FROM edits WHERE rating IS NOT NULL) AS rated,
                             (SELECT COUNT(*) FROM edits WHERE downloaded>0) AS downloaded,
                             (SELECT COUNT(*) FROM pairs) AS pairs,
                             (SELECT COUNT(*) FROM feedback) AS feedback""")
        return r[0]

    def reset(self):
        with self._lock:
            self._db.executescript("DELETE FROM edits; DELETE FROM pairs; DELETE FROM feedback; DELETE FROM kv;")
            self._db.commit()
