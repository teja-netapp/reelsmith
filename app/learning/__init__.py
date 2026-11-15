"""Learning from your feedback (see learner.py)."""
from __future__ import annotations

import threading

from .. import config as C
from .learner import Learner

_learner: Learner | None = None
_lock = threading.Lock()


def get_learner() -> Learner:
    global _learner
    with _lock:
        if _learner is None:
            _learner = Learner(C.WORKSPACE / "learning" / "reelsmith.db")
        return _learner


__all__ = ["Learner", "get_learner"]
