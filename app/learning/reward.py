"""Reward model: a linear score over edit features, learned online from your feedback.

  pairwise   "B beat A" (you rated B higher, downloaded B, picked B's look)  -> Bradley-Terry update
  absolute   star ratings                                                     -> regression on (stars-3)/2
  stated     revision requests ("faster", "no glitch")                        -> direct nudge of the weights

It is deliberately small (dozens of weights) so it learns from a handful of edits, stays explainable
("you like warm looks for travel"), and can be inspected or reset at any time.
"""
from __future__ import annotations

import math


def _dot(w, f):
    return sum(w.get(k, 0.0) * v for k, v in f.items())


class RewardModel:
    L2 = 0.004

    def __init__(self, state: dict | None = None):
        state = state or {}
        self.w: dict[str, float] = dict(state.get("w", {}))
        self.n: int = int(state.get("n", 0))

    def state(self):
        return {"w": {k: round(v, 5) for k, v in self.w.items() if abs(v) > 1e-5}, "n": self.n}

    @property
    def lr(self):
        return 0.35 / math.sqrt(1 + self.n / 15)

    @property
    def confidence(self) -> float:
        return 1 - math.exp(-self.n / 10)

    def score(self, f: dict) -> float:
        return _dot(self.w, f)

    def _decay(self, keys):
        for k in keys:
            if k in self.w:
                self.w[k] *= (1 - self.L2)

    def update_pair(self, f_win: dict, f_lose: dict, weight: float = 1.0) -> float:
        keys = set(f_win) | set(f_lose)
        diff = {k: f_win.get(k, 0.0) - f_lose.get(k, 0.0) for k in keys}
        diff = {k: v for k, v in diff.items() if abs(v) > 1e-6}
        if not diff:
            return 0.5
        z = max(-30.0, min(30.0, _dot(self.w, diff)))
        p = 1 / (1 + math.exp(-z))
        g = (1 - p) * self.lr * weight
        for k, v in diff.items():
            self.w[k] = self.w.get(k, 0.0) + g * v
        self._decay(diff)
        self.n += 1
        return p

    def update_rating(self, f: dict, stars: float, weight: float = 1.0):
        y = max(-1.0, min(1.0, (stars - 3) / 2))
        pred = math.tanh(_dot(self.w, f))
        g = (y - pred) * (1 - pred * pred) * self.lr * weight * 0.6
        for k, v in f.items():
            if v:
                self.w[k] = self.w.get(k, 0.0) + g * v
        self._decay(f)
        self.n += 1

    def nudge(self, deltas: dict, strength: float = 0.5):
        """A stated preference ('faster', 'warmer') moves the matching weights directly."""
        for k, d in deltas.items():
            self.w[k] = self.w.get(k, 0.0) + strength * d * max(0.15, self.lr)
        self.n += 1

    def top(self, n=12, min_abs=0.05):
        items = sorted(((k, v) for k, v in self.w.items() if abs(v) >= min_abs), key=lambda kv: -abs(kv[1]))
        return items[:n]
