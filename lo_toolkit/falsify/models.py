"""Benchmark models for the falsification lab.

Every model maps a draw history to per-number inclusion probabilities for
the next draw (summing to the draw size).  The point is not to predict —
it is to demonstrate, with strict walk-forward scoring, that popular
"pattern" heuristics do not beat the fair null model.
"""

from __future__ import annotations

import numpy as np

from ..games.ruleset import Ruleset


class NullModel:
    """The fair baseline: every number equally likely, every draw."""

    name = "fair_null"

    def predict(self, history: np.ndarray, rules: Ruleset) -> np.ndarray:
        n, d = rules.main.pool_size, rules.draw_size
        return np.full(n, d / n)


class HotColdModel:
    """'Hot numbers' heuristic: recent frequency, Laplace-smoothed.

    The archetypal lottery-strategy claim.  Expected walk-forward result:
    indistinguishable from or worse than the null.
    """

    name = "hot_cold"

    def __init__(self, window: int = 50, strength: float = 1.0):
        self.window = window
        self.strength = strength

    def predict(self, history: np.ndarray, rules: Ruleset) -> np.ndarray:
        n, d = rules.main.pool_size, rules.draw_size
        recent = history[-self.window:]
        counts = np.bincount(recent.ravel(), minlength=n + 1)[1:].astype(float)
        base = len(recent) * d / n
        weights = (counts + 1.0) / (base + 1.0)
        p = (d / n) * weights**self.strength
        return p * (d / p.sum())


class MarkovParityModel:
    """First-order Markov chain on the count of even numbers per draw.

    A state-compression Markov model of the kind the research reviews:
    legitimate machinery, near-zero edge on fair draws.
    """

    name = "markov_parity"

    def predict(self, history: np.ndarray, rules: Ruleset) -> np.ndarray:
        n, d = rules.main.pool_size, rules.draw_size
        states = (history % 2 == 0).sum(axis=1)          # evens per draw: 0..d
        trans = np.ones((d + 1, d + 1))                  # Laplace prior
        for a, b in zip(states[:-1], states[1:]):
            trans[a, b] += 1
        next_dist = trans[states[-1]] / trans[states[-1]].sum()
        exp_evens = float(next_dist @ np.arange(d + 1))

        numbers = np.arange(1, n + 1)
        n_even = int((numbers % 2 == 0).sum())
        p = np.empty(n)
        p[numbers % 2 == 0] = exp_evens / n_even
        p[numbers % 2 == 1] = (d - exp_evens) / (n - n_even)
        np.clip(p, 1e-9, 1 - 1e-9, out=p)
        return p * (d / p.sum())
