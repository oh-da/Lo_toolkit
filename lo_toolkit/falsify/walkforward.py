"""Strict walk-forward evaluation against the fair null.

Chronological one-step-ahead scoring only — no shuffling, no look-ahead.
Metric is mean per-number binary log loss (and Brier); hit-rate is banned
because it rewards overconfident noise.  A model 'wins' only if its mean
log loss is materially below the null's out of sample — which, on a fair
game, none should be.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..games.ruleset import Ruleset


@dataclass
class ModelScore:
    name: str
    log_loss: float
    brier: float
    delta_vs_null: float             # negative = better than null

    def verdict(self, tol: float = 1e-3) -> str:
        if self.delta_vs_null < -tol:
            return "beats null — suspect leakage/overfitting before believing it"
        if self.delta_vs_null <= tol:
            return "indistinguishable from fair null"
        return "worse than fair null"


@dataclass
class WalkForwardResult:
    n_scored: int
    scores: list[ModelScore]

    def summary(self) -> str:
        lines = [
            f"Walk-forward falsification ({self.n_scored} one-step-ahead draws)",
            f"{'model':16s} {'log loss':>10s} {'brier':>10s} {'vs null':>10s}  verdict",
        ]
        for s in self.scores:
            lines.append(
                f"{s.name:16s} {s.log_loss:10.6f} {s.brier:10.6f} "
                f"{s.delta_vs_null:+10.6f}  {s.verdict()}"
            )
        return "\n".join(lines)


def _score_draw(p: np.ndarray, drawn: np.ndarray, pool_size: int) -> tuple[float, float]:
    y = np.zeros(pool_size)
    y[drawn - 1] = 1.0
    p = np.clip(p, 1e-9, 1 - 1e-9)
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    brier = float(((p - y) ** 2).mean())
    return ll, brier


def walk_forward(
    models: list,
    draws: np.ndarray,
    rules: Ruleset,
    warmup: int = 100,
) -> WalkForwardResult:
    """Evaluate `models` (must include predict(history, rules)) on `draws`."""
    n = draws.shape[0]
    if n <= warmup + 10:
        raise ValueError(f"need more than {warmup + 10} draws (got {n})")

    from .models import NullModel

    all_models = [NullModel()] + [m for m in models if not isinstance(m, NullModel)]
    ll = {m.name: [] for m in all_models}
    br = {m.name: [] for m in all_models}
    for t in range(warmup, n):
        history, actual = draws[:t], draws[t]
        for m in all_models:
            p = m.predict(history, rules)
            l, b = _score_draw(p, actual, rules.main.pool_size)
            ll[m.name].append(l)
            br[m.name].append(b)

    null_ll = float(np.mean(ll["fair_null"]))
    scores = [
        ModelScore(
            name=m.name,
            log_loss=float(np.mean(ll[m.name])),
            brier=float(np.mean(br[m.name])),
            delta_vs_null=float(np.mean(ll[m.name])) - null_ll,
        )
        for m in all_models
    ]
    return WalkForwardResult(n_scored=n - warmup, scores=scores)
