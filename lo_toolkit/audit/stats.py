"""Audit test statistics.

Each function maps an (n_draws, d) matrix of sorted main numbers to a scalar
statistic.  None of them carries its own p-value: significance always comes
from Monte Carlo calibration under the exact game law (see montecarlo.py),
which sidesteps the broken iid assumptions flagged in the lottery-audit
literature.
"""

from __future__ import annotations

import numpy as np

from ..games.ruleset import Ruleset
from .nullmodel import ball_counts, ball_count_moments, pair_cooccurrence_mean


def marginal_chi2(draws: np.ndarray, rules: Ruleset) -> float:
    """Chi-square-style statistic on per-ball counts vs exact expectation."""
    n = draws.shape[0]
    counts = ball_counts(draws, rules.main.pool_size)
    mean, _ = ball_count_moments(rules, n)
    return float(((counts - mean) ** 2 / mean).sum())


def pairwise_max_dev(draws: np.ndarray, rules: Ruleset) -> float:
    """Max absolute deviation of pair co-occurrence counts from expectation."""
    n_pool = rules.main.pool_size
    n = draws.shape[0]
    co = np.zeros((n_pool + 1, n_pool + 1))
    for row in draws:
        co[np.ix_(row, row)] += 1
    np.fill_diagonal(co, 0)
    expected = pair_cooccurrence_mean(rules, n)
    real = co[1:, 1:]                     # drop the unused 0 index
    iu = np.triu_indices(n_pool, k=1)
    return float(np.abs(real[iu] - expected).max())


def mean_min_spacing(draws: np.ndarray, rules: Ruleset) -> float:
    """Mean over draws of the minimal gap between adjacent sorted numbers.

    Spacing statistics (Drakakis et al.) are sensitive to tampering that
    frequency tests miss, e.g. draws avoiding adjacent numbers.
    """
    gaps = np.diff(np.sort(draws, axis=1), axis=1)
    return float(gaps.min(axis=1).mean())


def repeat_count(draws: np.ndarray, rules: Ruleset) -> float:
    """Total numbers shared between consecutive draws."""
    total = 0
    for prev, cur in zip(draws[:-1], draws[1:]):
        total += np.isin(cur, prev).sum()
    return float(total)


def sum_dispersion(draws: np.ndarray, rules: Ruleset) -> float:
    """Variance of per-draw sums; detects over/under-dispersion."""
    return float(draws.sum(axis=1).var(ddof=1))


ALL_TESTS = {
    "marginal_frequency": marginal_chi2,
    "pairwise_cooccurrence": pairwise_max_dev,
    "min_spacing": mean_min_spacing,
    "repeats_prev_draw": repeat_count,
    "sum_dispersion": sum_dispersion,
}

# Direction of suspicion: "upper" = only large values are anomalous,
# "two" = both tails matter.
TAILS = {
    "marginal_frequency": "upper",
    "pairwise_cooccurrence": "upper",
    "min_spacing": "two",
    "repeats_prev_draw": "two",
    "sum_dispersion": "two",
}
