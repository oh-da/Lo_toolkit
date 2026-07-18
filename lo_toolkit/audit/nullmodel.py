"""Exact null expectations for k/N draws.

For a fair game drawing d numbers from 1..N, over n independent draws:

* each ball appears with probability p = d/N per draw, so its count is
  Binomial(n, p) — but counts across balls are *negatively correlated*
  within a draw (hypergeometric covariance), which is why naive iid
  chi-square thresholds are wrong for lotto data (Joe 1993).  The toolkit
  therefore calibrates every statistic by Monte Carlo under the exact law
  rather than trusting asymptotic distributions.
* an ordered pair (i, j) co-occurs with probability d(d-1) / (N(N-1)).
"""

from __future__ import annotations

import numpy as np

from ..games.ruleset import Ruleset


def ball_count_moments(rules: Ruleset, n_draws: int) -> tuple[float, float]:
    """(mean, variance) of a single ball's appearance count over n draws."""
    n_pool, d = rules.main.pool_size, rules.draw_size
    p = d / n_pool
    return n_draws * p, n_draws * p * (1 - p)


def pair_cooccurrence_mean(rules: Ruleset, n_draws: int) -> float:
    n_pool, d = rules.main.pool_size, rules.draw_size
    return n_draws * d * (d - 1) / (n_pool * (n_pool - 1))


def within_draw_covariance(rules: Ruleset) -> float:
    """Cov(1{i drawn}, 1{j drawn}) for i != j within one draw."""
    n_pool, d = rules.main.pool_size, rules.draw_size
    p = d / n_pool
    p_both = d * (d - 1) / (n_pool * (n_pool - 1))
    return p_both - p * p


def ball_counts(draws: np.ndarray, pool_size: int) -> np.ndarray:
    """Appearance count per ball (index 0 = ball 1) from an (n, d) draw matrix."""
    return np.bincount(draws.ravel(), minlength=pool_size + 1)[1:]
