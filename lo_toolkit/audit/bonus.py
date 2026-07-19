"""Uniformity check for a separate-pool bonus ball (e.g. Powerball, the
Israeli 'strong number').

A plain multinomial chi-square applies here — the bonus is one independent
uniform draw per drawing — but the p-value is still Monte Carlo-calibrated
for consistency with the rest of the battery.
"""

from __future__ import annotations

import numpy as np


def bonus_uniformity(
    bonus_values: np.ndarray,
    pool_size: int,
    n_sims: int = 2000,
    seed: int | None = 0,
) -> tuple[float, float]:
    """(chi-square statistic, MC p-value) for uniformity over 1..pool_size."""
    values = np.asarray(bonus_values)
    n = len(values)
    expected = n / pool_size
    counts = np.bincount(values, minlength=pool_size + 1)[1:]
    obs = float(((counts - expected) ** 2 / expected).sum())

    rng = np.random.default_rng(seed)
    sims = rng.integers(1, pool_size + 1, size=(n_sims, n))
    null = np.empty(n_sims)
    for i in range(n_sims):
        c = np.bincount(sims[i], minlength=pool_size + 1)[1:]
        null[i] = ((c - expected) ** 2 / expected).sum()
    p = (1 + (null >= obs).sum()) / (n_sims + 1)
    return obs, float(p)
