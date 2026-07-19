"""Monte Carlo calibration of audit statistics under the exact game law."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..games.ruleset import Ruleset
from ..sim.engine import DrawSimulator

StatFn = Callable[[np.ndarray, Ruleset], float]


def mc_pvalue(
    stat_fn: StatFn,
    observed: np.ndarray,
    rules: Ruleset,
    n_sims: int = 2000,
    tail: str = "upper",
    seed: int | None = None,
) -> tuple[float, float]:
    """(observed statistic, MC p-value) for `stat_fn` on `observed` draws.

    Simulates `n_sims` fair histories of the same length and compares.  Uses
    the add-one (permutation-style) estimator so p-values are never zero.
    """
    obs = stat_fn(observed, rules)
    sim = DrawSimulator(rules, seed=seed)
    n = observed.shape[0]
    null = np.empty(n_sims)
    for i in range(n_sims):
        null[i] = stat_fn(sim.draw_main(n), rules)

    if tail == "upper":
        p = (1 + (null >= obs).sum()) / (n_sims + 1)
    elif tail == "lower":
        p = (1 + (null <= obs).sum()) / (n_sims + 1)
    else:  # two-sided: double the smaller tail
        hi = (1 + (null >= obs).sum()) / (n_sims + 1)
        lo = (1 + (null <= obs).sum()) / (n_sims + 1)
        p = min(1.0, 2 * min(hi, lo))
    return obs, float(p)
