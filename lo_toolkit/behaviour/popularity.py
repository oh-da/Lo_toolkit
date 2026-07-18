"""Player number-popularity surface.

Player microdata are rarely public, so the default surface encodes the
robust findings from the literature (Wang et al.; Polin et al.; Matheson):

* numbers <= 31 (birthdays) are over-picked; <= 12 (months/days) even more
* culturally lucky numbers (especially 7, also 3, 11, 13 in some markets)
  are over-picked; the most popular number appeared ~16.5% of the time vs
  13.3% under uniform play, the least popular ~10.3% (a ~1.24x / ~0.77x
  relative range)
* large numbers (> 31) are under-picked

The surface can be refitted from per-tier winner counts when available.
Weights are *relative pick rates*: 1.0 = picked at the uniform rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np

_LUCKY = {3: 1.10, 7: 1.25, 8: 1.10, 9: 1.05, 11: 1.10, 13: 0.95, 17: 1.05}


def _default_weights(pool_size: int) -> np.ndarray:
    w = np.ones(pool_size)
    for n in range(1, pool_size + 1):
        if n <= 12:
            w[n - 1] = 1.20          # day-and-month range, most over-picked
        elif n <= 31:
            w[n - 1] = 1.12          # birthday range
        else:
            w[n - 1] = 0.85          # above-31 numbers are under-picked
        if n in _LUCKY:
            w[n - 1] *= _LUCKY[n]
    return w / w.mean()              # normalise: uniform play == 1.0


@dataclass
class PopularityModel:
    weights: np.ndarray              # index 0 = number 1; mean 1.0

    @classmethod
    def default(cls, pool_size: int) -> "PopularityModel":
        return cls(_default_weights(pool_size))

    @classmethod
    def from_weights(cls, weights: list[float]) -> "PopularityModel":
        w = np.asarray(weights, dtype=float)
        return cls(w / w.mean())

    def weight(self, number: int) -> float:
        return float(self.weights[number - 1])

    def combination_factor(self, line: tuple[int, ...], picks: int) -> float:
        """Relative pick rate of a whole line vs a uniform random line.

        First-order independence approximation (product of per-number
        weights), with pattern multipliers for combination-level effects
        documented in the literature.
        """
        factor = prod(self.weight(n) for n in line)
        s = sorted(line)
        # all-birthday lines are strongly over-represented
        if all(n <= 31 for n in s):
            factor *= 1.5
        # arithmetic progressions (e.g. 7-14-21-28-35-42) are heavily played
        if len(s) >= 3:
            diffs = {b - a for a, b in zip(s, s[1:])}
            if len(diffs) == 1:
                factor *= 5.0
        # long runs of consecutive numbers look "special" and attract picks
        if len(s) >= 4 and all(b - a == 1 for a, b in zip(s, s[1:])):
            factor *= 3.0
        return factor
