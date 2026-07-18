"""Jackpot -> ticket-sales forecasting.

Sales respond super-linearly to advertised jackpots; a log-log linear model
captures the documented elasticity well enough for split-risk estimation.
Fit on 1-3 years of (jackpot, sales) pairs from official open-data sources.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SalesModel:
    intercept: float
    elasticity: float
    resid_std: float
    n_obs: int

    @classmethod
    def fit(cls, jackpots: list[float], sales: list[float]) -> "SalesModel":
        j = np.log(np.asarray(jackpots, dtype=float))
        s = np.log(np.asarray(sales, dtype=float))
        if len(j) < 3:
            raise ValueError("need at least 3 (jackpot, sales) observations")
        slope, intercept = np.polyfit(j, s, 1)
        resid = s - (intercept + slope * j)
        return cls(float(intercept), float(slope), float(resid.std(ddof=2)), len(j))

    def predict(self, jackpot: float) -> float:
        """Median sales forecast for an advertised jackpot."""
        return float(np.exp(self.intercept + self.elasticity * np.log(jackpot)))

    def predict_mean(self, jackpot: float) -> float:
        """Mean forecast under log-normal errors (median x exp(sigma^2/2))."""
        return self.predict(jackpot) * float(np.exp(self.resid_std**2 / 2))
