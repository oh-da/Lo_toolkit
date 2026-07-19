"""Multiple-testing control (Benjamini–Hochberg)."""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05):
    """Returns (rejected_flags, adjusted_pvalues) under BH FDR control."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / (np.arange(m) + 1)
    # enforce monotonicity from the largest p downwards
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adj
    rejected = adjusted <= alpha
    return rejected.tolist(), adjusted.tolist()
