"""Audit battery for CARDS-family games (Chance-type: one symbol per position).

The right null here is multinomial, not hypergeometric: each draw is an
ordered tuple of independent uniform symbols.  Tests cover the failure
modes such a game can actually have:

* per-position uniformity  — a biased shoe/RNG for one suit
* position independence    — correlated suits within a draw
* serial dependence        — carry-over between consecutive draws
* full-outcome uniformity  — bias only visible at the whole-tuple level

All p-values are Monte Carlo-calibrated and pass through the same BH FDR
and split-half replication discipline as the k/N battery.
"""

from __future__ import annotations

import numpy as np

from ..games.ruleset import GameFamily, Ruleset
from .fdr import benjamini_hochberg
from .report import AuditResult, TestResult


def _position_chi2(draws: np.ndarray, symbols: int, pos: int) -> float:
    n = draws.shape[0]
    expected = n / symbols
    counts = np.bincount(draws[:, pos], minlength=symbols + 1)[1:]
    return float(((counts - expected) ** 2 / expected).sum())


def _pair_independence_chi2(draws: np.ndarray, symbols: int) -> float:
    """Sum over position pairs of contingency chi-square vs independence."""
    n, k = draws.shape
    total = 0.0
    expected = n / (symbols * symbols)
    for a in range(k):
        for b in range(a + 1, k):
            table = np.zeros((symbols, symbols))
            np.add.at(table, (draws[:, a] - 1, draws[:, b] - 1), 1)
            total += ((table - expected) ** 2 / expected).sum()
    return float(total)


def _serial_matches(draws: np.ndarray, symbols: int) -> float:
    """Total positions equal to the same position in the previous draw."""
    return float((draws[1:] == draws[:-1]).sum())


def _outcome_chi2(draws: np.ndarray, symbols: int) -> float:
    """Chi-square over all symbols^k whole-tuple outcomes."""
    n, k = draws.shape
    cells = symbols**k
    idx = np.zeros(n, dtype=np.int64)
    for pos in range(k):
        idx = idx * symbols + (draws[:, pos] - 1)
    expected = n / cells
    counts = np.bincount(idx, minlength=cells)
    return float(((counts - expected) ** 2 / expected).sum())


def _mc_pvalue(stat_fn, observed, symbols, k, n_sims, tail, rng) -> tuple[float, float]:
    obs = stat_fn(observed, symbols)
    n = observed.shape[0]
    null = np.empty(n_sims)
    for i in range(n_sims):
        sim = rng.integers(1, symbols + 1, size=(n, k))
        null[i] = stat_fn(sim, symbols)
    if tail == "upper":
        p = (1 + (null >= obs).sum()) / (n_sims + 1)
    else:
        hi = (1 + (null >= obs).sum()) / (n_sims + 1)
        lo = (1 + (null <= obs).sum()) / (n_sims + 1)
        p = min(1.0, 2 * min(hi, lo))
    return obs, float(p)


def run_cards_audit(
    rules: Ruleset,
    draws: np.ndarray,
    n_sims: int = 1000,
    alpha: float = 0.05,
    seed: int | None = 0,
    position_names: tuple[str, ...] | None = None,
) -> AuditResult:
    """Run the multinomial battery on an (n_draws, positions) rank matrix."""
    if rules.family != GameFamily.CARDS:
        raise ValueError("run_cards_audit is for CARDS-family games")
    if draws.shape[0] < 50:
        raise ValueError("need at least 50 draws for a meaningful audit")
    symbols, k = rules.symbols, rules.positions
    names = position_names or tuple(f"position_{i}" for i in range(k))

    tests: list[tuple[str, object, str]] = [
        (f"uniformity_{names[i]}", (lambda p: lambda d, s: _position_chi2(d, s, p))(i), "upper")
        for i in range(k)
    ]
    tests += [
        ("pair_independence", _pair_independence_chi2, "upper"),
        ("serial_dependence", _serial_matches, "two"),
        ("outcome_uniformity", _outcome_chi2, "upper"),
    ]

    half = draws.shape[0] // 2
    rng = np.random.default_rng(seed)
    results, pvals = [], []
    for name, fn, tail in tests:
        stat, p = _mc_pvalue(fn, draws, symbols, k, n_sims, tail, rng)
        _, p1 = _mc_pvalue(fn, draws[:half], symbols, k, n_sims, tail, rng)
        _, p2 = _mc_pvalue(fn, draws[half:], symbols, k, n_sims, tail, rng)
        results.append((name, stat, p, p1, p2))
        pvals.append(p)

    rejected, adjusted = benjamini_hochberg(pvals, alpha)
    return AuditResult(
        rules.game_id,
        draws.shape[0],
        [
            TestResult(name, stat, p, p_adj, rej, p1, p2)
            for (name, stat, p, p1, p2), p_adj, rej in zip(results, adjusted, rejected)
        ],
    )
