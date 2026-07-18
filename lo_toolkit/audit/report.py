"""Full audit battery with FDR control and split replication.

The expected outcome on a well-run lottery is *no anomaly*.  A finding is
only flagged as replicated when it is BH-significant on the full sample AND
nominally significant in both independent halves of the history — the
research document's guard against window cherry-picking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..games.ruleset import GameFamily, Ruleset
from .fdr import benjamini_hochberg
from .montecarlo import mc_pvalue
from .stats import ALL_TESTS, TAILS


@dataclass
class TestResult:
    name: str
    statistic: float
    pvalue: float
    p_adjusted: float
    significant: bool
    p_first_half: float
    p_second_half: float

    @property
    def replicated(self) -> bool:
        return self.significant and self.p_first_half < 0.05 and self.p_second_half < 0.05


@dataclass
class AuditResult:
    game_id: str
    n_draws: int
    tests: list[TestResult]

    @property
    def anomalies(self) -> list[TestResult]:
        return [t for t in self.tests if t.replicated]

    def summary(self) -> str:
        lines = [
            f"Fairness audit: {self.game_id} ({self.n_draws} draws, "
            f"MC-calibrated, BH FDR 5%)",
            f"{'test':26s} {'stat':>12s} {'p':>8s} {'p_adj':>8s} "
            f"{'p_h1':>8s} {'p_h2':>8s}  verdict",
        ]
        for t in self.tests:
            verdict = (
                "ANOMALY (replicated)"
                if t.replicated
                else ("significant, not replicated" if t.significant else "consistent with fair")
            )
            lines.append(
                f"{t.name:26s} {t.statistic:12.3f} {t.pvalue:8.4f} "
                f"{t.p_adjusted:8.4f} {t.p_first_half:8.4f} {t.p_second_half:8.4f}  {verdict}"
            )
        if not self.anomalies:
            lines.append(
                "\nNo replicated anomaly: the history is consistent with a fair "
                "game. No number-selection strategy can improve draw odds."
            )
        else:
            lines.append(
                "\nReplicated anomalies found. Verify against operator/regulator "
                "procedures and rule changes before assuming exploitability."
            )
        return "\n".join(lines)


def run_audit(
    rules: Ruleset,
    draws: np.ndarray,
    n_sims: int = 2000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> AuditResult:
    """Run the full battery on an (n_draws, d) matrix of main numbers."""
    if rules.family == GameFamily.DIGITS:
        raise ValueError("digit-game audits need the multinomial variant (not yet built)")
    if draws.shape[0] < 50:
        raise ValueError("need at least 50 draws for a meaningful audit")

    half = draws.shape[0] // 2
    results = []
    pvals = []
    for name, fn in ALL_TESTS.items():
        tail = TAILS[name]
        stat, p = mc_pvalue(fn, draws, rules, n_sims=n_sims, tail=tail, seed=seed)
        _, p1 = mc_pvalue(fn, draws[:half], rules, n_sims=n_sims, tail=tail, seed=seed)
        _, p2 = mc_pvalue(fn, draws[half:], rules, n_sims=n_sims, tail=tail, seed=seed)
        results.append((name, stat, p, p1, p2))
        pvals.append(p)

    rejected, adjusted = benjamini_hochberg(pvals, alpha)
    tests = [
        TestResult(name, stat, p, p_adj, rej, p1, p2)
        for (name, stat, p, p1, p2), p_adj, rej in zip(results, adjusted, rejected)
    ]
    return AuditResult(rules.game_id, draws.shape[0], tests)
