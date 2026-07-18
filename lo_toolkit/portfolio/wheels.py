"""Wheels: priced, guarantee-verified ticket bundles over a chosen pool.

A wheel never improves the probability of any individual line being drawn;
it trades stake for structured lower-tier coverage within the pool.  Reports
always state both.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb

from ..games.ruleset import Ruleset
from .covering import greedy_cover, verify_guarantee


@dataclass
class Wheel:
    rules: Ruleset
    pool: list[int]
    lines: list[tuple[int, ...]]
    guarantee_t: int

    @property
    def cost(self) -> float:
        return len(self.lines) * self.rules.ticket_price

    def pool_hit_probability(self, m: int) -> Fraction:
        """P(exactly m of the drawn numbers land inside the pool)."""
        n, d, v = self.rules.main.pool_size, self.rules.draw_size, len(self.pool)
        if m < 0 or m > min(v, d):
            return Fraction(0)
        return Fraction(comb(v, m) * comb(n - v, d - m), comb(n, d))

    def summary(self) -> str:
        t = self.guarantee_t
        p_t = float(sum(self.pool_hit_probability(m) for m in range(t, self.rules.draw_size + 1)))
        return "\n".join(
            [
                f"Wheel over pool {sorted(self.pool)} for {self.rules.name}",
                f"  lines: {len(self.lines)}  cost: {self.cost:.2f}",
                f"  guarantee: a {t}-match whenever >= {t} drawn numbers fall in the pool",
                f"  P(>= {t} drawn numbers in pool) = {p_t:.4%}",
                "  note: per-line draw odds are unchanged; the wheel only structures coverage",
            ]
        )


def build_wheel(rules: Ruleset, pool: list[int], guarantee_t: int) -> Wheel:
    """Build and verify a t-if-t covering wheel for the game's line size."""
    k = rules.main.picks
    pool = sorted(set(pool))
    if len(pool) < k:
        raise ValueError(f"pool must contain at least {k} numbers")
    if max(pool) > rules.main.pool_size or min(pool) < 1:
        raise ValueError("pool numbers outside game range")
    lines = greedy_cover(pool, k, guarantee_t)
    if not verify_guarantee(pool, lines, guarantee_t):
        raise AssertionError("constructed wheel failed guarantee verification")
    return Wheel(rules, pool, lines, guarantee_t)
