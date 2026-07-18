"""Covering designs C(v, k, t) via greedy construction.

A covering design is a set of k-number lines over a v-number pool such that
every t-subset of the pool appears in at least one line.  It guarantees a
t-match whenever t of the drawn numbers land in your pool ("t-if-t").
Greedy construction is within a log factor of optimal and is exact enough
for practical wheel sizes (v <= ~20).  For record-size designs, import
lines from the covering-design repositories instead.
"""

from __future__ import annotations

from itertools import combinations


def greedy_cover(pool: list[int], k: int, t: int) -> list[tuple[int, ...]]:
    """Greedy C(v,k,t): lines of size k covering all t-subsets of `pool`."""
    v = len(pool)
    if not t <= k <= v:
        raise ValueError("need t <= k <= len(pool)")
    if v > 22:
        raise ValueError("greedy solver capped at pool size 22; import known designs")

    uncovered: set[tuple[int, ...]] = set(combinations(sorted(pool), t))
    candidates = [tuple(c) for c in combinations(sorted(pool), k)]
    lines: list[tuple[int, ...]] = []
    while uncovered:
        best, best_gain = None, -1
        for cand in candidates:
            gain = sum(1 for sub in combinations(cand, t) if sub in uncovered)
            if gain > best_gain:
                best, best_gain = cand, gain
        assert best is not None
        lines.append(best)
        for sub in combinations(best, t):
            uncovered.discard(sub)
    return lines


def verify_guarantee(
    pool: list[int], lines: list[tuple[int, ...]], t: int, m: int | None = None
) -> bool:
    """Check the 't-if-m' guarantee: whenever m drawn numbers fall in the
    pool, at least one line matches >= t of them.  Default m = t."""
    m = t if m is None else m
    line_sets = [set(l) for l in lines]
    for scenario in combinations(sorted(pool), m):
        s = set(scenario)
        if not any(len(ls & s) >= t for ls in line_sets):
            return False
    return True
