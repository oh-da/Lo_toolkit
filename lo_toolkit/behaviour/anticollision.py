"""Anti-collision ticket generation.

Samples lines uniformly at random (preserving fair-draw probability), scores
each with the popularity surface, and keeps the least popular ones.  This
cannot change the chance a line is drawn — it lowers the expected number of
co-winners if it is.
"""

from __future__ import annotations

import random

from ..games.ruleset import Ruleset
from .popularity import PopularityModel


def generate_anticollision_lines(
    rules: Ruleset,
    n_lines: int,
    popularity: PopularityModel | None = None,
    candidate_multiple: int = 200,
    seed: int | None = None,
) -> list[tuple[int, ...]]:
    """Generate `n_lines` distinct low-popularity lines for the game."""
    pop = popularity or PopularityModel.default(rules.main.pool_size)
    rng = random.Random(seed)
    pool = range(1, rules.main.pool_size + 1)
    k = rules.main.picks

    candidates: set[tuple[int, ...]] = set()
    while len(candidates) < n_lines * candidate_multiple:
        candidates.add(tuple(sorted(rng.sample(pool, k))))
    scored = sorted(candidates, key=lambda l: pop.combination_factor(l, k))
    return scored[:n_lines]


def line_popularity_report(
    rules: Ruleset, lines: list[tuple[int, ...]], popularity: PopularityModel | None = None
) -> str:
    pop = popularity or PopularityModel.default(rules.main.pool_size)
    k = rules.main.picks
    rows = [f"{'line':30s} {'popularity vs uniform':>22s}"]
    for line in lines:
        f = pop.combination_factor(tuple(line), k)
        rows.append(f"{'-'.join(f'{n:02d}' for n in sorted(line)):30s} {f:>21.2f}x")
    rows.append(
        "\n(popularity < 1.0x means fewer expected co-winners than a quick pick;"
        "\n draw odds are identical for every line)"
    )
    return "\n".join(rows)
