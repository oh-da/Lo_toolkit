"""Jackpot-collision (sharing) estimation.

Splits the ticket pool into quick picks (uniform over combinations) and
manual picks (weighted by the popularity surface).  The expected number of
co-winners on a given line is the sum of both segments' collision rates.
"""

from __future__ import annotations

from ..behaviour.popularity import PopularityModel
from ..games.ruleset import Ruleset
from .odds import tier_probability


def expected_cowinners(
    rules: Ruleset,
    line: tuple[int, ...],
    tickets_sold: float,
    quick_pick_share: float = 0.7,
    popularity: PopularityModel | None = None,
) -> float:
    """Expected number of OTHER tickets holding the same jackpot-winning line."""
    top = next((t for t in rules.tiers if t.is_jackpot), rules.tiers[0])
    p = float(tier_probability(rules, top))
    pop = popularity or PopularityModel.default(rules.main.pool_size)
    factor = pop.combination_factor(line, rules.main.picks)
    qp = tickets_sold * quick_pick_share * p
    manual = tickets_sold * (1 - quick_pick_share) * p * factor
    return qp + manual
