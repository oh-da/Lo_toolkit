"""Exact tier probabilities derived from a ruleset.

All probabilities are computed as exact rational numbers (``fractions.Fraction``)
from the hypergeometric law of the game, then exposed as floats where handy.
These are the ground truth every other module (EV, simulation checks, audit
null models) is validated against.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb

from ..games.ruleset import BonusMode, GameFamily, PrizeTier, Ruleset


def match_probability(rules: Ruleset, m: int) -> Fraction:
    """P(exactly m of the player's main numbers are drawn)."""
    if rules.family == GameFamily.DIGITS:
        raise ValueError("digit games have no hypergeometric main field")
    k = rules.main.picks          # player's picks
    d = rules.draw_size           # operator's draw size
    n = rules.main.pool_size
    if m < 0 or m > min(k, d):
        return Fraction(0)
    return Fraction(comb(k, m) * comb(n - k, d - m), comb(n, d))


def bonus_probability(rules: Ruleset, m: int) -> Fraction:
    """P(bonus matched | m main numbers matched)."""
    if rules.bonus_mode == BonusMode.NONE:
        return Fraction(0)
    if rules.bonus_mode == BonusMode.SEPARATE_POOL:
        return Fraction(1, rules.bonus_pool_size)
    # FROM_REMAINING: bonus is one of the N - d numbers not in the main draw;
    # the player holds k - m of those.
    remaining = rules.main.pool_size - rules.draw_size
    return Fraction(rules.main.picks - m, remaining)


def tier_probability(rules: Ruleset, tier: PrizeTier) -> Fraction:
    """Exact probability of a single line hitting `tier`."""
    if rules.family == GameFamily.DIGITS:
        if tier.match_main != rules.digit_count:
            raise ValueError("only straight tiers supported for digit games")
        return Fraction(1, 10**rules.digit_count)
    p_main = match_probability(rules, tier.match_main)
    if tier.match_bonus is None:
        return p_main
    pb = bonus_probability(rules, tier.match_main)
    return p_main * (pb if tier.match_bonus else (1 - pb))


def tier_probabilities(rules: Ruleset) -> dict[str, Fraction]:
    """Probability of each prize tier, keyed by tier name."""
    return {t.name: tier_probability(rules, t) for t in rules.tiers}


def jackpot_odds(rules: Ruleset) -> float:
    """'1 in X' odds of the top (jackpot or first-listed) tier."""
    top = next((t for t in rules.tiers if t.is_jackpot), rules.tiers[0])
    p = tier_probability(rules, top)
    return float(1 / p)


def any_prize_probability(rules: Ruleset) -> Fraction:
    """P(a line wins any listed tier).  Tiers must be mutually exclusive."""
    return sum((tier_probability(rules, t) for t in rules.tiers), Fraction(0))
