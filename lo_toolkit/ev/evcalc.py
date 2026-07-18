"""Net expected value per line, including jackpot-sharing risk.

The number of *other* jackpot winners is modelled as Poisson with rate
lambda = (other tickets sold) x P(jackpot) x popularity factor.  Conditional
on winning, the expected share of the jackpot is E[1/(1+K)] which has the
closed form (1 - exp(-lambda)) / lambda for Poisson K.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..games.ruleset import Ruleset
from .odds import tier_probability


def sharing_factor(lam: float) -> float:
    """E[1 / (1 + K)] with K ~ Poisson(lam): expected fraction of the jackpot
    you keep, given that you won."""
    if lam <= 0:
        return 1.0
    return -math.expm1(-lam) / lam


@dataclass
class EVBreakdown:
    ev_fixed_tiers: float
    ev_jackpot: float
    ticket_price: float
    p_jackpot: float
    expected_share: float

    @property
    def ev_total(self) -> float:
        return self.ev_fixed_tiers + self.ev_jackpot

    @property
    def ev_net(self) -> float:
        return self.ev_total - self.ticket_price

    def summary(self) -> str:
        return (
            f"EV per line: {self.ev_total:.4f} (fixed tiers {self.ev_fixed_tiers:.4f}"
            f" + jackpot {self.ev_jackpot:.4f}), price {self.ticket_price:.2f},"
            f" net {self.ev_net:+.4f}; P(jackpot)=1 in {1 / self.p_jackpot:,.0f},"
            f" expected jackpot share if won: {self.expected_share:.1%}"
        )


def expected_value(
    rules: Ruleset,
    jackpot_cash: float,
    tickets_sold: float,
    popularity_factor: float = 1.0,
    parimutuel_values: dict[str, float] | None = None,
) -> EVBreakdown:
    """EV of one line given the current jackpot (cash value) and sales.

    `popularity_factor` scales collision risk for the specific line played:
    1.0 for a quick pick, <1.0 for an anti-collision line, >1.0 for popular
    (birthday-shaped) lines.  Parimutuel tiers use `parimutuel_values` or the
    ruleset's placeholder amounts and are treated as fixed (their own
    sharing dynamics are second-order for single-line EV).
    """
    ev_fixed = 0.0
    ev_jackpot = 0.0
    p_jackpot = 0.0
    share = 1.0
    for tier in rules.tiers:
        p = float(tier_probability(rules, tier))
        if tier.is_jackpot:
            p_jackpot = p
            lam = max(tickets_sold - 1, 0.0) * p * popularity_factor
            share = sharing_factor(lam)
            ev_jackpot = p * jackpot_cash * share
        elif tier.is_parimutuel and parimutuel_values and tier.name in parimutuel_values:
            ev_fixed += p * parimutuel_values[tier.name]
        else:
            ev_fixed += p * tier.prize
    return EVBreakdown(ev_fixed, ev_jackpot, rules.ticket_price, p_jackpot, share)
