"""Roll-down / mandatory-payout scenario EV.

In a roll-down (Cash WinFall-style), when the jackpot is not won under
qualifying conditions its value is distributed to lower tiers, multiplying
their prizes.  This is the canonical *legal* positive-EV situation: the
improved paytable is published and available to every player.
"""

from __future__ import annotations

from ..games.ruleset import Ruleset
from .odds import tier_probability


def rolldown_ev(
    rules: Ruleset,
    rolldown_pool: float,
    tier_allocation: dict[str, float],
    tickets_sold: float,
) -> dict[str, float]:
    """EV per line under a roll-down event.

    `tier_allocation` maps tier name -> fraction of `rolldown_pool` allocated
    to that tier.  Each tier's pool is split parimutuelly among expected
    winners (sales x tier probability), which is what makes roll-downs
    lucrative at low sales.  Returns per-tier EV plus 'total' and 'net'.
    """
    out: dict[str, float] = {}
    total = 0.0
    for tier in rules.tiers:
        if tier.is_jackpot:
            continue
        p = float(tier_probability(rules, tier))
        ev = p * tier.prize
        alloc = tier_allocation.get(tier.name, 0.0)
        if alloc > 0:
            # Parimutuel roll-down: P(win) x pool / E[winners] with
            # E[winners] = tickets_sold x p collapses to pool / tickets_sold.
            ev += (rolldown_pool * alloc) / tickets_sold
        out[tier.name] = ev
        total += ev
    out["total"] = total
    out["net"] = total - rules.ticket_price
    return out
