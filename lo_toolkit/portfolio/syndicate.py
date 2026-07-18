"""Syndicate modelling: pooled play with documented share accounting.

Buying more unique lines scales the group's absolute win probability
linearly with spend; per-ticket EV is unchanged (before fees).  The value
is variance-sharing — which the simulation makes visible.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..games.ruleset import Ruleset
from ..sim.engine import simulate_prizes


@dataclass
class SyndicateResult:
    n_members: int
    n_lines: int
    n_draws: int
    group_stake: float
    group_return: float
    tier_hits: dict[str, int]

    @property
    def per_member_stake(self) -> float:
        return self.group_stake / self.n_members

    @property
    def per_member_return(self) -> float:
        return self.group_return / self.n_members

    def summary(self) -> str:
        roi = (self.group_return - self.group_stake) / self.group_stake
        hits = ", ".join(f"{k}:{v}" for k, v in self.tier_hits.items() if v)
        return "\n".join(
            [
                f"Syndicate: {self.n_members} members, {self.n_lines} lines, "
                f"{self.n_draws} simulated draws",
                f"  group stake {self.group_stake:,.0f}, return {self.group_return:,.0f}"
                f" (ROI {roi:+.1%})",
                f"  per member: stake {self.per_member_stake:,.2f},"
                f" return {self.per_member_return:,.2f}",
                f"  tier hits: {hits or 'none'}",
            ]
        )


def simulate_syndicate(
    rules: Ruleset,
    lines: list[tuple[int, ...]],
    n_members: int,
    n_draws: int,
    jackpot_value: float,
    seed: int | None = None,
) -> SyndicateResult:
    res = simulate_prizes(rules, lines, n_draws, jackpot_value=jackpot_value, seed=seed)
    return SyndicateResult(
        n_members=n_members,
        n_lines=len(lines),
        n_draws=n_draws,
        group_stake=res.total_stake,
        group_return=res.total_return,
        tier_hits=res.tier_hits,
    )
