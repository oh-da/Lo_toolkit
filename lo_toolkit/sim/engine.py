"""Vectorized Monte Carlo draw simulation.

Used for audit p-value calibration, prize-distribution simulation, and
bankroll paths.  Simulation is always driven by the ruleset so the null
distribution matches the exact game law.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..games.ruleset import BonusMode, GameFamily, Ruleset


class DrawSimulator:
    """Simulates draws for a ruleset as numpy arrays of sorted numbers."""

    def __init__(self, rules: Ruleset, seed: int | None = None):
        self.rules = rules
        self.rng = np.random.default_rng(seed)

    def draw_main(self, n_draws: int) -> np.ndarray:
        """(n_draws, draw_size) array of sorted main numbers in 1..N."""
        r = self.rules
        if r.family == GameFamily.DIGITS:
            return self.rng.integers(0, 10, size=(n_draws, r.digit_count))
        n, d = r.main.pool_size, r.draw_size
        # Vectorized sampling without replacement via argpartition of random keys.
        keys = self.rng.random((n_draws, n))
        idx = np.argpartition(keys, d, axis=1)[:, :d]
        return np.sort(idx + 1, axis=1)

    def draw_bonus(self, main: np.ndarray) -> np.ndarray | None:
        """Bonus number per draw, consistent with the game's bonus mode."""
        r = self.rules
        if r.bonus_mode == BonusMode.NONE:
            return None
        n_draws = main.shape[0]
        if r.bonus_mode == BonusMode.SEPARATE_POOL:
            return self.rng.integers(1, r.bonus_pool_size + 1, size=n_draws)
        # FROM_REMAINING: uniform over numbers not in the main draw.
        n = r.main.pool_size
        out = np.empty(n_draws, dtype=np.int64)
        for i in range(n_draws):
            pool = np.setdiff1d(np.arange(1, n + 1), main[i], assume_unique=True)
            out[i] = self.rng.choice(pool)
        return out


@dataclass
class PrizeSimResult:
    n_draws: int
    total_stake: float
    total_return: float
    tier_hits: dict[str, int]

    @property
    def roi(self) -> float:
        return (self.total_return - self.total_stake) / self.total_stake


def simulate_prizes(
    rules: Ruleset,
    lines: list[tuple[int, ...]],
    n_draws: int,
    jackpot_value: float = 0.0,
    seed: int | None = None,
) -> PrizeSimResult:
    """Simulate playing `lines` for `n_draws` draws and tally prize returns.

    Bonus-dependent tiers are handled by simulating the bonus per draw; each
    line's bonus pick is assumed uniform (bonus choice does not affect EV).
    """
    if rules.family == GameFamily.DIGITS:
        raise ValueError("use exact odds for digit games; simulation adds nothing")
    sim = DrawSimulator(rules, seed=seed)
    main = sim.draw_main(n_draws)
    bonus = sim.draw_bonus(main)
    rng = sim.rng

    line_arr = np.array([sorted(l) for l in lines])           # (L, k)
    # hits[i, j] = how many of line j's numbers appear in draw i
    hits = np.zeros((n_draws, len(lines)), dtype=np.int64)
    for j, line in enumerate(line_arr):
        hits[:, j] = np.isin(main, line).sum(axis=1)

    if rules.bonus_mode == BonusMode.SEPARATE_POOL:
        line_bonus = rng.integers(1, rules.bonus_pool_size + 1, size=len(lines))
        bonus_hit = bonus[:, None] == line_bonus[None, :]
    elif rules.bonus_mode == BonusMode.FROM_REMAINING:
        bonus_hit = np.zeros_like(hits, dtype=bool)
        for j, line in enumerate(line_arr):
            bonus_hit[:, j] = np.isin(bonus, line)
    else:
        bonus_hit = np.zeros_like(hits, dtype=bool)

    tier_hits: dict[str, int] = {t.name: 0 for t in rules.tiers}
    total_return = 0.0
    for t in rules.tiers:
        mask = hits == t.match_main
        if t.match_bonus is True:
            mask &= bonus_hit
        elif t.match_bonus is False:
            mask &= ~bonus_hit
        count = int(mask.sum())
        tier_hits[t.name] = count
        value = jackpot_value if t.is_jackpot else t.prize
        total_return += count * value

    stake = rules.ticket_price * len(lines) * n_draws
    return PrizeSimResult(n_draws, stake, total_return, tier_hits)
