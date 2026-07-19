"""Simulator validated against exact combinatorics."""

import numpy as np
import pytest

from lo_toolkit.ev.odds import match_probability
from lo_toolkit.games.registry import LOTTO_649, POWERBALL
from lo_toolkit.sim.engine import DrawSimulator, simulate_prizes


def test_draws_are_valid(mini_game):
    draws = DrawSimulator(mini_game, seed=0).draw_main(1000)
    assert draws.shape == (1000, 5)
    assert draws.min() >= 1 and draws.max() <= 20
    # sorted, distinct within each draw
    assert (np.diff(draws, axis=1) > 0).all()


def test_empirical_match_distribution(mini_game):
    """Frequency of m-matches against a fixed line agrees with exact odds."""
    sim = DrawSimulator(mini_game, seed=123)
    draws = sim.draw_main(200_000)
    line = np.array([1, 5, 9, 13, 17])
    hits = np.isin(draws, line).sum(axis=1)
    for m in (2, 3):
        exact = float(match_probability(mini_game, m))
        empirical = (hits == m).mean()
        assert empirical == pytest.approx(exact, rel=0.05)


def test_separate_pool_bonus_uniform():
    sim = DrawSimulator(POWERBALL, seed=5)
    main = sim.draw_main(20_000)
    bonus = sim.draw_bonus(main)
    assert bonus.min() >= 1 and bonus.max() <= 26
    counts = np.bincount(bonus, minlength=27)[1:]
    assert counts.std() / counts.mean() < 0.1


def test_from_remaining_bonus_excludes_main():
    sim = DrawSimulator(LOTTO_649, seed=5)
    main = sim.draw_main(500)
    bonus = sim.draw_bonus(main)
    for i in range(500):
        assert bonus[i] not in main[i]


def test_simulate_prizes_roi_close_to_exact(mini_game):
    """Fixed-tier return rate from simulation matches exact EV."""
    lines = [(1, 2, 3, 4, 5)]
    res = simulate_prizes(mini_game, lines, n_draws=300_000, jackpot_value=0, seed=9)
    exact_ev = sum(
        float(match_probability(mini_game, t.match_main)) * t.prize
        for t in mini_game.tiers
        if not t.is_jackpot
    )
    simulated_ev = res.total_return / res.n_draws
    assert simulated_ev == pytest.approx(exact_ev, rel=0.05)
