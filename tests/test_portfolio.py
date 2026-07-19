"""Covering designs, wheels, syndicate simulation."""

from fractions import Fraction
from itertools import combinations

import pytest

from lo_toolkit.games.registry import LOTTO_649
from lo_toolkit.portfolio.covering import greedy_cover, verify_guarantee
from lo_toolkit.portfolio.syndicate import simulate_syndicate
from lo_toolkit.portfolio.wheels import build_wheel


def test_greedy_cover_produces_valid_design():
    pool = list(range(1, 10))                 # v=9
    lines = greedy_cover(pool, k=6, t=4)
    assert verify_guarantee(pool, lines, t=4)
    # every line is a valid 6-subset of the pool
    for line in lines:
        assert len(set(line)) == 6 and set(line) <= set(pool)


def test_greedy_cover_trivial_case():
    # v == k: single line covers everything
    pool = [3, 7, 11, 19, 22, 41]
    lines = greedy_cover(pool, k=6, t=3)
    assert lines == [tuple(sorted(pool))]


def test_verify_guarantee_detects_gap():
    pool = [1, 2, 3, 4, 5, 6, 7]
    # a single line cannot 3-if-3 cover a 7-number pool
    assert not verify_guarantee(pool, [(1, 2, 3, 4, 5, 6)], t=3)


def test_build_wheel_649():
    pool = [4, 9, 17, 23, 32, 38, 41, 45]
    wheel = build_wheel(LOTTO_649, pool, guarantee_t=4)
    assert verify_guarantee(pool, wheel.lines, t=4)
    assert wheel.cost == len(wheel.lines) * LOTTO_649.ticket_price
    # pool-hit distribution is a proper probability distribution
    total = sum(
        wheel.pool_hit_probability(m) for m in range(0, LOTTO_649.draw_size + 1)
    )
    assert total == Fraction(1)


def test_build_wheel_rejects_bad_pool():
    with pytest.raises(ValueError):
        build_wheel(LOTTO_649, [1, 2, 3], guarantee_t=3)     # too small
    with pytest.raises(ValueError):
        build_wheel(LOTTO_649, [1, 2, 3, 4, 5, 99], guarantee_t=3)  # out of range


def test_wheel_guarantee_holds_in_simulation(mini_game):
    """When >= t drawn numbers land in the pool, some line matches >= t."""
    import numpy as np

    from lo_toolkit.sim.engine import DrawSimulator

    pool = [2, 5, 8, 11, 14, 17, 20]
    lines = greedy_cover(pool, k=mini_game.main.picks, t=3)
    draws = DrawSimulator(mini_game, seed=3).draw_main(2000)
    pool_set = set(pool)
    for draw in draws:
        in_pool = pool_set & set(draw.tolist())
        if len(in_pool) >= 3:
            best = max(len(set(l) & set(draw.tolist())) for l in lines)
            assert best >= 3


def test_syndicate_accounting(mini_game):
    lines = [(1, 2, 3, 4, 5), (6, 7, 8, 9, 10), (11, 12, 13, 14, 15)]
    res = simulate_syndicate(
        mini_game, lines, n_members=10, n_draws=5000, jackpot_value=10_000, seed=0
    )
    assert res.group_stake == pytest.approx(3 * 1.0 * 5000)
    assert res.per_member_stake == pytest.approx(res.group_stake / 10)
    assert res.per_member_return == pytest.approx(res.group_return / 10)
