"""Odds engine validated against officially published odds."""

from fractions import Fraction

import pytest

from lo_toolkit.ev.odds import (
    any_prize_probability,
    jackpot_odds,
    match_probability,
    tier_probabilities,
)
from lo_toolkit.games.registry import KENO_10, LOTTO_649, MEGA_MILLIONS, PICK3, POWERBALL


def test_powerball_jackpot_odds_official():
    # Official: 1 in 292,201,338 (C(69,5) * 26)
    assert jackpot_odds(POWERBALL) == pytest.approx(292_201_338, abs=1)


def test_powerball_lower_tiers_official():
    p = {k: float(1 / v) for k, v in tier_probabilities(POWERBALL).items()}
    # Official prize-chart odds
    assert p["5+0"] == pytest.approx(11_688_053.52, rel=1e-6)
    assert p["4+PB"] == pytest.approx(913_129.18, rel=1e-6)
    assert p["4+0"] == pytest.approx(36_525.17, rel=1e-6)
    assert p["3+PB"] == pytest.approx(14_494.11, rel=1e-6)
    assert p["3+0"] == pytest.approx(579.76, rel=1e-4)
    assert p["2+PB"] == pytest.approx(701.33, rel=1e-4)
    assert p["1+PB"] == pytest.approx(91.98, rel=1e-4)
    assert p["0+PB"] == pytest.approx(38.32, rel=2e-4)


def test_megamillions_jackpot_odds_official():
    # 2025 rules: 1 in 290,472,336 (C(70,5) * 24)
    assert jackpot_odds(MEGA_MILLIONS) == pytest.approx(290_472_336, abs=1)


def test_lotto649_official():
    assert jackpot_odds(LOTTO_649) == pytest.approx(13_983_816, abs=1)
    p = tier_probabilities(LOTTO_649)
    # Official 5/6+bonus odds: 1 in 2,330,636
    assert float(1 / p["5/6+B"]) == pytest.approx(2_330_636, rel=1e-6)


def test_keno_10spot_official():
    # Official 10-spot solid-10 odds: 1 in 8,911,711.18
    assert jackpot_odds(KENO_10) == pytest.approx(8_911_711.18, rel=1e-6)


def test_pick3_straight():
    assert jackpot_odds(PICK3) == pytest.approx(1000)


def test_match_probabilities_sum_to_one(mini_game):
    total = sum(match_probability(mini_game, m) for m in range(0, 6))
    assert total == Fraction(1)


def test_any_prize_probability_bounds():
    p = float(any_prize_probability(POWERBALL))
    # Official overall odds of winning any Powerball prize: about 1 in 24.87
    assert 1 / p == pytest.approx(24.87, rel=1e-3)
