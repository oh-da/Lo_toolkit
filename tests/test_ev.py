"""EV engine: sharing factor, EV breakdown, sales model, roll-downs."""

import math

import pytest

from lo_toolkit.ev.collision import expected_cowinners
from lo_toolkit.ev.evcalc import expected_value, sharing_factor
from lo_toolkit.ev.rolldown import rolldown_ev
from lo_toolkit.ev.sales import SalesModel
from lo_toolkit.games.registry import POWERBALL


def test_sharing_factor_limits():
    assert sharing_factor(0.0) == 1.0
    assert sharing_factor(1e-12) == pytest.approx(1.0, abs=1e-6)
    # heavy collision: with lam co-winners expected you keep ~1/lam
    assert sharing_factor(50.0) == pytest.approx(1 / 50, rel=0.01)


def test_sharing_factor_matches_poisson_expectation():
    lam = 1.7
    exact = sum(
        (1 / (1 + k)) * math.exp(-lam) * lam**k / math.factorial(k) for k in range(60)
    )
    assert sharing_factor(lam) == pytest.approx(exact, rel=1e-9)


def test_ev_grows_with_jackpot():
    lo = expected_value(POWERBALL, 100e6, 20e6)
    hi = expected_value(POWERBALL, 500e6, 20e6)
    assert hi.ev_total > lo.ev_total
    assert lo.ev_net < 0  # normal draws are negative EV


def test_ev_sharing_hurts_at_high_sales():
    quiet = expected_value(POWERBALL, 500e6, 10e6)
    frenzy = expected_value(POWERBALL, 500e6, 300e6)
    assert frenzy.expected_share < quiet.expected_share
    assert frenzy.ev_jackpot < quiet.ev_jackpot


def test_anticollision_popularity_lowers_lambda():
    ev_qp = expected_value(POWERBALL, 800e6, 200e6, popularity_factor=1.0)
    ev_ac = expected_value(POWERBALL, 800e6, 200e6, popularity_factor=0.5)
    assert ev_ac.expected_share > ev_qp.expected_share


def test_sales_model_recovers_elasticity():
    # synthetic: sales = 1000 * jackpot^1.5
    jackpots = [20e6, 50e6, 100e6, 200e6, 400e6, 800e6]
    sales = [1000 * j**1.5 for j in jackpots]
    m = SalesModel.fit(jackpots, sales)
    assert m.elasticity == pytest.approx(1.5, rel=1e-6)
    assert m.predict(300e6) == pytest.approx(1000 * (300e6) ** 1.5, rel=1e-6)


def test_expected_cowinners_scales_with_sales():
    line = (5, 10, 23, 33, 60)
    low = expected_cowinners(POWERBALL, line, 10e6)
    high = expected_cowinners(POWERBALL, line, 100e6)
    assert high == pytest.approx(10 * low, rel=1e-9)


def test_popular_line_has_more_cowinners():
    birthday = (7, 14, 21, 28, 31)     # all <= 31, includes lucky 7
    unpopular = (38, 43, 54, 61, 68)
    n = 100e6
    assert expected_cowinners(POWERBALL, birthday, n) > expected_cowinners(
        POWERBALL, unpopular, n
    )


def test_rolldown_can_flip_ev_positive():
    """Cash WinFall economics: a rolled-down pool at modest sales beats price."""
    ev = rolldown_ev(
        POWERBALL,
        rolldown_pool=2_000_000,
        tier_allocation={"4+0": 0.3, "3+0": 0.7},
        tickets_sold=500_000,
    )
    assert ev["net"] > 0
    baseline = rolldown_ev(POWERBALL, 0.0, {}, 500_000)
    assert baseline["net"] < 0
