"""Popularity surface and anti-collision generation."""

from lo_toolkit.behaviour.anticollision import generate_anticollision_lines
from lo_toolkit.behaviour.popularity import PopularityModel
from lo_toolkit.games.registry import POWERBALL


def test_default_weights_shape_and_norm():
    pop = PopularityModel.default(69)
    assert len(pop.weights) == 69
    assert abs(pop.weights.mean() - 1.0) < 1e-9


def test_literature_ordering():
    pop = PopularityModel.default(69)
    assert pop.weight(7) > pop.weight(40)        # lucky 7 over-picked
    assert pop.weight(12) > pop.weight(45)       # birthday range over-picked
    assert pop.weight(45) < 1.0                  # above-31 under-picked


def test_pattern_multipliers():
    pop = PopularityModel.default(49)
    arithmetic = (7, 14, 21, 28, 35, 42)
    plain = (2, 9, 24, 33, 41, 47)
    assert pop.combination_factor(arithmetic, 6) > pop.combination_factor(plain, 6)


def test_anticollision_lines_are_unpopular_and_valid():
    lines = generate_anticollision_lines(POWERBALL, 5, seed=1)
    pop = PopularityModel.default(69)
    assert len(lines) == 5
    for line in lines:
        assert len(set(line)) == 5
        assert all(1 <= n <= 69 for n in line)
        assert pop.combination_factor(line, 5) < 1.0


def test_generation_is_deterministic_with_seed():
    a = generate_anticollision_lines(POWERBALL, 3, seed=42)
    b = generate_anticollision_lines(POWERBALL, 3, seed=42)
    assert a == b
