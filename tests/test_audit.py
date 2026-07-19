"""Audit suite: no anomalies on fair histories, catches a rigged game."""

import numpy as np
import pytest

from lo_toolkit.audit.fdr import benjamini_hochberg
from lo_toolkit.audit.nullmodel import ball_count_moments, ball_counts
from lo_toolkit.audit.report import run_audit
from lo_toolkit.sim.engine import DrawSimulator


def rigged_draws(rules, n_draws, bias_strength=3.0, seed=1):
    """Simulate a defective machine that favours low numbers."""
    rng = np.random.default_rng(seed)
    n, d = rules.main.pool_size, rules.draw_size
    weights = np.linspace(bias_strength, 1.0, n)
    weights /= weights.sum()
    draws = np.empty((n_draws, d), dtype=np.int64)
    for i in range(n_draws):
        draws[i] = np.sort(
            rng.choice(np.arange(1, n + 1), size=d, replace=False, p=weights)
        )
    return draws


def test_fair_history_passes(mini_game):
    draws = DrawSimulator(mini_game, seed=42).draw_main(400)
    result = run_audit(mini_game, draws, n_sims=300, seed=7)
    assert result.anomalies == []


def test_rigged_history_flagged(mini_game):
    draws = rigged_draws(mini_game, 400)
    result = run_audit(mini_game, draws, n_sims=300, seed=7)
    flagged = {t.name for t in result.anomalies}
    assert "marginal_frequency" in flagged


def test_ball_count_moments(mini_game):
    mean, var = ball_count_moments(mini_game, 100)
    assert mean == pytest.approx(25.0)      # 100 * 5/20
    assert var == pytest.approx(18.75)      # 100 * .25 * .75


def test_ball_counts_shape(mini_game):
    draws = DrawSimulator(mini_game, seed=0).draw_main(50)
    counts = ball_counts(draws, mini_game.main.pool_size)
    assert counts.shape == (20,)
    assert counts.sum() == 50 * 5


def test_benjamini_hochberg_known_case():
    pvals = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    rejected, adjusted = benjamini_hochberg(pvals, alpha=0.05)
    # classic BH example: first two rejected at FDR 5%
    assert rejected[0] and rejected[1]
    assert not any(rejected[5:])
    assert adjusted == sorted(adjusted)


def test_audit_requires_min_draws(mini_game):
    draws = DrawSimulator(mini_game, seed=0).draw_main(10)
    with pytest.raises(ValueError):
        run_audit(mini_game, draws)
