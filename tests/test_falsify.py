"""Falsification lab: no heuristic beats the fair null on fair draws."""

import pytest

from lo_toolkit.falsify.models import HotColdModel, MarkovParityModel, NullModel
from lo_toolkit.falsify.walkforward import walk_forward
from lo_toolkit.sim.engine import DrawSimulator


def test_predictions_sum_to_draw_size(mini_game):
    draws = DrawSimulator(mini_game, seed=0).draw_main(200)
    for model in (NullModel(), HotColdModel(window=50), MarkovParityModel()):
        p = model.predict(draws, mini_game)
        assert p.shape == (20,)
        assert p.sum() == pytest.approx(5.0, rel=1e-9)
        assert (p > 0).all() and (p < 1).all()


def test_no_model_beats_null_on_fair_history(mini_game):
    draws = DrawSimulator(mini_game, seed=11).draw_main(400)
    result = walk_forward(
        [HotColdModel(window=50), MarkovParityModel()], draws, mini_game, warmup=100
    )
    assert result.n_scored == 300
    by_name = {s.name: s for s in result.scores}
    assert by_name["fair_null"].delta_vs_null == 0.0
    # On a fair game, heuristics must not materially beat the null.
    for name in ("hot_cold", "markov_parity"):
        assert by_name[name].delta_vs_null > -1e-3


def test_walk_forward_needs_enough_draws(mini_game):
    draws = DrawSimulator(mini_game, seed=0).draw_main(50)
    with pytest.raises(ValueError):
        walk_forward([HotColdModel()], draws, mini_game, warmup=100)
