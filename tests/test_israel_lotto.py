"""Israeli Lotto: ruleset odds, Pais CSV parsing, era filtering, bonus audit."""

import numpy as np
import pytest

from lo_toolkit.audit.bonus import bonus_uniformity
from lo_toolkit.ev.odds import jackpot_odds, tier_probabilities
from lo_toolkit.games.registry import ISRAEL_LOTTO
from lo_toolkit.ingest.sources.israel_lotto import load_pais_csv

HEADER = "הגרלה,תאריך,1,2,3,4,5,6,המספר החזק/נוסף,x,y,\n"


def _write(tmp_path, rows):
    p = tmp_path / "Lotto.csv"
    p.write_text(HEADER + "".join(rows), encoding="cp1255")
    return p


def test_jackpot_odds():
    # C(37,6) * 7 = 2,324,784 * 7
    assert jackpot_odds(ISRAEL_LOTTO) == pytest.approx(16_273_488, abs=1)


def test_second_prize_odds():
    p = tier_probabilities(ISRAEL_LOTTO)
    # 6 without strong: C(37,6) * 7/6
    assert float(1 / p["6"]) == pytest.approx(2_324_784 * 7 / 6, rel=1e-9)


def test_parse_modern_rows(tmp_path):
    path = _write(
        tmp_path,
        [
            "3947,18/07/2026,05,06,08,31,32,36,4,0,0,\n"
            "3946,16/07/2026,02,14,21,26,27,28,2,0,0,\n"
        ],
    )
    draws = load_pais_csv(path)
    assert len(draws) == 2
    # oldest-first after parsing a newest-first file
    assert draws[0].draw_date == "2026-07-16"
    assert draws[1].numbers == (5, 6, 8, 31, 32, 36)
    assert draws[1].bonus == 4


def test_era_filter_stops_at_format_break(tmp_path):
    path = _write(
        tmp_path,
        [
            "3947,18/07/2026,05,06,08,31,32,36,4,0,0,\n"
            "2234,05/03/2011,01,02,03,04,05,06,7,0,0,\n"
            "2233,01/03/2011,13,20,25,28,33,35,8,0,0,\n"   # strong 8: old format
            "2232,26/02/2011,19,23,24,28,29,33,5,0,0,\n"
        ],
    )
    draws = load_pais_csv(path, modern_only=True)
    assert len(draws) == 2
    assert all(1 <= d.bonus <= 7 for d in draws)
    everything = load_pais_csv(path, modern_only=False)
    assert len(everything) == 4


def test_bonus_uniformity_fair_and_rigged():
    rng = np.random.default_rng(0)
    fair = rng.integers(1, 8, size=1500)
    _, p_fair = bonus_uniformity(fair, 7, n_sims=500, seed=1)
    assert p_fair > 0.05
    rigged = np.where(rng.random(1500) < 0.5, 7, rng.integers(1, 8, size=1500))
    _, p_rigged = bonus_uniformity(rigged, 7, n_sims=500, seed=1)
    assert p_rigged < 0.01
