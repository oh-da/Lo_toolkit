"""Chance (CARDS family): odds, parser, ordered storage, multinomial audit."""

import numpy as np
import pytest

from lo_toolkit.audit.cards import run_cards_audit
from lo_toolkit.ev.odds import jackpot_odds
from lo_toolkit.games.registry import CHANCE
from lo_toolkit.ingest.schema import Draw, Store
from lo_toolkit.ingest.sources.chance import SUITS, format_card_line, load_chance_csv

HEADER = "תאריך,הגרלה,תלתן,יהלום,לב,עלה,\n"


def test_odds():
    assert jackpot_odds(CHANCE) == pytest.approx(4096)


def test_parser(tmp_path):
    p = tmp_path / "Chance8.csv"
    p.write_text(
        HEADER + "19/07/2026,53251,8,K,Q,Q,\n18/07/2026,53250,10,K,K,K,\n",
        encoding="cp1255",
    )
    draws = load_chance_csv(p)
    assert len(draws) == 2
    assert draws[0].draw_date == "2026-07-18"          # oldest-first
    assert draws[0].numbers == (4, 7, 7, 7)            # 10,K,K,K
    assert draws[1].numbers == (2, 7, 6, 6)            # 8,K,Q,Q
    assert format_card_line(draws[1].numbers) == "8 of clubs, K of diamonds, Q of hearts, Q of spades"


def test_ordered_storage_preserves_position_and_repeats(tmp_path):
    store = Store(tmp_path / "t.db")
    d = Draw("chance_il", "2026-07-18", (4, 7, 7, 7))
    rep = store.upsert_draws([d], pool_size=8, ordered=True)
    assert rep.inserted == 1 and not rep.rejected
    assert store.draws("chance_il")[0].numbers == (4, 7, 7, 7)


def test_multiple_draws_per_day_are_kept(tmp_path):
    store = Store(tmp_path / "t.db")
    same_day = [
        Draw("chance_il", "2026-07-18", (4, 7, 7, 7), draw_no=53250),
        Draw("chance_il", "2026-07-18", (1, 8, 1, 5), draw_no=53249),
    ]
    store.upsert_draws(same_day, pool_size=8, ordered=True)
    loaded = store.draws("chance_il")
    assert len(loaded) == 2
    # ordered by draw_no within the day
    assert [d.draw_no for d in loaded] == [53249, 53250]


def test_unordered_storage_still_rejects_duplicates(tmp_path):
    store = Store(tmp_path / "t.db")
    rep = store.upsert_draws([Draw("x", "2026-01-01", (1, 1, 2))], pool_size=8)
    assert rep.inserted == 0


def test_fair_cards_history_passes():
    rng = np.random.default_rng(5)
    draws = rng.integers(1, 9, size=(3000, 4))
    result = run_cards_audit(CHANCE, draws, n_sims=300, seed=2, position_names=SUITS)
    assert result.anomalies == []


def test_rigged_suit_flagged():
    rng = np.random.default_rng(5)
    draws = rng.integers(1, 9, size=(3000, 4))
    # rig the hearts column: ace comes up twice as often as it should
    rig = rng.random(3000) < 0.12
    draws[rig, 2] = 8
    result = run_cards_audit(CHANCE, draws, n_sims=300, seed=2, position_names=SUITS)
    flagged = {t.name for t in result.anomalies}
    assert "uniformity_hearts" in flagged
    assert "uniformity_clubs" not in flagged


def test_correlated_suits_flagged():
    rng = np.random.default_rng(6)
    draws = rng.integers(1, 9, size=(3000, 4))
    # rig: diamonds copies clubs 20% of the time
    copy = rng.random(3000) < 0.2
    draws[copy, 1] = draws[copy, 0]
    result = run_cards_audit(CHANCE, draws, n_sims=300, seed=2, position_names=SUITS)
    flagged = {t.name for t in result.anomalies}
    assert "pair_independence" in flagged
