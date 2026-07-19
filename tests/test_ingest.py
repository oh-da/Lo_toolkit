"""Store roundtrip, validation, CSV import."""

from lo_toolkit.ingest.schema import Draw, SalesSnapshot, Store
from lo_toolkit.ingest.sources.csvfile import load_csv


def test_store_roundtrip(tmp_path):
    store = Store(tmp_path / "t.db")
    draws = [
        Draw("powerball", "2026-01-01", (5, 12, 23, 41, 60), bonus=7),
        Draw("powerball", "2026-01-04", (2, 9, 33, 48, 66), bonus=13),
    ]
    rep = store.upsert_draws(draws, pool_size=69)
    assert rep.inserted == 2 and not rep.rejected
    loaded = store.draws("powerball")
    assert [d.numbers for d in loaded] == [(5, 12, 23, 41, 60), (2, 9, 33, 48, 66)]
    assert loaded[0].bonus == 7


def test_store_idempotent_upsert(tmp_path):
    store = Store(tmp_path / "t.db")
    d = Draw("powerball", "2026-01-01", (5, 12, 23, 41, 60), bonus=7)
    store.upsert_draws([d], pool_size=69)
    store.upsert_draws([d], pool_size=69)     # re-ingest must not duplicate
    assert len(store.draws("powerball")) == 1


def test_store_rejects_invalid(tmp_path):
    store = Store(tmp_path / "t.db")
    bad = [
        Draw("powerball", "2026-01-01", (5, 5, 23, 41, 60)),      # duplicate
        Draw("powerball", "2026-01-04", (2, 9, 33, 48, 99)),      # out of range
    ]
    rep = store.upsert_draws(bad, pool_size=69)
    assert rep.inserted == 0
    assert len(rep.rejected) == 2


def test_sales_roundtrip(tmp_path):
    store = Store(tmp_path / "t.db")
    store.upsert_sales(
        [SalesSnapshot("powerball", "2026-01-01", 21e6, 300e6, 140e6)]
    )
    snaps = store.sales("powerball")
    assert snaps[0].jackpot == 300e6


def test_csv_import(tmp_path):
    p = tmp_path / "draws.csv"
    p.write_text(
        "date,numbers,bonus,multiplier\n"
        "2026-01-01,5 12 23 41 60,7,2\n"
        "2026-01-04,2-9-33-48-66,13,\n"
    )
    draws = load_csv(p, "powerball")
    assert draws[0].numbers == (5, 12, 23, 41, 60)
    assert draws[0].bonus == 7 and draws[0].multiplier == 2
    assert draws[1].numbers == (2, 9, 33, 48, 66)
    assert draws[1].multiplier is None
