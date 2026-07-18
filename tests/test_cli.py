"""CLI smoke tests."""

import pytest

from lo_toolkit.cli import main


def test_games(capsys):
    main(["games"])
    out = capsys.readouterr().out
    assert "powerball" in out and "292,201,338" in out


def test_odds(capsys):
    main(["odds", "powerball"])
    out = capsys.readouterr().out
    assert "5+PB" in out


def test_ev(capsys):
    main(["ev", "powerball", "--jackpot-cash", "200000000", "--sales", "20000000"])
    out = capsys.readouterr().out
    assert "EV per line" in out


def test_tickets(capsys):
    main(["tickets", "powerball", "-n", "3", "--seed", "1"])
    out = capsys.readouterr().out
    assert "anti-collision" in out


def test_audit_simulated(capsys):
    main(["audit", "lotto649", "--simulate", "200", "--sims", "100"])
    out = capsys.readouterr().out
    assert "Fairness audit" in out


def test_wheel(capsys):
    main(["wheel", "lotto649", "--pool", "4,9,17,23,32,38,41,45", "-t", "4"])
    out = capsys.readouterr().out
    assert "guarantee" in out


def test_falsify_simulated(capsys):
    main(["falsify", "lotto649", "--simulate", "200", "--warmup", "100"])
    out = capsys.readouterr().out
    assert "fair_null" in out


def test_unknown_game():
    with pytest.raises(KeyError):
        main(["odds", "nope"])
