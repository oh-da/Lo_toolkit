"""Mifal HaPayis Chance results-archive CSV parser.

The official export is cp1255-encoded, newest-first, with columns:
date (dd/mm/yyyy), draw id, then one card rank per suit in the order
clubs, diamonds, hearts, spades.  Ranks are 7, 8, 9, 10, J, Q, K, A and
are stored as 1..8 in that order (position in the tuple = suit).
"""

from __future__ import annotations

import csv
from pathlib import Path

from ..schema import Draw

GAME_ID = "chance_il"
RANKS = ("7", "8", "9", "10", "J", "Q", "K", "A")
SUITS = ("clubs", "diamonds", "hearts", "spades")
_RANK_INDEX = {r: i + 1 for i, r in enumerate(RANKS)}


def _iso(date_ddmmyyyy: str) -> str:
    d, m, y = date_ddmmyyyy.strip().split("/")
    return f"{y}-{m.zfill(2)}-{d.zfill(2)}"


def load_chance_csv(path: str | Path, encoding: str = "cp1255") -> list[Draw]:
    """Parse the official archive; returns draws oldest-first."""
    draws: list[Draw] = []
    with open(path, encoding=encoding, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # Hebrew header row
        for row in reader:
            if not row or not row[0].strip():
                continue
            ranks = tuple(_RANK_INDEX[x.strip().upper()] for x in row[2:6])
            draws.append(
                Draw(
                    game_id=GAME_ID,
                    draw_date=_iso(row[0]),
                    numbers=ranks,
                    source=str(path),
                    draw_no=int(row[1]),
                )
            )
    return list(reversed(draws))


def format_card_line(ranks: tuple[int, ...]) -> str:
    return ", ".join(f"{RANKS[r - 1]} of {s}" for r, s in zip(ranks, SUITS))
