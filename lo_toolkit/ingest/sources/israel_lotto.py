"""Mifal HaPayis (Israeli Lotto) results-archive CSV parser.

The official export is cp1255-encoded, newest-first, with columns:
draw id, date (dd/mm/yyyy), six main numbers, the strong number, and
winner-count columns.  The game has changed format over its history;
`modern_only=True` keeps the contiguous latest era of 6 numbers from 1..37
plus a strong number from 1..7 (in force since draw 2234, 2011-03-05).
"""

from __future__ import annotations

import csv
from pathlib import Path

from ..schema import Draw

GAME_ID = "lotto_il"
MODERN_POOL = 37
MODERN_STRONG = 7


def _iso(date_ddmmyyyy: str) -> str:
    d, m, y = date_ddmmyyyy.strip().split("/")
    return f"{y}-{m.zfill(2)}-{d.zfill(2)}"


def load_pais_csv(
    path: str | Path, modern_only: bool = True, encoding: str = "cp1255"
) -> list[Draw]:
    """Parse the official archive; returns draws oldest-first."""
    parsed: list[Draw] = []
    with open(path, encoding=encoding, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # Hebrew header row
        for row in reader:
            if not row or not row[0].strip():
                continue
            numbers = tuple(int(x) for x in row[2:8])
            strong = int(row[8]) if row[8].strip() else None
            parsed.append(
                Draw(
                    game_id=GAME_ID,
                    draw_date=_iso(row[1]),
                    numbers=numbers,
                    bonus=strong,
                    source=str(path),
                    draw_no=int(row[0]),
                )
            )

    if modern_only:
        # File is newest-first: keep the contiguous run of modern-format rows.
        modern: list[Draw] = []
        for d in parsed:
            ok = (
                len(set(d.numbers)) == 6
                and min(d.numbers) >= 1
                and max(d.numbers) <= MODERN_POOL
                and d.bonus is not None
                and 1 <= d.bonus <= MODERN_STRONG
            )
            if not ok:
                break
            modern.append(d)
        parsed = modern
    return list(reversed(parsed))
