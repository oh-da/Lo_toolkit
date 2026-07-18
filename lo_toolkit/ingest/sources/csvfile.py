"""Generic CSV draw importer.

Expected columns: ``date`` (ISO), ``numbers`` (space- or dash-separated main
numbers), optional ``bonus``, ``multiplier``.  Extra columns are ignored.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from ..schema import Draw

_SPLIT = re.compile(r"[\s,;\-]+")


def load_csv(path: str | Path, game_id: str) -> list[Draw]:
    draws: list[Draw] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            numbers = tuple(int(x) for x in _SPLIT.split(row["numbers"].strip()) if x)
            bonus = row.get("bonus")
            mult = row.get("multiplier")
            draws.append(
                Draw(
                    game_id=game_id,
                    draw_date=row["date"].strip(),
                    numbers=numbers,
                    bonus=int(bonus) if bonus not in (None, "") else None,
                    multiplier=int(mult) if mult not in (None, "") else None,
                    source=str(path),
                )
            )
    return draws
