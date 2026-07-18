"""Canonical storage: SQLite-backed draw and sales archive.

Ingestion is idempotent: the natural key is (game_id, draw_date) and upserts
replace prior rows, so re-running an ingestor never duplicates data.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS draws (
    game_id    TEXT NOT NULL,
    draw_date  TEXT NOT NULL,
    numbers    TEXT NOT NULL,          -- JSON array of main numbers (sorted)
    bonus      INTEGER,
    multiplier INTEGER,
    source     TEXT,
    PRIMARY KEY (game_id, draw_date)
);
CREATE TABLE IF NOT EXISTS sales (
    game_id       TEXT NOT NULL,
    draw_date     TEXT NOT NULL,
    sales         REAL,
    jackpot       REAL,               -- advertised (annuity) jackpot
    jackpot_cash  REAL,
    PRIMARY KEY (game_id, draw_date)
);
CREATE TABLE IF NOT EXISTS tier_results (
    game_id   TEXT NOT NULL,
    draw_date TEXT NOT NULL,
    tier      TEXT NOT NULL,
    winners   INTEGER,
    amount    REAL,
    PRIMARY KEY (game_id, draw_date, tier)
);
"""


@dataclass(frozen=True)
class Draw:
    game_id: str
    draw_date: str                     # ISO yyyy-mm-dd
    numbers: tuple[int, ...]
    bonus: Optional[int] = None
    multiplier: Optional[int] = None
    source: str = ""


@dataclass(frozen=True)
class SalesSnapshot:
    game_id: str
    draw_date: str
    sales: Optional[float] = None
    jackpot: Optional[float] = None
    jackpot_cash: Optional[float] = None


@dataclass
class ValidationReport:
    inserted: int = 0
    rejected: list[str] = field(default_factory=list)


class Store:
    def __init__(self, path: str | Path = "lo_toolkit.db"):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(_SCHEMA)

    # -- draws ------------------------------------------------------------
    def upsert_draws(
        self, draws: Iterable[Draw], pool_size: int | None = None
    ) -> ValidationReport:
        """Validate and store draws.  Invalid rows are rejected, not raised."""
        report = ValidationReport()
        rows = []
        for d in draws:
            problem = self._validate(d, pool_size)
            if problem:
                report.rejected.append(f"{d.game_id} {d.draw_date}: {problem}")
                continue
            rows.append(
                (
                    d.game_id,
                    d.draw_date,
                    json.dumps(sorted(d.numbers)),
                    d.bonus,
                    d.multiplier,
                    d.source,
                )
            )
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO draws VALUES (?,?,?,?,?,?)", rows
            )
        report.inserted = len(rows)
        return report

    @staticmethod
    def _validate(d: Draw, pool_size: int | None) -> str | None:
        if len(set(d.numbers)) != len(d.numbers):
            return "duplicate numbers"
        if any(n < 1 for n in d.numbers):
            return "numbers must be >= 1"
        if pool_size and any(n > pool_size for n in d.numbers):
            return f"number out of range 1..{pool_size}"
        return None

    def draws(self, game_id: str) -> list[Draw]:
        cur = self.conn.execute(
            "SELECT game_id, draw_date, numbers, bonus, multiplier, source "
            "FROM draws WHERE game_id=? ORDER BY draw_date",
            (game_id,),
        )
        return [
            Draw(g, dt, tuple(json.loads(nums)), b, m, s or "")
            for g, dt, nums, b, m, s in cur.fetchall()
        ]

    # -- sales ------------------------------------------------------------
    def upsert_sales(self, snaps: Iterable[SalesSnapshot]) -> int:
        rows = [
            (s.game_id, s.draw_date, s.sales, s.jackpot, s.jackpot_cash)
            for s in snaps
        ]
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO sales VALUES (?,?,?,?,?)", rows
            )
        return len(rows)

    def sales(self, game_id: str) -> list[SalesSnapshot]:
        cur = self.conn.execute(
            "SELECT game_id, draw_date, sales, jackpot, jackpot_cash "
            "FROM sales WHERE game_id=? ORDER BY draw_date",
            (game_id,),
        )
        return [SalesSnapshot(*row) for row in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()
