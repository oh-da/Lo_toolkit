"""Game rule definitions.

A :class:`Ruleset` is the single source of truth for a game's combinatorics:
every odds calculation, simulation, and audit null model is derived from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class GameFamily(str, Enum):
    LOTTO_KN = "lotto_kN"          # unordered k distinct from 1..N
    MULTI_POOL = "multi_pool"      # main field + independent bonus pool
    DIGITS = "digits"              # ordered digits with replacement (Pick 3/4)
    KENO = "keno"                  # player picks k, operator draws d from N
    RAFFLE = "raffle"              # serial-ticket draw
    CARDS = "cards"                # one symbol per position (Chance: rank per suit)


class BonusMode(str, Enum):
    NONE = "none"
    SEPARATE_POOL = "separate_pool"      # e.g. Powerball: 1 from 1..26
    FROM_REMAINING = "from_remaining"    # e.g. 6/49 bonus: drawn from remaining main pool


@dataclass(frozen=True)
class FieldSpec:
    """One drawn field: `picks` distinct numbers from 1..`pool_size`."""

    pool_size: int
    picks: int

    def __post_init__(self) -> None:
        if not 0 < self.picks <= self.pool_size:
            raise ValueError(f"invalid field: {self.picks} from {self.pool_size}")


@dataclass(frozen=True)
class PrizeTier:
    """Prize for matching `match_main` main numbers (+ bonus condition).

    `match_bonus` is None when the bonus is irrelevant to the tier, True when
    the bonus must be matched, False when it must not be matched.
    `prize` is a fixed amount; `is_jackpot`/`is_parimutuel` tiers get their
    value at EV time.
    """

    name: str
    match_main: int
    match_bonus: Optional[bool] = None
    prize: float = 0.0
    is_jackpot: bool = False
    is_parimutuel: bool = False


@dataclass(frozen=True)
class Ruleset:
    game_id: str
    name: str
    family: GameFamily
    main: FieldSpec
    bonus_mode: BonusMode = BonusMode.NONE
    bonus_pool_size: int = 0            # for SEPARATE_POOL
    bonus_picks: int = 1
    drawn: Optional[int] = None         # operator draw size when != player picks (Keno)
    digit_count: int = 0                # for DIGITS family
    positions: int = 0                  # for CARDS family: independent positions
    symbols: int = 0                    # for CARDS family: symbols per position
    ticket_price: float = 2.0
    tiers: tuple[PrizeTier, ...] = field(default_factory=tuple)
    notes: str = ""

    @property
    def draw_size(self) -> int:
        """How many main numbers the operator draws."""
        return self.drawn if self.drawn is not None else self.main.picks
