"""Built-in game definitions.

Odds-relevant structure (pool sizes, picks, bonus mode) follows the official
game rules.  Fixed prize amounts are representative published base amounts;
jackpot and parimutuel tiers are valued at EV time.  Always refresh prize
amounts from official sources before acting on EV output.
"""

from __future__ import annotations

from .ruleset import BonusMode, FieldSpec, GameFamily, PrizeTier, Ruleset

POWERBALL = Ruleset(
    game_id="powerball",
    name="Powerball (US)",
    family=GameFamily.MULTI_POOL,
    main=FieldSpec(pool_size=69, picks=5),
    bonus_mode=BonusMode.SEPARATE_POOL,
    bonus_pool_size=26,
    ticket_price=2.0,
    tiers=(
        PrizeTier("5+PB", 5, True, is_jackpot=True),
        PrizeTier("5+0", 5, False, prize=1_000_000),
        PrizeTier("4+PB", 4, True, prize=50_000),
        PrizeTier("4+0", 4, False, prize=100),
        PrizeTier("3+PB", 3, True, prize=100),
        PrizeTier("3+0", 3, False, prize=7),
        PrizeTier("2+PB", 2, True, prize=7),
        PrizeTier("1+PB", 1, True, prize=4),
        PrizeTier("0+PB", 0, True, prize=4),
    ),
)

MEGA_MILLIONS = Ruleset(
    game_id="megamillions",
    name="Mega Millions (US, 2025 rules)",
    family=GameFamily.MULTI_POOL,
    main=FieldSpec(pool_size=70, picks=5),
    bonus_mode=BonusMode.SEPARATE_POOL,
    bonus_pool_size=24,
    ticket_price=5.0,
    tiers=(
        PrizeTier("5+MB", 5, True, is_jackpot=True),
        PrizeTier("5+0", 5, False, prize=1_000_000),
        PrizeTier("4+MB", 4, True, prize=10_000),
        PrizeTier("4+0", 4, False, prize=500),
        PrizeTier("3+MB", 3, True, prize=200),
        PrizeTier("3+0", 3, False, prize=10),
        PrizeTier("2+MB", 2, True, prize=10),
        PrizeTier("1+MB", 1, True, prize=7),
        PrizeTier("0+MB", 0, True, prize=5),
    ),
    notes="Base prizes before the built-in random multiplier.",
)

LOTTO_649 = Ruleset(
    game_id="lotto649",
    name="Lotto 6/49 (Canada)",
    family=GameFamily.LOTTO_KN,
    main=FieldSpec(pool_size=49, picks=6),
    bonus_mode=BonusMode.FROM_REMAINING,
    ticket_price=3.0,
    tiers=(
        PrizeTier("6/6", 6, None, is_jackpot=True),
        PrizeTier("5/6+B", 5, True, is_parimutuel=True, prize=100_000),
        PrizeTier("5/6", 5, False, is_parimutuel=True, prize=2_500),
        PrizeTier("4/6", 4, None, is_parimutuel=True, prize=75),
        PrizeTier("3/6", 3, None, prize=10),
        PrizeTier("2/6+B", 2, True, prize=5),
        PrizeTier("2/6", 2, False, prize=3),  # free play, valued at ticket price
    ),
    notes="Parimutuel tier amounts are typical values used as EV placeholders.",
)

PICK3 = Ruleset(
    game_id="pick3",
    name="Pick 3 (straight)",
    family=GameFamily.DIGITS,
    main=FieldSpec(pool_size=10, picks=1),  # unused for digits; kept for schema
    digit_count=3,
    ticket_price=1.0,
    tiers=(PrizeTier("straight", 3, None, prize=500),),
)

KENO_10 = Ruleset(
    game_id="keno10",
    name="Keno 10-spot (20 of 80 drawn)",
    family=GameFamily.KENO,
    main=FieldSpec(pool_size=80, picks=10),
    drawn=20,
    ticket_price=1.0,
    tiers=(
        PrizeTier("10/10", 10, None, prize=100_000),
        PrizeTier("9/10", 9, None, prize=5_000),
        PrizeTier("8/10", 8, None, prize=500),
        PrizeTier("7/10", 7, None, prize=50),
        PrizeTier("6/10", 6, None, prize=10),
        PrizeTier("5/10", 5, None, prize=2),
        PrizeTier("0/10", 0, None, prize=5),
    ),
    notes="Representative Ohio-style paytable.",
)

REGISTRY: dict[str, Ruleset] = {
    r.game_id: r for r in (POWERBALL, MEGA_MILLIONS, LOTTO_649, PICK3, KENO_10)
}


def get_game(game_id: str) -> Ruleset:
    try:
        return REGISTRY[game_id]
    except KeyError:
        raise KeyError(
            f"unknown game {game_id!r}; available: {', '.join(sorted(REGISTRY))}"
        ) from None
