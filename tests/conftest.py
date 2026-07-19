import pytest

from lo_toolkit.games.ruleset import FieldSpec, GameFamily, PrizeTier, Ruleset


@pytest.fixture
def mini_game() -> Ruleset:
    """Small 5-from-20 game: fast to simulate, exact odds easy to check."""
    return Ruleset(
        game_id="mini",
        name="Mini 5/20",
        family=GameFamily.LOTTO_KN,
        main=FieldSpec(pool_size=20, picks=5),
        ticket_price=1.0,
        tiers=(
            PrizeTier("5/5", 5, None, is_jackpot=True),
            PrizeTier("4/5", 4, None, prize=50),
            PrizeTier("3/5", 3, None, prize=5),
        ),
    )
