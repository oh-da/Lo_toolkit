"""`lo` command-line interface.

Subcommands mirror the decision pipeline: games/odds (know the base odds),
ingest (build the archive), audit (is the game fair?), ev (is this draw
worth entering?), tickets/wheel (how to structure entries), falsify (kill
pattern claims).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .audit.report import run_audit
from .behaviour.anticollision import generate_anticollision_lines, line_popularity_report
from .ev.evcalc import expected_value
from .ev.odds import jackpot_odds, tier_probabilities
from .falsify.models import HotColdModel, MarkovParityModel
from .falsify.walkforward import walk_forward
from .games.registry import REGISTRY, get_game
from .ingest.schema import Store
from .ingest.sources.csvfile import load_csv
from .portfolio.syndicate import simulate_syndicate
from .portfolio.wheels import build_wheel
from .sim.engine import DrawSimulator


def _draws_matrix(args, rules) -> np.ndarray:
    """Load draws from the DB, or simulate a fair history for demos."""
    if getattr(args, "simulate", None):
        sim = DrawSimulator(rules, seed=getattr(args, "seed", None))
        return sim.draw_main(args.simulate)
    store = Store(args.db)
    draws = store.draws(rules.game_id)
    if not draws:
        sys.exit(
            f"no draws for {rules.game_id!r} in {args.db}; ingest data first "
            f"or pass --simulate N for a fair-history demo"
        )
    return np.array([d.numbers for d in draws])


def cmd_games(args) -> None:
    for g in REGISTRY.values():
        print(f"{g.game_id:14s} {g.name:38s} jackpot odds: 1 in {jackpot_odds(g):,.0f}")


def cmd_odds(args) -> None:
    rules = get_game(args.game)
    print(f"{rules.name} — exact tier odds (price {rules.ticket_price:.2f})")
    for name, p in tier_probabilities(rules).items():
        print(f"  {name:8s} 1 in {float(1 / p):>18,.2f}")


def cmd_ingest(args) -> None:
    store = Store(args.db)
    if args.source == "csv":
        rules = get_game(args.game)
        draws = load_csv(args.path, args.game)
        rep = store.upsert_draws(draws, pool_size=rules.main.pool_size)
    elif args.source == "pais":
        from .ingest.sources.israel_lotto import GAME_ID, load_pais_csv

        rules = get_game(GAME_ID)
        draws = load_pais_csv(args.path)
        rep = store.upsert_draws(draws, pool_size=rules.main.pool_size)
    elif args.source == "chance":
        from .ingest.sources.chance import GAME_ID, load_chance_csv

        rules = get_game(GAME_ID)
        draws = load_chance_csv(args.path)
        rep = store.upsert_draws(draws, pool_size=rules.symbols, ordered=True)
    elif args.source == "ny":
        from .ingest.sources.nyopendata import (
            IngestError,
            fetch_ny_megamillions,
            fetch_ny_powerball,
        )

        fetch = {"powerball": fetch_ny_powerball, "megamillions": fetch_ny_megamillions}[
            args.game
        ]
        rules = get_game(args.game)
        try:
            draws = fetch()
        except IngestError as e:
            sys.exit(f"{e}\nfalling back is possible via: lo ingest csv {args.game} --path FILE")
        rep = store.upsert_draws(draws, pool_size=rules.main.pool_size)
    else:
        sys.exit(f"unknown source {args.source}")
    print(f"inserted {rep.inserted} draws into {args.db}")
    for r in rep.rejected:
        print(f"  rejected: {r}")


def cmd_audit(args) -> None:
    from .audit.bonus import bonus_uniformity
    from .games.ruleset import BonusMode, GameFamily

    rules = get_game(args.game)
    if rules.family == GameFamily.CARDS:
        from .audit.cards import run_cards_audit
        from .ingest.sources.chance import SUITS

        if getattr(args, "simulate", None):
            rng = np.random.default_rng(args.seed)
            draws = rng.integers(1, rules.symbols + 1, size=(args.simulate, rules.positions))
        else:
            store = Store(args.db)
            records = store.draws(rules.game_id)
            if not records:
                sys.exit(f"no draws for {rules.game_id!r} in {args.db}; ingest data first")
            draws = np.array([d.numbers for d in records])
        result = run_cards_audit(
            rules, draws, n_sims=args.sims, seed=args.seed, position_names=SUITS
        )
        print(result.summary())
        return

    bonuses = None
    if getattr(args, "simulate", None):
        draws = _draws_matrix(args, rules)
    else:
        store = Store(args.db)
        records = store.draws(rules.game_id)
        if not records:
            sys.exit(f"no draws for {rules.game_id!r} in {args.db}; ingest data first")
        draws = np.array([d.numbers for d in records])
        if rules.bonus_mode == BonusMode.SEPARATE_POOL and all(
            d.bonus is not None for d in records
        ):
            bonuses = np.array([d.bonus for d in records])
    result = run_audit(rules, draws, n_sims=args.sims, seed=args.seed)
    print(result.summary())
    if bonuses is not None:
        stat, p = bonus_uniformity(
            bonuses, rules.bonus_pool_size, n_sims=args.sims, seed=args.seed
        )
        verdict = "consistent with uniform" if p >= 0.05 else "NON-UNIFORM (investigate)"
        print(
            f"\nBonus-ball uniformity (1..{rules.bonus_pool_size}): "
            f"chi2={stat:.2f}, MC p={p:.4f} — {verdict}"
        )


def cmd_ev(args) -> None:
    rules = get_game(args.game)
    ev = expected_value(
        rules,
        jackpot_cash=args.jackpot_cash,
        tickets_sold=args.sales,
        popularity_factor=args.popularity,
    )
    print(rules.name)
    print(ev.summary())
    if ev.ev_net > 0:
        print("positive net EV — verify rules, taxes, and sales estimate before acting")


def cmd_tickets(args) -> None:
    rules = get_game(args.game)
    lines = generate_anticollision_lines(rules, args.count, seed=args.seed)
    print(f"{rules.name} — {args.count} anti-collision lines")
    print(line_popularity_report(rules, lines))


def cmd_wheel(args) -> None:
    rules = get_game(args.game)
    pool = [int(x) for x in args.pool.split(",")]
    wheel = build_wheel(rules, pool, args.guarantee)
    print(wheel.summary())
    for line in wheel.lines:
        print("  " + "-".join(f"{n:02d}" for n in line))
    if args.members > 1:
        res = simulate_syndicate(
            rules, wheel.lines, args.members, args.draws, args.jackpot_cash, seed=args.seed
        )
        print()
        print(res.summary())


def cmd_falsify(args) -> None:
    rules = get_game(args.game)
    draws = _draws_matrix(args, rules)
    models = [HotColdModel(window=args.window), MarkovParityModel()]
    result = walk_forward(models, draws, rules, warmup=args.warmup)
    print(result.summary())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="lo",
        description="Audit-first, EV-first lottery analysis toolkit. "
        "It will not predict winning numbers — nothing can, in a fair game.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("games", help="list built-in games").set_defaults(func=cmd_games)

    sp = sub.add_parser("odds", help="exact tier odds for a game")
    sp.add_argument("game")
    sp.set_defaults(func=cmd_odds)

    sp = sub.add_parser("ingest", help="ingest draw history")
    sp.add_argument("source", choices=["csv", "ny", "pais", "chance"])
    sp.add_argument("game")
    sp.add_argument("--path", help="CSV path (source=csv)")
    sp.add_argument("--db", default="lo_toolkit.db")
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("audit", help="fairness audit battery")
    sp.add_argument("game")
    sp.add_argument("--db", default="lo_toolkit.db")
    sp.add_argument("--simulate", type=int, help="audit a simulated fair history of N draws")
    sp.add_argument("--sims", type=int, default=1000, help="Monte Carlo replications")
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_audit)

    sp = sub.add_parser("ev", help="expected value of one line for a draw")
    sp.add_argument("game")
    sp.add_argument("--jackpot-cash", type=float, required=True, help="jackpot cash value")
    sp.add_argument("--sales", type=float, required=True, help="expected tickets sold")
    sp.add_argument(
        "--popularity",
        type=float,
        default=1.0,
        help="line popularity vs uniform (anti-collision < 1.0)",
    )
    sp.set_defaults(func=cmd_ev)

    sp = sub.add_parser("tickets", help="generate anti-collision lines")
    sp.add_argument("game")
    sp.add_argument("-n", "--count", type=int, default=5)
    sp.add_argument("--seed", type=int, default=None)
    sp.set_defaults(func=cmd_tickets)

    sp = sub.add_parser("wheel", help="build a covering wheel over a number pool")
    sp.add_argument("game")
    sp.add_argument("--pool", required=True, help="comma-separated numbers")
    sp.add_argument("-t", "--guarantee", type=int, default=3)
    sp.add_argument("--members", type=int, default=1, help="syndicate size to simulate")
    sp.add_argument("--draws", type=int, default=100_000, help="simulated draws")
    sp.add_argument("--jackpot-cash", type=float, default=10_000_000)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_wheel)

    sp = sub.add_parser("falsify", help="walk-forward benchmark vs the fair null")
    sp.add_argument("game")
    sp.add_argument("--db", default="lo_toolkit.db")
    sp.add_argument("--simulate", type=int, help="use a simulated fair history of N draws")
    sp.add_argument("--window", type=int, default=50, help="hot/cold lookback window")
    sp.add_argument("--warmup", type=int, default=100)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_falsify)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
