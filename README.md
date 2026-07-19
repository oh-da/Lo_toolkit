# Lo_toolkit

A Python toolkit for lawfully analyzing lotteries and improving expected outcomes —
**audit-first and EV-first, not a number-prediction oracle**.

Design and methodology follow the research document *"Lottery Research Methodologies
for Lawfully Improving Lottery Chances"*; see [PLAN.md](PLAN.md) for the full
implementation plan and rationale.

## What it does

- **Exact odds** — tier probabilities computed as exact rationals from game rules,
  validated in tests against officially published odds (Powerball, Mega Millions,
  Lotto 6/49, Keno).
- **Fairness audits** — a Monte Carlo-calibrated test battery (marginal frequency,
  pairwise co-occurrence, spacing, repeats, dispersion) against the exact
  hypergeometric null, with Benjamini–Hochberg FDR control and split-half
  replication.
- **EV & split-risk analysis** — jackpot-sharing via a Poisson collision model,
  jackpot→sales elasticity fitting, roll-down (Cash WinFall-style) scenario EV.
- **Anti-collision tickets** — uniform-random lines filtered by a literature-based
  popularity surface (birthday range, lucky numbers, arithmetic patterns) to reduce
  expected jackpot splitting.
- **Wheels & syndicates** — greedy covering designs C(v,k,t) with verified
  t-if-t guarantees, bundle pricing, and Monte Carlo prize/ROI simulation.
- **Falsification lab** — walk-forward log-loss/Brier benchmarking of "hot numbers"
  and Markov heuristics against the fair null, to kill pattern claims with evidence.

## What it deliberately won't do

Predict winning numbers. In a fair lottery every combination is equally likely each
draw; the toolkit's predictive models exist only to demonstrate that apparent
patterns don't survive out-of-sample testing.

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest          # 56 tests, incl. validation against official odds
```

## Usage

```bash
lo games                                     # built-in games and jackpot odds
lo odds powerball                            # exact tier odds

# Build a draw archive (official NY Open Data mirrors, or any CSV)
lo ingest ny powerball --db data.db
lo ingest csv lotto649 --path draws.csv --db data.db

# Is the game fair?  (expected answer: yes — and that's the point)
lo audit lotto649 --db data.db

# Is this draw worth entering?
lo ev powerball --jackpot-cash 300000000 --sales 40000000
lo ev powerball --jackpot-cash 300000000 --sales 40000000 --popularity 0.5

# How should entries be structured?
lo tickets powerball -n 5                    # anti-collision lines
lo wheel lotto649 --pool 4,9,17,23,32,38,41,45 -t 4 --members 8

# Do "patterns" survive out-of-sample testing?  (they don't)
lo falsify lotto649 --db data.db
```

Sample audit output on 600 draws:

```
Fairness audit: lotto649 (600 draws, MC-calibrated, BH FDR 5%)
test                               stat        p    p_adj  verdict
marginal_frequency               45.246   0.3912   0.8224  consistent with fair
pairwise_cooccurrence            10.347   0.6347   0.8224  consistent with fair
min_spacing                       1.867   0.8224   0.8224  consistent with fair
repeats_prev_draw               445.000   0.7784   0.8224  consistent with fair
sum_dispersion                 1092.309   0.6946   0.8224  consistent with fair

No replicated anomaly: the history is consistent with a fair game.
```

## Legal & responsible play

Use public rules, public data, and lawful ticket purchases only. Follow each
lottery's rules, age restrictions, and claim/tax requirements; document syndicate
shares cleanly. Every EV report shows the unchangeable base odds — lottery play
should stay affordable, and no strategy here changes the probability that any
line is drawn.
