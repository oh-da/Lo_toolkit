# Lo_toolkit — Implementation Plan

A Python toolkit for lawfully analyzing lotteries and improving expected outcomes,
based on the research document *"Lottery Research Methodologies for Lawfully Improving
Lottery Chances"* (see `docs/` for the source analysis summary).

## 1. Guiding principles (from the research)

The research is unambiguous, and the toolkit's design follows it:

1. **No number-prediction oracle.** In a fair lottery every combination is equally
   likely each draw. Historical frequencies do not forecast future numbers. Any
   "predictive" module exists only to *falsify* pattern claims.
2. **Audit-first.** The scientifically strongest first layer is a fairness audit
   suite built on the exact combinatorial null model (hypergeometric for k/N games),
   with Monte Carlo calibration and multiple-testing control.
3. **EV-first.** The most realistic public edge is *market-facing*: identifying rare
   positive-EV situations (roll-downs, mandatory payouts, promotions), timing entry
   via jackpot/sales/split-risk modelling, and choosing games with materially better
   base odds.
4. **Improve conditional payout, not raw odds.** Split-avoidance (picking unpopular
   combinations) and wheeling/covering designs don't change the probability a line is
   drawn — they improve take-home value if you win, or lower-tier coverage per unit
   of stake.
5. **Lawful and ethical only.** Public rules, public data, lawful purchases. Auditing
   randomness is legitimate research; exploiting published prize rules open to every
   player (Cash WinFall-style) is legal; anything touching draw equipment, insider
   data, or retailer-procedure bypass is out of scope by design.

## 2. What the toolkit can and cannot improve

| Lever | Improves | Does not improve |
|---|---|---|
| Game selection (odds/prize structure) | Expected return per dollar | — |
| Roll-down / positive-EV detection | Expected return, sometimes materially | Raw combinatorial odds of a line |
| Split-avoidance ticket generation | Conditional payout if you win | Probability your line is drawn |
| Wheeling / covering designs | Lower-tier coverage guarantees for a chosen pool | Per-line odds or EV "by magic" |
| Syndicate modelling | Group's absolute chance; variance sharing | EV per ticket before fees |
| Fairness audit | Confidence the game is fair; anomaly detection | A forecasting edge (unless a real defect exists) |

## 3. Architecture

Five loosely-coupled pipelines over a shared canonical data layer, per the research's
recommended build plan:

```
                    ┌─────────────────────────────────────────┐
                    │  Ingestion: official archives & APIs    │
                    │  (MUSL API, NY Open Data, operator CSVs)│
                    └───────────────────┬─────────────────────┘
                                        │
                    ┌───────────────────▼─────────────────────┐
                    │  Canonical schema + validation/dedup    │
                    │  (SQLite; games, rulesets, draws, sales)│
                    └───────────────────┬─────────────────────┘
              ┌───────────┬─────────────┼─────────────┬──────────────┐
              ▼           ▼             ▼             ▼              ▼
        ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌───────────┐ ┌────────────┐
        │  Audit  │ │ EV/split │ │ Behaviour │ │ Portfolio │ │ Falsifier  │
        │  suite  │ │  engine  │ │bias model │ │ optimizer │ │  bench lab │
        └────┬────┘ └────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬──────┘
             └───────────┴─────────────┼─────────────┴─────────────┘
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │  Decision layer + reports/dashboard     │
                    │  "don't play" / "enter with anti-       │
                    │  collision tickets" / "enter via wheel  │
                    │  or syndicate"                          │
                    └─────────────────────────────────────────┘
```

### Package layout

```
lo_toolkit/
├── pyproject.toml
├── lo_toolkit/
│   ├── games/            # Game rule definitions & registry
│   │   ├── ruleset.py    #   dataclasses: fields, pools, prize tiers, price
│   │   └── registry.py   #   Powerball, Mega Millions, 6/49, Keno, Pick 3/4...
│   ├── ingest/           # Data acquisition
│   │   ├── sources/      #   MUSL numbers API, NY Open Data, CSV importers
│   │   ├── schema.py     #   canonical entities (see §4)
│   │   └── validate.py   #   range checks, dedup, gap detection
│   ├── audit/            # Fairness audit suite (exact null models)
│   │   ├── nullmodel.py  #   hypergeometric moments, exact expectations
│   │   ├── tests_marginal.py   # per-ball frequency vs exact envelope
│   │   ├── tests_pairwise.py   # co-occurrence vs hypergeometric covariance
│   │   ├── tests_spacing.py    # minimal-distance / spacing statistics
│   │   ├── tests_runs.py       # runs, over/under-dispersion, repeats
│   │   ├── montecarlo.py       # simulated null distributions, MC p-values
│   │   └── fdr.py              # Benjamini–Hochberg, replication splits
│   ├── ev/               # EV & split-risk engine
│   │   ├── odds.py       #   exact tier probabilities from ruleset
│   │   ├── sales.py      #   jackpot→sales forecasting (statsmodels)
│   │   ├── collision.py  #   sharing probability under pick-bias + quick-pick share
│   │   ├── rolldown.py   #   roll-down / mandatory-payout EV scenarios
│   │   └── evcalc.py     #   net EV: cash value, taxes, annuity, splits
│   ├── behaviour/        # Player-choice bias models
│   │   ├── popularity.py #   number-popularity surface (priors from literature,
│   │   │                 #   fitted from winner-count data where available)
│   │   └── anticollision.py  # unpopular-combination ticket generator
│   ├── portfolio/        # Ticket-construction engine
│   │   ├── covering.py   #   covering designs (v,k,t): known designs + greedy/ILP
│   │   ├── wheels.py     #   full/abbreviated wheels, pricing
│   │   └── syndicate.py  #   pooled play, share accounting, variance modelling
│   ├── falsify/          # Benchmark lab (kills bad ideas)
│   │   ├── baselines.py  #   fair-probability null predictor
│   │   ├── models.py     #   Markov/HMM, gradient boosting, optional LSTM
│   │   └── walkforward.py#   strict chronological eval: log loss / Brier vs null
│   ├── sim/              # Monte Carlo backtesting engine (shared)
│   │   └── engine.py     #   draw simulator, bankroll paths, drawdown stats
│   ├── report/           # Outputs
│   │   ├── plots.py      #   frequency-vs-envelope bars, co-occurrence heatmaps,
│   │   │                 #   rolling z-scores, jackpot-vs-EV curves, wheel matrices
│   │   └── dashboard.py  #   static HTML report generation
│   └── cli.py            # `lo` command: ingest / audit / ev / tickets / falsify
└── tests/
```

## 4. Canonical data model

Mirrors the entity model recommended in the research:

- **LotteryGame** — id, name, family (`lotto_kN`, `multi_pool`, `digits`, `keno`, `raffle`, `rolldown`)
- **Ruleset** — game FK, effective dates, field specs (pool size, picks, bonus pool),
  ticket price, prize tiers (fixed vs parimutuel), roll-down/mandatory-payout rules
- **Draw** — game FK, ruleset FK, draw date/number, numbers (main + bonus),
  machine/ball-set metadata when published
- **PrizeTierResult** — per-draw winners count and amounts per tier (key input for
  both split-risk estimation and popularity inference)
- **SalesSnapshot** — per-draw sales, advertised jackpot, cash value
- **TicketStrategy / TicketLine** — generated ticket bundles and their provenance
  (which module and parameters produced them)
- **NumberScore** — popularity/anti-collision scores per number/combination pattern

Storage: SQLite via a thin layer (sqlite3 or SQLAlchemy) — portable, no server, easy
to version. Ingestion is idempotent (natural keys: game + draw date/number).

## 5. Module specifications

### 5.1 Audit suite (build first)
- Exact null expectations for k/N games: per-ball inclusion probability k/N,
  hypergeometric covariance for pair tests (never naive iid chi-square — Joe's
  correction; Coronel-Brizio audit framework).
- Tests: marginal frequency, pairwise co-occurrence, runs/repeats across draws,
  over/under-dispersion, minimal-distance/spacing (Drakakis).
- All p-values Monte Carlo-calibrated under exact game rules; Benjamini–Hochberg FDR
  across the test battery; findings must replicate on a held-out, non-overlapping
  period before being reported as anomalies.
- Digit games (Pick 3/4) get a multinomial variant (order matters, replacement).
- Output: "fairness report" per game — expected result is *no exploitable anomaly*;
  the value is ruling bias out before spending effort downstream.

### 5.2 EV & split-risk engine (the realistic edge)
- Exact odds/tier probabilities computed from the ruleset (validated against
  published odds tables).
- Sales model: jackpot size → expected ticket sales (log-linear regression on 1–3
  years of jackpot/sales history; NY Open Data provides Powerball/Mega Millions).
- Collision model: P(share | win) from sales forecast + pick-bias distribution
  (quick-pick share uniform; manual picks weighted by popularity surface). Kim–Skiena
  anti-collision logic.
- Roll-down scenario module: encode special payout rules and compute EV under them
  (Cash WinFall-style analysis, applied to any current game with such rules).
- Net-EV calculator: cash vs annuity, jurisdiction tax stub, expected splits.
- Output: per-draw "is this worth entering?" table across configured games.

### 5.3 Behaviour / anti-collision module
- Popularity priors from published research (small numbers, birthdays ≤31, lucky 7,
  playslip layout effects — Wang et al., Polin et al.) since player microdata are
  rarely public.
- Where per-tier winner counts exist, fit/refine the popularity surface: draws whose
  numbers are "popular-shaped" should show systematically more winners per dollar of
  sales.
- Anti-collision generator: sample tickets uniformly, then filter/penalize
  popular patterns (all ≤31, arithmetic sequences, visual playslip shapes, previous
  draws, fortune-cookie classics like 7-14-21-28-35-42).

### 5.4 Portfolio / wheeling engine
- Covering designs C(v,k,t): bundled tables of known-good designs (La Jolla /
  Covering Repository data where licensing permits) + greedy and ILP (pulp) solvers
  as fallback.
- Wheel pricing and guarantee verification by exact combinatorics ("3-if-4 within
  pool" style guarantees), plus Monte Carlo prize-distribution simulation for
  solo vs syndicate play.
- Explicit honesty in output: wheels never improve per-line odds; reports always show
  cost escalation vs coverage.

### 5.5 Falsification lab (build last)
- Baseline: the fair null model predicting exact combinatorial probabilities.
- Challengers: frequency-based "hot/cold" heuristics, Markov/HMM on derived states,
  gradient boosting on engineered features, optional small LSTM.
- Strict walk-forward chronological evaluation; metric is log loss / Brier vs the
  null, never hit-rate. Expected (and useful) result: nothing beats the null on fair
  games. This is the toolkit's built-in defense against seductive pattern claims.

### 5.6 Simulation engine (shared)
- Vectorized numpy draw simulator per ruleset; used for MC p-value calibration,
  prize-distribution simulation, bankroll/drawdown paths, and the "fair histories
  still look streaky" demonstrations.

## 6. Data sources & ingestion priority

| Priority | Source | Use |
|---|---|---|
| Highest | Official operator archives/rules (Powerball, Mega Millions, national lotteries) | Ground truth for draws, odds, tiers, rule changes |
| High | MUSL Numbers/Draw Report APIs; NY Open Data (Powerball, Mega Millions); Data.gov mirrors | Automated ingestion + sales/jackpot backfills |
| High | Regulator/audit documents | Context for judging whether an anomaly is plausibly exploitable |
| Medium | Academic literature parameters | Null models, popularity priors, effect-size expectations |
| Lower | News/informal sources | Legacy cases (e.g., Cash WinFall) only |

Ingestors are per-source plugins with a common interface (`fetch() -> raw`,
`parse(raw) -> canonical rows`), cached raw responses, and idempotent upserts.

## 7. Evaluation & validation standards

- **Audits:** exact/MC-calibrated p-values, FDR control, replication across
  non-overlapping periods. No result reported from a single window.
- **Probability models:** log loss, Brier, calibration curves — never raw hit-rate.
- **Market strategies:** EV, variance, max drawdown, jackpot-sharing rate, realised
  ROI under walk-forward simulation. A strategy that raises hit-rate but worsens
  post-split EV is rejected.
- **Engineering:** pytest suite; odds calculators validated against official
  published odds; simulator validated against closed-form probabilities.

## 8. Roadmap

**Phase 0 — Skeleton (0.5 week)**
Repo scaffolding, pyproject, CI (lint + pytest), game ruleset dataclasses, registry
with Powerball / Mega Millions / 6/49 / Pick 3 / Keno definitions, exact odds
calculator validated against official odds.

**Phase 1 — Data layer (1 week)**
Canonical SQLite schema, CSV importer, MUSL + NY Open Data ingestors, validation/
dedup, backfill of several years of Powerball/Mega Millions draws + sales.

**Phase 2 — Audit suite (1 week)**
Null models, full test battery, MC calibration, FDR, fairness report + plots.
*Milestone: `lo audit powerball` produces a defensible fairness report.*

**Phase 3 — EV & split-risk engine (1–2 weeks)**
Sales model, collision model, roll-down scenarios, net-EV calculator, jackpot-vs-EV
curves. *Milestone: `lo ev` ranks configured games by current-draw EV.*

**Phase 4 — Behaviour + tickets (1–2 weeks)**
Popularity surface, anti-collision generator, covering/wheel engine, syndicate
simulator. *Milestone: `lo tickets --budget 20 --pool 1,4,38,40,46,49 --guarantee 3if4`.*

**Phase 5 — Falsification lab + dashboard (2–3 weeks)**
Walk-forward benchmark harness, Markov/HMM/GBM challengers, static HTML dashboard
consolidating audit, EV, and ticket outputs into the decision layer
(don't play / play anti-collision / play wheel-syndicate).

## 9. Tech stack

Python ≥3.11 · pandas, numpy, scipy · statsmodels (sales/EV) · pulp (covering ILP) ·
mlxtend/scikit-learn, hmmlearn (falsification lab) · matplotlib (reports) · SQLite ·
pytest · optional: pymc (Bayesian bias shrinkage), torch (LSTM benchmark only).

## 10. Legal & ethical guardrails (encoded in the toolkit)

- Public data and published rules only; no scraping that violates operator ToS —
  prefer official APIs/open-data portals.
- Reports always display the unchangeable base odds alongside any strategy output,
  and label what each lever does and does not improve.
- Syndicate module produces share-accounting documentation.
- Responsible-play framing: bankroll simulator surfaces variance and drawdown, not
  just upside; no output ever claims a predictive edge without walk-forward proof
  against the fair null.
