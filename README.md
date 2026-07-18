# Lo_toolkit

A Python toolkit for lawfully analyzing lotteries and improving expected outcomes —
audit-first and EV-first, not a number-prediction oracle.

**Status: planning.** See [PLAN.md](PLAN.md) for the full implementation plan, derived
from the research document *"Lottery Research Methodologies for Lawfully Improving
Lottery Chances"*.

## What it will do

- **Fairness audits** — test draw histories against the exact combinatorial null model
  (hypergeometric moments, spacing tests, Monte Carlo calibration, FDR control).
- **EV & split-risk analysis** — exact odds, jackpot→sales forecasting, jackpot-sharing
  probability, roll-down/mandatory-payout scenario EV.
- **Anti-collision ticket generation** — avoid popular human-picked patterns to reduce
  jackpot-splitting risk.
- **Wheeling & syndicates** — covering designs with verified lower-tier guarantees,
  bundle pricing, and Monte Carlo prize simulation.
- **Falsification lab** — walk-forward benchmarks (Markov/HMM/GBM/LSTM) against the
  fair null, to kill "pattern" claims with evidence.

## What it deliberately won't do

Predict winning numbers. In a fair lottery every combination is equally likely each
draw; the toolkit's predictive models exist only to demonstrate that apparent
patterns don't survive out-of-sample testing.
