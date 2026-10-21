---
id: "00154"
slug: differentiable-physics-n0-gate-2-heating-error-against-truth
theme: differentiable-physics-pooling-across-substations
also_relevant_to: [hierarchy-and-pooling]
status: planned
size: >-
  simple. Stored: private files under
  data/ml-experiments/00154-differentiable-physics-n0-gate-2-heating-error-against-truth/ (one
  result file per seed); no contract or production table. Serving path: not touched. Degradation
  rule: not touched. Designs: one defensible design, fixed in this plan (score the heating error
  against the injected truth, with the fair oracle and both controls as references). Callers: this
  experiment's scripts only; the gate code is copied from experiment 150 and the pooled fit it
  calls lives in packages/studies/ and is not changed.
hypothesis: >-
  When the partly pooled convex model of experiment 109 holds only 4 winter weeks of one
  substation, its fitted heating curve is closer to the injected truth than the curve of no pooling
  and of complete pooling, as close to the truth as a fair oracle's curve (within 10%), and it
  recovers the sign of the substation's own heating offset from the fleet's curve; the fleet's
  shared curve is recovered as well as an average of per-site fits recovers it.
prediction: >-
  The pooled fit's heating error is below both controls' on 4 or 5 of 5 seeds (experiment 150's
  post hoc January rerun had it below complete pooling on 4 of 5 and below no pooling on 5 of 5),
  and within 10% of the fair oracle's on 5 of 5 seeds (the oracle's January error was the larger
  one). The sign of the substation's own offset is right at 6 or more of 8 substations on 3 or 4 of
  5 seeds: offsets are small relative to the noise at 4 weeks, so the sign line has little power.
decision_rule: >-
  Four lines W1 to W3 below, each counted over 5 seeds, and a line holds when it passes on at least
  4 of 5 seeds. Verdict `supported` if all four lines hold; `partly supported` if exactly three
  hold and W2b is one of them; `inconclusive` if two or fewer hold, checked after W2b; `not
  supported` if W2b fails, checked first, because then a control is as good as the pooled fit. No
  smaller threshold is worth acting on: the gate decides whether the partly pooled GAM may start
  from the pooled convex model.
benchmark: >-
  A fair oracle (the posterior mean of a Gaussian model that knows the fleet mean and every
  family's true spread and the noise variance, and estimates every coefficient, capacity included,
  from the same rows) and two controls (no pooling, penalty 0 rows; complete pooling, penalty
  1,000,000 rows). Experiment 150 is the earlier record.
row_set_id: >-
  not applicable (synthetic fleets on the real calendar and weather of experiment 109's MW
  Primaries; no scored forecast rows)
metric: >-
  The root mean square of the fitted heating contribution minus the injected one over the winter
  rows, in points of effective capacity, pooled over the 8 substations' turns, per seed
date: 2026-10-21
code: ml-experiments/00154-differentiable-physics-n0-gate-2-heating-error-against-truth/
---

# Differentiable physics, N0 gate 2: the partly pooled convex model's heating error against the injected truth

**We are testing a forecasting method on made-up data whose right answer we know.** The method is
the convex demand model of [experiment
109](../00109-differentiable-physics-n0-partially-pooled-convex-model/result.md), which gives each
substation the fleet's shared coefficients plus a penalised deviation of its own ("partial
pooling"). [Experiment
150](../00150-differentiable-physics-n0-gate-rerun-fair-oracle-five-seeds/result.md) could not tell
whether the method shrinks a substation's own heating coefficients correctly, because on 4 June weeks
the heating signal is weak. Here we score the heating error against the injected truth on the 4
weeks before 2025-02-01, when heating dominates demand.

## Design

- **Fleet.** Experiment 150's: 8 synthetic substations built from the real calendar and weather of
  the MW Primaries, each with injected heating coefficients (a shared fleet curve plus a known
  per-substation offset) and noise made of whole weeks of another substation's residuals. The five
  seeds 20261101 to 20261105 give five fleets. The injection is into the synthetic series, not into
  the residuals of a fitted real substation, so that the truth for every coefficient is known.
- **Turns.** Each substation in turn holds only its last 4 weeks before 2025-02-01; the other seven
  keep two years. The deviation penalty is the one experiment 150 tuned on each seed's fleet
  (read from its stored result), so no tuning runs here.
- **Arms.** The pooled fit at that penalty; no pooling (penalty 0 rows); complete pooling (penalty
  1,000,000 rows); and the fair oracle.
- **Heating error.** The root mean square, over the November to March rows of the held-out
  substation's two-year window, of (fitted heating coefficients minus injected) times the heating
  design columns, in points of effective capacity. Measured on the contribution to the prediction,
  because the four smoothing windows of an hour band are nearly collinear. Pooled over the 8 turns
  as a root mean square, per seed. The coefficient-space error is also printed.
- **Offset.** The substation's own offset is the mean heating contribution of its coefficients
  minus the fleet curve's, over the same rows.

## Pre-registered lines

- **W1, the shared curve.** The fitted fleet curve's error against the injected fleet mean, averaged
  over the 8 turns (the fleet curve from each turn's joint fit), is at most 1.1 times the error of
  the average of the per-site fits (the penalty-0 fit of the same turn).
- **W2, close to the fair oracle.** The pooled fit's heating error is at most 1.1 times the fair
  oracle's.
- **W2b, better than both controls.** The pooled fit's heating error is below no pooling's and below
  complete pooling's.
- **W3, the offset's sign.** The pooled fit's offset has the injected offset's sign at 6 or more of
  the 8 substations, and the slope of the fitted offsets on the injected offsets (through the
  origin) is positive.

## Predictions against outcome

(To be filled in after the run.)
