---
id: '00264'
slug: dp-gas-engine-asymmetric-bands-at-d8
title: >-
  Can D8's forecast bands carry the uncertainty that an undetectable gas engine adds, by widening in
  the direction the engine pushes?
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - calibration-and-uncertainty
  - curtailment-cost
status: planned
size: >-
  complex, by one trigger: more than one design would defensibly test the hypothesis (a lower-tail
  recalibration, a two-sided recalibration, and a mixture over running engine units, each with
  several fitting choices). Stored: private prediction files under
  data/ml-experiments/00264-dp-gas-engine-asymmetric-bands-at-d8/ only. Serving path: not touched.
  Degradation rule: not touched; the "engine visible" switch is a proposal that this experiment
  scores and does not install. Callers: this folder's scripts, which read experiment 191's stored
  forecasts and call reviewed `studies.*` modules unchanged. The maintainer's brief limits the
  reviewers to Sonnet, so the plan and code reviews are Sonnet reviews.
hypothesis: >-
  D8's forecast bands are too narrow in both directions, and part of the lower-tail miss is a 6 MW
  gas engine (probably three units of about 2 MW) that runs in spells and cannot be detected. A
  spread that widens as-of, using only residuals already resolved at the forecast origin, and that
  widens most in the direction the engine pushes (net demand falls), gives honest q05 to q95 bands
  at D8 without raising the quantile score at the Primaries that have no engine.
prediction: >-
  Written before any forecast is scored, but after the descriptive table of experiment 191's stored
  reference forecasts was read (monthly coverage and tail shares, see Plan, "What was seen first"),
  so P1 and P2 are not blind. Each prediction can fail. (P1) Arm 2, which widens only the lower
  quantiles, does not bring the q05-to-q95 coverage within 5 points of 0.90 in any quarter, because
  22% to 44% of D8 readings in the scored year lie above the reference's q95. (P2) Arm 2b, which
  widens both sides by the same as-of method, brings the coverage within 5 points of 0.90 in at
  least 3 of the 4 quarters. (P3) Arm 3, the mixture over 0 to 3 running units, does the same in at
  least 3 quarters, and its quantile score is within 2% of arm 2b's, so the mixture does not repay
  its added complexity. (P4) Arm 4 (arm 3 with a price known only in hindsight) lowers the quantile
  score by under 1% against arm 3, because the engine's chance of running rose only from 9% to 27%
  across price fifths in the one summer we hold its output for. (P5) The "engine visible" switch
  turns off at D8 between 2026-03-01 and 2026-05-15, and stays on at D8 before then; it is off at
  D10 and D16 throughout. (P6) With the switch forced on, the mixture's fitted chance of at least
  one running unit averages under 0.05 at D10 and at D16.
decision_rule: >-
  Reference: arm 1, experiment 191's reference forecast (arm a0, the GAM-K engine of experiment
  152, no price input) with its residual-table spread, on `fold_a_daily_gap_v1`, seed 0, scored
  year 2025-07-02 to 2026-06-30, leads 0.5 to 168 hours, outside the hand-filtered switching events.
  The quantile score is twice the mean pinball loss over 11 quantiles, in points of effective
  capacity. `supported`: arm 2 or arm 3 has q05-to-q95 coverage in [0.85, 0.95] in each of the four
  quarters of the scored year (valid times before 2025-08-01 are excluded from the first quarter, as
  a 30-day burn-in while the as-of fit gathers its first residuals, and the full-quarter value is
  reported beside it), at D8; and its D8 quantile score is at most 0.5% above arm 1's; and at D10
  and at D16 its quantile score is at most 0.1% above arm 1's and its q05-to-q95 coverage is no
  more than 1 point below arm 1's, in both the free-switch and the forced-switch runs. `partly
  supported`: only arm 2b passes, or a candidate passes the coverage test and misses the score
  limit. `not supported`: otherwise. The mixture earns its place over the plain widening only if
  arm 3's D8 quantile score is at least 1% below arm 2b's with the 95% month-block interval wholly
  below zero; if not, the recommendation is arm 2b. The 1% threshold is set by cost: arm 3 adds an
  expectation-maximisation fit and a visibility switch to the forecast, and arm 2b adds a
  rolling table of quantile ratios.
benchmark: >-
  Arm 1 at the same site, rows and leads; experiment 191's reference forecast is the stored
  prediction. Experiment 251's best-guess stack does not cover D8 (its six substations are D5, D9,
  D10, D11, D14, and D16), so the median here is the GAM-K reference and not the stack. This is a
  departure from the brief and the plan says so.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score and q05-to-q95 coverage by quarter at D8; coverage and pinball loss at q01, q05,
  q95, and q99; mean absolute error of the mean forecast; the placebo sites D10 and D16 on the same
  columns.
code: ml-experiments/00264-dp-gas-engine-asymmetric-bands-at-d8/
date: 2026-10-23
---

**This experiment asks whether D8's forecast bands can carry the uncertainty that an undetectable
gas engine adds.** D8 is a primary substation of National Grid Electricity Distribution (NGED).
Its reading appears to include the output of one 6 MW gas engine, probably three units of about 2
MW each, whose own meter reading stopped arriving in August 2023 (the design is in the private
brief `price_sensitive_generation_design.md`). Three attempts to detect the engine from D8's readings
failed ([experiment 191](../00191-dp-price-sensitive-assets-in-the-history/result.md), [experiment
198](../00198-d8-summer-spikes-cooling-load-and-residual-test/result.md), and [experiment
199](../00199-ps2b-engine-detector-v2-ramp-night-price/result.md)). So this experiment does not
detect the engine. It asks whether the forecast's bands can carry the uncertainty that the engine
adds, by widening as-of: using only the residuals that are already known when each forecast is made.

## Plan

### What was seen first

- **The reference forecast is too narrow in both directions, not only downwards.** Before writing
  this plan we tabulated experiment 191's stored reference forecasts at D8 by month, using for each
  valid time the forecast with the shortest lead. In the 12 months the reference's q05-to-q95
  coverage runs from 0.27 to 0.51. Between 22% and 44% of readings lie above q95, and 25% to 47%
  lie below q05. The median distance from q50 to q05 is only 6.8 to 9.8 points (0.7 to 1.0 MW),
  against a mean absolute error of about 15 points. The share of readings more than 25 points (2.5
  MW) below q50 falls from 5% to 16% in July to March to 3.5% to 5.8% in April to June 2026. At D10
  and D16 the same table shows coverage of 0.77 to 0.97, so the narrow bands are particular to D8.
  The lower-tail-only widening that the maintainer's brief names as arm 2 therefore cannot be
  enough on its own, and we added arm 2b, which widens both sides, to separate "the band is too
  narrow" from "the band is too narrow in the direction the engine pushes".
- **These numbers are descriptive, were read before the arms were fixed, and fix the thresholds
  below only where this plan says so.** The visibility threshold of arm 3 (4%) was chosen after
  seeing that table, and the plan reports the result at 2% and 6% as well.

### Inputs

- **Forecasts.** Experiment 191's stored arm `a0` (gated variant, which equals the ungated variant at
  D8 because D8 has no pre-connection rows) at D8 (`time_series_id` 26), and at the placebo sites D10
  (10) and D16 (18). Each row holds `q01` to `q99`, `mean`, `actual`, `lead_hours`, and `run`.
  The files are `data/ml-experiments/00191-*/work/preds/gated/scored_<site>.parquet`.
- **Units.** One point is 1% of the site's effective capacity, which is 0.1 MW at D8. The engine unit
  size of 1.97 MW is 19.7 points, and the "drop" threshold of -2.5 MW is -25 points. The placebo
  sites use the same constants in points, because the question there is whether the same algorithm
  with the same settings leaves a site without an engine alone.
- **Price columns for arm 4 only.** The stored `spread_positive` and `price_p75` flags in the same
  files. They are hindsight inputs and never run in a live service.

### The as-of rule

**A residual enters a fit at origin O only if its valid time is at or before O.** For each valid
time, the training row is the stored forecast with the shortest lead, so each valid time counts
once. The residual is `actual` minus `q50`. Every fit is refitted every 7 origins and applied to the
next 7 daily origins, and fit windows are the trailing 365 days of resolved valid times. A fit
needs at least 28 days (1,300 rows) of resolved valid times. Before that, every arm equals arm 1.
This is the cold start. D8's scored forecasts begin on 2025-07-02, and no earlier D8 forecast is
stored, so the first weeks of the first quarter cannot be improved. The burn-in rule in the
decision rule handles that quarter.

### Arms

1. **Control.** Arm 1, as above.
2. **Lower-tail widening** (the maintainer's arm 2). For each row, the normalised shortfall is
   `s = (q50 - actual) / (q50 - q05)`. In each cell of hour-of-day block (00-06, 06-12, 12-18, and
   18-24 UTC) by season (December-February, March-May, June-August, September-November), and
   with the cell falling back to the hour block alone when it holds under 300 resolved rows, let
   `S_tau` be the empirical `1 - tau` quantile of `s` for tau in {0.01, 0.05, 0.10, 0.25}. The new
   lower quantile is `q50 - max(r_tau, S_tau) * (q50 - q05)`, where `r_tau` is the reference's own
   ratio `(q50 - q_tau) / (q50 - q05)`. The widening never narrows a band, and the upper quantiles
   and the mean are unchanged.
3. **Arm 2b (added by this plan).** The same method applied to both sides: the upper quantiles
   `q75`, `q90`, `q95`, and `q99` are recalibrated the same way from the normalised overshoot
   `(actual - q50) / (q95 - q50)`.
4. **Mixture over running units (arm 3).** The reading is modelled as the reference's distribution
   shifted by `delta - k * g`, where `k` is the number of running units (0, 1, 2, or 3) and `g` is
   1.97 MW (19.7 points), and with the reference's own spread multiplied by a noise scale `c`. The
   reference distribution is the piecewise-linear cumulative distribution through its 11 quantiles
   with exponential tails. The probability of `k` follows a multinomial logistic model of five
   inputs: hour block, season, weekend, the share of half-hours in the 14 days before the origin
   with a residual below -25 points, and the visibility switch below. The model is fitted by
   expectation-maximisation on the resolved residuals, with `c` on the grid {1, 1.5, 2, 3, 4, 6}
   and the likelihood-maximising `c` kept, and with `delta` set so that the mixture's mean equals the
   reference's mean on the training rows (the reference was trained on readings that include the
   engine, so its mean already sits at the average engine effect). The forecast quantiles come from
   inverting the mixture's cumulative distribution. Ridge penalty 1.0 on the logistic weights,
   30 expectation-maximisation iterations.
   **The engine visibility switch** is on when the share of half-hours in the 30 days before the
   origin with a residual below -25 points is at least 4%. When it is off, `pi_0 = 1`, `delta = 0`,
   and `c = 1`, so arm 3 equals arm 1. This protects against the April 2026 change. Report every
   date the switch changes state at each site, and the sensitivity at 2% and 6% (post hoc).
5. **Arm 4.** Arm 3 with `spread_positive` and `price_p75` at the valid time added to the logistic
   model. It is a hindsight upper bound for the value of a price forecast.
6. **Placebo (arm 5).** Arms 2, 2b, and 3 at D10 and D16, with the switch free, and with the switch
   forced on (a stress test that is the real placebo for the mixture, because the free switch is
   off at both sites).
7. **Not run.** The small neural network for the mixture probabilities. The brief welcomes it only if
   it beats arm 3 by a pre-registered margin, and the plan sets none, so it is listed under "What
   to try next" in the write-up.

### Contrasts and scoring

- **Contrasts.** At D8: arms 2, 2b, 3, and 4 each minus arm 1 on the quantile score, arm 3 minus arm
  2b, and arm 4 minus arm 3. Each is a mean paired difference with a 95% interval from 20,000
  resamples of the 12 calendar months of the scored year, and a Bonferroni-corrected 99.2% interval
  (6 contrasts) beside it.
- **Cuts.** All rows; the four quarters; leads 0 to 24, 24 to 72, and 72 to 168 hours; months with the
  switch on and off; winter (November to March) evenings (16:00 to 20:00 UTC).
- **Tail table.** Coverage and pinball loss at q01, q05, q95, and q99 per arm, with
  `studies.tail_calibration`.
- **Checks.** The median forecast minus the mean forecast, per site and arm. The error of the mean
  forecast beside each quantile-score contrast. The leave-one-month-out sign of each D8 contrast. The
  audit triggers, listed in the write-up.
- **Curtailment.** `studies.node_curtailment` needs a bulk supply point's forecast and its generators'
  forecasts, and this experiment forecasts a single Primary, so the metric cannot be run. The
  stand-in is the Brier score of the event "the reading falls below D8's 2nd percentile of readings
  in the 12 months before 2025-07-02", computed from each arm's mixture of quantiles by linear
  interpolation of its cumulative distribution. The write-up labels it a stand-in and does not call
  it a curtailment cost.

### Compute

CPU only, 4 cores at most, thread caps of 1, inside `systemd-run --user --scope -p MemoryMax=16G -p
OOMPolicy=continue`. The fits use the 17,000 shortest-lead rows per site and refit 52 times, so the
whole run should take minutes. The three sites run in parallel workers.

### Stop rule

Stop and report if arm 1, rebuilt from the stored files, does not reproduce experiment 191's D8
absolute scores (quantile score 10.95, mean absolute error 15.18 points, q05-to-q95 coverage 0.42)
to within 0.01, or if any arm's row keys differ from arm 1's.

### Anonymisation

The metered generator is named G2 only. No generator time series with a name or an identifier is
published. D8, D10, and D16 are substations and may be named.
