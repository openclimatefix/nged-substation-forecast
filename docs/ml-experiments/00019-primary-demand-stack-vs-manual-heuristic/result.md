---
id: "00019"
slug: primary-demand-stack-vs-manual-heuristic
status: done
size: complex
hypothesis: >-
  For the 16 Primary substation demand series, stacking three changes on the single-stage
  ENS-trained quantile model N1 (a 14-day correction from the model's own recent errors, a recency
  half-life of 0.5 years, and a spread factor of 1.15) beats `manual_heuristic` on ENS weather,
  which N1 alone does not.
prediction: >-
  N1 scores 0.44 points of capacity above `manual_heuristic` on the quantile score over all leads
  (6.33 against 5.89). The full stack scores 0.6 to 1.2 below N1 and so 0.2 to 0.8 below
  `manual_heuristic`. The correction supplies most of the gain (0.4 to 0.9 below), the half-life
  between 0.2 below and 0.1 above (it is tested on a frozen model, so a loss is possible), and the
  spread factor 0.03 to 0.05 below. The three single gains add to within 0.2 points of the stack's
  gain. The 99.44% interval on C1 is about plus or minus 0.7 points (00013's C1 was plus or minus
  0.67 at 99%), so C1 may well straddle zero even when the point estimate is as predicted; C2, a
  paired contrast against N1, is expected to be much tighter.
decision_rule: >-
  Nine planned contrasts on the 16 Primary substations, quantile score over all leads, read at
  99.44% (Bonferroni across nine), whole months and seeds resampled. C1 (primary) is the full stack
  minus `manual_heuristic`. C2 is the full stack minus N1 (the combined gain), and C3 is the full
  stack minus the production recipe. C4 to C6 are the full stack with the correction, the half-life,
  or the spread factor dropped, minus the full stack (the loss from dropping one component). C7 to
  C9 are N1 plus one component minus N1. The stack is supported when C1 and C2 have intervals wholly
  below zero and C2's point estimate is at most -0.3 points, a threshold that justifies the effort
  of porting a pipeline step that reads observed power; it is `small gain` when both intervals are
  wholly below zero and C2's point estimate is above -0.3; it is refuted when C1's interval is
  wholly above zero or C2's interval is wholly above zero (the stack is worse than N1); otherwise it
  is inconclusive. A component earns a place in the release when its drop contrast (C4 to C6) has an
  interval wholly above zero and a point estimate of at least the threshold: 0 for the half-life and
  the spread factor (they change a number in the fit or the output) and 0.10 points for the
  correction (a new pipeline step). C3 and C7 to C9 decide no verdict but count in the Bonferroni
  family. The mean absolute error of the median is reported beside the quantile score and decides
  nothing. benchmark: >-
  `manual_heuristic`, `climatology`, and the production recipe (stored, 00013), and N1 (00013).
row_set_id: fold_a_every5d_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over 9 levels) and mean absolute error
  of the median, in points of capacity (1 point is 1% of the series' effective capacity)
verdict: supported
best_arm: >-
  The full stack (5.32 points on the quantile score over all leads, against N1's 6.33), level with
  N1 plus the correction alone (5.34)
headline: >-
  C1, the full stack minus manual_heuristic, quantile score over all leads: -0.56 points of capacity
  (99.44% interval -0.93 to -0.26). C2, the stack minus N1: -1.00 (-1.88 to -0.14). The correction
  supplies nearly all of it (C4, +0.97 when dropped, +0.15 to +1.81).
lessons: >-
  N1's forecasts over the validation year sat 2.6 points too low, and the 14-day correction cut that
  bias to 0.5. The half-life of 0.5 years added nothing to a model frozen at its training end, and
  the spread factor of 1.15 added 0.03 to 0.05 points and lifted 5% to 95% coverage from 80% to 85%.
advice:
  - >-
    Add the 14-day error correction to the Primary demand model before any other change, because it
    lowered N1's quantile score by 0.99 (C7) and put N1 below manual_heuristic (C1).
  - >-
    Do not add the 0.5-year half-life to a model that is frozen at its training end, because it
    moved the quantile score by -0.01 (-0.03 to +0.02, C5); test it with monthly retraining instead.
  - >-
    Apply the 1.15 spread factor to N1-style quantile output, because dropping it from the stack
    raised the quantile score by 0.03 (+0.005 to +0.05, C6), though 5% to 95% coverage is still 85%.
  - >-
    Test the correction with forecasts made daily, not every 5 days, and with monthly retraining
    in place, because both change how much bias is left to remove (limits).
production_status: candidate
adopted_in: none
pull_request: https://github.com/openclimatefix/nged-substation-forecast/pull/1225
code: ml-experiments/00019-primary-demand-stack-vs-manual-heuristic/
---

# Experiment 19: does stacking small gains beat `manual_heuristic` for Primary demand?

We asked whether stacking several small improvements gives a demand forecast for Primary substations
(substations that step power down to the local distribution network) that beats a stored baseline
built from the power at the same time in earlier weeks. The stack added to a single-step model a
correction using the previous 14 days of forecast errors, a weighting of recent years, and a 15%
widening of the forecast range. We scored it on 16 Primary substations over 73 forecast runs from
July 2025 to June 2026. The stack beat the baseline and the single-step model, and the 14-day
correction supplied nearly all of the gain.

**Supported: the full stack beats `manual_heuristic` and N1, and the 14-day error correction
supplies nearly all of the gain (-0.99 of -1.00).** N1 is the single-stage quantile model trained on
the control member of the ensemble weather forecast (ENS) of the European Centre for Medium-Range
Weather Forecasts. Scores are the quantile score over all leads, in points of capacity (1 point is
1% of a series' effective capacity), on 16 Primary substations, 73 forecast runs every 5 days from
July 2025 to June 2026, with ENS weather. C1, the stack minus `manual_heuristic`, is -0.56 (99.44%
interval -0.93 to -0.26), and the mean absolute error of the median agrees (-0.88, -1.51 to -0.30).
C2, the stack minus N1, is -1.00 (-1.88 to -0.14), which meets the -0.3 threshold. The stack's
quantile score is 5.32, N1's is 6.33, and `manual_heuristic`'s is 5.89. Up to 48 hours the stack
scores 4.61 against 5.68 (C1, 95% interval -1.38 to -0.79), and from 3 days on 5.49 against 5.94.
The quantile score measures how well the forecast range fits what happened, and smaller is better.
`manual_heuristic` is a stored baseline that forecasts from the power at the same time of day in
earlier weeks and years.

![Quantile score by arm](figure_1.svg)

| Arm | Quantile score | Median error | 5%-95% band covers | 10%-90% band covers |
|---|---|---|---|---|
| `manual_heuristic` | 5.89 | 8.23 | 81% | 72% |
| `climatology` | 7.46 | 10.33 | 83% | 74% |
| Production recipe | 9.52 | 9.92 | 10% | 8% |
| N1 | 6.33 | 8.45 | 80% | 71% |
| N1 plus correction | 5.34 | 7.34 | 80% | 70% |
| N1 with half-life 0.5 years | 6.34 | 8.46 | 80% | 70% |
| N1 plus spread factor | 6.28 | 8.45 | 85% | 77% |
| Full stack | 5.32 | 7.35 | 85% | 76% |
| Stack without correction | 6.29 | 8.46 | 85% | 77% |
| Stack without half-life | 5.32 | 7.34 | 85% | 76% |
| Stack without spread factor | 5.35 | 7.35 | 80% | 70% |

![The nine planned contrasts](figure_2.svg)

**The correction removes a level bias.** N1, frozen at 2025-06-30, forecast the validation year
2.6 points too low on average. The correction cuts that bias to 0.5, and the median error falls by
1.1 points (C7: -0.99 on the quantile score, -1.84 to -0.15 at 99.44%, 7 of 12 months below
zero). Dropping the correction from the stack raises the score by 0.97 (C4, +0.15 to +1.81).
**The half-life does nothing here.** C8 is +0.01 (-0.01 to +0.05) and C5 is -0.005 (-0.026 to
+0.016), so the 0.5-year half-life earns no place. **The spread factor is worth a small gain.** C9
is -0.05 (-0.09 to -0.01), and dropping the factor raises the stack's score by 0.026 (C6, +0.005
to +0.049, threshold zero), though 5% to 95% coverage is only 85%.

**The singles add up to the stack's gain.** The three single gains sum to -1.02 (C7 -0.99, C8 +0.01,
C9 -0.05) against the stack's -1.00 (C2), so the components overlap by at most 0.02 points.

**The stack's gains fall in the predicted ranges, except the correction's, which is larger.** The
stack's gain over N1 (-1.00) falls in the predicted -0.6 to -1.2, and over `manual_heuristic`
(-0.56) in -0.2 to -0.8. The correction's gain (-0.99) is larger than the predicted drop of 0.4 to
0.9, the half-life's (+0.01) is inside its range, and the spread factor's (-0.05) is at the top of
its range. C1's interval is narrower than the plus or minus 0.7 predicted.

## Limits

- **The correction's history comes from 5-day-spaced runs.** A past half-hour's forecast has a
  lead of 0 to 5 days, so the history errors are larger than a daily service would see, and the
  correction is applied at every lead to 15 days. A daily service would use leads under 1 day.
- **N1 is never retrained.** The correction partly stands in for monthly retraining, which
  experiment 00017 found lowers the error by 0.5 points (retrained monthly against yearly); the two
  have not been combined.
- **The production row's coverage of 10% is a property of the stored forecast, not of this
  experiment.** Its stored quantiles come from ensemble members and are not a calibrated band.
- **The half-life was chosen on the scored year,** and 26 of 1,460 series-runs (73 runs times 20
  series; the first runs of some series) had no history and received a correction of zero.
- **The 3.5-hour telemetry delay is assumed,** as in experiment 00013.

## Plan, written before any code ran

## Plan: sizing

- **Stored data:** prediction sets only, under `data/ml-experiments/`.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **Defensible designs:** more than one; the correction's history source and window, and the
  half-life, are fixed below before any fit.
- **Callers:** the scripts copy 00013's helpers and read its stored predictions, which the author
  could name without searching.

Size: complex, because more than one defensible design tests the hypothesis. One Sonnet review of
the plan, one of the code (leakage, fit stopping at the training end), and one of the write-up
(Opus is kept for work that is truly vital).

## Plan: arms

Rows, series, columns, lags, seeds (0, 1, 2), and pooling over the 51 members are those of
experiment 00013 (73 runs of `fold_a_every5d_v1`, valid times before 2026-07-01).

- **Stored:** `manual_heuristic`, `climatology`, the production recipe, and N1, read from 00013.
- **N1H:** N1 refitted with a recency half-life of 0.5 years in place of 2 years, anchored at
  2025-06-30. Nothing else changes. The model is frozen at that date, as N1 is.
- **Correction (C):** for each series and run with origin `T`, the mean of (observed power minus the
  median forecast) over the half-hours in the 14 days ending at the newest usable observation
  (00013's telemetry assumption: 3.5 hours before `T`), added to every quantile. The forecast for a
  past half-hour comes from the validation run with the latest origin at or before that half-hour,
  so only earlier runs are used. A run with fewer than 3 days of scored history gets no correction.
  The runs are 5 days apart, so a past forecast has a lead of 0 to 5 days; a daily service would
  use leads under 1 day, and the error of the correction on these rows is probably larger than the
  service's.
- **Spread (S):** each quantile `q` becomes `q50 + 1.15 * (q - q50)`, with `q50` the corrected
  median where the correction applies. The factor is 00018's, fitted on the staged chain, not on N1.
- **Arms by component:** N1, N1+C, N1+H, N1+S, the full stack F = H+C+S, and the three drops
  F-C (H+S), F-H (N1+C+S), F-S (H+C).

## Plan: leakage checks

The scoring script asserts that every observation entering a correction has a grid index at or
before the origin minus 7 half-hours, that the run supplying the forecast for a past half-hour has
an origin at or before that half-hour, that a run with fewer than 3 days of history gets a
correction of zero (and reports how many rows that is), and that N1H's training rows end at
2025-06-30 23:30. Observed power is read through `load_series_arrays`, as in 00013, and the median
is the pooled mixture's q50. The spread factor is not re-tuned on N1; the 5% to 95% coverage of
every arm is reported.

## Plan: limits stated in advance

- **The half-life was chosen on the scored year.** 00017 chose 0.5 years on origins through June
  2026, which overlap this validation year, and 00017 retrained monthly. N1H is frozen and trained
  on 15 months, so its weights fall to about a quarter at the start of the archive.
- **The correction's history has leads of 0 to 5 days** (runs are 5 days apart) and is applied to
  leads up to 15 days, and N1 already carries origin lags, so 00017's 1.5-point gain, measured on a
  model with no power lags, may not transfer.

## Plan: past weather

ENS weather in training and scoring; lags and the correction use observed power before the origin.
No past weather is used.

## Plan: earlier advice

Followed: compare against `manual_heuristic` on both metrics (00013, 00018); test a deployable
recent-error correction (00013, 00017); the 0.5-year half-life (00017); the 1.15 spread factor
(00018). Departure: N1 is not retrained monthly, so the half-life is tested without 00017's
moving anchor.
