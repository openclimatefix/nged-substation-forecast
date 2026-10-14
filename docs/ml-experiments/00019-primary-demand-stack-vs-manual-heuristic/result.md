---
id: "00019"
slug: primary-demand-stack-vs-manual-heuristic
status: planned
size: complex
hypothesis: >-
  For the 16 Primary substation demand series, stacking three changes on the single-stage ENS-trained
  quantile model N1 (a 14-day correction from the model's own recent errors, a recency half-life of
  0.5 years, and a spread factor of 1.15) beats `manual_heuristic` on ENS weather, which N1 alone
  does not.
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
  below zero and C2's point estimate is at most -0.3 points, a threshold that pays for porting a
  pipeline step that reads observed power; it is `small gain` when both intervals are wholly below
  zero and C2's point estimate is above -0.3; it is refuted when C1's interval is wholly above zero
  or C2's interval is wholly above zero (the stack is worse than N1); otherwise it is inconclusive.
  A component earns a place in the release when its drop contrast (C4 to C6) has an interval wholly
  above zero and a point estimate of at least the threshold: 0 for the half-life and the spread
  factor (they change a number in the fit or the output) and 0.10 points for the correction (a new
  pipeline step). C3 and C7 to C9 decide no verdict but count in the Bonferroni family. The mean absolute error of the median is reported beside the quantile score and
  decides nothing.
benchmark: >-
  `manual_heuristic`, `climatology`, and the production recipe (stored, 00013), and N1 (00013).
row_set_id: fold_a_every5d_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over 9 levels) and mean absolute error
  of the median, in points of capacity (1 point is 1% of the series' effective capacity)
---

# Experiment 19: does stacking small gains beat `manual_heuristic` for Primary demand?

## Plan: sizing

- **Stored data:** prediction sets only, under `data/ml-experiments/`.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **Defensible designs:** more than one; the correction's history source and window, and the
  half-life, are fixed below before any fit.
- **Callers:** the scripts copy 00013's helpers and read its stored predictions, which the author
  could name without searching.

Size: medium. One Sonnet review of the plan, one of the code (leakage, fit stopping at the training
end), and one of the write-up.

## Plan: arms

Rows, series, columns, lags, seeds (0, 1, 2), and pooling over the 51 members are those of
experiment 00013 (73 runs of `fold_a_every5d_v1`, valid times before 2026-07-01).

- **Stored:** `manual_heuristic`, `climatology`, the production recipe, and N1, read from 00013.
- **N1H:** N1 refitted with a recency half-life of 0.5 years in place of 2 years, anchored at
  2025-06-30. Nothing else changes. The model is frozen at that date, as N1 is.
- **Correction (C):** for each series and run with origin `T`, the mean of (observed power minus the
  median forecast) over the half-hours in the 14 days ending at the newest usable observation (00013's
  telemetry assumption: 3.5 hours before `T`), added to every quantile. The forecast for a past
  half-hour comes from the validation run with the latest origin at or before that half-hour, so
  only earlier runs are used. A run with fewer than 3 days of scored history gets no correction.
  The runs are 5 days apart, so a past forecast has a lead of 0 to 5 days; a daily service would
  use leads under 1 day, and the error of the correction on these rows is probably larger than the
  service's.
- **Spread (S):** each quantile `q` becomes `q50 + 1.15 * (q - q50)`, with `q50` the corrected median
  where the correction applies. The factor is 00018's, fitted on the staged chain, not on N1.
- **Arms by component:** N1, N1+C, N1+H, N1+S, the full stack F = H+C+S, and the three drops
  F-C (H+S), F-H (N1+C+S), F-S (H+C).

## Plan: leakage checks

The scoring script asserts that every observation entering a correction has a grid index at or
before the origin minus 7 half-hours, that the run supplying the forecast for a past half-hour has
an origin at or before that half-hour, that a run with fewer than 3 days of history gets a
correction of zero (and reports how many rows that is), and that N1H's training rows end at
2025-06-30 23:30. Observed power is read through `load_series_arrays`, as in 00013, and the median
is the pooled mixture's q50. The spread factor is not re-tuned on N1; the 5% to 95% coverage of every
arm is reported.

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
