---
id: "00090"
slug: sw2-28-lag-unusualness-inputs-all-rows-and-switching
status: planned
size: simple
hypothesis: >-
  For the 16 Primary substation demand series, telling the monthly-refitted (P-roll) XGBoost model
  how unusual each of its three lags is (the probability integral transform, PIT, of each lag
  against a probabilistic climatology of the series) lowers the quantile score on all half-hours
  outside switching, and widens the quantiles on the half-hours whose lag lies inside a logged
  switching event, when both arms train on normal-arrangement targets only.
prediction: >-
  On all rows outside switching the PIT arm changes the quantile score by -0.05 to +0.01 points of
  capacity (experiment 18's residual lags: -0.04, -0.12 to +0.07, with ERA5 at lag times). On the
  rows outside switching with a lag inside a logged event it changes the score by -0.3 to 0.0 and
  moves the coverage of the 5%-to-95% band from about 0.78 towards 0.90. The switching gain is small
  because masked training holds few rows with an unusual lag and a normal target.
decision_rule: >-
  Two planned contrasts, `clim_pit` minus `plain`, quantile score over all leads, 97.5% interval
  (Bonferroni across two), whole validation months resampled. C1, all rows outside switching
  (the general headline). C2, the rows outside switching whose lag is inside a logged (oracle)
  event (the switching headline). The PIT inputs are `supported` for the general question when
  C1's upper end is below -0.02 points (0.4% of the plain score), `small gain` when C1's upper end
  is below zero but not below -0.02, `refuted` when C1's lower end is above zero, and
  `inconclusive` otherwise. The same words, from C2, answer the switching question. The threshold
  is -0.02 because the inputs add a climatology table and three columns to the serving path, which
  has to pay for that with a gain whose interval excludes zero. Tail calibration (coverage and
  pinball loss at q01, q05, q95, q99) is descriptive.
benchmark: >-
  The `plain` arm refitted by this experiment's own code, 11 quantile levels, targets inside any
  switching event or closure removed from training.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over 11 levels), in points of capacity
code: ml-experiments/00090-sw2-28-lag-unusualness-inputs-all-rows-and-switching/
verdict: pending
production_status: pending
adopted_in: none
---

# Experiment 90: does telling the model how unusual each lag is improve the forecast?

Status: planned. The result is written after the fits.

- **Sizing** (the five `plan-issue` triggers). Stores: nothing new beyond private predictions and
  boosters. Production serving path: no (R&D only; a later release would add a climatology table).
  Degradation rule: no. More than one defensible design: yes in principle (the PIT reference could
  be a climatology or a lag-free booster), so this experiment runs only the climatology reference,
  the cheapest, and leaves the others to follow if it helps. Unfamiliar callers: no, the code
  copies [experiment 86](../00086-sw2-12-detector-labels-for-nra-forecast/result.md). The size is
  simple, reviewed by one Sonnet code reviewer and one Sonnet write-up reviewer.
- **Arms.** `plain`: the N1 inputs and raw lags. `clim_pit`: the same plus the PIT of the three lags
  (origin, same time yesterday, same time last week) against the series' climatology from the
  refit's training window, in cells of local month, weekday or weekend, and half-hour of day (each
  pooled with the two half-hours either side; non-switching power only). Both arms drop training
  rows whose valid time is inside an oracle event, a logged closure, or an XL event.
- **Schedule.** Monthly refits on a rolling 12-month window, 150 rounds, eta 0.15 (P-roll), one
  seed, 11 levels. The contrast could depend on the schedule if weekly refits tracked drift better.
- **Past-weather caveat.** The PIT uses power only, no weather. The XGBoost inputs are those of
  experiment 86 (ERA5 at lag times, a ceiling for that reason).
- **Not run here.** The stage 1 PIT and neighbour arms need the lag-free booster of SW2-27, which
  does not exist yet. Experiment 88's injected cases are not rescored here.
