---
id: '00274'
slug: lsc-long-arrangement-audit-gam-refit
title: >-
  How long can an arrangement last before a no-lag forecast learns it as normal?
theme: lsc-detectors
also_relevant_to:
  - retraining-and-training-history
status: planned
size: >-
  medium. Stored: private bias tables under
  data/ml-experiments/00274-lsc-long-arrangement-audit-gam-refit/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one is defensible for how a transfer is
  injected and for which training window the fit uses, so both are fixed below before any score is
  read. Callers: this folder's scripts, which call the reviewed `studies.dp_gam` fit and predict
  functions unchanged. Step 8 of the LSC decision program (private brief
  `briefs/sw_lsc_decision_program_brief.md`).
hypothesis: >-
  A forecast with no lagged power learns a long-lasting change of running arrangement as the
  normal pattern, because the monthly refit trains on whatever the meter recorded. The learned share
  grows with the duration of the arrangement, and the duration beyond which the forecast of the
  normal running arrangement (NRA) is wrong by more than a stated tolerance sets the limit that can
  be told to NGED.
prediction: >-
  Written before any score is read. A 30-day transfer of 25% of the receiver's load, injected into
  the training data of the lagless convex additive model (GAM_ONLY) and refitted monthly, biases
  the forecast of the true NRA by under 0.1 points of effective capacity in the 48 hours after the
  transfer ends (95% interval for the mean over substations and onset dates below 0.1). This is
  expected to fail: the 24-month window gives 30 days about 4% of the weight, so a bias of 0.5 to 1
  points is expected right after the end. Further predictions: the bias in the 48 hours after the
  end grows roughly in proportion to the duration up to 90 days; a permanent step is learned to
  half of its size within 12 months with the 24-month window and within 6 months with the 12-month
  window; and the rows before the onset have a bias of exactly zero (positive control).
decision_rule: >-
  Pre-registered before any score is read. Bias is the signed mean of (forecast trained on the
  injected series minus forecast trained on the clean series) over the rows of a group, multiplied
  by the sign of the transfer, so that a positive bias means the transfer was learned as normal.
  The two forecasts come from the same cut, weather, and settings, so the bias is exactly what
  the injection changed, and the bias against the true NRA series equals it. The tolerance for
  "learned as normal" is 0.1 points of effective capacity (the brief's figure), and 0.5 points is
  reported beside it. The limit for NGED is the longest tested duration whose bias in the 48 hours
  after the end has a 95% interval upper bound under the tolerance, per size and window, with the
  duration interpolated linearly in log duration between tested values. The threshold is a
  statement to NGED, so no porting cost applies. Verdict: `supported` when the 30-day, 25% bias
  after the end is under 0.1 with the interval upper bound under 0.1; `not supported` when the lower
  bound is above 0.1; `partly supported` otherwise.
benchmark: >-
  The experiment's own zero-size control and the clean-trained forecast of each scenario. Reference
  for the setting: the GAM_ONLY arm of experiments 208 and 258 (a fixed fit, no refit).
row_set_id: >-
  Own rows: monthly forecast periods at 10 Primary substations from onset dates in December 2022,
  June 2023, December 2023, and June 2024; not comparable with the leaderboard
metric: >-
  Signed mean of (forecast trained on injected series minus forecast trained on clean series), in
  points of effective capacity, by phase.
code: ml-experiments/00274-lsc-long-arrangement-audit-gam-refit/
production_status: not-for-production
adopted_in: none
---

# Experiment 274: how long can an arrangement last before a no-lag forecast learns it as normal?

*(Planned. The finished write-up replaces this page.)*

## Design

- **Forecast under test.** GAM_ONLY is the convex additive model of `studies.dp_gam` with the GAM-K
  settings of experiment 252 (ridge 0.1, no drift, fitted solar capacity, no lagged power), refitted
  on the first of every month. The default window is the last 24 months; the 12-month and
  whole-history windows are run as sensitivity arms, fixed here. The weather is the feature cache's
  actual ERA5 temperature and CAMS irradiance (lab condition: ERA5 arrives about 5 days late, so a
  live service would use ensemble weather; this isolates the effect of the training data).
- **Series.** D2, D3, D4, D5, D10, D11, D12, D19, D21, and D22: Primary substations with a full
  history from 2019-09. D8 is excluded (its series changes meaning in 2025).
- **Injection.** The transfer adds `size x y(t)` to the receiver's power in points of effective
  capacity (a share of the receiver's own load, so it follows the daily shape, as in experiment
  141), where `y` is the cached power; rows with no reading stay missing. Sizes are 10%, 25%, and
  40%. Durations are 7, 14, 30, 60, and 90 days, and a permanent step (from the onset to the end
  of the data). Onsets are 1 December 2022, 1 June 2023, 1 December 2023, and 1 June 2024 (winter
  and summer, two years each). The donor case (the transfer subtracted, so the series loses
  `size x y`) runs for the 25% size only.
- **Refits.** For each scenario, a fit at every monthly cut from the onset month to 12 months after
  the end (the end of the data for a permanent step), each predicting the whole following month.
  A fit at a cut uses only rows before the cut.
- **Phases of a forecast row**, fixed by the target time: BEFORE (before the onset), DURING,
  AFTER_48H (within 48 hours of the end), and LATER (more than 48 hours after the end, split into
  the first 3 months and months 4 to 12). A permanent step has no AFTER phase.
- **Reported:** bias by phase, size, duration, season, substation, window, and sign; the share of
  the transfer learned (bias divided by the mean transfer in points); the mean absolute error
  against the true NRA series for the clean-trained and injected-trained forecasts; and the time
  for a permanent step to reach 50% learned. Intervals resample whole units (substation and onset
  date), 20,000 draws. Metrics of the wider decision program that need an ensemble or a detector
  (CRPS, twCRPS, curtailment cost) do not apply, because GAM_ONLY here is a point forecast and no
  detector is involved.
- **Controls.** (1) BEFORE rows must show a bias of exactly zero. (2) A transfer of size 0 must show
  exactly zero in every phase. A failure of either makes the page `invalid`.
- **Labels.** Substations by D-labels only. No generator is involved.

## Checks (to fill in)

- The median forecast minus the mean forecast does not apply: GAM_ONLY is a least-squares mean
  forecast.
- Which of the audit triggers fired: to fill in.
