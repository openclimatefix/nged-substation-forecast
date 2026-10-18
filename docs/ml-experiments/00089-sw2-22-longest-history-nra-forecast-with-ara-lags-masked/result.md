---
id: "00089"
slug: sw2-22-longest-history-nra-forecast-with-ara-lags-masked
status: planned
size: medium
hypothesis: >-
  For the 16 Primary substations, an XGBoost demand model refitted monthly and trained on the
  longest history of ECMWF ensemble control-member forecasts (2021-03 onward), with every
  half-hour inside an abnormal running arrangement (ARA) removed from the training targets, scores
  below the same model on the standard 12-month window, and setting to missing every lagged power
  input that lies inside an ARA lowers the score on the half-hours whose lags lie inside an ARA.
prediction: >-
  Long history minus 12-month window (L minus R) is -0.05 to -0.20 points of capacity on all rows
  outside switching, because the window holds about 40 logged events instead of about 24 and
  experiments 34 and 49 saw the 2-year half-life matter little. Masking minus no masking (LM minus
  L) is -0.1 to -0.5 points on the rows whose lag lies inside an ARA (experiment 75 saw -0.42 for
  corrected lags and experiment 86 saw an unresolved -0.29) and within plus or minus 0.02 on all
  rows outside switching.
decision_rule: >-
  Three planned contrasts, quantile score over all leads, 98.33% interval (Bonferroni across
  three), whole validation months resampled, seed 0, paired on the 292 runs of
  `fold_a_daily_gap_v1`. C1, LM minus L, on the in-event-lag rows (headline). C2, L minus R, on
  all rows outside switching. C3, LM minus R, on the in-event-lag rows. Masking is supported when
  C1's upper end is below zero and its no-harm read on all rows outside switching has an upper end
  of at most +0.02 points (0.37% of R's score); the long history is supported when C2's upper end
  is below zero. Both thresholds are zero because neither change adds a model: masking adds one
  array lookup per lag at serving time (and so an ARA label on the serving path, which makes its
  threshold a gain whose interval excludes zero), and a longer window only changes the training
  slice. C3 is descriptive. Verdict `supported` when both hold, `mixed` when one holds and the
  other's lower end is above zero, `small gain` when one holds, `inconclusive` when neither
  interval excludes zero, `refuted` when a lower end is above zero. The in-event-lag rows
  are rows whose valid time lies outside every ARA and whose origin lag, same-time-latest-day lag,
  or same-time-latest-week lag lies inside an ARA. The arms M, L_ens, and LM_ens are descriptive.
  The masks use hindsight ARA labels, so C1 and C3 are upper bounds on what a live service, which
  cannot know an ongoing event, would achieve.
benchmark: >-
  Arm R, the 12-month-window booster with ARA targets removed and lags unmasked, refitted by this
  experiment. The plain booster of experiment 86, `manual_heuristic`, and `climatology` are
  descriptive references on the same rows.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over 9 levels), in points of capacity,
  on rows outside switching; tail calibration at 0.05 and 0.95 (the nine-level mixture has no
  0.01 or 0.99 level).
code: ml-experiments/00089-sw2-22-longest-history-nra-forecast-with-ara-lags-masked/
sizing:
  stored: Boosters and predictions under data/ml-experiments/00089-*/; nothing in a Delta table.
  serving_path: No; masking would need an ARA label at serving time, which is a later decision.
  degradation_rule: No.
  more_than_one_design: Yes (label source, window start), so the arms and labels are fixed here.
  callers_not_nameable: No; the scripts copy experiment 86's and import reviewed packages only.
advice: []
---

# Experiment 89: does the longest history, with switching masked, improve the NRA forecast?

Planned. The plan, written before any fit, is in the front matter and in the bullets below.

- **Question.** With half-hours inside an ARA removed from the training targets and not corrected
  by any constant, does training on the longest history of control-member forecasts (2021-03
  onward) beat the 12-month window, and does setting ARA lags to missing help on rows whose lags
  lie in an ARA?
- **Arms** (Primary N1 XGBoost, monthly P-roll refits, 150 rounds, eta 0.15, control member only,
  seed 0, 16 series). Every arm drops training rows whose target is inside an ARA. R: the last 365
  days, recency half-life 0.5 years (experiment 86's recipe), lags unmasked. M: R with lags inside
  an ARA set to missing. L: all rows since 2021-03, recency half-life 2 years, lags unmasked. LM:
  L with masked lags. L_ens and LM_ens: L and LM using only rows from 2024-04-01, the clean check
  on one weather product. R and M are descriptive about masking alone; L_ens and LM_ens are
  descriptive about the product seam. L and R differ in window and half-life together, and L_ens
  separates the two partly.
- **ARA label.** The union of the intervals of the refit-offset neighbour-count (RO-NC) detector
  (called XL in experiments 70, 71, and 82; hindsight end rule), NGED's logged events, and
  NGED's logged closures. Neither source labels anything before 2022-01, so 2021-03 to
  2021-12 is unlabelled and its switching stays in the training targets. The write-up reports
  labelled events per training year and per source.
- **Training rows.** The control member of the 18 km runs (2021-03 to 2023-06-26), the 9 km runs
  (2023-06-27 to 2024-03-31), and the Dynamical.org archive from 2024-04-01, which is a different
  product (seam at 2024-04-01). The first two come from the backfill that experiment 81 built into
  frames. The write-up labels which window each number uses; L_ens and LM_ens (runs initialised from
  2024-04-01) are the clean check on one product.
- **Scored rows.** The 292 daily runs of `fold_a_daily_gap_v1` (2025-07 to 2026-06) with a valid
  time outside every ARA. Headline rows: those with a lag inside an ARA.
- **Weather at lag times.** The forecast's own weather columns from the same run; nothing from
  ERA5 enters.
