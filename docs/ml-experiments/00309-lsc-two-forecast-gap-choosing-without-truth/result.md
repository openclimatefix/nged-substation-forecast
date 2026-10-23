---
id: "00309"
slug: lsc-two-forecast-gap-choosing-without-truth
title: >-
  Can the gap between two forecasts tell NGED which one to use, without knowing the truth?
theme: lsc-detectors
also_relevant_to:
  - curtailment-cost
status: planned
size: >-
  medium. Stored: private score tables under
  data/ml-experiments/00309-lsc-two-forecast-gap-choosing-without-truth/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one is defensible for the choosing
  rule and for the "better fixed choice", so each is fixed below before any score is read.
  Callers: this folder's scripts only, which read the stored rows of experiments 297 and 258 and
  the gap scores of experiment 308. Asset graph: not touched. The maintainer's brief limits the
  work to Sonnet reviewers and CPU.
hypothesis: >-
  NGED sees two forecasts and not the true switching state. A forecast with no lagged power
  (GAM_ONLY) forecasts the normal running arrangement, and a forecast with lags (K0) forecasts what
  will flow while an abnormal arrangement lasts. A rule that reads only the gap between the two
  forecasts, how long it has lasted, and a neighbour check can choose between them nearly as well as
  knowing the truth. This is part 2 of the brief
  briefs/sw_lsc_two_forecasts_divergence_and_selection_brief.md, and it builds on experiment 308.
prediction: >-
  Written before any score is read. RULE recovers at least 60% of ORACLE's gain over the better
  fixed choice, on the threshold-weighted CRPS (twCRPS) at the 95th percentile against observed
  power, in the rows near a shift (G5, and separately the rows before the alarm, during the shift,
  and within 48 hours after it), without raising the quantile score of ordinary rows by more than
  0.02 points. It fails if the neighbour check cannot tell a switch from a persistent change or a
  weather error; the page then says NGED needs an attribution flag as well as the two forecasts
  and what the flag must say.
decision_rule: >-
  Fixed before any score is read. Rows, injected worlds: the 44 worlds of experiment 141, six
  Primary substations, and the cached ensemble-weather forecasts of experiment 297 (GAM_ONLY with
  bands fitted on odd months, K0, and the band quantile head BAND_HEAD), unchanged. Rows, real
  record: experiment 258's harness (15 Primary series, past weather, 6-hourly origins), scored
  against observed power only, because the real record has no separate normal-arrangement series.
  Arms: ALWAYS_NRA (GAM_ONLY); ALWAYS_ASIS (K0); BAND_HEAD (injected worlds only); RULE at three
  thresholds; ORACLE (per row, the arm with the lower quantile score against the target scored,
  the ceiling); ORACLE_STATE (K0 while a true shift is on at the origin, GAM_ONLY otherwise, for
  observed power; GAM_ONLY always for the normal-arrangement target). RULE: the plain gap of
  experiment 308 (mean of K0 minus GAM_ONLY over leads 0 to 6 hours, in units of the series'
  robust noise scale) enters an abnormal state when its absolute value exceeds k at two
  consecutive origins (6 hours apart) and leaves it when the value stays under k/2 at two
  consecutive origins or after 14 days. In the abnormal state RULE forecasts with K0; otherwise
  with GAM_ONLY. The three thresholds are k = 2.3, 3.2, and 4.5, the plain gap's thresholds at the
  false alarm rates of RO-NC and of the learned score in experiment 308, and a stricter one. The
  threshold is chosen on rows whose forecast origin falls in odd calendar months (lowest pooled
  twCRPS against observed power in G5) and every contrast is read on even-month rows only. On the
  real record RULE_NB adds the neighbour check of experiment 308 at the moment the abnormal state
  is entered; it cannot apply in the injected worlds, whose receivers are unmetered. A forecast
  origin is mapped to the latest 6-hourly origin at or before it, so the state uses readings that
  ended at least 3.5 hours before the forecast. Targets: observed power, which is what flows, and
  the true normal-arrangement series, which is what NGED wants for scenario studies, reported
  separately with the arm that serves each. Row groups G0 to G5 of the LSC decision program
  (all, ordinary, before the alarm, during, 48 hours after the end, within 7 days) and the splits
  by size, duration, season, lead, and substation. Metrics: the quantile score of the 11 quantiles
  (an approximation to the CRPS), twCRPS at the 95th percentile, MAE of the median and of the
  mean, the curtailment cost at the scaled limit 0.5 for 12 to 36 hour leads, and the share of
  readings above q99. Better fixed choice: whichever of ALWAYS_NRA and ALWAYS_ASIS has the lower
  pooled twCRPS in the group and target scored. Recovery: (better fixed minus RULE) divided by
  (better fixed minus ORACLE), resampled by whole shifts and whole months, 95%, 20,000 draws.
  Descriptive: the permanent-step and D8 engine-spell cases, from experiment 274's stored bias
  table and experiment 308's gap scores. Verdict `supported` when recovery is at least 60% in G5
  and ordinary rows are within 0.02 points; `partly supported` when it holds in one of the three
  near-shift groups or fails only on the ordinary-row clause; `not supported` otherwise.
benchmark: >-
  The better fixed choice of ALWAYS_NRA and ALWAYS_ASIS, ORACLE as the ceiling, and BAND_HEAD of
  experiment 297 as the strongest single forecast.
row_set_id: >-
  Experiment 297's injected-world rows (44 worlds of experiment 141, six Primary substations) and
  experiment 258's harness rows; not comparable with the leaderboard
metric: >-
  twCRPS at the 95th percentile, quantile score, MAE of the median and mean, curtailment cost at
  the scaled limit 0.5, share of readings above q99
code: ml-experiments/00309-lsc-two-forecast-gap-choosing-without-truth/
---

**Planned.** The plan is the front matter above; the write-up replaces this body when the
experiment finishes.
