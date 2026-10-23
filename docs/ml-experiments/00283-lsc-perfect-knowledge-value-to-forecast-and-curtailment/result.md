---
id: "00283"
slug: lsc-perfect-knowledge-value-to-forecast-and-curtailment
title: >-
  What is perfect knowledge of load shifts worth to MAE, CRPS, twCRPS and curtailment cost?
theme: lsc-detectors
also_relevant_to:
  - curtailment-cost
  - switching-labels-in-the-forecast
status: planned
size: >-
  medium. Stored: private score tables under
  data/ml-experiments/00283-lsc-perfect-knowledge-value-to-forecast-and-curtailment/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: more than one is defensible for the
  Primary-level curtailment proxy and for the ordinary-row group, so both are fixed below before any
  score is read. Callers: this folder's scripts only; they are copies of the modules of experiments
  208 and 277 and read the stored worlds of experiment 141. Asset graph: not touched.
hypothesis: >-
  Knowing every load shift exactly (its true start and end) lowers the error of the forecast of the
  normal running arrangement (NRA) in the 48 hours after a shift, and lowers the curtailment cost
  of the forecast of observed power in the days around a shift, while leaving ordinary rows
  unchanged. This is question 1 of the LSC decision program.
prediction: >-
  Written before any score is read. On experiment 141's injected worlds, six Primary substations,
  12 to 36 hour leads for curtailment and all leads for the other metrics: PERFECT lowers the
  threshold-weighted CRPS (twCRPS, 95th percentile of earlier power) against the true NRA series in
  the AFTER group by at least 3% with a 95% interval above zero; PERFECT lowers curtailment cost
  against observed power in the near group at the 0.5 scaled limit with a 95% interval above zero;
  and G0 (all rows) and G1 (ordinary rows) change by under 0.5%. The prediction fails if the
  near-group curtailment gain interval includes zero, and then the answer to question 1 for money
  is "not shown".
decision_rule: >-
  Fixed before any score is read. Arms: K0 (Kalman filter on recent power, no knowledge), PERFECT
  (true span known at the true onset, end known at once, MASK design) and PERFECT_LAG (the same
  spans known after a live service's delays, experiment 151's lags), the machinery of experiment
  208 unchanged except that all 11 quantiles are stored. Targets, always both: observed power (the
  real series plus the injected megawatts) and the true NRA series (the real series). Row groups on
  injected worlds follow experiment 208: BEFORE_ALARM, DURING, AFTER, near (the union of those
  three plus 7 days either side), ordinary (outside every true shift and the 144 half-hours after
  it, every fourth daily run), all rows; plus size, duration, season, lead and substation strata.
  The real record (experiment 163 and 171 rows, logged shifts: LOGGED and LOGGED_LAG against K0,
  observed power only) is descriptive and not part of any decision. Metrics: MAE of the median and
  of the mean, CRPS (twice the mean pinball loss over 11 quantiles), the quantile score of
  experiment 208 (the same quantity, so reported once), twCRPS at the 95th and 99th percentile of
  power before the scored year (experiment 277's definition), and curtailment cost from
  studies.node_curtailment on the quantile path with sorted quantiles, null q01 and q99 repaired
  to q05 and q95 before any cap or sort, node-only, scaled limit f times the largest flow before
  the first scored day, f = 0.5 pre-registered and 0.75 and 0.9 swept, GBP 2,000 per MWh. A Primary
  has no published limit and no known farm, so its curtailment is a relative comparison between
  arms and states no money saved; this goes at the top of the page. Experiment 277 found the real
  record resolves no curtailment difference, so curtailment is decided on injected worlds, with
  CRPS and twCRPS as the fallback. Bulk supply points D13 and D17 are not scored here: step 7
  reports how many shifts cross a boundary, and BSP-level curtailment is added only for cross-
  boundary shifts if few cross and that result has landed. Intervals: whole shifts resampled (units:
  world and event) for shift groups, whole calendar months for ordinary rows, 95%, 20,000 draws.
  Gain is the K0 value minus the arm's value, as a percentage of K0 for the percent claims. The
  threshold is zero for the curtailment contrast, because the question is whether any benefit
  exists; the 3% bar on twCRPS is the pre-registered size from the decision program. Verdict:
  `supported` when the twCRPS bar and the curtailment contrast both hold and G0 and G1 move by
  under 0.5%; `partly supported` when exactly one of the two holds; `not supported` when neither
  holds. Reported beside the contrasts: the median forecast minus the mean forecast per site, the
  error of the mean forecast beside each quantile-score contrast, and a leave-one-substation-out
  check of each headline contrast.
benchmark: >-
  Experiment 208's K0, PERFECT and PERFECT_LAG gains on the quantile score (1.50 points after a
  shift, 0.005 over the year); experiment 277's detectability limits for the curtailment metric.
row_set_id: >-
  Experiment 208's injected-world rows (44 worlds of experiment 141, six Primary substations) and
  experiment 163 and 171's real-record rows; not comparable with the leaderboard
metric: >-
  Avoidable curtailment cost (GBP, node-only, scaled limit), MAE of the median and of the mean,
  CRPS, twCRPS at the 95th and 99th percentile of earlier power
code: ml-experiments/00283-lsc-perfect-knowledge-value-to-forecast-and-curtailment/
date: 2026-10-23
---

Plan only. The result is written after the code has run.
