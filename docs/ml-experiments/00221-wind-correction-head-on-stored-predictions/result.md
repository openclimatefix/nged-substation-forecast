---
id: "00221"
slug: wind-correction-head-on-stored-predictions
title: >-
  Does a small neural-network head on the stored wind best guess lower the three wind farms' error at
  lead days 1 and 2, and how much of any gain does a blind head reproduce?
theme: generator-forecasting
also_relevant_to: [calibration-and-uncertainty]
status: planned
date: 2026-10-22
size: >-
  medium. Stored: private files under data/ml-experiments/00221-wind-correction-head-on-stored-predictions/
  only. Serving path: not touched. Degradation rule: not touched. Designs: several are defensible
  (head inputs, training window, whether the head is pooled), and this plan pre-registers one. Callers:
  this folder's scripts, which copy experiment 200's head and experiment 52's scoring code and read
  the stored best guess of experiment 33. A new application of a known method, so one Sonnet code
  review and one Sonnet write-up review; no Opus unless an audit trigger fires.
hypothesis: >-
  A small neural-network head, fed the nine quantiles of the wind best guess (experiment 33's
  ERA5-trained mixture over the 51 ENS members, with the power-curve base margin and the spread
  factor) plus the ensemble mean and spread of the 100 m wind speed and the direction sector, lowers
  the quantile score at lead days 1 and 2 against the best guess by at least 0.1 points of capacity.
  The head is pooled over the three farms and trained on earlier months only.
prediction: >-
  Written before any fit. The head with weather (A2) minus the best guess (A0) at lead days 1 and 2
  is -0.05 to -0.20 points, the gain concentrated at the 0.05 and 0.95 quantiles. The blind head (A3,
  lead and calendar only) captures more than half of the gain of A2, as in experiments 201 and 209,
  which would mean the head repairs calibration and does not use physics. The table (A5) captures
  less than the heads. Only 9 scored months and 3 farms are available, so `inconclusive` is the
  likeliest verdict.
decision_rule: >-
  One decisive contrast, C1: A2 minus A0 on the quantile score at lead days 1 and 2 (calendar days
  from the ENS run's date to the valid time), on the masked rows of the 9 scored months, read at 95%
  with whole months and the three seeds resampled, 50,000 resamples. The threshold is 0.1 points,
  because the head is a new trained component on the serving path. C1 is `supported` when the
  interval lies wholly below -0.1, `small gain` when it lies wholly below zero but not wholly below
  -0.1, `refuted` when it lies wholly above zero, and `inconclusive` otherwise. Descriptive
  contrasts, all at 95% and none changing the verdict: C2 A3 minus A0, C3 A2 minus A3, C4 A1 minus
  A0, C5 A4 minus A2, C6 A5 minus A0, each at lead days 1 and 2, days 3 to 6, and over all leads.
  Blind share: (A3 minus A0) divided by (A2 minus A0) at lead days 1 and 2, reported when C1's point
  estimate is below zero; a share above 0.5 means the head repairs calibration. Stop rules: stop and
  report if A0 recomputed from the stored table differs by more than 0.01 points from experiment 52's
  printed score on the same rows (6.701 at lead days 1 and 2, 11.037 over all leads); stop and report
  if A3 scores better than A2 at both lead days 1 and 2 and lead days 3 to 6, because the head then
  does not use the weather. The intervals describe these three farms, because farms are not
  resampled.
benchmark: >-
  The experiment 33 wind best guess (`bg_full`) on the same rows, with the row mask of experiment 210
  applied, read from its stored table. A0 is the table as stored, and the tables of experiments 52 and
  54 are not used.
row_set_id: >-
  experiment 33's wind cases (`fold_a_tuesdays_v1`: 53 Tuesday runs from 2025-07-01 to 2026-06-23,
  W1 to W3, three seeds) minus the rows of experiment 210's `mask.parquet`; the head arms are scored
  from the October 2025 runs onward, so the first 3 months only train
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels, the estimate of the
  CRPS), in points of capacity; pinball loss at 0.05 and 0.95 and coverage of the 5% to 95% band by
  lead-day band
code: ml-experiments/00221-wind-correction-head-on-stored-predictions/
production_status: not-for-production
adopted_in: none
---

# Experiment 221: does a small neural-network head lower the wind farms' error at 1 to 2 days ahead?

**Plan only; the results are not in yet.**

## Plan

- **Rows.** Experiment 33's wind best guess, three farms (W1 to W3), three seeds, the 53 Tuesday runs
  of July 2025 to June 2026. The rows of experiment 210's `mask.parquet` (idle-band and suspected
  outage half-hours) are dropped from training and from scoring in every arm. The head arms are scored
  for the runs of October 2025 to June 2026 (9 months), each month by a head trained on all earlier
  months, whose valid times lie at least 4 hours before the month's first run.
- **Head.** Experiment 200's form: a small network (two hidden layers of 64 units, GELU) emits a
  median shift `c` and a log-scale `s`, and the nine quantiles are the clipped, sorted
  `median + c + exp(s) * (q - median)` of the base forecast, in fractions of capacity. The loss is
  the metric's mass-weighted pinball loss. No farm embedding, so the head is pooled and can be tested
  on a farm it never saw. Fixed 12 epochs, no early stopping, to avoid tuning on neighbouring months.
- **Arms.** A0 best guess. A1 head on the nine quantiles and lead. A2 A1 plus the ensemble mean and
  standard deviation of the 100 m wind speed and the direction sector (8 sectors, circular mean of
  the members). A3 blind: lead and calendar only. A4 A2 with the base forecast replaced by the
  ERA5-trained wind model fed the ensemble-mean wind (refit per seed as in experiment 52), plus the
  head-free mean-only forecast as a descriptive row. A5 descriptive table: per lead band, a median
  shift and an offset scale chosen on a grid to minimise the pinball loss on earlier months.
- **Leave one farm out.** A1 to A3 trained on two farms' earlier months and scored on the third farm.
- **Weather at forecast time.** The ensemble mean, spread, and direction come from the ENS run the
  best guess already used, so they exist at forecast time.
- **Not scored.** The stored best guess holds nine levels (0.05 to 0.95), so pinball at 0.01 and 0.99
  cannot be computed for A0; tails are read at 0.05 and 0.95. The node curtailment metric needs a
  node and its generators and does not apply to three separate farms.
- **Advice read.** The plan follows experiment 52's scoring rows and spread conventions, experiment
  210's mask advice, and experiments 200, 201, and 209's finding that a blind copy reproduces much of
  a head's gain. It departs from experiment 52 by scoring nine months and not twelve.
- **Retraining schedule.** The head is refitted monthly on all earlier months, an expanding window.
  The base forecast is not refitted (it is the stored table), so the contrast does not depend on the
  base forecast's retraining cadence.
