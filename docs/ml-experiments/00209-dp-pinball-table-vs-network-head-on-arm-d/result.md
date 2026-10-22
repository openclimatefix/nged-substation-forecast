---
id: "00209"
slug: dp-pinball-table-vs-network-head-on-arm-d
title: "Differentiable physics: is a neural network needed to recalibrate arm D, or does a pinball-loss table do the same?"
theme: differentiable-physics-forecast-engine
also_relevant_to: [calibration-and-uncertainty]
status: planned
size: >-
  simple. Stored: private files under data/ml-experiments/00209-dp-pinball-table-vs-network-head-on-arm-d/
  only. Serving path: not touched. Degradation rule: not touched. Designs: one table design is
  pre-registered and a second arm adds one term. Callers: this folder's scripts, which read the
  stored rows of experiments 200 and 201. No new data and no new method (a table fitted by gradient
  descent with the pinball loss, the head of experiment 200 without its hidden layers).
hypothesis: >-
  The gain that experiment 200's network head and its blind copy get over arm D (0.100 to 0.125
  points) is a recalibration that a table without hidden layers can reproduce. The table gives arm
  D's median a shift that depends on site, half-hour, day of week, lead band, and arm D's forecast
  level by an ensemble-era flag, and scales arm D's quantile offsets by site, lead band, and
  six-hour block. It is fitted by gradient descent on the pinball loss over the 11 quantiles, on the
  same rows as the network. A second arm adds a temperature-extremes term (a median lift below -3
  degrees Celsius and above 24 degrees Celsius, from experiment 201's finding).
prediction: >-
  Written before any fit. The table scores within 0.02 points of the blind network (table minus
  blind network between -0.02 and +0.02), most likely about +0.01. The temperature-extremes term
  lowers the table's score by 0.01 to 0.04 points. An additive table without the era term scores
  worse than arm D, as experiment 200's ridge did when fitted to all years.
decision_rule: >-
  The table matches the blind network if the point estimate of table minus blind network lies within
  0.02 points and the 99% month-resampled paired interval includes zero. A network is then not
  needed for this gain. If the table is worse than the blind network by more than 0.02 with an
  interval wholly above zero, a network is needed (some interaction the table lacks). The
  temperature-extremes arm is a gain if its interval against the plain table lies wholly below zero.
  The threshold 0.02 is set by cost: a network head is a new trained component on the serving path,
  and a table is not, so a gain under 0.02 points does not pay for the network.
benchmark: >-
  Arm D of experiment 155 (2.432 points at day D+1 on the five decision substations), the blind
  network (2.332) and the head (2.307) of experiment 200's 2020-onward fit
row_set_id: >-
  experiment 200's scored rows: arm D of the scored year 2025-07-02 to 2026-06-29, leads 15.5 to 39
  hours, outside switching events, six substations of which five are decision substations
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of effective capacity,
  day D+1, outside switching events
code: ml-experiments/00209-dp-pinball-table-vs-network-head-on-arm-d/
---

**Planned, not run.** Experiment 200 found that a network head beats arm D, that a copy given no
recent errors gets most of the gain, and that a pinball-loss table comes within 0.01 to 0.02 points
of the blind network on one year. This experiment tests the same table on all years, with the
forecast level by era term the audit found necessary, and adds a temperature-extremes arm.

## Plan

- **Rows.** Experiment 201's shared feature files (arm D rows with the filter's level and the
  driver temperature), the same training rows, validation block, and scored rows as experiment 200.
- **Arms.** T0, the table; T1, T0 plus the temperature-extremes term; T2, T0 without the forecast
  level by era term (repeats the failure of the squared-error ridge).
- **Fit.** Torch embedding sums trained on CPU (4 threads) with the pinball loss, early stopping on
  the validation block, 3 seeds.
- **Contrasts (99% month-resampled paired intervals).** T0 minus blind network (decides the verdict),
  T0 minus head, T1 minus T0, T0 minus arm D, T2 minus T0.
