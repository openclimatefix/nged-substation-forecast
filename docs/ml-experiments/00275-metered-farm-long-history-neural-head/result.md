---
id: '00275'
slug: metered-farm-long-history-neural-head
title: >-
  Does training the metered-farm neural head on years of past weather lower the day-ahead score?
theme: metered-solar-farms
also_relevant_to:
- calibration-and-uncertainty
- retraining-and-training-history
status: planned
size: >-
  medium. Stored: predictions and tables under
  data/ml-experiments/00275-metered-farm-long-history-neural-head/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one defensible way to give past-weather rows
  an ensemble spread, so the rule is fixed in this plan before any fit. Callers: this folder's
  scripts, copied from experiments 224 and 243, and no new reviewed module. Asset graph: not
  touched.
hypothesis: >-
  Experiment 243 fitted H5 (the solar-farm power model plus a neural network that outputs all 11
  quantile offsets) on about 17 months of ensemble forecasts. Adding each farm's rows built from
  past weather (Copernicus Atmosphere Monitoring Service irradiance and ERA5 temperature, back to
  2019-09) as extra training rows, with a flag that marks them, lowers H5's quantile score on the
  ensemble-driven scored year, and the neural head gains more from the extra rows than XGBoost does.
prediction: >-
  H5 trained on past-weather plus ensemble rows with an era flag (arm L2) beats H5 trained on
  ensemble rows only (arm L0) by at least 2% in quantile score, with a 95% month-block interval
  wholly below zero. L2's gain over L0 is at least 1 percentage point larger than XGBoost's gain
  from the same rows (X2 against X0).
decision_rule: >-
  Quantile score (twice the mean pinball loss over 11 levels, in points of capacity), equal weight
  per farm, on the daylight half-hours of experiment 243's scored year at 12 to 36 hour leads,
  2025-09 to 2026-08, with intervals from resampling the 12 scored months 20,000 times, at the 98.3%
  level (Bonferroni over three planned contrasts). C1: L2 minus L0. C2: (L2 minus L0) minus (X2
  minus X0). C3: L2 minus the shuffled-power control. Supported when C1 is at least 2% below zero
  with an interval wholly below zero and C2 is at least 1 percentage point below zero. The 2%
  threshold pays for keeping a second, longer training table and a stand-in spread in the
  retraining pipeline. Stop rule: if L2 gains under 1% over L0 at 4 or more of the 6 farms, report
  that more past-weather rows do not help a single farm's head and skip the zero-history arms. If
  the shuffled-power control gains as much as L2, report that the gain is a prior on the power
  level, not the weather-to-power relation.
benchmark: >-
  H5 and XGBoost from experiment 243 (stored, own history), the plant alone, and experiment 224's
  manual_heuristic and climatology floors
row_set_id: >-
  Experiment 243's scored-year rows at the 12 to 36 hour lead band (daylight half-hours, sun more
  than 5 degrees above the horizon, farms A to F, 2025-09 to 2026-08)
metric: >-
  Quantile score in points of capacity, MAE of the median and of the mean forecast, and pinball
  loss and coverage at 0.01, 0.05, 0.95, and 0.99, per farm and arm; the quantile-crossing rate
  per arm
code: ml-experiments/00275-metered-farm-long-history-neural-head/
---

**Planned, not yet run.** We ask whether a neural network that adjusts a solar-farm power model's
forecast band gets better when it trains on years of past weather as well as on the 17 months of
ensemble forecasts. The idea follows [experiment
200](../00200-dp-pooled-correction-head-on-kalman-state/result.md), where past-weather rows raised
a substation head's gain from 0.071 to 0.125 points, and [experiment
211](../00211-dp-boosted-tree-calibrator-and-input-ablations/result.md), where an era flag carried
about 45% of such a gain. [Experiment 257](../00257-metered-farm-fair-plant-arm/result.md) found
that a linear fit recovers 90% of the neural head's gain, so a gain here may also show up for the
linear fit.

## Arms (five seeds for neural arms; one fit per arm on rows before 2025-09)

- **L0:** H5 on ensemble tuning rows only, refitted (compared with the stored H5 of experiment 243).
- **L1:** H5 on past-weather rows plus ensemble rows, no era flag.
- **L2:** L1 plus an era flag input (1 for past weather, 0 for ensemble).
- **L3:** L2 pretrained on past-weather rows, then fine-tuned on the ensemble rows only.
- **X0, X2:** XGBoost on ensemble rows only, and on both with the era flag.
- **Control:** L2 with the past-weather rows' power shuffled across days of the same farm, month,
  and half-hour.
- **Zero history:** L2 and X2 leave-one-farm-out, only if the stop rule does not fire.

## Inputs and rows

- **Past-weather rows:** CAMS irradiance (about 2 days late) and ERA5 temperature (about 5 days
  late), neither of which exists at forecast time. They appear only in training rows before 2025-09.
  Each row holds the daylight half-hours of every day from the farm's first clean month; the
  lead is the one the ensemble row for that valid time has (12 to 36 hours).
- **Spread stand-in:** the median ensemble-era spread for the same farm, month, half-hour, and
  clear-sky-index tercile, from rows before 2025-09. The era flag tells the network the spread is a
  stand-in.
- **Lagged power:** the farm's own power one day earlier (two when the lead is over 24 hours); it
  exists at forecast time.
- **Validation:** the last two tuning months (ensemble rows only) pick the epoch; past-weather rows
  in those months are left out of training.
- **Quantile order:** H5 sorts its outputs; the crossing rate before sorting is reported per arm.
