---
id: '00257'
slug: metered-farm-fair-plant-arm
title: >-
  Is the neural head needed once the solar-farm power model sees the same inputs?
theme: metered-solar-farms
also_relevant_to:
- calibration-and-uncertainty
status: planned
size: >-
  medium. Stored: predictions and tables under data/ml-experiments/00257-metered-farm-fair-plant-arm/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one defensible
  way to give the plant the lagged power (a ratio or a difference), so the ratio is fixed in this
  plan before any fit. Callers: this folder's scripts, copied from experiments 224 and 243, and no
  new reviewed module. Asset graph: not touched.
hypothesis: >-
  The 4.4% gain of H5 (the plant plus a neural network that outputs all 11 quantile offsets) over
  the plant alone in experiment 243 came from the inputs that H5 saw and the plant did not (the
  lagged power and the ensemble spread), so a convex linear quantile regression of the 11 offsets
  on those same inputs (arm P2) recovers most of it, and the neural network is not needed.
prediction: >-
  H5 beats P2 by at least 2% in quantile score, with a 95% month-block interval wholly below zero,
  and by at least 10% in pinball loss at the 0.99 quantile (the prediction that could fail).
  Experiment 243 found H5 4.4% better than the plant alone, and experiment 224's two-number plant
  (P1) already recovers about 1.7% of that, so P2 within 1% of H5 is plausible.
decision_rule: >-
  Quantile score (twice the mean pinball loss over 11 levels, in points of capacity), equal weight
  per farm, on the 41,654 daylight half-hours of experiment 243, 12 to 36 hours ahead, 2025-09 to
  2026-08, with intervals from resampling the 12 scored months 20,000 times, at the 99% level
  (Bonferroni over five planned contrasts). C1: H5 minus P2, quantile score. C2: H5 minus P2,
  pinball loss at 0.99. C3: P2 minus P1. C4: P3 minus H5 with zero history. C5: P2 minus P2 without
  the availability ratio. Supported when C1 is at least 2% below zero with an interval wholly below
  zero and C2 is at least 10% below zero. The 2% threshold is the cost of serving and retraining a
  neural network beside a convex fit. Stop rule: if P2 is within 1% of H5 in quantile score,
  report that the gain came from the inputs and not from the network, and that P2 replaces H5 as
  the metered-farm candidate. If P2 is worse than P1, report a fitting fault before any verdict.
benchmark: >-
  H5 and XGBoost from experiment 243 (stored, own and zero history), the plant alone (P0), the
  two-number plant (P1), and experiment 224's manual_heuristic and climatology floors
row_set_id: >-
  Experiment 243's rows: daylight half-hours (sun more than 5 degrees above the horizon) at 12 to
  36 hour leads, 2025-09 to 2026-08, farms A to F
metric: >-
  Quantile score, MAE of the median, MAE of the mean forecast, and pinball loss and coverage at
  0.01, 0.05, 0.95, and 0.99, per farm and arm
code: ml-experiments/00257-metered-farm-fair-plant-arm/
---
 **This page is the plan; the write-up replaces it when the experiment finishes.**

## Question

**We ask whether a neural network is needed on top of the solar-farm power model once the power
model sees the same inputs as the network.** [Experiment
243](../00243-metered-farm-architecture-comparison/result.md) found that H5, the plant plus a neural
network that outputs all 11 quantile offsets, scores 4.4% better than the plant alone. The plant arm
saw neither the lagged power nor the ensemble spread, while H5 saw both, so the gain may belong to
the inputs. A convex quantile regression ([Koenker and Bassett
(1978)](https://doi.org/10.2307/1913643)) on the same inputs, with no network, tests that.

## Arms

- **P0:** the plant alone, with experiment 224's offsets (experiment 243's `plant` arm, stored).
- **P1:** the two-number plant (the mean and standard deviation of the 51 members' plant responses,
  with the band scaled by a fitted spread factor per lead band; experiment 224 `a3_two_number`,
  stored).
- **P2:** P0's forecast plus a linear quantile regression of each of the 11 offsets on the plant
  forecast, the 51 members' spread, the availability ratio, and dummies for the lead band, the
  clear-sky-index band, and the sun-elevation band. The availability ratio is the farm's power one
  day earlier (two days when the lead is over 24 hours) over the plant forecast at that earlier
  half-hour, clipped to 0 to 1.5, with 1 and a missing-data flag when it cannot be formed. Each
  level is a convex linear programme. The quantiles are sorted to remove crossing, then clamped to
  zero and to the farm's ceiling, as in experiment 243.
- **P3:** P2 fitted on the other five farms' tuning months only, with the plant scale and ceiling
  the median over those five farms (experiment 243's zero-history setting), against H5 with zero
  history.
- **P2 without the availability ratio,** to show whether the lagged power carries the gain.
- **H5 and XGBoost:** experiment 243's stored predictions, own and zero history, unchanged.

## Design

- **Rows and fit:** experiment 243's rows and one fit on the tuning months before 2025-09 (the
  screening schedule), so that P2 compares like for like with the stored H5. The inner choice of the
  regression's penalty is none: it is an unpenalised linear programme. Tuning rows are grouped by
  valid time, never split by run, if any split is made.
- **Calendar:** no calendar columns enter P2. The clear-sky index and elevation bands are weather
  and sun position, as in experiment 243.
- **Weather and forecast time:** ECMWF ensemble 00 UTC run at the nearest 0.25 degree cell, and farm
  power one or two days earlier. Both exist at forecast time.
- **Scoring:** copied from experiment 243's scorer, with the same month-resampling. The predictions
  carry `q01`, `q05` to `q95`, `q99`, `actual`, `lead_hours`, and `time_series_id` (farm label), and
  `studies.tail_calibration` reports coverage and pinball loss at 0.01, 0.05, 0.95, and 0.99. The
  quantile score, the MAE of the median, and the MAE of the mean forecast are reported for every
  arm.
- **Anonymisation:** farms A to F only.

## Earlier advice followed

- Experiment 243: give the plant forecast the lagged power and the ensemble spread before crediting
  physics for the gain, and use the same rows and the same scorer.
- Experiment 224: the two-number plant is the stronger plant reference.
