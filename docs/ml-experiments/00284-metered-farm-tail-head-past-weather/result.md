---
id: '00284'
slug: metered-farm-tail-head-past-weather
title: >-
  Does a network trained only for the tails beat XGBoost at the 0.01 and 0.99 quantiles of a metered solar farm?
theme: metered-solar-farms
also_relevant_to:
- calibration-and-uncertainty
- curtailment-cost
status: planned
size: >-
  medium. Stored: predictions and tables under
  data/ml-experiments/00284-metered-farm-tail-head-past-weather/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one defensible way to build the tail quantiles
  (a network correction to a binned plant, or a direct fit), so both are arms. Callers: this
  folder's scripts, copied from experiments 270 and 275, and no new reviewed module. Asset graph:
  not touched.
hypothesis: >-
  Experiment 275 found that adding years of past-weather rows lowered the 0.99 pinball loss of the
  neural head H5 by 4.6% while its overall score moved by 0.47%. A small network trained with a loss
  restricted to the tail quantiles (0.01, 0.02, 0.05, 0.95, 0.98, 0.99), on the same past-weather
  plus ensemble rows with an era flag, lowers the 0.01 and 0.99 pinball loss of a metered solar
  farm's forecast below that of XGBoost trained on the same rows.
prediction: >-
  The tail head (T3) lowers the 0.99 pinball loss by at least 5% against XGBoost on past-weather
  plus ensemble rows (X2), with a 98.75% month-block interval wholly below zero. The prediction in
  the brief that the node curtailment cost at 0.75 times the node's earlier maximum export is no
  worse than XGBoost's holds by construction and is not a test (see the decision rule).
decision_rule: >-
  Pinball loss in points of capacity at one quantile, equal weight per farm, on experiment 275's
  scored rows (daylight half-hours at 12 to 36 hour leads, 2025-09 to 2026-08, farms A to F), with
  intervals from resampling the 12 scored months 20,000 times at the 98.75% level (Bonferroni over
  four planned contrasts). C1: T3 minus X2 at 0.99. C2: T3 minus X2 at 0.01. C3: T3 minus T5
  (XGBoost on the tail quantiles, same inputs) at 0.99. C4: T3 minus the shuffled-power placebo at
  0.99. Supported when C1 is at least 5% below zero with an interval wholly below zero. The 5%
  threshold pays for a second model in the retraining pipeline. Stop rule 1: if the C1 interval
  includes zero at 4 or more of the 6 farms, report that the tails are already as good as XGBoost's
  and stop. Stop rule 2: if T3's coverage at 0.01 is outside 0.5% to 1.5% at 3 or more farms,
  report the miscalibration before any value result. The node curtailment rule of
  studies.node_curtailment reads only each farm's median forecast, and T3 keeps arm L2's median
  and middle quantiles, so T3's curtailment cost equals L2's; it is reported from experiment 255's
  method only as a pointer, not as a contrast. Experiment 255 also found under 20 events at 0.75 times the
  maximum export, so the metric cannot rank the arms.
benchmark: >-
  X2 and L2 from experiment 275 (stored, same rows), the spread-binned plant P2 (experiment 270's
  pq_spread), and experiment 224's manual_heuristic and climatology floors
row_set_id: >-
  Experiment 275's scored rows (experiment 243's row set: daylight half-hours at 12 to 36 hour
  leads, sun more than 5 degrees above the horizon, farms A to F, 2025-09 to 2026-08)
metric: >-
  Pinball loss and coverage at 0.01, 0.05, 0.95, and 0.99 per farm and arm, the number of tail
  exceedances, and the quantile score of the 11-level frame
code: ml-experiments/00284-metered-farm-tail-head-past-weather/
---

**Planned.** This page is the plan; the write-up replaces it when the fits finish.

## Question

- National Grid Electricity Distribution (NGED) meters six solar farms (labelled A to F). Curtailing
  a farm depends on the upper tail of its forecast, so the 0.01 and 0.99 quantiles matter most.
- [Experiment 275](../00275-metered-farm-long-history-neural-head/result.md) trained the neural head
  H5 on 17 months of ensemble rows plus nearly six years of rows built from past weather (Copernicus
  Atmosphere Monitoring Service irradiance and ERA5 temperature), with an era flag. H5's 0.99 pinball
  loss fell 4.6% and its overall score 0.47%. XGBoost gained 1.7% overall from the same rows.
- We ask whether a network trained only for the tails beats XGBoost at 0.01 and 0.99.

## Arms (own history, experiment 275's rows, folds, and one fit before 2025-09; five seeds for networks)

1. **X2:** XGBoost on past-weather plus ensemble rows with the era flag (stored in experiment 275).
2. **P2 (reference):** the plant with offsets binned by the forecast (6 bins) and ensemble spread (3
   bins), as experiment 270's `pq_spread`, from ensemble tuning rows.
3. **L2 (reference):** H5 as in experiment 275 (stored).
4. **T3:** a network on P2's forecast, the ensemble spread, lagged power, the clear-sky index, the
   lead, the cell temperature, the sun elevation, and the era flag. It adds a bounded correction
   to P2's offsets at the six tail levels, with pinball loss at those levels plus a penalty on
   quantiles that cross before sorting. The middle quantiles (0.10 to 0.90) come from L2.
5. **T0:** T3 trained on ensemble rows only.
6. **T4a:** T3 without the era flag. **T4b (placebo):** T3 with the past-weather rows' power
   shuffled within farm, month, and half-hour.
7. **T5:** XGBoost on the six tail levels with T3's inputs.

## Weather and its availability at forecast time

- Training rows from the past use CAMS irradiance (about 2 days late) and ERA5 temperature (about 5
  days late). Neither exists at forecast time, and they appear only in training rows.
- Scored rows use the ECMWF ensemble (00 UTC run), which exists at forecast time. Lagged power
  (one day, or two when the lead exceeds 24 hours) exists at forecast time.
- Past-weather rows take a stand-in ensemble spread (the median of the same farm, month,
  half-hour, and cloud tercile), as in experiment 275.

## Schedule

One fit on rows before 2025-09, no retraining in the scored year (the screening schedule). Absolute
tail scores could change under weekly refits.
