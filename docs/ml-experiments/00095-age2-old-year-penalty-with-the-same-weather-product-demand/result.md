---
id: "00095"
slug: age2-old-year-penalty-with-the-same-weather-product-demand
status: planned
size: simple
hypothesis: >-
  The extra forecast error of an old training year (experiment 93) comes mostly from the change of
  weather product between the old years (18 km grid spacing) and the recent year (9 km), not from
  the age of the training data, so an old year and a recent year that use the same weather product
  score about the same.
prediction: >-
  Trained and scored on ERA5, the 2021-07 year scores less than half as far behind the 2024-07 year
  as it does on the control-member forecasts (C2 under half of C1), and the 2023-07 year (9 km)
  scores within 0.3 points of the 2024-07 year (9 km) (C4 under 0.3).
decision_rule: >-
  Five planned contrasts on the quantile score, in points of capacity, paired on series, run, and
  valid time, with whole calendar months resampled (99% interval, the Bonferroni level for five
  contrasts). C1 is the 2021-07 year minus the 2024-07 year, both on the control member (the
  experiment 93 contrast). C2 is the same two years, both on ERA5. C3 is C1 minus C2, the part of
  the penalty that disappears when both years use one weather product. C4 is the 2023-07 year minus
  the 2024-07 year, both 9 km control member. C5 is the 2021-07 year minus the 2022-07 year, both
  18 km control member. Supported when C1 is above zero, C2 is under half of C1, and C3 is above
  zero. The weather product is the main cause of the penalty when C2 is under half of C1 and C4 is
  near zero; age is the main cause when C2 is at least half of C1 or C4 is clearly above zero. No
  threshold for porting is needed, because the ERA5 arms are diagnostic and nothing is ported.
benchmark: >-
  The 2024-07 year on the same weather product as the arm it is compared with. `manual_heuristic`
  and `climatology` rows are not scored, because the question is a contrast between training years.
row_set_id: age1_daily_2025_07_to_2026_06_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over 11 levels, 0.01 to 0.99) in
  points of capacity.
production_status: not-for-production
adopted_in: none
---

# Experiment 95: is the old-year penalty the weather product or the age? (AGE2)

**Planned. This page holds the plan; the write-up replaces it when the fits have run.**

Experiment 93 found that a Primary demand model trained on 2021-07 to 2022-06 scores 1.57 points of
capacity worse than one trained on 2024-07 to 2025-06, and that the old years used a coarser
weather forecast (18 km grid spacing against 9 km). Experiment 95 separates the two causes.

## Design

- **Arms (13 Primary substations, one frozen fit per series and arm, 150 rounds at learning rate
  0.15, seed 0, no refits).** Each arm is scored on the 365 daily runs of 2025-07 to 2026-06, using
  the weather product it was trained on at inference.
    - ERA5 arms (diagnostic, not for production, because ERA5 arrives about 5 days late):
      trained on 2021-07, 2022-07, 2023-07, or 2024-07 years, with ERA5 weather at the valid time.
      The rows are the control member's (run, valid time) pairs, so only the weather columns differ.
    - Control-member arms: 2021-07, 2022-07 (18 km) and 2024-07 (9 km) reuse the experiment 93 raw
      predictions. The 2023-07 to 2024-06 year (9 km) is new: the control member from the daily
      frames of experiment 81 up to 2024-03-31, then the control member rows of the stage 1 pilot
      from 2024-04-01.
- **Reported:** the absolute score of every arm; the old-minus-recent gap per weather product; the
  share of the experiment 93 gap that survives on ERA5 (C2 over C1); all contrasts with and without
  the substation that carries most of the penalty (D8, post hoc); a check of D8's history for a
  step change; tail calibration at 0.01, 0.05, 0.95, and 0.99.
- **Not scored:** the curtailment-cost metric and the `manual_heuristic` baseline, because the
  question compares training years of one model.

## Sizing

- **Stored:** nothing beyond private predictions.
- **Serving path:** untouched.
- **Degradation rule:** untouched.
- **Designs:** one defensible design (the experiment 93 recipe with the weather product swapped).
- **Callers:** the scripts copy experiment 93's, and no package code changes.
