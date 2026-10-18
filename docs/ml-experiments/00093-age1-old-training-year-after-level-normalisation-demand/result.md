---
id: "00093"
slug: age1-old-training-year-after-level-normalisation-demand
status: done
size: simple
hypothesis: >-
  A year of training data hurts a Primary substation demand forecast with its age only because the
  substation's average demand level differed in that year, so subtracting a trailing 365-day level
  from the targets and the power lags removes most of the penalty for an old training year.
prediction: >-
  A frozen XGBoost model trained on 2021-07 to 2022-06 scores at least 0.5 points of capacity worse
  on 2025-07 to 2026-06 than one trained on 2024-07 to 2025-06, and level normalisation cuts that
  gap by at least half.
decision_rule: >-
  Three planned contrasts on the quantile score, in points of capacity, paired on series, run, and
  valid time, with whole calendar months resampled (95% interval). C1 is the old-year raw arm
  minus the recent-year raw arm. C2 is the old-year normalised arm minus the recent-year
  normalised arm. C3 is C1 minus C2, the part of the age penalty that normalisation removes.
  Supported when C1 is at least 0.5 with its interval above zero and C3 is at least half of C1
  with its interval above zero. If C1 is below 0.5 or C3 is under half of C1, the penalty is not
  mostly level, and the next steps are the heating offset, solar vintage checks, and weather
  product reconciliation, not the level-normalised long history of LV1. The threshold for porting
  is zero for the level subtraction itself (it adds one cheap per-series step), but the verdict
  uses the 0.5-point size from the design brief.
benchmark: >-
  The recent-year raw arm (trained 2024-07 to 2025-06), plus `manual_heuristic` and `climatology`
  scored on the same rows.
row_set_id: age1_daily_2025_07_to_2026_06_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over 11 levels, 0.01 to 0.99) in
  points of capacity; the 9-level score of earlier experiments is reported beside it.
verdict: refuted
best_arm: >-
  Trained 2024-07 to 2025-06, raw: 4.78 (95% interval 4.45 to 5.14) points of capacity, the best of
  the eight models.
headline: >-
  Subtracting a trailing 365-day level removes 42% of the old year's penalty (C1 1.57, C2 0.91
  points), under the half the prediction required, and without the one series that carries most of
  the penalty it removes none (0.61 to 0.68).
lessons: >-
  An old training year hurts mainly for reasons other than its average level. One substation
  carries most of the raw penalty, and for the other 12 the old year scores about 0.6 points worse
  whether or not the level is removed. The old year also changes the weather product (18 km control
  member against 9 km), so the weather seam is a candidate that this experiment could not separate
  from age.
advice:
  - >-
    Do not expect a trailing-level subtraction to make older training years usable for Primary
    demand, because it removed 42% of the penalty with all 13 series and none without the largest
    one (C1, C2, C3).
  - >-
    Keep training on the recent year or two until the weather-product seam is tested, because the
    2021-07 and 2022-07 years score 1.3 to 1.6 points worse than the 2024-07 year (C1, R1).
  - >-
    Report every old-year contrast with and without the substation that carries most of it, because
    one substation moved the all-series penalty from 0.6 to 1.6 points (figure 3).
  - >-
    Test the 18 km to 9 km weather seam directly before building a long-history model, for example
    by training the recent year on the 18 km control member (post hoc reading of C1).
production_status: not-for-production
adopted_in: none
---

# Experiment 93: does an old training year still hurt once the level is removed? (AGE1)

**Taking each substation's average demand level out of the targets removes under half of the harm an
old training year does, and without the one substation that carries most of the harm it removes
none.** The old year therefore differs from the recent year in more than its level. The weather
product also changed between the years (18 km control member in the old years, 9 km in the recent
year), so the next experiment should test that seam before any long-history model is built. The
verdict is refuted: the penalty reached the size predicted, but level removal did not take off half
of it.

We trained the same XGBoost quantile model once per substation on one year of ENS control-member
forecasts, for 13 Primary substations. We scored each model on 365 daily forecast runs from July 2025
to June 2026. Two of the four planned models trained on 2021-07 to 2022-06, and two on 2024-07 to
2025-06. For each year, one model used the raw demand and one had the substation's trailing
365-day average demand subtracted from the target and the recent-power inputs, then added back to
the forecast.

![Age penalty, raw and level removed](figure_1.svg)

![Absolute score of every model](figure_2.svg)

![Penalty by substation](figure_3.svg)

![Tail calibration](figure_4.svg)

## Results

- **Penalty (C1, C2, C3).** Old minus recent, quantile score in points of capacity, 98.3% interval:
  raw 1.57 (1.14 to 2.00); level removed 0.91 (0.69 to 1.20); the difference 0.65 (0.04 to 1.23),
  which is 42% of the raw penalty. The prediction needed the penalty at least 0.5 and the share at
  least 50%, so the second half failed.
- **Without the largest series (post hoc, 95%).** Raw 0.61 (0.41 to 0.81); level removed 0.68
  (0.47 to 0.94). Level removal took off none of the penalty. One substation has a raw penalty of
  13.1 points, against 0.0 to 2.2 for the other 12.
- **Repeat with 2022-07 to 2023-06 (post hoc, 95%).** Raw 1.28 (1.04 to 1.55); level removed 0.86
  (0.60 to 1.17); the share removed is 33%.
- **Half-hourly level (post hoc, 95%).** Subtracting a level for each half-hour of the day gives an
  old-year penalty of 0.75 (0.47 to 1.09) for the 2021-07 year, close to the all-hours level's
  0.91.
- **Normal-arrangement rows only (95%).** Raw 1.65 (0.85 to 2.03); level removed 0.72 (0.56 to 1.00).
- **Absolute scores (points, 95% interval).** Recent raw 4.78 (4.45 to 5.14); recent, level removed
  4.84 (4.49 to 5.22); 2021-07 raw 6.34 (6.01 to 6.69); 2021-07, level removed 5.75 (5.29 to 6.32);
  2022-07 raw 6.05; 2022-07, level removed 5.70. Level removal does not help the recent year.
- **Tail calibration (share of actual values at or below the quantile; ideal 0.01, 0.05, 0.95,
  0.99).** Recent raw 0.015, 0.077, 0.913, 0.984. 2021-07 raw 0.091, 0.196, 0.924, 0.984. 2021-07
  with the level removed 0.041, 0.109, 0.838, 0.958. Level removal halves the old year's lower-tail
  miss and widens its upper-tail miss. The 0.01 pinball loss is 1.30 for 2021-07 raw against 0.27
  for recent raw.

## Caveats

- **Schedule.** Frozen fits at 150 rounds and `eta` 0.15 (the P-roll recipe), seed 0, one fit per
  substation and arm. The first 500-round fit took 155 s on the shared GPU, so the recipe was
  changed before any result existed. The contrast is a comparison of frozen models, so the
  retraining schedule does not enter.
- **Weather product.** The old years use the 18 km control member and the recent year the 9 km one,
  so age and weather product cannot be separated here.
- **Switching labels.** Labels start in 2022-01, so the first half of the 2021-07 year is not
  masked. The 2022-07 repeat is fully masked and gives the same picture.
- **Intervals.** Twelve calendar months are resampled, so the 98.3% intervals are coarse.
- **Not scored.** The climatology and `manual_heuristic` benchmark rows and the curtailment-cost
  metric were not computed, because the question is a contrast between training years of one model.

## What to try next

The penalty is not mostly a level. For the 12 substations without the large outlier, the old year
is about 0.6 points worse with or without the level, so the next experiment should separate age
from weather product, for example by training the recent year on the 18 km control member, and then
check the heating and solar terms. LV1's level-normalised long history is not supported by this
result.
