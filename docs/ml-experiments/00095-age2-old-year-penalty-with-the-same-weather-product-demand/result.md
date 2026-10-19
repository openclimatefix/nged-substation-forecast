---
id: "00095"
slug: age2-old-year-penalty-with-the-same-weather-product-demand
theme: retraining-and-training-history
also_relevant_to: []
status: done
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
verdict: refuted
best_arm: >-
  The 2024-07 year on ERA5 (diagnostic, not for production): 4.51 (95% interval 4.18 to 4.89) points
  of capacity. The best control-member arm is the same year: 4.78 (4.45 to 5.14).
headline: >-
  The old-year penalty does not shrink when both years use ERA5: the 2021-07 year is 1.71 points
  worse than the 2024-07 year on ERA5 (99% interval 1.15 to 2.29) and 1.57 worse on the control
  member (1.12 to 2.03), so the weather product is not the main explanation and the penalty goes
  with how old the training year is. The gap shrinks with each newer year (ERA5: 1.71, 1.26, 0.46
  for the 2021-07,
  2022-07, and 2023-07 years).
lessons: >-
  Changing the weather product from 18 km to 9 km does not explain why an old training year scores
  worse. The penalty grows by about 0.5 to 0.8 points for each year older on ERA5 and 0.3 to 0.8 on
  the control member. It is not mainly an average level shift (experiment 93) or the weather product
  (this experiment). The largest-gap substation (D8) shows a smooth fall in demand with no step or
  stuck meter. The remaining candidates are demand that has changed shape, such as embedded solar or
  heating, and changes to the electricity network.
advice:
  - >-
    Train Primary demand models on the newest year or two of data until the cause of the age penalty
    is found, because each older year scores worse on ERA5 and on the control member (C1, C2, figure
    2).
  - >-
    Do not run more experiments on the 18 km to 9 km weather seam as an explanation of the age
    penalty, because the ERA5 gap is no smaller than the control-member gap (C3, figure 1).
  - >-
    Test explicit demand-shape terms, such as heating and embedded solar, before a long-history
    model, because the penalty grows by 0.5 to 0.8 points per year of age on ERA5 with no level or
    product explanation (figure 2, figure 4).
  - >-
    Report every old-year contrast with and without D8, because its gap is 13.1 points while the
    other 12 substations' gaps are 0.0 to 2.6 (figure 3).
production_status: not-for-production
adopted_in: none
---

# Experiment 95: does an old training year score worse because of the weather product, or its age? (AGE2)

**A model trained on demand from July 2021 to June 2022 forecasts demand in 2025-26 worse than a
model trained on 2024-25 demand, and the coarser weather forecast of the old years is not the main
reason.** The extra error ("gap") was 1.71 points of capacity when both models were trained and
scored on ERA5, the European Centre for Medium-Range Weather Forecasts (ECMWF) estimate of past
weather, against 1.57 points with the ECMWF ensemble forecast's unperturbed run ("control member").
Here "points of capacity" are percentage points of a substation's effective capacity on the
quantile score, where smaller is better. The gap also grew with each year older. Experiment 93
found that a difference in the average demand level between the years does not explain the gap.
The cause is therefore probably a change in how the substations' demand depends on weather and time
of day, such as growing embedded solar output or heating, or a change to the electricity network.
Explicit solar and heating terms are the candidate explanations the next experiment could test.

**The prediction was refuted on both counts.** The ERA5 gap was 109% of the control-member gap, not
under half. The model trained on July 2023 to June 2024 was 0.50 points behind the 2024-25 one, not
under 0.3.

We trained one XGBoost quantile model per Primary substation (13 substations, each named in the
figures by an anonymised code such as D8) on one year of demand and weather. "The 2021-07 year"
means the year of data from 2021-07 to 2022-06, and likewise for the other years. Experiment 93 had
shown that the 2021-07 year scores 1.57 points worse than the 2024-07 year, and that its years
also used a coarser weather forecast (18 km grid spacing, against 9 km from July 2023). Here each
arm (one training year with one weather product) was trained once, not retrained, and scored on the
365 daily forecast runs from July 2025 to June 2026. The ERA5 arms used ERA5 weather at the valid
time. The control-member arms used the control member's forecast. The quantile score is twice the
mean pinball loss over 11 quantile levels from 0.01 to 0.99, weighted by the probability each level
represents, in percentage points of capacity.

![Old-year gap on the control member and on ERA5](figure_1.svg)

![Quantile score by training year](figure_2.svg)

![Gap by substation](figure_3.svg)

![Demand of substation D8](figure_4.svg)

![Tail calibration](figure_5.svg)

## Results

- **The ERA5 gap is no smaller than the control-member gap (99% intervals, the Bonferroni level
  for five planned contrasts).** C1, the 2021-07 year minus
  the 2024-07 year on the control member, is 1.57 (1.12 to 2.03). C2, the same on ERA5, is 1.71
  (1.15 to 2.29). C3, C1 minus C2, is -0.15 (-0.32 to -0.002), so the ERA5 gap is slightly larger;
  C2 is 109% of C1 (1.00 to 1.18). C4, the 2023-07 year minus the 2024-07 year on the 9 km control
  member, is 0.50 (0.28 to 0.76). C5, the 2021-07 year minus the 2022-07 year on the 18 km control
  member, is 0.29 (-0.35 to 0.85), which cannot separate one year of age from zero.
- **The gap grows with each year of age on ERA5 (95%).** Against the 2024-07 year the 2022-07 year
  is 1.26 points behind (1.10 to 1.44) and the 2023-07 year is 0.46 behind (0.36 to 0.57). On the
  control member the 2022-07 year is 1.28 behind (1.04 to 1.55).
- **Absolute quantile score (points, 95% interval).**
    - ERA5: 2021-07 6.22 (5.82 to 6.60); 2022-07 5.77 (5.37 to 6.27); 2023-07 4.97 (4.58 to 5.43);
      2024-07 4.51 (4.18 to 4.89).
    - Control member: 2021-07 6.34 (6.01 to 6.69); 2022-07 6.05 (5.60 to 6.62); 2023-07 5.28
      (4.84 to 5.77); 2024-07 4.78 (4.45 to 5.14).
- **Without D8, the substation that carries most of the penalty (post hoc, 95%).** C1 is 0.61 (0.41
  to 0.81), C2 is 0.78 (0.53 to 1.04), C4 is 0.35 (0.19 to 0.53), and C5 is -0.05 (-0.52 to 0.37).
  The ERA5 gap is larger than the control-member gap here: C3 is -0.18 (-0.28 to -0.08) and C2 is
  129% of C1 (116% to 144%), so D8 dilutes the ratio with all 13 series. D8's own gap is 13.1 points
  on the control member and 12.9 on ERA5.
- **D8's demand fell smoothly; the checks found no step change or stuck meter.** Its monthly mean
  demand fell from 0.82 of capacity in 2021-07 to 0.34 in 2026-06, with no month under 99.3%
  observed half-hours and under 0.6% of half-hours repeating the previous value. The largest change
  in the 60-day mean demand over 60 days is 0.22 of capacity, a rise ending 2024-08-25. Summer
  demand fell faster than winter demand (August 0.77 in 2021 and 0.31 in 2025; January 0.76 in 2022
  and 0.48 in 2026), which fits growing embedded solar output, but this experiment did not test
  whether embedded solar output grew.
- **Rows in the normal switching arrangement only (95%).** C1 is 1.65 (0.84 to 2.03) and C2 is 1.81
  (0.79 to 2.26), so labelled switching events are not the main explanation, though the intervals
  are wide. The switching-event labels start in 2022-01, so the first half of the 2021-07 year is
  not masked.
- **Weather product within one year (post hoc, 95%).** ERA5 gives a quantile score 0.31 points lower
  (better) than the control member for the 2023-07 year (0.16 to 0.46) and 0.27 lower for the
  2024-07 year (0.13 to 0.42).
- **Tail calibration (share of actual values at or below the quantile; ideal 0.01, 0.05, 0.95,
  0.99).**

| Model | 0.01 | 0.05 | 0.95 | 0.99 |
|---|---|---|---|---|
| 2021-07, control member | 0.091 | 0.196 | 0.924 | 0.984 |
| 2021-07, ERA5 | 0.098 | 0.233 | 0.917 | 0.983 |
| 2024-07, control member | 0.015 | 0.077 | 0.913 | 0.984 |
| 2024-07, ERA5 | 0.019 | 0.096 | 0.893 | 0.980 |

  The pinball loss at the 0.01 quantile is 1.30 (control member) and 1.35 (ERA5) for the 2021-07
  year, and 0.27 and 0.28 for the 2024-07 year, in points of capacity.

## Caveats

- **Schedule.** Frozen fits of 150 boosting rounds at learning rate 0.15, seed 0, one fit per
  substation and arm. Weekly or monthly refits would give an old year's model newer data, so the
  contrasts describe how old a frozen model's training data is, not a retraining schedule.
- **ERA5 arms are diagnostic.** ERA5 arrives about 5 days after the weather, so no live service
  could use it at forecast time. The ERA5 arms score the same runs and valid times as the control
  member arms, with a single weather value per time and no ensemble spread. Absolute scores and tail
  calibration are therefore not like for like across the two products, and C3 compares two
  contrasts, each computed within one weather product.
- **The 2023-07 year.** Its rows come from experiment 81's daily frames up to 2024-03-31 and from
  the stage 1 pilot's training file from 2024-04-01. The 2024-01-03 run is missing, so the year has
  one run fewer. The 18 km to 9 km change took place in June 2023, so the 2023-07 year is 9 km.
- **Intervals.** They resample 12 whole calendar months, so they are coarse. The share C2 over C1 is
  a ratio of two means and its interval is only informative because C1's interval excludes zero.
- **Not scored.** The curtailment-cost metric and the `manual_heuristic` and climatology rows, as in
  experiment 93.

## What to try next

The age penalty is about 0.5 to 0.8 points for each year older on ERA5 and 0.3 to 0.8 on the control
member, and a level shift does not explain it. Two experiments could test what does: an explicit
heating offset (the subject of the reserved experiment 94) and an explicit embedded-solar term, with
the old years re-scored. If either term shrinks the gap across the four years, the result would
support trying a longer training history. If neither does, the data give no reason to move away from
the newest year or two.
