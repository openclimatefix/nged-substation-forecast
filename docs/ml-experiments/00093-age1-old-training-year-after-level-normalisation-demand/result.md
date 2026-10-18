---
id: "00093"
slug: age1-old-training-year-after-level-normalisation-demand
status: planned
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
production_status: not-for-production
adopted_in: none
---

# Experiment 93: does an old training year still hurt once the level is removed? (AGE1)

**Plan, committed before any fit runs.** Experiment 17 found that adding older years to the training
history made the Primary substation demand forecast worse, and LV0 (experiment 92) found the drift
is specific to each substation and is not explained by a plain level shift. This experiment asks
directly whether the damage done by an old training year is only a matter of that year's average
demand level.

## Design

- **Arms (one frozen fit per series each, no refits, 500 rounds, eta 0.05, equal row weights).**
  Old year raw; recent year raw; old year normalised; recent year normalised. The old year is
  2021-07 to 2022-06 and the recent year is 2024-07 to 2025-06. A repeat uses 2022-07 to 2023-06
  as the old year.
- **Training rows.** The daily ENS control member runs whose run start falls in the window and whose
  valid time lies inside it. The old years use the 18 km control member built for experiment 81, and
  the recent year uses the 9 km control member of the stage 1 pilot, so the change of weather
  product sits inside the contrast and cannot be separated from age here.
- **Normalisation.** For each row, the level is the mean of the series' power over capacity during
  the 365 days before the newest usable observation (3.5 hours before the forecast origin). The
  target and the three power lags have that level subtracted in training, and it is added back to
  the prediction. Half-hours inside a labelled abnormal arrangement are left out of the level.
  The choice of one all-hours level follows LV0 (experiment 92): a rolling mean carries over half
  the drift variance in 12 of 13 series. LV0 also found a summer-midday fall of 1.3 points a year
  that an all-hours level cannot carry, so a post hoc arm subtracts a level for each half-hour of
  the day, which is labelled post hoc and leaves the three contrasts and the verdict unchanged.
- **Switching.** Targets inside a labelled abnormal arrangement (experiment 41's labelled events
  and logged closures) are dropped from training. Labels start in 2022-01, so the first half of the
  2021-07 to 2022-06 window is unmasked, which the page states.
- **Scoring.** All 365 daily 00:00 UTC ENS runs of 2025-07 to 2026-06, 51 members mixed, valid times
  before 2026-07-01 only. Alongside the headline: the absolute score of every arm, tail
  calibration at the 0.01, 0.05, 0.95, and 0.99 levels, and the score by series. The models predict
  11 quantile levels, adding 0.01 and 0.99 to the usual 9.
- **Series.** The Primary substations with power in both the old and the recent window.
- **Schedule.** Frozen fits, so no retraining schedule is involved and the contrast does not depend
  on one. Seed 0.

## The five size triggers

- **Stored.** Predictions only, under the experiment's private data folder; no Delta table.
- **Serving path.** Not touched; the experiment trains frozen models offline.
- **Degradation rule.** Not touched.
- **More than one defensible design.** Yes for the normalisation (all-hours level against a level
  per half-hour), which the post hoc arm covers; the three pre-registered contrasts use the
  all-hours level.
- **Unnamed callers.** No: the scripts copy experiment 85's fitting helpers and call only
  `studies.*`.

The change is a permutation of settings the pipeline already has, plus one subtraction, so the size
is simple.

## Advice followed

Experiment 92's advice: run AGE1 before LV1 (followed), and treat the explicit solar term as the
likeliest carrier of the drift (followed by the post hoc half-hourly arm).
