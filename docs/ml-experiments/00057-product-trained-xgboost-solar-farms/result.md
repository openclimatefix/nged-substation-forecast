---
id: "00057"
slug: product-trained-xgboost-solar-farms
status: planned
size: complex
hypothesis: >-
  For the six metered solar farms, an XGBoost quantile model trained on ENS and given UKV's
  shortwave radiation, total cloud, and temperature as extra input columns scores lower on the
  quantile score at lead hours 24 to 54 than the same model trained on ENS alone, with every arm
  trained on the same window, rows, and seeds. UKV is read from the CEDA archive, which is not the
  live product, so the contrast is a ceiling read.
prediction: >-
  The ENS-plus-UKV arm scores -0.05 to -0.20 points of capacity against the ENS-only arm at lead
  hours 24 to 54 after the ENS run's 00 UTC. The sources for that range are experiment 53, whose
  fitted ENS and UKV blend lowered the mean squared error of the solar anomaly by 4% to 14% within
  54 hours but raised the correlation with CAMS by only 0.01 to 0.03, and experiment 54, where UKV
  lowered the wind-farm score by 0.2. Solar is expected to gain no more than wind, because
  experiment 50 found that a second global ensemble (GEFS) added nothing to the solar anomaly at
  day 1 to 2. The UKV-alone arm
  scores worse than the ENS-only arm. The shuffled-UKV placebo scores within 0.04 points of the
  ENS-only arm. A `supported` verdict (interval wholly below -0.15) needs a point estimate near
  -0.25 or better, so `small gain` and `inconclusive` are the likeliest verdicts.
decision_rule: >-
  One decisive contrast, C1: the ENS-plus-UKV arm minus the ENS-only arm on the quantile score for
  cases with a valid time 24 to 54 hours after the 00 UTC start of the ENS run's date (lead day 1,
  and the first 6 hours of lead day 2), pooled over the six farms and read at 95% with whole months
  and the three seeds resampled, 50,000 resamples. The interval is conditional on the six farms.
  The threshold is 0.15 points, about 3% of the ENS-only arm's expected score of 5 points. It is
  the same relative size as experiment 54's wind threshold (2.5% of 6.1 points), so that the two
  technologies are read on one scale; an added weather feed and three added model input columns
  are a pipeline step each, and a smaller gain does not justify them. C1 is `supported` when its
  interval lies wholly below -0.15, `small gain` when it lies wholly below zero but not wholly
  below -0.15, `refuted` when it lies wholly above zero, and `inconclusive` otherwise. Because the
  CEDA archive is not the live UKV product, a `supported` C1 is reported as `small gain` for
  production and `supported` as a ceiling, and the verdict line says both. Every other read (all
  leads within UKV's range, the UKV-alone arm, the placebo, the 3-hourly control, per farm,
  daylight-only, against the ERA5-trained best guess) is exploratory, labelled so, and read at 95%.
benchmark: >-
  The ENS-trained XGBoost arm with the same training protocol (arm A), on the same cases. The
  ERA5-trained best guess of experiment 33 is a reference row, not the benchmark.
row_set_id: fold_a_tuesdays_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels, the estimate of the
  CRPS), in points of capacity; coverage of the 5% to 95% band; share of cases above the 0.95
  quantile
code: ml-experiments/00057-product-trained-xgboost-solar-farms/
production_status: not-for-production
adopted_in: none
---

# Experiment 57: does UKV, learned from its own history, help the solar-farm forecast?

**This page is the plan, committed before any code runs.** The write-up replaces it when the
experiment finishes.

We ask whether the Met Office's UKV forecast of sunshine, cloud, and temperature lowers the error of
a solar-farm forecast that already uses the European Centre's ensemble (ENS). The design is
[experiment 54's](../00054-product-trained-xgboost-wind-farms/result.md) design for the three wind
farms, applied to the six metered solar farms. Each XGBoost model is trained on the history of the
weather product it is given, so a model is never trained on one product and fed another.

## Plan

- **Arms.** All arms use experiment 33's recipe for its ENS-trained model: one XGBoost model per
  farm and seed with nine quantile levels, 500 rounds, the ENS control member of the daily 00 UTC
  runs from 2024-04-01 to 2025-06-30, recency weights with a half-life of 2 years, `lead_hours`,
  and three lags of power from the 09:00 UTC origin. No calendar column is used. Seeds are 0, 1,
  and 2.
    - **A:** the nine ENS solar columns of experiment 33 (shortwave, longwave, 2 m temperature, dew
      point, surface pressure, precipitation, 10 m wind speed, solar zenith, solar azimuth).
    - **B:** A plus three UKV columns: downward shortwave flux, total cloud cover, and 1.5 m
      temperature, as separate columns.
    - **C:** the UKV columns alone, plus the solar zenith and azimuth (deterministic geometry,
      which every arm has).
    - **Placebo:** B with the UKV columns shuffled among rows of the same farm, month, hour of day,
      and lead day.
    - **B3h (control):** B with the UKV columns rebuilt from UKV's steps at multiples of 3 hours
      and joined linearly. UKV is hourly and ENS is 3-hourly, so this separates a gain from UKV's
      time resolution from a gain from its different forecast.
    - **Reference rows:** experiment 33's ERA5-trained best guess and its ENS-trained model, on the
      same cases.
- **Scoring.** The 6 solar farms (anonymised labels), `fold_a_tuesdays_v1`, valid times 2025-07-01
  to 2026-06-30, cases within UKV's 54-hour range. Arms A, B, the placebo, and B3h run once per ENS
  member and pool the 51 member quantile sets; arm C runs once.
- **Clear-sky features.** Experiment 33's best guess for solar contains no clear-sky feature
  (experiment 12's clear-sky-ratio target gained -0.048, -0.093 to +0.024), so this experiment
  adds none.
- **UKV source.** The CEDA archive's 00 UTC run of the ENS run's date at each farm's nearest 2 km
  cell. The assumption that the run is available by 09:00 UTC is not measured.

## Earlier advice followed and departed from

- **Followed:** train each model on its own product's history (experiment 54); read UKV as a
  ceiling (experiments 53 and 54); use a shuffled-UKV placebo and per-farm reads (experiment 54);
  no calendar columns (experiment 1).
- **Departed from:** experiment 54 advised a live UKV history before any port. None exists for
  lead days 1 and 2, so this experiment stays a ceiling read.

## Size

- **What gets stored:** predictions under `data/ml-experiments/`, which are private; nothing in a
  Delta table.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes (which UKV columns, how to join the hourly steps), so
  the size is complex; the columns are fixed here, before any result.
- **Code whose callers could not be named without searching:** no; the scripts copy experiment 54's.
