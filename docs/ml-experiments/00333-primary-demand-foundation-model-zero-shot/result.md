---
id: '00333'
slug: primary-demand-foundation-model-zero-shot
title: >-
  Does a pre-trained time-series foundation model, used without training, forecast Primary demand
  as well as Primary demand research default 3?
theme: production-reference-and-best-guess
also_relevant_to:
  - weather-products-and-blending
status: planned
kind: change
size: >-
  medium. Stored: private files under data/ml-experiments/00333-*/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one model is defensible, but the brief fixes
  the arms and this plan fixes the model (Chronos-2). Callers: this folder's scripts, which import
  the folders of experiments 317 and 322 and read their stored files. One new dependency
  (`chronos-forecasting`), added with `uv run --with` and not to any `pyproject.toml`. No change
  under `packages/`.
hypothesis: >-
  A pre-trained time-series foundation model (Chronos-2, 120 million parameters, Apache-2.0)
  forecasts a Primary substation's half-hourly demand from the substation's own history. Without
  weather inputs it cannot match Primary demand research default 3 at leads of 1 to 10 days. With
  the ensemble-mean temperature, a solar-elevation term, and calendar columns passed as covariates,
  it closes part of the gap, but not all of it.
prediction: >-
  Written before any score is read; each can fail. (P1) Arm Z0 (history only) trails Primary demand
  research default 3 by more than 0.50 points of effective capacity pooled at 0.5 to 240 hours, and
  trails the manual heuristic at leads beyond 48 hours. (P2) Arm Z1 (with covariates) closes at
  least half of the gap between Z0 and the default at 24 to 120 hours. (P3) No arm is within 0.03
  points of the default at 0.5 to 240 hours. Stop rule: the reproduction gate fails (the default
  rows equal experiment 322's default to within 0.01 points); or Z0 and Z1 both trail the manual
  heuristic in every 24-hour lead band, in which case the result is reported and Z2 is skipped.
decision_rule: >-
  No arm is a candidate for production, by the maintainer's instruction. The experiment reports each
  arm's gain over Primary demand research default 3 with its interval, with and without D10, and the
  maintainer decides what counts as decisive. Z2 (a light fine-tune) runs only if Z1 is within 0.30
  points of the default in some 24-hour lead band, with the number of steps fixed before any score
  is read. Intervals resample the 12 months of the origin, at the 95% level.
benchmark: >-
  Primary demand research default 3 (experiment 251's arm below 168 hours, experiment 322's d1
  above), the manual heuristic, and climatology, all from stored files.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score over 11 levels (twice the mean pinball loss), in points of effective capacity;
  twCRPS at the 90th-percentile threshold of each Primary; pinball loss and coverage at q01, q05,
  q95, q99.
code: ml-experiments/00333-primary-demand-foundation-model-zero-shot/
date: 2026-10-24
---

## Plan

Primary demand research default 3 is our research demand forecast for six Primary substations. This
experiment asks whether a time-series foundation model, pre-trained on other series and used
without training on ours, gets near it. The model is
[Chronos-2](https://arxiv.org/abs/2510.15821) (Ansari et al., 2025), which accepts past and known
future covariates.

### Model choice

- **Chronos-2** (`amazon/chronos-2`): Apache-2.0 licence, 478 MB of weights, covariates supported.
  It is the only candidate that takes covariates, which Z1 needs.
- Not used: TimesFM, Moirai, and Toto. Z0 on a second model would test the same hypothesis and
  would not give Z1. The download costs nothing.

### The five plan items

1. **Question.** How far behind Primary demand research default 3 is a zero-shot foundation model
   at 0.5 to 240 hours, by 24-hour lead band?
2. **Prediction that can fail.** P1 to P3 in the front matter.
3. **Rows and period.** `fold_a_daily_gap_v1`, six Primaries (D5, D9, D10, D11, D14, D16), scored
   year 2025-07-02 to 2026-06-30, origin 09:00 UTC, rows outside hand-filtered events.
4. **Metric.** As in the front matter, with the research-default page's protected groups.
5. **Stop rule.** As in the front matter.

### Arms

- **Z0:** the context is each Primary's half-hourly demand in the 2688 half-hours (8 weeks) that
  end 3.5 hours before the origin, the telemetry delay of experiment 320. The model forecasts 487
  half-hours ahead, and the first 7 are dropped. The 11 quantile levels come from the model.
- **Z1:** Z0 plus covariates: the ensemble-mean 24-hour temperature (the forecast covariate of
  experiment 317, available at forecast time; past values come from the run with the shortest lead
  for each past half-hour), the solar elevation, and the local half-hour, weekday, and bank-holiday
  calendar columns. The solar elevation stands in for the default's solar response.
- **Z2:** only if the rule in the front matter holds.
- **References:** the default, the manual heuristic, climatology.

### Checks to run

The reproduction gate, median minus mean forecast per Primary, mean-forecast error beside each
quantile-score contrast, leaving D10 out, cold winter nights, sunny daylight, tail calibration, and
the seven audit triggers.
