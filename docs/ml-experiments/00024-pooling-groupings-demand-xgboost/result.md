---
id: "00024"
slug: pooling-groupings-demand-xgboost
status: planned
size: complex
hypothesis: >-
  For the 20 demand series of Primary, BSP and GSP substations, one XGBoost quantile model trained
  across a group of series (all series, one model per substation type, or one model per size band),
  given the series identifier as a categorical column, scores within 0.15 points of capacity of one
  model per series, so that the number of models can fall from one per series to one per group.
prediction: >-
  One model per substation type, with the identifier, scores within plus or minus 0.3 points of the
  per-series models (Primary-MW and Primary-MVA pools of 8 series each share a load shape). One
  model across all 20 series scores 0.2 to 1.0 points worse than the per-series models, mostly on
  the 2 BSP and 2 GSP series, whose load shapes differ from Primary substations. Dropping the
  identifier raises the score of the all-series pool by 0.2 to 1.0 points and the by-type pools by
  at most 0.2. The size-band pools fall between the by-type pools and the all-series pool.
decision_rule: >-
  Six planned contrasts on the quantile score over all leads of the full stack (14-day error
  correction and spread factor 1.15 on each model's quantiles), 20 series, read at 99.17%
  (Bonferroni across six), whole validation months and the three seeds resampled. C1 is by-type
  with the identifier minus per-series. C2 is all-series with the identifier minus per-series. C3
  is size-band with the identifier minus per-series. C4 is by-type minus all-series (both with the
  identifier). C5 is by-type without the identifier minus by-type with it. C6 is all-series without
  the identifier minus all-series with it. A pooled arm matches the per-series models when its
  contrast against per-series (C1 to C3) has an upper limit of at most +0.15 points; the margin is
  half of the 0.3 points that one validation year of 20 series resolves, and pooling removes models
  from production rather than adding a pipeline step, so no gain is needed. The experiment is
  supported when at least one pooled arm matches. The winner is the arm with the fewest models among
  the arms that match; if an arm's interval lies wholly below zero it is called better than
  per-series and wins when its point estimate is lowest. It is refuted when C1, C2 and C3 all lie
  wholly above zero, and inconclusive otherwise. C4 to C6 decide no verdict. Per-type breakdowns,
  the uncorrected models, and the median error are exploratory at 95%.
benchmark: >-
  Per-series models (the same recipe), `manual_heuristic`, and `climatology`.
row_set_id: fold_a_every5d_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over 9 levels) and mean absolute error
  of the median, in points of capacity (1 point is 1% of the series' effective capacity)
verdict: pending
production_status: pending
adopted_in: none
code: ml-experiments/00024-pooling-groupings-demand-xgboost/
---

# Experiment 24: which groupings of substations can share one XGBoost model?

**Status: planned.** The write-up replaces this page when the fits finish.

## Plan: sizing

- **Stored data:** prediction sets only, under `data/ml-experiments/`.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **Defensible designs:** more than one; the groupings, the pooled models' columns, and the
  per-series reference are fixed below before any fit.
- **Callers:** the scripts copy 00019's fit and 00021's scoring, which the author could name without
  searching.

Size: complex, because more than one design tests the hypothesis. One Sonnet review of the plan, one
of the code (leakage, the fit stopping at 2025-06-30, the correction's history bound), and one of
the write-up. Opus is kept for work that is truly vital.

## Plan: arms

**Recipe.** Every model is 00019's single-stage quantile model: the 9-level quantile loss on the ENS
control member, 00006's columns, the lead in hours, and the three power lags measured from the
forecast origin (00013's 3.5-hour telemetry delay). The recency half-life is 0.5 years (00017), the
training data are daily runs from 2024-04-01 with valid times up to 2025-06-30 23:30, and the model
is frozen there. Targets and lags are power over effective capacity, so series of different sizes
share one scale. Predictions pool the 51 ENS members into the 9-quantile mixture. The 14-day
additive error correction and the spread factor of 1.15 (00019's stack, correction history measured
with 00019's 3.5-hour delay) are then applied per series to each arm's quantiles.

**Series.** The 20 demand series of 00013 and 00021: 8 Primary substations that meter in MW, 8
Primary substations that meter in MVA, 2 BSP, and 2 GSP.

**Groupings.** The models differ only in which series a model trains on.

| Arm | Models | Rows per model |
|---|---|---|
| `own`: one model per series | 20 | one series |
| `type_id`: one per substation type (BSP, GSP, Primary-MW, Primary-MVA), with the identifier | 4 | 2, 2, 8, 8 series |
| `type_noid`: the same without the identifier | 4 | as above |
| `all_id`: one model across all series, with the identifier | 1 | 20 series |
| `all_noid`: the same without the identifier | 1 | 20 series |
| `size_id`: one per capacity band (terciles of effective capacity, 7, 7, and 6 series), with the identifier | 3 | 7, 7, 6 series |

The identifier is the series' label as a categorical column that XGBoost splits on natively.
`own` has no identifier because each model sees one series. The size bands use only the capacity,
which is fixed before any outcome. Primary substations' units (MW or MVA) come from the metadata
table. Each arm is fitted for seeds 0, 1 and 2. References: `manual_heuristic` and `climatology`
stored by 00013.

**Cost.** For each arm the experiment records the number of models and the wall-clock fit seconds
(training only, GPU), summed over models for one seed.

## Plan: leakage and scoring checks

- The scoring script asserts the correction's history bounds as 00019's and 00021's do: every
  observation in a correction is at or before the origin minus 7 half-hours, and the forecast for a
  past half-hour comes from a run that began at or before it.
- Training rows end at 2025-06-30 23:30 and only the control member is used; the scripts assert both.
- The validation rows are `fold_a_every5d_v1` (73 runs, every 5 days, valid times before
  2026-07-01), with ENS weather. Past weather is not used at lag times.
- A check refits `own` and compares it with 00019's N1H (same recipe): the two should agree within
  seed noise on the 16 Primary series.
- No calendar control is added: the calendar columns of 00006's set are in every arm, so the
  contrasts compare groupings, not the value of the calendar.

## Plan: earlier advice

Followed: 00006's columns, 00017's half-life of 0.5 years, 00019's 14-day correction and spread
factor of 1.15 (stack, not tested separately here), 00021's finding that the correction is unproven
for BSP and GSP (reported per type, not assumed to help), and a climatology row. Departure: the
half-life is tested on a frozen model, which 00019 found neutral.

## Plan: limits stated in advance

- **BSP and GSP have 2 series each,** so their per-type intervals are wide and the by-type pools for
  them are tiny pools of 2 series; the verdict is driven by the 16 Primary series.
- **Only 20 series,** so conclusions about 2,500 series are an extrapolation.
- **The correction's history comes from 5-day-spaced runs** (00019's limit).
