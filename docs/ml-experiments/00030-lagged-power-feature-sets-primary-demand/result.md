---
id: "00030"
slug: lagged-power-feature-sets-primary-demand
status: planned
size: simple
hypothesis: >-
  For the 16 Primary substation demand series, giving the single-stage ENS-trained quantile model N1
  the 13 power lags that `manual_heuristic` uses (the same weekday and time of day 1 to 6 weeks back,
  and 49 to 55 weeks back), as separate columns beside N1's three lags measured from the forecast
  origin, lowers the quantile score.
prediction: >-
  The 6 weekly analogues lower N1's quantile score by 0.1 to 0.3 points of capacity (C1), mostly at
  leads beyond 3 days, where N1's latest same-weekday lag is already 2 or more weeks old. The 7 annual
  analogues lower it by a further 0.05 to 0.2 points (C2). The summaries (mean and median of the
  weekly analogues, mean of the annual analogues) score within 0.05 points of the separate columns
  (C4). The gain shrinks by at least a third when the 14-day error correction is applied to both arms
  (C8), because the correction already removes the level offset that the annual analogues carry.
decision_rule: >-
  Eight planned contrasts on the 16 Primary substations, quantile score over all leads, read at
  99.375% (Bonferroni across eight), whole months and the three seeds resampled. C1 is A1 minus A0,
  C2 is A2 minus A1, C3 (primary) is A2 minus A0, C4 is A3 minus A2, C5 is A4 minus A2, C6 is A5
  minus A2, C7 is A2 minus `manual_heuristic`, and C8 is A2 with the correction minus A0 with the
  correction. The hypothesis is supported when C3's interval is wholly below zero. The threshold
  for porting a lag column is zero, because a lag column costs one more column in the existing
  feature pipeline and no new data source or step, so an interval that excludes zero is enough; the
  arm with the lowest score among those whose step over its predecessor is a gain is the
  recommended set. The residual-lag arm A6 needs a new pipeline step that stores past forecasts, so
  it is exploratory and would need a gain of at least 0.10 points against A2 with the correction (95%
  interval wholly below -0.10) to be worth a plan. The verdict is refuted when C3's interval is
  wholly above zero, and inconclusive otherwise. The mean absolute error of the median is reported
  beside the quantile score and decides nothing.
benchmark: >-
  N1 (A0, 00013), `manual_heuristic` and `climatology` (stored, 00013), and the 00019 stack (N1H with
  the 14-day correction and the 1.15 spread factor)
row_set_id: fold_a_every5d_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over 9 levels), in points of capacity
production_status: not-for-production
adopted_in: none
code: ml-experiments/00030-lagged-power-feature-sets-primary-demand/
---

# Which lagged power columns lower the error of Primary substation demand forecasts?

**Status: planned.** This page is the plan, committed before any code runs.

## Plan: sizing

- **Stored data:** prediction sets only, under `data/ml-experiments/`; no production table changes.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **Defensible designs:** more than one for the lag definitions; each is fixed below before a fit.
- **Callers:** the scripts copy 00013's and 00019's helpers, which the plan's author could name
  without searching.

Size: simple, because arms A1 to A5 permute the input columns of N1 with no new data and no new
method. Arm A6 stores past forecasts as a model input, so it is exploratory. Review: one Sonnet
reviewer reads the code for leakage and for the fit stopping at the training end, and one Sonnet
reviewer checks the write-up against the printed report. The earlier advice that this plan follows:
compare every Primary arm against `manual_heuristic` (00013); add lagged power measured from the
forecast origin (00013); report the interval over validation months (00013). It departs from none.

## Plan: arms

Each arm is N1 (00013: the 00006 columns, the lead in hours, recency half-life 2 years, 500 rounds,
the 9-level quantile loss, trained on the ENS control member from 2024-04-01 to 2025-06-30)
with a different lag block. `colsample_bytree` and `colsample_bynode` are 1.0 in every arm, so a wider
arm has no subsampling advantage. Every lag is read from the observed power of the same series on the
dense half-hourly grid and is null (NaN) when its grid index is later than the newest usable
observation, 3.5 hours before the forecast origin (00021's and 00013's telemetry delay). No lag
column looks past that index, in training or in validation.

| Arm | Lag columns |
|---|---|
| A0 (N1, 00013's stored fits) | 3: the newest usable value, and the latest earlier same-half-hour and same-weekday values |
| A1 | A0 plus the 6 weekly analogues: power at the valid time minus 1, 2, 3, 4, 5, and 6 weeks |
| A2 | A1 plus the 7 annual analogues: power at the valid time minus 49 to 55 weeks (the 13 lags of `manual_heuristic` plus A0's 3) |
| A3 | A0 plus 3 summaries of A2's 13 lags: the mean and the median of the usable weekly analogues, and the mean of the usable annual analogues |
| A4 | A2 plus 3 recent values: the mean of the 12 half-hours ending at the newest usable one, and the values 2 and 4 half-hours before it |
| A5 | A2 plus 3 same-day-type analogues: power at the same half-hour on the latest 3 earlier days of the valid time's day type (weekday against weekend or bank holiday, from the England holiday calendar) |
| A6 (exploratory) | A2 plus the mean residual of the last 14 days (actual minus the A2 median forecast of each half-hour, taken from cross-fitted fits of A2 over 3 blocks of training months) |

A lag of a fixed number of weeks is null whenever the lead is long enough for it to fall after the
newest usable observation, so the 1-week analogue is null beyond about 6.9 days of lead, which is
also what the pipeline's `_nullify_leaky_lags` does. A0 already holds the latest day and the latest
week analogue at any lead, so A4 and A5 test only what is added beyond them. Every arm is also
scored with the 14-day error correction of 00019 (the mean error of the arm's median over the 14
days up to the newest usable observation, added to every quantile), and the report shows each arm
with and without it. The 00019 stack is rebuilt from 00019's stored N1H fits. The arms are fitted
for seeds 0, 1, and 2, and a climatology and the stored baselines are repeated per seed as in 00013.

## Plan: planned contrasts

The eight contrasts in the decision rule are read over all leads at 99.375%. Every other contrast
(each lead band, BSP and GSP, with and without the correction, A6, against the 00019 stack) is
exploratory, read at 95%. BSP (2 series) and GSP (2 series) are descriptive and never averaged
into the Primary figure.

## Plan: past weather

No past weather is used at lag times; the lags are of power, observed before the origin.
