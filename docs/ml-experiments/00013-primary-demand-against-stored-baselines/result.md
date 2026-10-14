---
id: "00013"
slug: primary-demand-against-stored-baselines
status: planned
size: complex
hypothesis: >-
  For the 16 Primary substation demand series, a single-stage XGBoost quantile model trained on the
  ENS control member (N1: the 00006 columns, recency weights, and power lags measured from the
  forecast origin) beats `manual_heuristic`, the best stored baseline, and the production recipe
  loses to `manual_heuristic` because it has no usable power persistence at short leads.
prediction: >-
  N1 lowers the mean absolute error of the median (MAE) below `manual_heuristic` by at least 0.5
  points of capacity at leads up to 48 hours and by at least 0.2 points over all leads. The
  production recipe scores 1.5 to 3 points worse than `manual_heuristic` (the stored leaderboard
  shows 2.2), and N1 without origin lags recovers at least half of that gap at leads beyond 3 days.
decision_rule: >-
  Five planned contrasts, read at 99% (Bonferroni across five), on Primary substations, months
  resampled and the three seeds. C1, N1 minus `manual_heuristic`, is the primary contrast and counts
  as a gain when the MAE interval over all leads is wholly below zero and the quantile-score interval
  is wholly below zero. N1 passes the gate for P7 and P8 when C1 is a gain. If only the MAE interval
  is wholly below zero, the verdict is partly supported and P7 compares against `manual_heuristic`
  too. If C1's MAE interval is wholly above zero, or C3's all-lead MAE interval is not wholly above
  zero (production does not lose to `manual_heuristic` on these rows), the hypothesis is refuted.
  The clause that N1-nolag recovers half the production gap is exploratory: C5's all-lead point
  estimate beyond 3 days must be at most -50% of C3's.
benchmark: >-
  `manual_heuristic` (stored), `climatology` (stored), and the production recipe (stored), all
  re-scored on the row set; reference arms from 00005 (R0 lag-free ERA5-trained stage 1, R1
  ENS-trained with lags as inputs, R2 climatology).
row_set_id: fold_a_every5d_v1
metric: >-
  Mean absolute error of the median, and the quantile score (twice the mass-weighted mean pinball
  loss over 9 levels, an estimate of the continuous ranked probability score), in points of capacity
verdict: null
code: ml-experiments/00013-primary-demand-against-stored-baselines/
---

# Where does Primary substation demand stand against the stored baselines?

**Plan, written before any code runs.** The result is added below when the experiment finishes.

## Sizing

- **Stored data:** prediction sets only, under `data/ml-experiments/`; no production table changes.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **Defensible designs:** more than one. N1's weather pooling, the telemetry delay, and the lag
  definitions are choices; each is fixed below before a fit.
- **Callers:** the scripts copy 00005's helpers and read its stored predictions, which the author of
  this plan could name without searching.

Size: complex. The plan gets two Sonnet reviews (Opus is kept for work that is truly vital), the code
gets two Sonnet reviews (leakage and the fit stopping at `train_end`; then correctness), and the
write-up gets one.

## Arms

All arms are scored on the 73 runs of `fold_a_every5d_v1`, on the rows every arm holds and an
observed power value, for valid times before 2026-07-01. Primary substations (16 series) are the
headline; BSP and GSP (2 series each) are reported separately and never averaged in.

- **MH, CL, PROD (stored):** `manual_heuristic`, `climatology`, and the production recipe
  (`xgboost_baseline`, fold `mid_2025_to_mid_2026`) read from the `power_forecasts` Delta table;
  the quantiles of each arm's members. They have no seed, so each is repeated once per seed.
- **S1 (00005's `r0_from2023`):** stage 1, ERA5-trained from 2023-09, no lags, pooled over the 51
  members. R1 (`r1`) and R2 (`r2`) are reported as references.
- **N1:** one XGBoost model per series with the 9-level quantile loss, trained on the ENS control
  member of every daily run from 2024-04-01 (the start of the stored ENS archive, so 2023-09 is not
  available to ENS-trained arms), valid times up to 2025-06-30 23:30. Columns: the 00006 Primary set
  (weather block, solar position, hour, weekday, COVID indicator, windowed days to Christmas), the
  lead in hours, and three lags measured from the forecast origin. Recency weights use a half-life
  of 2 years, anchored at 2025-06-30 (the last training valid time), not at each row's own date.
  The fit runs a fixed 500 rounds with no early stopping and no validation split. A training row is
  kept only when its valid time is on or before 2025-06-30 23:30, and every lag of a training row
  lies before that row's origin.
- **Pooling.** The 51 members' 9 quantiles each give 459 values with equal member weights and the
  9-level atom masses; the forecast quantiles are the weighted quantiles of those values at the 9
  levels, sorted so none cross. A model trained on the control member and applied to perturbed
  members adds its own spread to the ensemble's spread, so the bands may be too wide, which raises
  the quantile score. N1-control, N1 applied to the control member alone, checks this
  (exploratory).
- **N1-nolag:** N1 without the lag columns (isolates the lags, and ENS-trained against ERA5-trained
  once compared with S1).
- **N1-nocal:** N1 without hour, weekday, and days to Christmas (the no-calendar control).

**The three lags.** The origin `T` is the NWP run time plus 9 hours. The newest usable observation
is the latest half-hour at or before `T` minus 3.5 hours. NGED's files cover 6-hour windows, and the
live ingest downloads each window 3 hours after it ends, so at 09:00 UTC the 03:00 to 06:00 window has just arrived.
The plan keeps one half-hour of margin, so the newest value is the one stamped 05:30. The lags are the power at that newest half-hour, the power at the
latest earlier time with the same half-hour of day as the valid time, and the power at the latest
earlier time with the same weekday and half-hour. All three are zero-lookahead by construction. The
telemetry delay is an assumption, not a measurement of NGED's delay; the live ingest's download time
is the only evidence in the repository. Every N1 result is conditional on the 3.5-hour figure, and
no sensitivity run is planned.

**Deviation from P6.** P6 names a 2023-09 start for N1. The stored ENS archive starts on 2024-04-01,
so N1 trains on 15 months where S1 trains on 22. C5 therefore mixes the effect of training on ENS
with the effect of the shorter window. The stored arms have no seed, so each is repeated once per
seed as the skill requires; the three copies are identical, and the intervals resample months.

**Diagnosis of the production gap.** The stored forecasts are split by lead band and series; the
gap between production and `manual_heuristic` is decomposed into bias (mean error) and the rest;
and the ladder PROD, S1, N1-nolag, N1 shows which step recovers the gap.

## Planned contrasts (99%, Primary substations)

- **C1:** N1 minus `manual_heuristic` (primary).
- **C2:** N1 minus PROD.
- **C3:** PROD minus `manual_heuristic` (the gap to diagnose).
- **C4:** N1 minus N1-nolag (what origin lags add).
- **C5:** N1-nolag minus S1 (ENS-trained quantile against ERA5-trained, both lag-free).

The planned contrasts are read at 99% over all leads, over leads up to 48 hours, and over leads of
3 days or more. Every other contrast is exploratory and read at 95%, including each lead band.

## Past weather

N1 uses ENS weather in training and in scoring, so no past weather is used at lag times. The lags
are of power, observed before the origin.
