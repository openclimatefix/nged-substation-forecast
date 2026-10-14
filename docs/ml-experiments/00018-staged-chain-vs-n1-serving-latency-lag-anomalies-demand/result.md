---
id: "00018"
slug: staged-chain-vs-n1-serving-latency-lag-anomalies-demand
status: planned
size: complex
hypothesis: >-
  For the 16 Primary substation demand series, the staged chain (an ERA5-trained stage 1 and the
  stage 2 calibrator, with lag anomalies built only from weather that serving has at the forecast
  origin, and NGED's real telemetry delay) beats the single-stage ENS-trained quantile model N1 and
  `manual_heuristic` on the quantile score.
prediction: >-
  The chain with serving-latency lag anomalies scores 0.3 to 0.8 points of capacity below N1 on the
  quantile score over all leads, and 0.3 to 0.5 points better than `manual_heuristic` (a lower score is better). It keeps at least
  half of the 1.80-point gain over the lag-free calibrated stage 1 that experiment 00005 measured
  with ERA5 weather and no telemetry delay. The 5% to 95% band covers 83% to 87% before a spread
  factor and 88% to 91% after.
decision_rule: >-
  Six planned contrasts on the 16 Primary substations, read at 99.17% (Bonferroni across six),
  months and seeds resampled. C1 (primary) is the chain with serving-latency lag anomalies (S)
  minus N1, and C3 is S minus `manual_heuristic`, both on the quantile score over all leads. The
  staged design is supported on demand when both C1 and C3 have a point estimate of at most -0.3
  points and an interval wholly below zero. The threshold of 0.3 points (about 5% of N1's 6.33)
  is set by the cost of porting: a second model per series, a second weather product at lag times,
  and a calibrator that must be refitted. The verdict is `small gain` when both intervals are wholly
  below zero and either point estimate is above -0.3, `refuted` when the C1 interval is wholly above
  zero or the C3 interval is wholly above zero, and `inconclusive` otherwise. A split outcome, in which
  one of C1 and C3 is wholly below zero and the other wholly above, is `refuted`. The mean absolute
  error of the median is reported beside the quantile score and decides nothing.
benchmark: >-
  N1 and `manual_heuristic` from experiment 00013 (stored predictions, same row set), plus the
  lag-free calibrated stage 1 R0c of experiment 00005, the production recipe, and `climatology`.
row_set_id: fold_a_every5d_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over 9 levels, an estimate of the
  continuous ranked probability score) and mean absolute error of the median, in points of capacity
  (1 point is 1% of the series' effective capacity); coverage of the 5% to 95% band; mean signed
  error reported separately as bias
---

# Does the staged chain beat the single-stage model for Primary substation demand?

**Status: planned.** This page holds the pre-registered plan. Results are added after the run.

## Plan: sizing

- **Stored data:** prediction sets under `data/ml-experiments/`; no production table changes.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **Defensible designs:** more than one. The weather source at each lag time, the telemetry delay,
  and the spread factor each have more than one defensible form; each is fixed below before a fit.
- **Callers:** the scripts copy the helpers of experiments 00005 and 00013, which the author of this
  plan could name without searching.

Size: complex. The plan gets one Sonnet review, the code gets two Sonnet reviews (leakage and the
fit stopping at the fold's end; then correctness), and the write-up gets one. Opus is not used,
because the maintainer asked for Opus only where vital.

## Plan: what differs from experiment 00005

Experiment 00005 measured the lag gain with a ceiling, because the lag anomaly (observed power at a
lag time minus stage 1's expectation at that time) used final ERA5 weather at past times and a
forecast origin with no telemetry delay. This experiment changes three inputs and nothing else in
the chain.

- **Telemetry delay.** NGED's files cover 6-hour windows ending at 02:00, 08:00, 14:00, and 20:00
  UTC, and in this project's downloaded-files list each file lands 3.0 hours after its window ends
  (median over the 26,697 files from 2026-03-26). The forecast origin is 09:00 UTC, 9 hours after
  the 00:00 UTC ENS run. The newest observed half-hour usable at the origin is therefore the one
  ending 02:00 UTC, 7 hours before the origin. Experiment 00013 assumed 3.5 hours. A lag is usable
  only when its time is at or before that newest half-hour.
- **Weather at lag times.** Stage 1's expectation at a lag time uses one of four weather sources,
  all emulated from data on disk, and every source respects what serving has at 09:00 UTC. ERA5T is
  emulated by final ERA5 and is available for times at least 5 days before the origin. CAMS
  irradiance (hourly, at the series' location) is available up to 00:00 UTC of the origin day. The
  analysis proxy is the ENS control member of the freshest run that had started by the lag time
  (lead 0 to 24 hours, so a run's first day), as `weather_utils.analysis_proxy` defines it, and is
  available to the origin. ERA5T differs slightly from final ERA5, and the emulation ignores that.
- **Lag times.** The anomalies and residuals use the times 24, 48, 168, and 336 hours before the
  valid time, the newest usable half-hour and the ones 1 and 3 hours before it, and the latest
  earlier time with the valid time's half-hour of day and of week. A lag is null when its time is
  after the newest usable half-hour (origin minus 7 hours), so the 24-hour lag is unusable at leads
  up to 17 hours.
- **Which stage 1 models.** Calibration origins use the forward-chain stage 1 model of their
  quarter, and validation origins use the final model, both from experiment 00005. A chain model stops
  15 days before its quarter starts, and an origin reads lag times no earlier than 14 days before
  itself, so every expectation is out of sample. The experiment refits
  nothing in stage 1.

## Plan: arms

All arms use stage 1a from experiment 00005 (`from2023`: ERA5-trained from 2023-09, 13 columns)
the same ENS statistics (mean, standard deviation, 10th, 50th,
and 90th percentile of the 51 members, plus the lead), and the same stage 2 calibrator
(multi-quantile XGBoost, depth 4, 300 rounds, 9 levels, per series, trained on every third daily run
from 2024-04-01 to 2025-06-30). They differ only in the recent-origin inputs. Seeds are 0, 1, and 2.

- **CH-none:** raw recent power only: the newest usable power, the power 1 and 3 hours before it,
  the power at the same time of day and of week, and the four power lags (24, 48, 168, and 336
  hours), no lag anomalies.
- **CH-S (serving):** the same inputs plus the stage 1 residuals and lag anomalies. A lag time at
  least 5 days before the origin takes ERA5T. A later one up to 00:00 UTC of the origin day takes
  the analysis proxy with its shortwave replaced by CAMS. A later one still takes the analysis
  proxy. This is the arm whose contrasts decide the verdict.
- **CH-S-noCAMS:** as CH-S, without the CAMS swap.
- **CH-P (proxy):** the analysis proxy at every lag time.
- **CH-E (ceiling, real delay):** final ERA5 at every lag time, with the 7-hour delay.
- **CH-S-delay3.5:** as CH-S with the 3.5-hour delay that experiment 00013 gave N1 and
  `manual_heuristic`, so C1 and C3 can be read at equal telemetry freshness. It is exploratory.
- **CH-E-nodelay (replicates 00005's L3):** final ERA5 and a 0.5-hour delay, as 00005 had.
- **CH-S-spread:** CH-S quantiles with the spread scaled about the median, `q50 + k (q - q50)`, by
  one factor `k` for the Primary substations (and one each for the bulk and grid supply points). `k` is chosen from the grid 1.00 to 1.60 in steps of 0.05 on
  held-out calibration rows: a calibrator fitted on the calibration runs before 2025-01-01 scored on
  the runs from 2025-01-01 to 2025-06-30 by quantile score. The scored arm uses the full-window
  calibrator and that `k`. No validation row influences `k`. N1 gets no equivalent tuning, so C6 is descriptive and the
  verdict never rests on it.
- **References from stored predictions:** N1 (00013), `manual_heuristic`, `climatology`, and the
  production recipe (00013), and R0c, the lag-free calibrated stage 1 (00005), each repeated once
  per seed where unseeded.

## Plan: planned contrasts

Quantile score over all leads and by lead band, on the 16 Primary substations, 99.17% intervals.

- **C1 (primary):** CH-S minus N1.
- **C2:** CH-S minus R0c (the lag-free calibrated stage 1; 00005 measured -1.80 with ERA5 and no
  delay).
- **C3:** CH-S minus `manual_heuristic`.
- **C4:** CH-S minus CH-E (what the serving sources cost against final ERA5 at the same delay).
- **C5:** CH-S minus CH-none (what lag anomalies add to raw lagged power).
- **C6:** CH-S-spread minus CH-S, on the quantile score; coverage is read beside it.

Exploratory (95%): CH-S minus CH-S-noCAMS, CH-S minus CH-P, CH-E minus CH-E-nodelay (the cost of the
real telemetry delay), CH-S minus the production recipe, and the same contrasts for the two bulk
supply points and the two grid supply points, which are descriptive.

## Plan: bias

Every arm in experiment 00013 under-forecast the validation year by 1.3 to 3.1 points. The page
reports the mean signed error of the median for every arm and lead band, and the mean absolute error
after subtracting each arm's per-series mean signed error (an oracle on the validation year, labelled
post hoc), so the drift is not mistaken for a difference between arms.

## Plan: design confounds

- **Training data differs by design.** Stage 1 trains on ERA5 from 2023-09, which predates the ENS
  archive that N1 trains on (from 2024-04). That difference is part of the staged design under
  test, not a defect.
- **Column choice.** Experiment 00006 chose stage 1's 13 columns on its own selection year
  (2024-07 to 2025-06), before any 2025-07 data was scored.
- **Residual scale.** Calibration residuals come from models at most 3 months past their fit, and
  validation residuals from a final model up to 12 months past its fit. The write-up reports the
  residual scale by months since fit.

## Plan: leakage and cleanliness

- Nothing is scored on or after 2026-07-01.
- Lag times, ERA5, CAMS, and proxy rows are all at or before the newest usable half-hour, which is at
  or before the origin minus 7 hours; a unit check asserts this on every case, and also each source's cutoff: ERA5T times at least
  5 days before the origin, CAMS times up to 00:00 UTC of the origin day, and proxy values from a
  run started by the lag time.
- Stage 1 models are never refitted. The calibrator trains on valid times up to 2025-06-30 23:30.
- The spread factor `k` uses calibration runs only.
- No calendar column enters the calibrator, as in 00005.

## Plan: ceiling labels and caveats

- CAMS irradiance enters an ERA5-trained stage 1 as shortwave; the two products differ in bias and
  scale, so the CH-S-noCAMS contrast shows whether the swap helps.
- The ENS archive starts on 2024-04-01, so calibration origins in the first 5 days have null
  proxy-window lags.
- ERA5T is emulated by final ERA5.

## Not tested here

Solar and wind series, a second fold, other weather models, and a recency-weighted stage 1.
