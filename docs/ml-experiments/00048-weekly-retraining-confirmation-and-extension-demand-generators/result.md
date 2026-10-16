---
id: "00048"
slug: weekly-retraining-confirmation-and-extension-demand-generators
status: planned
size: complex
hypothesis: >-
  On the 292 daily ENS forecast runs of the validation year that experiment 34's 73 runs leave out,
  weekly retraining of the Primary substation demand model N1, with no error correction, lowers the
  quantile score below the best-guess stack (frozen N1 with the 14-day correction and spread factor
  1.15) by more than zero points of capacity; monthly retraining of the ERA5-trained solar and wind
  stage 1a models changes the score by less than 0.05 points; and, once the model retrains weekly,
  neither a 7-day nor a 14-day error correction helps.
prediction: >-
  Primary weekly minus the stack is -0.20 to -0.30 (experiment 34's 73 runs gave 5.04
  against 5.32, which is -0.28), and monthly is 0.05 to 0.10 behind weekly. A 7-day or 14-day
  correction on the weekly arm adds +0.05 to +0.30, the 7-day window less than the 14-day window.
  For the 2 BSP and 2 GSP series, weekly minus frozen is -0.1 to -0.5, with intervals wide on 2
  series each. For solar and wind, monthly minus frozen lies within plus or minus 0.05, because the
  weather-to-power relation is physical and experiment 12 found the training start changed little.
decision_rule: >-
  Eight planned contrasts, quantile score over all leads on the 292 new runs, read at 99.375%
  (Bonferroni across eight), whole validation months resampled. Seed 0 only for every refit arm,
  because experiment 34 found three seeds narrowed its interval by under 0.01. Primary: C1 weekly
  (no correction, spread factor) minus the stack (frozen N1, 14-day correction, spread factor);
  C2 monthly minus the stack; C3 weekly minus monthly; C4 weekly with the 7-day correction minus
  weekly; C5 weekly with the 14-day correction minus weekly; C6 weekly with the 7-day correction
  minus weekly with the 14-day correction. BSP and GSP: C7 weekly minus frozen, pooling the 4
  series. Solar and wind: C8 monthly minus frozen, pooling the 6 solar and 3 wind farms. C1 is
  supported when its interval lies wholly below zero, refuted when wholly above zero, and
  inconclusive otherwise. The threshold is zero because retraining replaces the correction step in
  the pipeline and adds a scheduled job, and experiment 34 already read monthly retraining as
  worth porting at any gain. C3 earns weekly over monthly only when its interval lies wholly below
  -0.05, because weekly refits are 4.3 times as many fits and the service would run one refit per
  series per week. The correction is judged still helping when C4 or C5 lies wholly below zero and
  no longer helping when it lies wholly above zero. C8 is read as no change when its interval lies
  inside plus or minus 0.05. Every other comparison, including P-roll and the 73-run
  reproduction, is descriptive. Scores on the 73 runs of experiment 34 are recomputed from the same
  refits as a reproduction check and must match the stored scores to 0.001.
benchmark: >-
  The best-guess stack (experiment 33's `n1_cs`), frozen N1 for BSP, GSP, solar, and wind,
  `manual_heuristic`, and `climatology`, each scored on the same 292 runs.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over 9 levels), in points of capacity;
  MAE of the median, the exceedance rate of the 0.95 quantile, and the twCRPS at `historical_p99`
  are reported and decide nothing
code: ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators/
---

# Experiment 48: does weekly retraining still win on the runs experiment 34 left out?

## Plan

**Experiment 34 scored 73 forecast runs, 5 days apart, and found weekly retraining 0.28 below the
best-guess stack with no correction; this experiment asks whether the gain holds on the other 292
daily runs of the same year.** The 292 runs share their months and weather with the 73, so the test
guards against the choice of runs and seeds, not against an unusual year. Experiment 34 also left
three gaps: its correction read forecasts 5 days apart, retraining was not tried for BSP, GSP, solar,
and wind beyond monthly BSP and GSP, and the 7-day correction was not tried on a retrained model.

**Rows.** The row set `fold_a_daily_gap_v1` holds the daily 00:00 UTC ENS runs from 2025-07-01 to
2026-06-30 that `fold_a_every5d_v1` does not. All 365 daily runs exist in the ENS archive, so the set
holds 292 runs, and the 73 of `fold_a_every5d_v1` are forecast again from the same refits as a
reproduction check. The frames use the same builder as `studies/stage1_pilot/demand_build_ens.py`
(production's half-hourly upsampling, all 51 members, lags from the observed power). No experiment
scores a valid time on or after 2026-07-01.

**Arms.**

- **Primary (16 series):** the stack; weekly and monthly full refits with a half-life of 0.5 years
  and an expanding window, no correction, spread factor 1.15, using experiment 34's `n1h`
  recipe and its refit times (52 weekly, 12 monthly); P-roll (12-month window, 150 rounds, `eta`
  0.15, monthly); and the weekly arm with the 7-day and the 14-day correction, built from the 365
  daily forecasts of the weekly refits. The frozen model for the stack is refit with experiment 33's
  recipe, because it was stored only for the 73 runs.
- **BSP and GSP (2 each):** frozen, monthly, and weekly, with and without the correction.
- **Solar (6) and wind (3):** the ERA5-trained stage 1a, refitted monthly on ERA5 up to 5 days
  before each refit date (the date ERA5 arrives), against the frozen model trained to 2025-06-17,
  both applied to the ENS members and pooled as in experiment 33. Wind keeps the power-curve margin.
- **Baselines:** `manual_heuristic` and `climatology` on the same 292 runs.

**Resources.** About 5 to 7 GPU hours inside slot A (10 cores, at most 24 GB): weekly Primaries
2.4, monthly 0.6, BSP and GSP weekly 0.6, generators 1 to 2, P-roll 0.2. Fits run as parallel
workers, one output file per refit written atomically, so a run resumes. The GPU is shared with
other sessions, so the timings are not reproducible; the numbers are, because seeds are fixed per job.

**Sizing, by the five triggers.** Stored: predictions under `data/`, private, nothing in production.
Serving path: not touched. Degradation rule: not touched. More than one defensible design: no, the
approved ranking fixes arms and refit dates. Callers not nameable without searching: no, the scripts
are copies of experiment 33's and 34's. The experiment is complex only because it builds new
frames and a new solar and wind refit, so it gets the full review set: one plan review (Opus), one
code review (Sonnet), one write-up check (Sonnet), one prose review (Sonnet), and one result review
(Opus).

**Earlier advice followed.** Retrain monthly in the Primary stage 1 model and drop the correction
(00034); screen on P-roll (00034, used here only as a descriptive arm); read with the 99.375%
interval, because months set the width (00034). Departures: none.
