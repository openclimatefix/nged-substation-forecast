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
  Primary weekly minus the stack is -0.10 to -0.30 (experiment 34's 73 runs gave 5.04 against 5.32,
  a descriptive -0.28 against a three-seed stack; the daily runs let the stack's correction read
  forecasts from the last day, not 5 days back, which probably improves the stack and shrinks the
  gap), and monthly is 0.05 to 0.10 behind weekly. A 7-day or 14-day
  correction on the weekly arm adds +0.05 to +0.30, the 7-day window less than the 14-day window.
  For the 2 BSP and 2 GSP series, weekly minus frozen is -0.1 to -0.5, with intervals wide on 2
  series each. For solar and wind, monthly minus frozen lies within plus or minus 0.05, because the
  weather-to-power relation is physical and experiment 12 found the training start changed little.
decision_rule: >-
  Nine planned contrasts, quantile score over all leads on the 292 new runs, read at 99.44%
  (Bonferroni across nine), whole validation months resampled, each run counted in its origin
  month. Seed 0 only for every refit arm,
  because experiment 34 found three seeds narrowed its interval by under 0.01. Primary: C1 weekly
  (no correction, spread factor) minus the stack (frozen N1, 14-day correction, spread factor);
  C2 monthly minus the stack; C3 weekly minus monthly; C4 weekly with the 7-day correction minus
  weekly; C5 weekly with the 14-day correction minus weekly; C6 weekly with the 7-day correction
  minus weekly with the 14-day correction. BSP and GSP: C7 weekly (no correction, spread factor) minus
  frozen with the 14-day correction and spread factor, the deployable reference, pooling the 4
  series. Solar and wind: C8s monthly minus frozen for the 6 solar farms, and C8w the same for the 3
  wind farms, each pooled over its farms. C1 is
  supported when its interval lies wholly below zero, refuted when wholly above zero, and
  inconclusive otherwise. The threshold is zero because retraining replaces the correction step in
  the pipeline and adds a scheduled job, and experiment 34 already read monthly retraining as
  worth porting at any gain. C3 is read like C1, supported when its interval lies wholly below zero and
  refuted when wholly above, because 16 refits a week cost seconds of compute in production; it is
  expected to resolve only if the gain is near the upper end of the prediction.
  The correction is judged still helping when C4 or C5 lies wholly below zero and
  no longer helping when it lies wholly above zero. C8s and C8w are read as no change when the
  interval lies inside plus or minus 0.05, and as a change when it excludes zero; with 6 solar and 3
  wind farms the wind interval will probably be too wide to read as no change, and the page says so.
  Every series is also reported on its own, because the pooled interval resamples months only and
  says nothing about other series. Every other comparison, including P-roll, is descriptive. The 73
  runs of experiment 34 are scored again from the same refits as a reproduction check: the arms
  without a correction must match experiment 34's stored seed-0 scores to 0.001.
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
guards against the choice of origins and seeds, not against an unusual year. Experiment 34 also left
three gaps: its correction read forecasts 5 days apart, retraining was not tried for BSP, GSP, solar,
and wind beyond monthly BSP and GSP, and the 7-day correction was not tried on a retrained model.

**Rows.** The row set `fold_a_daily_gap_v1` holds the daily 00:00 UTC ENS runs from 2025-07-01 to
2026-06-30 that `fold_a_every5d_v1` does not. All 365 daily runs exist in the ENS archive, so the set
holds 292 runs, and the 73 of `fold_a_every5d_v1` are forecast again from the same refits as a
reproduction check. Each run spans about 14.6 days, so the 73 runs already cover every valid time of
the year: the 292 runs score the same observed demand at other origins and leads, and guard against
the choice of origins, not against the outcome data. The demand frames use the same builder as `studies/stage1_pilot/demand_build_ens.py`
(production's half-hourly upsampling, all 51 members, lags from the observed power). No experiment
scores a valid time on or after 2026-07-01. The generators need frames for their 9 series over all
365 runs and 51 members, built the same way for the generators' H3 cells and the technology's
columns (experiment 33 holds only the 53 Tuesday runs).

**Arms.**

- **Primary (16 series):** the stack; weekly and monthly full refits with a half-life of 0.5 years
  and an expanding window, no correction, spread factor 1.15, using experiment 34's `n1h`
  recipe and its refit times (52 weekly, 12 monthly); P-roll (12-month window, 150 rounds, `eta`
  0.15, monthly); and the weekly arm with the 7-day and the 14-day correction, built from the 365
  daily forecasts as issued: each past half-hour's forecast comes from the refit that served that
  run, never from a re-forecast by the current refit, and no correction reads a run before
  2025-07-01. The stack's frozen model is refit with experiment 33's recipe (half-life 2 years,
  anchored at 2025-06-30, 500 rounds) because it was stored only for the 73 runs, and forecasts all
  365 runs.
- **BSP and GSP (2 each):** frozen (with the correction as the deployable reference, and without),
  monthly, and weekly, with and without the correction.
- **Solar (6) and wind (3):** the ERA5-trained stage 1a (solar `base`, with no CAMS and no stage 0;
  wind `margin`, with the power-curve margin), refitted monthly on ERA5 up to 5 days before each
  refit date (when ERA5 arrives), against the frozen model trained to 2025-06-17, both applied to the
  ENS members and pooled as in experiment 33. Both arms take the spread factor the same way as
  experiment 33's `apply_spread`, fitted only on cases whose valid time precedes the origin. The
  prediction of no change rests on experiment 12's training-start result, not on a retraining
  measurement.
- **Baselines:** `manual_heuristic` and `climatology` on the same 292 runs.

**Resources.** About 5 to 7 GPU hours inside slot A (10 cores, at most 24 GB): weekly Primaries
2.4, monthly 0.6, BSP and GSP weekly 0.6, generators 1 to 2, P-roll 0.2.
The 292 new demand frames add about 8 GB of disk, and the generator frames some more; each
worker loads the runs of one refit at a time, so memory stays inside 24 GB, and the fits run as
parallel workers, one output file per refit written atomically, so a run resumes. The GPU is shared with
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

**Plan review (Opus), triaged.** Accepted: state that the correction reads forecasts as issued and
nothing before 2025-07-01; predict a smaller Primary gap because the daily stack's correction is
fresher; set C3's threshold to zero; split C8 into solar and wind and make "no change" conditional on
the interval; name which frozen reference C7 uses; report each series; limit the reproduction check
to arms without a correction; add the generator frame builder, the disk and memory budget, and the
spread-factor rule. Rejected: none.
