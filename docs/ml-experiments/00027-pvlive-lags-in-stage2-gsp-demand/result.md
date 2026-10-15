---
id: "00027"
slug: pvlive-lags-in-stage2-gsp-demand
status: planned
size: medium
hypothesis: >-
  For the two grid supply points (GSPs) Walpole and Bicker Fen, giving the stage 2 calibrator
  Sheffield Solar's PV_Live estimate of the GSP's solar generation at the lag times, next to the
  lagged delta (metered demand minus the stage 1a expectation), lowers the quantile score below
  that of the same calibrator without PV_Live.
prediction: >-
  The PV_Live-adjusted delta (delta plus PV_Live, which separates solar from other demand) lowers
  the GSP quantile score over all leads by 0.5 points of capacity (plausible range 0 to 1.5),
  mostly at midday and in summer, because the delta of a lag time mixes solar error with demand
  error and the adjusted delta removes the solar part. The PV_Live level and the share of capacity
  help less (0.2 points each), because the calibrator must learn the subtraction itself from 2
  series. The interval will be wide, because 2 series carry the result.
decision_rule: >-
  Three planned contrasts on the GSP quantile score over all leads (arm minus A0, the calibrator
  without PV_Live), read at 98.33% (a 5% level with a Bonferroni correction across three, two
  sided), resampling whole validation months and the three seeds: C1 PV_Live level, C2 PV_Live share
  of capacity, C3 PV_Live-adjusted delta. The verdict follows the arm with the lowest estimate.
  Supported when its interval is wholly below zero and its estimate is at most -0.30 points,
  because the change adds a data source and a serving step that reads a second published series. A
  small gain when the interval is wholly below zero and the estimate is above -0.30, refuted when
  the interval's lower bound is above -0.30, and inconclusive otherwise.
  The stage 1 variant S1 and every other contrast are exploratory at 95%.
benchmark: >-
  A0, the L3 calibrator of experiment 00005 refitted here for the two GSPs (checked against the
  stored `l3_from2023` arm), and the day-type climatology (`r2`) of experiment 00005.
row_set_id: fold_a_every5d_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over 9 levels), in points of capacity,
  and mean absolute error of the median
---

# Does PV_Live at the lag times help the stage 2 forecast of GSP demand?

## Plan, written before any code ran

### Plan: sizing

- **Stored data:** prediction sets only, under `data/ml-experiments/`; no production table changes.
- **Production serving path:** not touched.
- **Degradation rule:** not touched. A missing PV_Live value is a missing input of the calibrator.
- **Defensible designs:** more than one (which PV_Live transform to give the calibrator). The
  transforms are the arms, fixed below.
- **Callers:** the scripts reuse experiment 00005's calibrator, which the author could name without
  searching.

Size: medium. One Sonnet review of the plan and one of the code.

### Plan: arms

Series: Walpole (series D6) and Bicker Fen (series D1), the two GSPs of the 20 demand series, matched
to PV_Live's `WALP_B` and `BICF_1`. The calibrator is experiment 00005's `l3` arm for the primary
training window (from 2023-09): stage 2 XGBoost models, one per series and seed, on the stage 1a
member statistics, the lead, and the recent-origin inputs, with 00005's hyperparameters. The stage
1a statistics, the out-of-fold ERA5 predictions, and the observed-power lags are read from experiment
00005's stored work folder, so no stage 1a is refitted for the calibrator arms.

- **A0:** the `l3` inputs, no PV_Live.
- **A1, PV_Live level:** `l3` plus PV_Live's `generation_mw` divided by the series' effective capacity
  at six positions: the lags of 24, 48, 168, and 336 hours, and the same time of day and the same
  time of week as the target, at the newest usable day and week.
- **A2, PV_Live share:** `l3` plus `generation_percent_of_capacity` divided by 100 at the same six
  positions. The share is of PV_Live's own installed capacity, so it stays comparable as capacity
  grows.
- **A3, PV_Live-adjusted delta:** `l3` plus, at the same six positions, the delta of that position
  (the observed power minus the stage 1a prediction at ERA5 weather, over capacity) plus A1's
  PV_Live level. The sum separates solar from other demand at the lag time.
- **S1, stage 1 variant (exploratory, no verdict):** the final stage 1a model of each GSP and seed
  refitted with PV_Live's level at the valid time as one more input, against the same model without
  it. Both are scored by the mean absolute error of the point prediction at observed ERA5 weather
  over the validation year (July 2025 to June 2026). The observed PV_Live value at the valid time is
  not known at the forecast origin, so S1 says only whether stage 1a understands the GSP's solar
  better, and not what a forecast gains.

**Known at the forecast origin.** A lag position counts only when it is a past half-hour that
the PV_Live estimate had time to settle. PV_Live revises each half-hour's estimate as more
data arrive, and the download keeps only the latest estimate, so the experiment uses a conservative
delay: a PV_Live value is usable at origin `T` only when its half-hour ended at least 1 day before
`T`. Power and the stage 1a delta at the lag positions keep experiment 00005's rule (before `T`), so
A1 to A3 use PV_Live values 1 day older than the newest delta. The script asserts the bound.

**Past weather.** The delta uses stage 1a's prediction at ERA5 weather at the lag times, as in
experiment 00005. ERA5 arrives about 5 days late, so every arm is a ceiling for the delta; PV_Live is
a measurement and has no such gap. The comparison across arms is unaffected, because every arm
shares the same delta.

**Rows and seeds.** The 73 runs of `fold_a_every5d_v1`, valid times before 2026-07-01, 3 seeds
that A0 to A3 share. Calibration runs are experiment 00005's (every third day from 2024-04-01 to
2025-06-30), and PV_Live is not used after the calibration cut-off.

**Resolution.** Two series give 2 independent clusters. Intervals resample months and seeds, not
series, so a result describes these two GSPs and may not carry to other GSPs. The interval will
resolve gains of roughly 1 point of capacity, not 0.3.

### Plan: references

Day-type climatology (`r2`) and A0, both from experiment 00005 or refitted identically.

Attribution: Sheffield Solar PV_Live, CC BY 4.0.
