---
id: "00141"
slug: switching-detector-fed-into-gam-k
title: "Switching detector fed into the GAM-K forecast"
theme: switching-labels-in-the-forecast
also_relevant_to:
  - switching-detectors
  - differentiable-physics-forecast-engine
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00141-switching-detector-fed-into-gam-k/
  only (alarm tables, corrections, prediction sets). Serving path: not touched. Degradation rule:
  not touched. Designs: two defensible ways to feed the detector (regime probability in the
  filter, or plain features), and this plan runs both. Callers: this folder's scripts only; they
  copy the modules of experiments 130 and 134 they need and read those experiments' stored outputs.
hypothesis: >-
  Telling GAM-K's Kalman filter when the RO-NC detector has seen a load transfer lowers the
  forecast's quantile score on days after a transfer begins, because the filter's level then
  adapts within hours instead of over days. The same signal given as plain features (event age and
  signed step size) does the same, more weakly.
prediction: >-
  Written before any fit. In the world with injected transfers (known size and timing), S-KF minus
  GAM-K without a detector (K0) is -0.05 to -0.20 points of quantile score on rows inside the
  first 48 hours of an injected headline-band transfer, and closes 10% to 40% of the gap between K0
  and the perfect-label arm (experiment 88 found 16% for RO-NC with a hindsight end). S-F closes 5%
  to 25%. On the real scored year, S-KF minus K0 is -0.010 to 0.000 points over all rows, because
  experiment 44 found that a perfect detector could lower the best guess's score by only 0.42%.
  False alarms cost at most +0.005 points over all rows.
decision_rule: >-
  See "Decision rule" in the plan below.
benchmark: >-
  K0 (GAM-K's Kalman filter without a detector, as stored by experiment 134), the hindsight-end
  ceiling arm (not deployable), and the perfect-label arm. Absolute skill beside them:
  climatology, the manual heuristic, and the best-guess model of experiment 136 where it is ready.
row_set_id: >-
  fold_a_daily_gap_v1 (292 runs, 09:00 UTC initialisation, 2025-07-02 to 2026-06-29), leads 0.5 to
  168 h, the five decision substations of experiment 134 plus D9
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of effective capacity;
  tail calibration (coverage and pinball loss at q01, q05, q95, q99) on cold nights.
code: ml-experiments/00141-switching-detector-fed-into-gam-k/
---

## Plan

**Question.** How much does the switching detector help when its output is fed into the forecast
itself? Earlier results measured the ceiling only: a perfect detector lowers the best guess's score
by 0.42% ([experiment 44](../00044-sw2-2-switching-share-of-best-guess-score/result.md)), and true
moved megawatts close 83% of the gap that the refit-offset neighbour-count detector (RO-NC) closes
16% of ([experiment 88](../00088-sw2-21-nra-forecast-recovers-normal-arrangement-during-and-after-injected-switches/result.md)).
RO-NC is the detector of [experiment 130](../00130-ro-nc-detector-on-the-gam-k-residual/result.md).
Lab conditions are allowed (maintainer, 2026-10-18); every shortcut is named in the write-up.

**Base forecast.** GAM-K member forecasts and the K0 filter as experiment 134 stores them (monthly
refits from experiment 111). The best-guess assembly of experiment 136 is not merged; if it is
ready before the real-year scoring, the same arms are re-run on it, labelled post hoc.

**Arms.**

- **K0** — GAM-K's filter with no detector input (control).
- **S-KF** — the filter becomes a two-regime mixture per hour band. Regime N ("normal running
  arrangement") uses the fitted level variance `q`; regime S ("switched") uses `q` times a
  multiplier chosen on the tuning months. The prior probability of S per half-hour is a logistic
  function of the causal RO-NC score, and the filter's likelihood updates it. The correction is
  the posterior-weighted mean of the two regimes' corrections.
- **S-F** — K0 plus a regression of the residual on two causal features: event age (hours since the
  causal onset, capped at 7 days) and signed step size. Fitted on the tuning months.
- **S-HE** — S-KF with the detector's hindsight end (the interval end from experiment 130's end
  rule). **NOT DEPLOYABLE.** It is the ceiling for the causal arms.
- **S-ORACLE** — S-KF whose prior is 1 inside the true transfer interval and 0 outside, from the
  injected truth (and from the logged events in the real year). **NOT DEPLOYABLE.** Upper bound.
- **Causal onset-only signal** is the deployable input. The harness takes the signal as a table of
  `(series, half-hour, score, signed step)`, so the causal live end rule of experiment 140 can be
  swapped in later without changing the arms.

**Known answer first (injected world).** Draw transfers of known size and timing in the real
residual stream using experiment 130's injection harness (headline band: moved load 25% or more
of the receiving load, lasting over 1 day; plus the other bands descriptively). The residual of
the frozen GAM-K forecast plus the moved megawatts is the fast path of experiment 130, valid
because the weather model is not refitted. The filter, the arms, and the scoring run on the
injected world, and the actuals are the injected power. Checks: (a) S-KF's posterior probability of
S passes 0.5 within 24 hours of the onset for at least 60% of detected headline-band events;
(b) the false-alarm bound below.

**False-alarm bound.** On quiet rows (no injected or logged event within 7 days), S-KF minus K0 is
at most +0.005 points over all rows of the real scored year, and at most +0.02 points on rows within
48 hours of a false alarm.

**Real year.** The same arms on the real scored year, with the logged switching events used only
descriptively (rows inside logged events are scored separately; rows outside are the headline, as
in experiment 134). Compare with experiments 44 and 88.

**Rows and scoring.** `fold_a_daily_gap_v1`; the five decision substations of experiment 134 plus
D9 beside them; leads 0.5 to 168 h with D+1 (15.5 to 39 h) beside; winter (November to March) and
cold nights (daily mean under about 3 C) separately; tail calibration with q95 coverage on cold
nights (73% to 78% in experiment 125); intervals resample whole calendar months; absolute scores
for K0, climatology, the manual heuristic, and (when ready) the best guess are shown beside every
contrast. Labels for the sites follow experiment 134; no metered-generator series is used.

**Contrasts (Bonferroni across the four).**

- C1: S-KF minus K0, injected world, rows within 48 hours of a headline-band onset.
- C2: S-F minus K0, same rows.
- C3: S-KF minus K0, real scored year, all rows outside logged events (the non-inferiority bound).
- C4: share of the K0-to-S-ORACLE gap closed by S-KF, injected world (descriptive, no verdict).

**Decision rule.** The threshold is set by the cost of porting: a two-regime filter plus a
detector pipeline step is a new component, so it must beat what a perfect detector could give on
the real year.

- `supported`: C1's 99% interval is wholly below zero and C3's upper end is at most +0.005 points.
- `small gain`: as `supported`, but C1's point estimate is under -0.05 points.
- `not supported`: C1's interval includes zero or lies above it.
- `refuted`: C3's lower end is above +0.005 points (the filter harms the real year).
- `inconclusive` otherwise. S-F, S-HE, and the real-event rows carry no verdict.

**Stop rule.** If the injected-world check (a) fails and C1's point estimate is above zero after the
tuning months, stop before scoring the real year, record `refuted` or `inconclusive`, and write up
what the failure rules out. Compute cap: 6 cores at one thread each, 16 GB per job.

**Compute.** CPU only (the corrections are small filters on cached forecasts; the GPU is not
needed). `systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue`, 6 workers,
`OMP_NUM_THREADS=1`.

**Lab shortcuts, named.** ERA5-driven residuals in the detector (as in experiment 130); hindsight
end in S-HE; true megawatts in S-ORACLE; fast-path injection.

**Data processing.** Reads `cleaned_power_time_series` with the `substation_zero` flags treated as
kept, as experiment 132 did.

## Reviews planned

Sonnet reviews the plan, then the code for leakage (the detector score at a half-hour may use
residuals ending no later than that half-hour minus the 3.5-hour telemetry delay), then the
write-up. One Opus review only with the maintainer's approval.
