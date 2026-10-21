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

**Causal score.** RO-NC's step statistic compares the mean of the residual after a stamp with the
mean before it, so a score stamped at the onset has seen up to 24 hours of the future. The
deployable score is therefore re-indexed: the score at stamp `s` uses only residuals ending no
later than `s` minus 3.5 hours (the telemetry delay), the alarm is stamped when the "after" window
is full and not at the onset, and the peak, partner, and veto tests use the same cut. A unit test
truncates the residual stream at `s` minus 3.5 hours and requires the score at `s` to be unchanged.
Only S-HE and S-ORACLE may read onset-stamped scores.

**New filter core (a deliverable).** Experiment 134's filter is single-regime, so S-KF needs a
two-regime core in this folder: a per-stamp prior input, a level-variance multiplier for regime S,
a posterior update from the two regimes' likelihoods, and moment-matched mixing of state and
covariance. Gate: S-KF with the prior fixed at 0 reproduces K0 to numerical tolerance. The
regime-S multiplier grid is 4, 16, 64, and 256 times `q`, chosen on the tuning months.

**Known answer first (injected world).** Draw transfers of known size and timing in the real
residual stream using experiment 130's injection harness (headline band: moved load 25% or more of
the receiving load, lasting over 1 day; other bands descriptively), with injection seeds disjoint
from the tuning months. The frozen GAM-K forecast is not refitted, so adding the moved megawatts to
the stream (experiment 130's fast path) is valid for the filters. Scoring uses actuals plus the
signed moved megawatts at the donor and each receiver, never the original actuals, normalised by
the same effective capacity as the score. The spread is shared across arms (K0's), so every
contrast is a location-only difference. The `scorable` mask is recomputed on the injected world:
injected rows stay in, and logged events overlapping an injection window are dropped. Two gates:
an injected event of size zero reproduces K0 exactly, and check (a): S-KF's posterior probability of
S passes 0.5 within 24 hours of the causal alarm for at least 60% of detected headline-band events.

**Controls.**

- **Placebo.** S-KF with the causal score shifted by a random whole number of weeks within each
  series, which keeps the alarm rate, durations, and hour-of-day structure. Predicted gain: about
  zero.
- **Variance-inflation control.** K0 with regime S's multiplier applied at the same fraction of
  stamps but unrelated to events (the placebo prior, with no likelihood update), to separate
  knowing when from simply adapting faster.
- **False alarms.** Report the alarm rate per series per year, the detection-delay distribution,
  and alarms on Economy 7 winter mornings and evenings.
- **S-F** is fitted on the injected tuning world (disjoint seeds), because the real tuning months
  hold few events; the number of events is stated.

**Real year.** The same arms on the real scored year. The real year can support only "no harm"
(C3) and a descriptive gain near events (C5), because experiment 44 bounds the whole-year gain at
about 0.42%. Logged events are used only descriptively. The expected whole-year change is
computed from the injected-world gain and experiment 44's event frequency and compared with C3's
interval width before the verdict.

**Rows and scoring.** `fold_a_daily_gap_v1`; the five decision substations of experiment 134 plus
D9 beside them; leads 0.5 to 168 h with D+1 (15.5 to 39 h) beside, and the gain by lead band;
winter (November to March) and cold nights (daily mean under about 3 C) separately; tail
calibration, with q95 and q99 coverage and pinball loss on cold nights reported separately (q95
coverage was 73% to 78% in experiment 125); intervals resample whole calendar months; absolute
scores for K0, climatology, the manual heuristic, and (when ready) the best guess shown beside every
contrast. "Within 48 hours of an onset" means the target valid time. A pooled figure weights the six
sites equally. Labels for the sites follow experiment 134; no metered-generator series is used.

**Contrasts (family-wise level 5%, Bonferroni across C1, C2, C3, and C5).**

- C1: S-KF minus K0, injected world, target rows within 48 hours of a causal alarm on a
  headline-band transfer.
- C2: S-F minus K0, same rows.
- C3: S-KF minus K0, real scored year, all rows outside logged events (non-inferiority).
- C4: share of the K0-to-S-ORACLE gap closed by S-KF, injected world (descriptive).
- C5: S-KF minus K0, real scored year, rows within 48 hours of a causal alarm or of a logged event.
- Placebo minus K0 and variance-inflation control minus K0 are descriptive, and S-KF must beat both
  on C1 for `supported`.

**Decision rule.** The threshold is set by the cost of porting: a two-regime filter plus a detector
pipeline step is a new component. Before the verdict, the plan fixes the minimum detectable effect
as the half-width of C3's month-resampled interval.

- `supported`: C1's interval is wholly below zero, S-KF beats the placebo and the variance-inflation
  control on C1, C3's upper end is at most +0.005 points, and the q95 and q99 coverage on cold
  nights is within 2 points of K0's.
- `not supported`: C1's interval includes zero or lies above it, or S-KF fails to beat the placebo.
- `inconclusive`: otherwise, including a C3 interval wider than +0.005 points.

C1 is an easy test by construction (large injected steps). A `supported` verdict therefore means
the mechanism works in the lab, and the write-up says plainly that the real-year benefit is
bounded by experiment 44 and judged by C3 and C5.

**Stop rule.** If check (a) fails, or C1's point estimate is above zero on the tuning months, stop
before scoring the real year, record `not supported` or `inconclusive`, and write up what the
failure rules out. Compute cap: 6 cores at one thread each, 16 GB per job.

**Compute.** CPU only (the corrections are small filters on cached forecasts).
`systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue`, 6 workers,
`OMP_NUM_THREADS=1`.

**Lab shortcuts, named.** ERA5-driven residuals in the detector (as in experiment 130), which are
cleaner than live ENS-driven residuals and so inflate C1; hindsight end in S-HE; true megawatts in
S-ORACLE; fast-path injection.

**Data processing.** Reads `cleaned_power_time_series` with the `substation_zero` flags treated as
kept, as experiment 132 did.

## Reviews planned

A Sonnet review of the plan is triaged above (leakage of the score, the new filter core, scoring
of injected actuals, the vacuous verdict rule, and the missing controls were accepted). Sonnet then
reviews the code for leakage (the detector score at a half-hour may use
residuals ending no later than that half-hour minus the 3.5-hour telemetry delay), then the
write-up. One Opus review only with the maintainer's approval.
