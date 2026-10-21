---
id: "00127"
slug: differentiable-physics-wu-weight-update-schemes-for-the-gam-weights
title: "Differentiable physics, WU: which way of updating the convex model's weights forecasts best"
part: "WU"
theme: differentiable-physics-forecast-engine
also_relevant_to: [differentiable-physics-long-history-and-drift, lagged-power-and-error-correction]
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00127-.../ only. Serving path: not
  touched. Degradation rule: not touched. More than one design is defensible (window length,
  how the block-fit noise enters the filter), and the plan pre-registers one of each. Callers:
  this experiment's scripts only; the scripts reuse the KFE harness and the module
  studies.dp_effects_filter, and need no new module unless the two-stage filter's measurement
  update is added to the module (open question 1).
hypothesis: >-
  Denoising block convex fits of the weights (28 to 56 days) with a Kalman filter whose
  measurement noise is the block fit's own coefficient covariance forecasts day-ahead demand on
  ENS better than frozen weights, better than a rolling convex refit with no filter, and better
  than the 14-weight filter on the residuals (KEZ, experiment 118), because a block fit
  identifies PV, heating, and profile weights that a day of residuals cannot, and the filter
  removes the block fit's sampling noise.
prediction: see "Predictions"
decision_rule: see "Decision rule"
benchmark: >-
  GAM-K as stored by experiment 111 (level and anomaly filter, 2.521 points at D+1 over the five
  decision substations in the scored year, as reproduced by experiment 118); frozen weights (arm
  W0); KEZ (experiment 118); climatology; manual_heuristic
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles) in points of
  effective capacity, on rows outside the hand-filtered switching events, at day D+1 (leads 15.5
  to 39 hours from the 09:00 UTC power_fcst_init_time); every lead band beside it
verdict: <supported | small gain | partly supported | not supported | refuted | inconclusive>
production_status: not-for-production
best_arm: <after the results>
headline: <after the results>
lessons: <after the results>
adopted_in: none
take_home_message: <after the results>
date: <date>
pr: <pr>
advice: <after the results>
code: ml-experiments/00127-differentiable-physics-wu-weight-update-schemes-for-the-gam-weights/
---

# Differentiable physics, WU: which way of updating the convex model's weights forecasts best

**This plan starts only when the KFE gate diagnosis (the block-fit yardstick step in
`briefs/dp_kfe_gate_diagnosis_run.md`) shows that block convex fits are a usable yardstick for the
weights.** If block weights vary as much from block to block as the filtered weights do, the
two-stage arms (W3) have nothing to denoise, and this experiment is not run; the diagnosis's finding
is then the answer. The manager checks this before assigning a slot. The plan was written by Sonnet
from the briefs and write-ups named below and awaits one Sonnet plan review; no code has run.

## What WU asks, and the evidence it starts from

**The convex model's weights (level, daily profile, heating, solar-panel capacity, holidays) can be
frozen, refitted on a window, or tracked by a filter, and nobody has compared the choices on one
set of rows.** Experiments 106, 107, 111, and 118 each tested one choice against a different
benchmark. WU compares them on the same rows, with the same tuning, replicate, and scored months and
the same five decision substations as experiment 118 (KFE).

**What the earlier experiments found, which this plan respects:**

- [Experiment 107](../00107-differentiable-physics-rolling-origin-history-and-tails/result.md):
  giving the model a drift term lowered the score at 10 of 12 quarterly refits (by 0.45 points), but
  about three-fifths of that size is one substation (D10) at the 2023-24 refits.
- [Experiment 106](../00106-differentiable-physics-4-all-years-drift-and-recent-level/result.md)
  (DP Part 4): once the Kalman filter on recent residuals runs, a drift term adds nothing.
- [Experiment 111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md): the
  level-and-anomaly filter (GAM-K) lowered the score by 0.151 points, mostly under 24 hours.
- Experiment 118 (KFE, `pr: 1409`): the 14-weight filter on the residuals (KEZ) scored 0.015 points
  worse than GAM-K at D+1 (interval -0.013 to +0.048; 1 of 5 substations better), and worse on cold
  winter nights at 4 of 5 substations. The known-answer gate failed on resampled real residuals, and
  the strict bar on the recovery of the noise variance (control P4) was missed because the likelihood
  is nearly flat (fitted PV variance scattered about 10 times either way). The diagnosis found no
  coding fault (`briefs/dp_kfe_p4_diagnose_result.md`).

**The method idea comes from the literature.** [de Vilmarest et al.
(2024)](https://doi.org/10.1109/TPWRS.2023.3310280) froze an additive model's effects, put a random
walk on one weight per effect, and fitted the noise by likelihood; [de Vilmarest and Goude
(2022)](https://doi.org/10.48550/arXiv.2110.00334) compared the noise choices; the R package
[viking](https://CRAN.R-project.org/package=viking) supplies the likelihood search. Their daily
refit and static-weight comparators correspond to arms W1 and W0 here. The two-stage arm W3 (a
Kalman filter whose measurement is a regression estimate and whose measurement noise is that
estimate's covariance) is the standard way of filtering a repeated regression estimate; we have not
checked whether the energy-forecasting literature has applied it to additive-model weights, and the
plan claims no novelty.

**A one-day convex fit cannot identify the PV, heating, and profile weights** (one day holds one
solar curve and one temperature), so there is no single-day arm. A daily refit is only affordable on
a window of 28 days or more, refitted each day (arm W1-d).

## Arms

All arms start from experiment 111's monthly frozen-effect decomposition of the convex model
(`studies.dp_effects_filter`: level per hour band, profile scale per day type, heating per hour
band, PV capacity, holiday), so each arm changes only the weights. All arms keep GAM-K's level and
anomaly filter underneath unless the table says otherwise, so every contrast measures the weights'
update beyond the filter that production-candidate already has.

| Arm | Weights | Update | Role |
|---|---|---|---|
| W0 | frozen at the monthly fit | none (GAM-K as stored) | benchmark, reproduced |
| W1-w28, W1-w56 | block convex fit | refit weekly on the last 28 or 56 days of residuals, no filter | scheme 1, windows |
| W1-d28 | block convex fit | refit daily on the last 28 days (run only if the timing check in "Compute" allows) | scheme 1, cadence |
| W2 | 14 weights | KEZ, experiment 118's filter, reproduced | scheme 2 |
| W3-28, W3-56 | 14 weights | block convex fit every 7 days on 28 or 56 days, then a Kalman filter on the block estimates with measurement noise equal to the fit's coefficient covariance | scheme 3 |
| W3-q | as W3, the better window | W3 plus a learned extra process noise per weight group, fitted on the tuning months | scheme 3 variant |

**Block convex fit.** The fit is a ridge-regularised non-negative least squares of the frozen
group contributions c_k(t) on the residuals of W0 over the window, with the same sign limits as
experiment 118 (heating, PV, profile weights floored at 0), a ridge towards a weight of 1 whose
strength is fitted on the tuning months, and the 3.5-hour data delay of experiment 118. The
covariance of the weights is σ²(CᵀC + λI)⁻¹ from the Gram matrix C, with σ² estimated from the
window's residuals and inflated by experiment 113's variance-inflation factor of 27 for
autocorrelated residuals; the inflation is reported as an assumption.

**W3's filter.** Each weight follows a random walk. Every 7 days the block estimate is a
measurement of the current weight with covariance equal to the block fit's. Overlapping blocks make
successive measurements correlated, so the filter uses only the non-overlapping part: with a 56-day
window refitted every 7 days, the measurement covariance is multiplied by 56/7 = 8, which is stated
as an approximation and tested in the gate. The base random-walk variance per weight comes from
the same marginal-likelihood fit as experiment 118's, using a day-ahead likelihood (a lesson of
experiment 118: the one-step likelihood was 0.127 points worse).

**W3-q** adds a learned variance per weight group to the random walk, fitted on the tuning months.
Groups with under 200 active half-hours in the tuning months take the shared prior mean, as in
experiment 118.

**Spread.** As in experiment 118 (`spread_with_factors`, spread factor per lead band from the same
grid, fitted on the tuning months), so arms differ only in the median path.

## Years, substations, and rows

- **Tuning:** the odd months of 2024-07 to 2025-06 (ridge strength, spread factors, noise).
  **Replicate:** the even months of the same year, opened once. **Scored:** the 292 runs,
  2025-07-02 to 2026-06-29. Same cuts as experiment 118; no row on or after 2026-07-01.
- **Substations:** the decision substations D5, D10, D11, D14, and D16 (4 of 5 needed), with D9
  beside them. D15 and D18 appear only as G-c with no dates; generator series are otherwise
  excluded.
- **Rows:** present rows outside the hand-filtered switching events, shared across a contrast's
  arms, with the all-rows score beside. Lead bands as in experiment 118: D+1 (headline), 0-24,
  24-48, 48-144, and 144-168 hours.

## Known-answer gate (blocking: a failure caps the verdict at `inconclusive`)

**The gate reuses experiment 118's gate harness (`e00118_gate.py`, `e00118_streams.py`,
`e00118_noise.py`) and its scenarios, and runs on every arm with time-varying weights before any
replicate or scored row is opened.** It must be able to fail: the bars are fixed now and the gate
has failed in experiment 118.

- **Streams:** the real regressors c_k(t) of the five decision substations over the tuning months'
  calendar, with injected weight changes: S0 no change; S1 PV weight rising by 30% over 180 days;
  S2 heating weights falling by 30% over 90 days; S3 both.
- **Noise, two forms:** (a) the filter's own AR(1) noise at each substation's fitted φ and σ; (b)
  a circular month-block resample of the substation's real tuning residuals. 20 draws per
  substation, scenario, and form.
- **Bars, per arm with moving weights, at least 80% of draws:**
    1. PV tracking (S1, S3): the arm's PV weight at day 200 recovers at least 70% of the injected
       change.
    2. Heating tracking (S2, S3): at least 50% by day 120. Not blocking; a failure labels the
       heating weight "not interpretable".
    3. No false movement (S0): no weight deviates from 1 by 0.10 or more after day 60 (PV,
       heating, holiday).
    4. No cross-talk (S1, S2): the other group's weight moves under 25% of the injected change.
    5. Forecast skill follows (S1 to S3): the arm's mean absolute error on the stream is below
       W0's.
- **Comparison with the yardstick:** the same table for the unfiltered block fits (W1), so the
  write-up can say whether denoising adds recovery or only lag. This is the block-fit evidence the
  start condition names.
- **Power check first:** before the gate, the code prints each arm's posterior sd of the PV
  weight at day 200 and the heating weight at day 120. If the injected change is under 3 sds the
  failure stands as "underpowered at the real amplitude", and a doubled-amplitude row is added.
- Gate failures cap the verdict at `inconclusive` for the failing arm; the other arms keep their
  labels.

**Plumbing controls (run first):** P1, W2 reproduces experiment 118's KEZ to 1e-6 and W0 reproduces
GAM-K's 2.521 at D+1; P2, a block fit of noise-free synthetic data with known weights returns them
to 1e-6; P3, causality: changing any reading after the cutoff, or any ENS run issued after it,
leaves every arm's forecast unchanged; P4, the block fit's covariance matches a bootstrap over days
within 30% on simulated data.

## Planned contrasts and margins

**Six planned contrasts, Bonferroni-corrected 99% intervals from `studies.bootstrap` (resampling
whole calendar months, paired across arms, 3 seeds),** with substation counts, every arm's
absolute score, the error of the median beside each, and both periods.

- **C1 (decision):** best W3 arm minus W0 at D+1.
- **C2:** best W3 arm minus W2 (KEZ): does denoising block fits beat filtering residuals.
- **C3:** best W3 arm minus best W1 arm: does the filter add anything to the block fit.
- **C4:** best W1 arm minus W0: does a plain rolling refit help.
- **C5:** W3-q minus the W3 arm it extends: does learned extra process noise help.
- **C6:** W2 minus W0, reproducing experiment 118 (+0.015) as a check.

**The smallest gain worth porting is 0.05 points at D+1** (about 2% of GAM-K's score). Any W3 arm
adds a weekly block fit, a covariance, and a filter to the serving path; the threshold is a
judgement about that cost, not a detection limit. Noise reference: experiment 118's C1 interval
was +-0.03 wide.

**Exploratory at 95%:** each W1 window against W0; daily against weekly refit (if W1-d runs);
leaving one substation out; November to March against April to October.

## Scoring that must be reported

- **Absolute skill beside every contrast:** each arm's score at D+1 and over 0.5 to 168 hours on
  shared rows, beside climatology and manual_heuristic.
- **Winter breakdown (November to March):** the cold-snap rule is fixed before any score is read:
  a cold snap is a run of at least 3 days whose daily mean ENS-control temperature at the substation
  is in the coldest 10% of tuning-month days. Evening peaks (16:00 to 20:00 local) are scored
  apart. **Independent winters are counted and stated:** the scored year holds one winter (2025-26)
  and the replicate months half of another, so every winter claim is "one winter, plus half of
  one".
- **Christmas and event days:** scored apart, with non-inferiority elsewhere at a margin of 0.02
  points (an arm that helps on event days must not lose more than 0.02 elsewhere). The
  holiday-day result of experiment 118 (zeroing the correction is much worse) is the prior.
- **Operator-visible cold-winter-night figure:** 22:00 to 08:00 local, the median over nights of
  the absolute error in points, cold nights defined by the rule above, per arm and substation.
- **Example weeks fixed before scores:** the weeks beginning 2025-07-14, 2025-10-13, 2026-01-12,
  and 2026-04-13, as in experiment 118, D+1 forecasts chained from each 09:00 UTC origin.
- **Tails:** coverage and pinball at q01, q05, q95, q99, every arm, with threshold-weighted CRPS
  (`studies.tail_calibration`).

## Weight plots (user rule)

**Every arm with time-varying weights gets a plot of its weights over time, per decision
substation:** one panel per weight group (level by hour band, profile scale by day type, heating,
PV capacity, holiday), the weight as a line with its band (the posterior sd for the filter arms,
the block fit's standard error for W1), the block-fit weights as points beside the filtered
weights in W3, bank holidays marked with vertical ticks and cold-snap spells shaded, with
D-labels only. A cross-arm figure per substation overlays W1, W2, W3-best on the same axes. The
write-up says whether the weights move by more than their bands.

## Predictions

Written before any fit; each can fail, and none overrides the decision rule.

- **Gate:** W3 passes readings 1 and 3 to 5 on the AR(1) form; on resampled real residuals it
  passes reading 5 and perhaps 1 (block fits use many days, so they should tolerate seasonal
  structure better than a day-by-day filter), with heating tracking the likeliest failure. W1
  passes reading 1 on block-resampled noise at least as often as W2 does.
- **C1:** best W3 minus W0 is -0.02 to -0.08 at D+1, with gains at 2 to 4 of 5 substations, below
  the 0.05 threshold at most substations.
- **C2:** W3 minus W2 is -0.02 to -0.06 (W3 removes the day-by-day noise that hurt KEZ).
- **C3:** W3 minus the best W1 arm is -0.02 to -0.10 (the filter removes sampling noise).
- **C4:** the best W1 arm minus W0 is within +-0.05, with 56 days better than 28.
- **C5:** within +-0.02.
- **C6:** +0.015 +- 0.03 (reproduction).
- **Cold winter nights:** no arm beats W0 by more than 0.1 points at 4 of 5 substations.
- **Weights:** PV weight within +-0.15 of 1 at 4 of 5 substations; heating weight higher in
  winter than the shoulder months; W1 weights vary from block to block by more than W3's.

## Decision rule

**The verdict is C1's label, with C2 and C3 deciding between `supported` and `partly supported`.**
The scored year is the main test and the replicate months a second.

- **`supported`:** the gate passed for the best W3 arm; C1 below zero in both periods with at least
  4 of 5 substations better in each; the scored-year interval wholly below zero with an estimate of
  at least 0.05 points; and C2 below zero in the scored year; and the non-inferiority margin on
  event days and elsewhere holds.
- **`small gain`:** the scored-year C1 interval wholly below zero with an estimate under 0.05, and
  the replicate estimate below zero.
- **`partly supported`:** C1's estimate is below zero in both periods but the interval includes
  zero or fewer than 4 substations improve; or C1 meets `supported` but C2 does not (the gain is the
  block fit and not the denoising).
- **`refuted`:** C1's interval lies wholly above zero in the scored year or the replicate months.
- **`not supported`:** both periods' C1 intervals lie within +-0.05 of zero.
- **`inconclusive`:** anything else, or a failed plumbing control or gate reading that cannot be
  fixed.

**Post hoc labelling and first run.** Anything not listed here (for example, an arm added after
seeing scores) is labelled post hoc; the first run's numbers are kept in the write-up even if a
rerun is made.

## Audit before merge: the seven triggers

"Checks" bullets: median minus mean forecast per substation and arm; mean error beside each
contrast.

1. **A control fails:** P1 to P4, any gate reading, W0's reproduction of 2.521, W2's of experiment
   118.
2. **A contradiction:** C1 against experiment 106's finding that drift adds nothing once the filter
   runs, or C2 against experiment 118's KEZ result, without a tested explanation.
3. **Quantile score and mean error disagree in sign** on any of C1 to C5.
4. **Leaving one substation out flips a sign or label** of C1 to C5 (experiment 107's gain was
   mostly D10, so this is the likeliest trigger).
5. **Nested arms:** W3 nests the level-only filter of W0, so W0 beating W3 by more than the noise,
   or a control that should change nothing moving a score by more than its interval's half-width.
6. **Interval width:** C1's interval over twice or under half the width of experiment 118's C1
   (+-0.03).
7. **Dependence:** the arm is adopted into a plan while the replicate months disagree or the
   weights are labelled "not interpretable".

## Privacy

**D-labels only, with dates, for D5, D9, D10, D11, D14, and D16.** D15 and D18 appear only as G-c
with no dates. No names; topology appears only as interconnected D-label pairs. Scores are in
points of effective capacity, the percentage-of-capacity unit of the series; MW is stated once
(first mention of a substation's capacity scale in the write-up) and not again.

## Compute

| Job | Bound by | Arithmetic | Estimate |
|---|---|---|---|
| Block fits, W1-w28/w56/W3 | CPU (NumPy solve of a 14x14 system) | 5 substations x 52 weekly fits x 3 windows x about 0.1 s (a monthly-sized fit takes about 3 s per substation-year, so 1 week of 2-day-stride windows is far less) | under 5 min |
| W1-d28 daily refit | CPU | 5 x 365 x 0.01 s per 14x14 solve over a 28-day window | under 2 min per pass; affordable |
| Noise fit (W3, W3-q) | CPU | about 16 s per fit (experiment 118) x 5 substations x 2 arms | under 5 min |
| Gate | CPU | 4 scenarios x 5 substations x 2 noise forms x 20 draws x 6 arms x about 2 s of block-fit work (the filter noise is refitted on W3 only: 1,600 fits x 16 s) | about 7 h on 1 core, about 3.5 h on 2 cores |
| Scoring and bootstraps | CPU | tens of seconds per contrast | under 30 min |
| Figures | CPU | trivial | minutes |

**Everything is CPU only; no GPU job.** The gate dominates at about 3.5 hours on 2 cores, so
the slot is 2 cores and under 4 GB while other jobs run (the block fits are chunked, float32
where possible, lazy Polars); each script under 8 GB. If a noise fit exceeds 2 minutes the gate
cuts to 10 draws and states the wider half-widths. **Cut first:** W1-d28, then W3-q. Agent time:
about 4 hours to write the scripts, 4 to 5 hours of wall clock to run.

**Pull requests: one, opened only when the result is ready, with the plan commit as its first
commit.** No draft plan PR. If W3's filter update is added to `studies.dp_effects_filter` it needs
its own reviewed module PR (open question 1).

## Review triage

Placeholder until the plan review (Sonnet) is back; findings are applied or rejected with a reason.

## Open questions

1. **Does the two-stage measurement update live in `studies.dp_effects_filter` or in the script?**
   The plan assumes a script-level Kalman step on 14-dimensional block estimates (a few lines of
   NumPy), reusing the module's decomposition. A module change needs its own PR and an Opus review.
2. **The inflation factor of 27** for the block fit's covariance comes from experiment 113's
   simulation, and may not carry to a 28-day window; the gate's coverage of the block estimate
   (bar 4 in P4) tests it.
3. **Is the KFE gate diagnosis done?** The start condition above. If it finds block fits as noisy
   as the filter, cancel this experiment.
4. **W1-d28 cadence:** a daily 28-day refit overlaps its predecessor on 27 of 28 days, so it
   roughly equals W1-w28 with a lag; whether to run it is a judgement left to the manager.
