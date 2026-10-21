---
id: "00127"
slug: differentiable-physics-wu-weight-update-schemes-for-the-gam-weights
part: "WU"
theme: differentiable-physics-forecast-engine
also_relevant_to: [differentiable-physics-long-history-and-drift, lagged-power-and-error-correction]
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00127-.../ only. Serving path: not
  touched. Degradation rule: not touched. More than one design is defensible (window length,
  how the block-fit noise enters the filter), and the plan pre-registers one of each. Callers:
  this experiment's scripts only; the scripts reuse the KFE harness and the module
  studies.dp_effects_filter. The two-stage filter's measurement update stays in the script, and
  moves into the module only if W3 is `supported`.
hypothesis: >-
  Denoising block convex fits of the weights (one window, 28 or 56 days) with a Kalman filter whose
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

**This plan starts only when the KFE gate diagnosis shows that block convex fits are a usable
yardstick for the weights.** The diagnosis is the block-fit yardstick step in
`briefs/dp_kfe_gate_diagnosis_run.md`. The test is numeric and is a pre-fit choice, written here
before the diagnosis is read. On the gate's injected-effect streams (the PV weight rising by 30%
over 180 days, and the heating weights falling by 30% over 90 days), the block fit's standard
error (one standard deviation of its coefficient covariance) must be under both 0.15 (half the
injected change) and the standard deviation of experiment 118's filter (KEZ) for the same weight
on the same day, at 4 of 5 decision substations. The PV weight is read at day 200 and the heating
weight at day 120. If the block fit fails the test, this experiment is not run, because the
two-stage arm (W3) has nothing to denoise, and the diagnosis's finding is the answer. The manager
checks this before assigning a slot. The plan was written by Sonnet from the briefs and write-ups
named below and has had one Sonnet plan review, triaged below; no code has run.

**The plan expects to land on `small gain` or `partly supported`, not `supported`.** The predicted
C1 range (-0.02 to -0.08 points) straddles the 0.05-point threshold for porting, so most outcomes
are a gain too small to port. The experiment is worth running because it also says whether
denoising block fits beats filtering residuals (C2) and whether the filter adds anything to the
block fit (C3).

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

**A one-day convex fit cannot identify the PV, heating, and profile weights** (one day holds one
solar curve and one temperature), so there is no single-day arm. A daily refit is not run either:
refitted every day on a window of 28 days or more, it overlaps its predecessor on all but one day
and would roughly equal a weekly refit with a lag.

## Arms

All arms start from experiment 111's monthly frozen-effect decomposition of the convex model
(`studies.dp_effects_filter`: level per hour band, profile scale per day type, heating per hour
band, PV capacity, holiday), so each arm changes only the weights.

**Two arms differ in which filter runs underneath, and the plan states how.** W0, W1, and W3 keep
GAM-K's level-and-anomaly filter underneath, and their block fits and Kalman filter act on the
weights other than the level weights. W2 (KEZ) is the exception: its 14 weights include the level
weights, and it runs no separate level-and-anomaly filter. C2 (W3 minus W2) therefore differs in
two ways, the update scheme and how the level is tracked, and the write-up says so beside C2. C1,
C3, and C4 compare arms that share the level-and-anomaly filter.

| Arm | Weights | Update | Role |
|---|---|---|---|
| W0 | frozen at the monthly fit | none (GAM-K as stored) | benchmark, reproduced |
| W1 | block convex fit on the non-level weights | refit weekly on the last w days of residuals (w is 28 or 56, chosen below), no Kalman filter | scheme 1 |
| W2 | 14 weights | KEZ, experiment 118's filter, reproduced | scheme 2 |
| W3 | block convex fit on the non-level weights | W1's weekly block fit, then a Kalman filter on the block estimates with measurement noise equal to the fit's coefficient covariance times one fitted multiplier | scheme 3 |

**The window w is chosen once, for W1 and W3 together, on the tuning months before any replicate
or scored row is scored.** The rule: score W1 with w = 28 and w = 56 at D+1 on the tuning months'
shared rows, and take the window with the lower mean score over the five decision substations; a
difference under 0.01 points goes to 56 days, which has the smaller sampling noise. The choice is
frozen before the gate runs. The window that is not chosen appears in no contrast, so no contrast
picks its best arm on the scored year.

**Block convex fit.** The fit is a ridge-regularised non-negative least squares of the frozen
group contributions c_k(t) on the residuals of W0 over the window, with the same sign limits as
experiment 118 (heating, PV, profile weights floored at 0), a ridge towards a weight of 1 whose
strength is fitted on the tuning months, and the 3.5-hour data delay of experiment 118. The fit
sees past weather as the analysis proxy (`weather_utils`), because that is all a forecast at the
cutoff holds; scoring then uses ENS forecasts, so the fitted PV and heating weights carry the
analysis-to-ENS mismatch, and the same mismatch applies to W2. The sandwich covariance of the
weights is σ²(CᵀC + λI)⁻¹ from the Gram matrix C, with σ² estimated from the window's residuals.

**W3's filter, with one fitted measurement-noise multiplier.** Each weight follows a random walk.
Every 7 days the block estimate is a measurement of the current weight. Its covariance is the
sandwich covariance times one scalar multiplier m, which stands in for three things at once:
autocorrelated residuals (experiment 113 estimated a variance inflation of 27 from pooled
simulation at about 11 months, and this plan does not import it), the overlap of successive
blocks (a 56-day window refitted every 7 days overlaps its predecessor by 49 days), and the
filter's own fitted noise. Only the product reaches the filter, so m is fitted once per window
and not as separate factors. The fit has two steps on the tuning months only. A day-block
bootstrap of the tuning-month residuals (resampling whole days) gives m_boot, the multiplier that
makes the sandwich covariance match the bootstrap variance of the block estimate. The day-ahead
likelihood (a lesson of experiment 118: the one-step likelihood was 0.127 points worse) is then
maximised over a grid of m from m_boot/4 to 4 m_boot, pooled over the five decision substations,
and that value is m. The write-up reports m and m_boot beside 27. The base random-walk variance
per weight comes from the same day-ahead marginal-likelihood fit as experiment 118's.

**Spread.** As in experiment 118 (`spread_with_factors`, spread factor per lead band from the same
grid, fitted on the tuning months), so arms differ only in the median path.

## Years, substations, and rows

- **Tuning:** the odd months of 2024-07 to 2025-06 (window choice, ridge strength, spread factors,
  noise multiplier). **Replicate:** the even months of the same year, opened once. **Scored:** the
  292 runs, 2025-07-02 to 2026-06-29. Same cuts as experiment 118; no row on or after 2026-07-01.
- **Block-fit history may span months of any group, because it is past data.** A 28-day or 56-day
  window ending in a tuning month usually reaches back into a replicate month, and the block fit
  reads that month's power as history. Only scoring is held back: no replicate month is scored
  until the window, ridge, multiplier, and spread factors are frozen.
- **Substations:** the decision substations D5, D10, D11, D14, and D16 (4 of 5 needed), with D9
  beside them. D15 and D18 appear only as G-c with no dates; generator series are otherwise
  excluded.
- **Rows:** present rows outside the hand-filtered switching events, shared across a contrast's
  arms, with the all-rows score beside. Lead bands as in experiment 118: D+1 (headline), 0-24,
  24-48, 48-144, and 144-168 hours.

## Known-answer gate (blocking: a failure caps the verdict at `inconclusive`)

**The gate reuses experiment 118's gate harness (`e00118_gate.py`, `e00118_streams.py`,
`e00118_noise.py`) and its scenarios, and runs on W1, W2, and W3 before any replicate or scored
row is scored.** It must be able to fail: the bars are fixed now and the gate has failed in
experiment 118.

- **Streams:** the odd tuning months hold no stretch of 200 days, so each stream is built on a
  synthetic contiguous calendar. The stream takes the real regressors c_k(t) of one decision
  substation over the contiguous year 2024-07-01 to 2025-06-30 (regressors are weather and
  calendar values, not outcomes), preceded by 56 days at weight 1 so that the first block fit
  exists on day 1. Injected weight changes: S0 no change; S1 PV weight rising by 30% over days 1
  to 180; S2 heating weights falling by 30% over days 1 to 90; S3 both. Each weight then holds
  its final value to day 240. Block-fit windows of 28 or 56 days are cut from the stream itself,
  so they involve no real power.
- **Noise, two forms:** (a) the filter's own AR(1) noise at each substation's fitted φ and σ; (b)
  a circular month-block resample of the substation's real tuning residuals (odd months only).
  20 draws per substation, scenario, and form: 4 × 5 × 2 × 20 = 800 stream-draws.
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
- **Comparison with the yardstick:** the same table for the unfiltered block fit (W1), so the
  write-up can say whether denoising adds recovery or only lag.
- **Power check first:** before the gate, the code prints the standard deviation across the 20
  draws of S0 of each arm's PV weight at day 200 and heating weight at day 120. This empirical
  spread does not depend on the arm's own covariance, which is the quantity in doubt for W3. If
  the injected change is under 3 of these standard deviations, the failure stands as
  "underpowered at the real amplitude", and a doubled-amplitude row is added.
- Gate failures cap the verdict at `inconclusive` for the failing arm; the other arms keep their
  labels.

**Plumbing controls (run first):**

- **P1:** W2 reproduces experiment 118's KEZ to 1e-6; W0 reproduces GAM-K's 2.521 at D+1; and W2
  minus W0 reproduces experiment 118's +0.015 (within +-0.03).
- **P2:** a block fit of noise-free synthetic data with known weights returns them to 1e-6.
- **P3:** causality. Changing any reading after the cutoff, or any ENS run issued after it, leaves
  every arm's forecast unchanged.
- **P4:** the sandwich covariance times m matches a day-block bootstrap within 30% on simulated
  data.

## Planned contrasts and margins

**Four planned contrasts, Bonferroni-corrected 98.75% intervals (1 - 0.05/4) from
`studies.bootstrap` (resampling whole calendar months, paired across arms, 3 seeds),** with
substation counts, every arm's absolute score, the error of the median beside each, and both
periods.

- **C1 (decision):** W3 minus W0 at D+1.
- **C2:** W3 minus W2 (KEZ): does denoising block fits beat filtering residuals. The level
  tracking differs between the arms (see "Arms").
- **C3:** W3 minus W1: does the Kalman filter add anything to the block fit.
- **C4:** W1 minus W0: does a plain rolling refit help.

**The smallest gain worth porting is 0.05 points at D+1** (about 2% of GAM-K's score). W3 adds a
weekly block fit, a covariance, and a filter to the serving path; the threshold is a judgement
about that cost, not a detection limit. Noise reference: experiment 118's C1 interval was +-0.03
wide.

**Exploratory at 95%:** each window against W0 on the tuning months; leaving one substation out;
November to March against April to October.

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
  the absolute error in points, cold nights defined by the rule above, per arm and substation. It
  enters the decision rule below.
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
D-labels only. A cross-arm figure per substation overlays W1, W2, and W3 on the same axes. The
write-up says whether the weights move by more than their bands.

## Predictions

Written before any fit; each can fail, and none overrides the decision rule.

- **Gate:** W3 passes readings 1 and 3 to 5 on the AR(1) form; on resampled real residuals it
  passes reading 5 and perhaps 1 (block fits use many days, so they should tolerate seasonal
  structure better than a day-by-day filter), with heating tracking the likeliest failure. W1
  passes reading 1 on block-resampled noise at least as often as W2 does.
- **C1:** W3 minus W0 is -0.02 to -0.08 at D+1, with gains at 2 to 4 of 5 substations, below the
  0.05 threshold at most substations.
- **C2:** W3 minus W2 is -0.02 to -0.06 (W3 removes the day-by-day noise that hurt KEZ).
- **C3:** W3 minus W1 is -0.02 to -0.10 (the filter removes sampling noise).
- **C4:** W1 minus W0 is within +-0.05.
- **Cold winter nights:** W3 does not beat W0 by more than 0.1 points at 4 of 5 substations.
- **Weights:** PV weight within +-0.15 of 1 at 4 of 5 substations; heating weight higher in
  winter than the shoulder months; W1 weights vary from block to block by more than W3's.

## Decision rule

**The verdict is the first label below whose conditions hold, tested in this order, so no two
labels can both apply.** Notation: C1's estimate and 98.75% interval in the scored year are E_s
and I_s, and in the replicate months E_r and I_r. The replicate interval rests on 6 bootstrap
blocks and is fragile.

1. **`inconclusive` (gate or control):** a plumbing control fails, or the gate fails for W3.
2. **`refuted`:** I_s lies wholly above zero. The replicate months alone never refute, because 6
   blocks cannot carry a verdict; `supported` needs both periods, and `refuted` needs the scored
   year, by design.
3. **`supported`:** all of: E_s at most -0.05 with I_s wholly below zero; E_r below zero; at least
   4 of 5 substations better in each period; C2's scored-year estimate below zero; the
   non-inferiority margin on event days and elsewhere holds; and the median cold-winter-night
   error of W3 is not worse than W0's by more than 0.1 points at 4 of 5 substations.
4. **`partly supported`:** I_s wholly below zero with E_s at most -0.05, but one of the other
   conditions of `supported` fails. The write-up names which one (for example, C2 not below zero
   means the gain is the block fit and not the denoising).
5. **`small gain`:** I_s wholly below zero, E_s between -0.05 and zero, and E_r below zero.
6. **`not supported`:** none of the above, and both I_s and I_r lie inside -0.05 to +0.05.
7. **`inconclusive`:** anything else.

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
3. **Quantile score and mean error disagree in sign** on any of C1 to C4.
4. **Leaving one substation out flips a sign or label** of C1 to C4 (experiment 107's gain was
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
| Block fits, W1 and W3, both windows on the tuning months | CPU (NumPy solve of a 14x14 system) | 5 substations x 52 weekly fits x 2 windows x about 0.1 s | under 5 min |
| Noise multiplier and base process variance (W3) | CPU | about 16 s per fit (experiment 118) x 5 substations | under 5 min |
| Gate, block-fit work | CPU | 800 stream-draws x 3 arms (W1, W2, W3) x about 2 s | about 1.3 h |
| Gate, W3 process-noise refit per stream-draw | CPU | 800 x about 16 s | about 3.6 h |
| Scoring and bootstraps | CPU | tens of seconds per contrast | under 30 min |
| Figures | CPU | trivial | minutes |

**The honest total is about 5.5 hours on 1 core, or about 3 hours on 2 cores.** The reviewed plan
with its extra arms (a second W3 window, a learned process noise, a daily refit) came to about 800
stream-draws at about 10 hours on 1 core; dropping those arms halves the figure. Everything is CPU
only; no GPU job. The slot is 2 cores and 8 GB, the same 8 GB that each script is limited to (the
block fits are chunked, float32 where possible, lazy Polars). The gate stays at 20 draws; cutting
it to 10 needs the manager's agreement and a statement of the wider half-widths. Agent time: about
4 hours to write the scripts, 3 to 4 hours of wall clock to run.

**Pull requests: one, opened only when the result is ready, with the plan commit as its first
commit.** No draft plan PR. The script keeps W3's Kalman step; the step moves into
`studies.dp_effects_filter` (own reviewed PR, Opus review) only if W3 is `supported`.

## Review triage

The Sonnet plan review (`data/ml-experiments/briefs/dp_wu_plan_review_result.md`) gave 12 ranked
findings, 4 open-question recommendations, and a simpler version.

**Accepted and applied:**

- Finding 1 (compute understated): the gate is 800 stream-draws and the honest total is written
  above; the slot memory is reconciled at 8 GB.
- Finding 2 (filter confound): W1 and W3 keep the level-and-anomaly filter; W2 does not, and C2 is
  labelled as differing in level tracking.
- Finding 3 (no selection rule): one window is chosen on the tuning months, and "best W3 arm" and
  "best W1 arm" no longer exist.
- Finding 4 (overlapping verdict words): the decision rule is an ordered list, and the asymmetry
  between `refuted` and `supported` is stated.
- Finding 5 (odd-month calendar): the gate streams are synthetic contiguous, and block-fit history
  may span replicate months.
- Finding 6 (stacked inflation factors): one fitted multiplier m replaces the factors 27 and 8.
- Finding 7 (no numeric start condition): written in the opening paragraph, labelled a pre-fit
  choice.
- Finding 8 (weather the fit sees): stated under "Block convex fit".
- Finding 9 (cold nights): added to `supported`.
- Finding 10 (plan expects not to clear its bar): stated in the lead.
- Finding 11 (C6 dilutes Bonferroni): C6 moved to P1; the level is 98.75% for four contrasts.
- Finding 12 (circular power check): the check uses the spread of the 20 S0 draws.
- Simpler version: W1-d28, W3-q, C5, the second W3 window, and the 20-to-10 draws default are
  dropped.
- Open questions: the Kalman step stays in the script; the factor 27 is not imported; the daily
  refit is dropped.

**Skipped:** nothing. The cold-night margin of 0.1 points and the start condition's 0.15 are
proposals of this revision, labelled pre-fit choices, not values the review fixed.
