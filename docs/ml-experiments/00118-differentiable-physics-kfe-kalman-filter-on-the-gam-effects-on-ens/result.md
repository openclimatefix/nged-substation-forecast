---
id: "00118"
slug: differentiable-physics-kfe-kalman-filter-on-the-gam-effects-on-ens
part: "KFE"
theme: differentiable-physics-forecast-engine
also_relevant_to:
  - lagged-power-and-error-correction
  - differentiable-physics-long-history-and-drift
  - differentiable-physics-embedded-generation
  - production-reference-and-best-guess
status: planned
size: >-
  complex for the new module studies.dp_effects_filter (its own pull request, with one Sonnet review
  and one Opus review before it merges); medium for the scripts. Stored: private files under
  data/ml-experiments/00118-differentiable-physics-kfe-kalman-filter-on-the-gam-effects-on-ens/
  only. Serving path: not touched. Degradation rule: not touched. Several designs are defensible
  (which effects get a weight, how the noise matrix is fitted, the anomaly state, sign limits), and
  the plan pre-registers one of each. Callers: this experiment's scripts only.
hypothesis: >-
  Freezing the convex model's fitted effects and letting a weight on each follow a random walk,
  with a diagonal noise matrix fitted by likelihood on the tuning months, forecasts day-ahead
  demand on ENS better than experiment 111's filter of one level and one anomaly per hour band
  (GAM-K), because heating slopes, PV capacity, profile scale, and holiday effects drift faster
  than a 24-month fit follows. The fitted noise matrix beats one step variance shared by every
  weight, so the gain comes from how each effect drifts and not from the weights' freedom alone.
prediction: see "Predictions"
decision_rule: see "Decision rule"
benchmark: >-
  GAM-K as stored by experiment 111 (2.800 points over leads 0.5 to 168 hours in the scored year),
  reproduced; the same filter with one shared step variance; GAM-0, lagless XGBoost with the
  filter (X-nl+K), lagless XGBoost with lags (Xlag-d), and BG2 as stored by experiment 111;
  climatology; manual_heuristic
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles) in points of
  effective capacity, on rows outside the hand-filtered switching events at the scored site, at day
  D+1 (leads 15.5 to 39 hours from the 09:00 UTC power_fcst_init_time); every lead band beside it
verdict: <supported, small gain, partly supported, not supported, refuted, or inconclusive>
production_status: not-for-production
best_arm: <after the results>
headline: <after the results>
lessons: <after the results>
adopted_in: none
take_home_message: <written after the results>
date: <date>
pr: <pr>
advice: <written after the results>
code: ml-experiments/00118-differentiable-physics-kfe-kalman-filter-on-the-gam-effects-on-ens/
---

# Differentiable physics, KFE: a Kalman filter on the convex model's effects, on ENS

**Status: planned.** This page is the plan, committed before any code runs. The write-up replaces
it when the experiment finishes.

**This experiment re-implements the method of [de Vilmarest et al.
(2024)](https://doi.org/10.1109/TPWRS.2023.3310280), on our convex model and our data.** They froze
the fitted effects of an additive model of regional electricity load, put a random walk on one
weight per effect, and fitted a diagonal noise matrix by likelihood. We apply the same idea to the
convex model of the differentiable-physics series for Primary substations, driven by ECMWF ensemble
(ENS) weather forecasts. The noise-matrix choices follow [de Vilmarest and Goude
(2022)](https://arxiv.org/abs/2110.00334), and the greedy likelihood search for the noise matrix
follows the procedure of the R package [viking](https://CRAN.R-project.org/package=viking) as de
Vilmarest et al. (2024) use it. Everything here builds on [experiment
111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md), whose monthly
convex fits, pinned capacity, ENS calibration, and stored predictions it reuses.

## What the experiment asks, and the evidence it starts from

**Experiment 111's filter adapts one number per hour band, and the literature adapts the effects
themselves.** That filter holds a level and a transient anomaly per hour band, so it cannot say that
the heating response has weakened or that more PV exists than the pinned capacity knows. On 14 GB
regions in 2021, de Vilmarest et al. (2024) report a normalised root mean square error of 0.332
for the Kalman filter on the effects, against 0.377 for daily refits and 0.380 with the noise matrix
set to zero (regional load, not Primary substations). [de Vilmarest and Goude
(2022)](https://arxiv.org/abs/2110.00334) found a noise matrix proportional to the identity best in
only one competition month.

**Three earlier findings set the risks.** In [experiment
106](../00106-differentiable-physics-4-all-years-drift-and-recent-level/result.md), a drift term
added nothing once the filter was running. In [experiment
113](../00113-differentiable-physics-n5-convex-identification-at-realistic-noise/result.md), a
heating slope was known to within about 38% of its size at real noise, so a heating weight may
wander without meaning. In experiment 111, the filter's gain sat mostly under 24 hours. This
experiment changes only the filter, at the screening tier with one seed.

## The effects filter

**Frozen effects.** For each site, experiment 111's decision-form convex model (24-month window, no
drift, pinned capacity) predicts net power in points of effective capacity as a sum of terms.
`studies.dp_effects_filter` returns each group's contribution for any weather member. The groups
plus a remainder sum to the model's prediction to 1e-6 (control P2). A weight multiplies a whole
group.

| Group | Weights | Regressor c_k(t) | Meaning of a weight of 1 |
|---|---|---|---|
| Level, per hour band | 4 | 1 in the band, else 0 | no level shift (this is GAM-K's level) |
| Profile scale, per day type | 4 | the day type's profile value at the half-hour, mean removed | the fitted weekday, Saturday, Sunday, and bank-holiday profiles |
| Heating, per hour band | 4 | the band's fitted heating contribution | the fitted slopes |
| PV capacity | 1 | the pinned capacity times the derated unit PV shape (negative sign) | the pinned vintage capacity |
| Holiday | 1 | the sum of the named-holiday and Christmas-tent contributions | the fitted holiday effects |

**There are 14 weights, plus 4 anomaly states in the arms that use them.** The day-of-year
harmonics, lighting ramps, signed tent, and profile means stay frozen in the remainder, so the level
is the one carrier of mean shifts.

**The state and the update.** Write δ_k for a weight minus 1 (prior mean 0). The residual is r(t) =
Σ_k δ_k c_k(t) + z_b(t) + ε(t), where z_b is the anomaly of hour band b: an AR(1) process with the
factor φ and innovation size σ_z of GAM-K. Each δ_k follows a random walk with step variance q_k per
day. Each regressor is divided by its root mean square over the tuning months' active half-hours,
so the q_k are in points squared per day of a typical contribution and are comparable. The page
reports each as a relative change in the weight per year, sqrt(365 q_k) divided by that root mean
square. The update is a rank-one Kalman step per half-hour reading, on residuals at least 3.5 hours
old (`DEFAULT_DELAY_HOURS`, an assumption), with no switching masks in the filter. It runs warm on
experiment 111's stitched out-of-sample residual stream.

**The forecast** at lead h for target s and ensemble member m is the model's prediction for member m
plus Σ_k δ_k(τ) c_k,m(s) plus φ^h z(τ), with the weights held at their value at the forecast time
τ. Each member's own heating and PV contributions enter c_k,m. The 51 members are pooled by
`studies.ens_members.mixture_quantiles`. The heating, PV, and profile-scale weights are floored at
0 in the forecast.

**Fitting the noise matrix.** For each site the 14 log q_k are fitted by maximum a posteriori on the
tuning months' stream, by a batched greedy search over a grid of values for each weight in turn,
followed by an L-BFGS polish. The state runs warm through the replicate months, which are not in the
likelihood. All log q_k carry a normal prior whose mean and spread are shared across sites and
updated from the sites' own fits (empirical Bayes), because experiment 106's drift step sat at a grid
edge. GAM-K's q, φ, and σ_z are experiment 111's, and σ_ε is refitted jointly. The noise matrix is
reported per site. **A group with under 200 regressor-active half-hours in the tuning stream has its
q set to the shared prior mean, not fitted,** and is labelled "q set by the prior". The holiday
group has about four events in odd months, so it is the group most likely to carry this label. The
counts are printed. **The regressors for the update** come from the analysis proxy (the control
member of the latest run, leads 9 to 33 hours), as experiment 111's residuals do.

## Arms

| Arm | Weights | Anomaly | Noise matrix | Role |
|---|---|---|---|---|
| GK | level only (4) | yes | experiment 111's | benchmark, reproduced (GAM-K) |
| KE | all 14 | no | fitted | the effects filter, white-noise likelihood |
| KE-infl | all 14 | no | fitted, σ_ε inflated by √27 | KE under a fairer likelihood (exploratory) |
| KEZ | all 14 | yes | fitted | the effects filter plus the anomaly state; decision arm |
| KEZ-Q0 | level only moves (10 weights fixed) | yes | fitted for the level | the paper's static control (descriptive) |
| KEZ-QI | all 14 | yes | one q for every weight, fitted | identity noise matrix: the test of a learned noise matrix |
| KEZ-drop-g | all but group g | yes | fitted | which group earns the gain; g in profile, heating, PV, holiday |
| KEZ-1s | all 14 | yes | fitted by the one-step likelihood, `log q` capped at 2 | the first run's KEZ, kept (pre-fit clarification below); descriptive |
| KEZ-DT | all 14 | yes | fitted | the day-type-aware arm (pre-fit clarification below); descriptive for the verdict |
| KEZ-sh | KEZ plus experiment W1's shape state | yes | fitted | labelled follow-up with no number, unless experiment 115 (W1) has merged a tested shape state by the end of the scored year |

**KEZ nests GK:** with q_k = 0 for the 10 non-level weights it reproduces GK exactly (control P1).
**KEZ-Q0 equals GK with σ_ε refitted,** so KEZ minus KEZ-Q0 is C1 plus a refit artefact and is
descriptive. The real test of a learned noise matrix is KEZ minus KEZ-QI (C5). **KE's likelihood has
no place for autocorrelated residuals** (experiment 113 measured a variance inflation of 27), so its
noise matrix is inflated and its weights chase noise. C2 and C3 are read with that caveat, and
KE-infl is the fairer arm without an anomaly.

**Spread.** Each arm's quantiles are rebuilt from residual quantiles per hour band, quarter, and lead
band from the tuning months (`studies.dp_ens.LeadHourSpread`), with a spread factor per lead band
from {0.8, 0.85, 0.9, 1.0, 1.15, 1.3}. Any factor at an edge is reported. A shift-only check (GK's
quantiles moved by KEZ's correction) is reported beside KEZ, tails included.

## Years, sites, and rows

- **Tuning:** the odd months of 2024-07 to 2025-06 (noise matrix, σ_ε, spread factors).
- **Replicate:** the even months of the same year, never opened for the noise matrix or the spread.
- **Scored:** the 292 runs of 2025-07-02 to 2026-06-29.
- **Rows.** A tuning or replicate row needs both its origin and its valid time in the period's
  months. Rows are present rows outside the hand-filtered events, shared across a contrast's arms;
  the all-rows score sits beside. Lead bands are D+1 (the headline), 0–24, 24–48, 48–144, 144–168,
  and 12–36 hours.
- **Sites.** Decision sites D5, D10, D11, D14, and D16 (4 of 5 must improve); D9 beside them. The
  generator series are excluded.
- **Robustness rows:** November to March against April to October; leaving one site out; each hour
  band; working days against weekends and bank holidays.

## Controls and the known-answer gate

**Plumbing controls run first; an unfixable failure makes the verdict `inconclusive`.**

- **P0, initial state:** weights start at δ = 0 with variance (0.1)² of a weight (zero for the 10
  non-level weights under P1); the level and anomaly start at GAM-K's stationary values. The
  capacity vintage date used for each cut is printed against its origin.
- **P1, nesting:** KEZ with the 10 non-level q_k at 0 equals GK's stored `gamk_tuning_<site>.parquet`
  from experiment 111: `level` and `anomaly` to 1e-9 in float64, quantiles to 1e-4.
- **P2, decomposition:** the groups plus the remainder sum to `predict_gam` for every member, site,
  and cut, to 1e-6 points; each regressor is zero where its term is zero (PV at night).
- **P3, causality:** changing any reading after τ minus the delay, or any ENS run issued after τ,
  leaves every arm's forecast at τ unchanged; the standardisation, the noise matrix, and the spread
  use tuning months only.
- **P4, filter and noise fit:** the rank-one update matches a dense Kalman filter on 200 random
  small problems to 1e-8 (tested in `packages/studies/tests/test_dp_effects_filter.py`); a stream
  simulated from known q_k (0, 1e-6, 1e-4 per day) on the real regressors is fitted within a factor
  of 3 for every group whose weight ends with posterior standard deviation under 0.5, and groups at
  0 fit below 1e-7.

**The known-answer gate is blocking: a failed reading stops scoring and caps the verdict at
`inconclusive`.** It runs before any replicate or scored run, on synthetic streams built on the real
regressors c_k(t) of the five decision sites over the tuning months' calendar, with noise in two
forms: (a) AR(1) at each site's fitted φ and σ, and (b) a circular month-block resample of the
site's real tuning residuals. **Regressors come in two conditions:** exact (the filter sees the c_k
that made the stream), and errors-in-variables (the stream is built from ERA5 and CAMS
contributions, and the filter sees the ENS control contributions, so weather error enters through
c_k). Each scenario uses 20 draws per site, noise form, and condition, and the noise matrix is
refitted on each stream, so the gate tests the whole pipeline.

Scenarios: S0 clean (all weights 1); S1 PV weight rising linearly from 1 to 1.30 over 180 days; S2
heating weights falling from 1 to 0.70 over 90 days; S3 both.

**A reading passes only if at least 80% of draws pass it,** at the real amplitude, in every noise
form and condition (the errors-in-variables condition has the same bars).

1. **Tracking (S1, S3):** the filtered PV weight at day 200 recovers at least 70% of the injected
   change, and the correlation of the filtered and true weight over days 0 to 200 is at least 0.7.
2. **Tracking (S2, S3):** the filtered heating weight (mean over bands) recovers at least 50% of the
   injected change by day 120. A failure here does not block: it labels the heating weight "not
   interpretable" (experiment 113 found heating weakly identified).
3. **No false movement (S0):** the largest absolute deviation of any filtered weight from 1 after
   day 60 is under 0.10 for PV, heating, and holiday, and each non-level weight's fitted q is
   under a tenth of its value in S1 to S3.
4. **No cross-talk (S1, S2):** with only PV drifting, the heating weight's mean absolute change is
   under 25% of the PV change; with only heating drifting, the PV weight's change is under 25% of
   the heating change.
5. **Forecast skill follows (S1 to S3):** KEZ's day-ahead mean absolute error on the stream is below
   GK's.

**Power check, sized before the gate runs.** The code computes the posterior standard deviation of
each weight at day 200 (PV) and day 120 (heating) from the fitted noise matrix. If the injected
change is under 3 posterior standard deviations, the real-amplitude failure stands ("underpowered
at the real amplitude"), the amplitude at which 80% of draws pass is printed, a doubled-amplitude
row is added, and the weight is labelled "not interpretable". The posterior standard deviation is
probably overconfident (variance inflation of 27), so the spread of recovered weights in S0 is the
yardstick for "a weight moved". With 100 draws per scenario, a pass rate of 0.8 has a 95%
half-width of about 0.08.

**Reported, not blocking:** recovery at 10% and 20% of the amplitude; a +2 point level step (the
non-level weights move under 0.10); a +10% holiday step on 4 holidays (otherwise the holiday weight
is "not validated"); the gate with the shared-step noise matrix.

## Planned contrasts and margins

**Five planned contrasts, Bonferroni-corrected (99% intervals from `studies.bootstrap`, resampling
whole calendar months, paired across arms, 3 seeds),** with site counts, every arm's absolute score,
the mean absolute error of the median beside each, and the robustness rows, in the replicate months
and the scored year. The noise reference is the contrast C4 of [experiment
106](../00106-differentiable-physics-4-all-years-drift-and-recent-level/result.md) (two filtered
convex arms), which had a half-width of about 0.023 at 97.5%, so about 0.03 is expected here. GK
minus GAM-0 (a half-width of 0.14 in experiment 111) is a descriptive row only.

- **C1, the effects filter plus anomaly against GAM-K (decision):** KEZ minus GK at D+1.
- **C2, the effects filter without the anomaly against GAM-K:** KE minus GK at D+1.
- **C3, the anomaly's added value:** KEZ minus KE at D+1 and at 0–24 hours.
- **C4 (descriptive):** KEZ minus KEZ-Q0 at D+1, the static control of de Vilmarest et al. (2024).
- **C5, does the learned noise matrix matter:** KEZ minus KEZ-QI at D+1.

**The smallest gain worth porting is 0.05 points at D+1** (about 2% of GAM-K's score). KEZ adds 10
weights, a per-member decomposition at serving time, and a fitted noise matrix. The threshold sits
above the noise reference, so it is a judgement about what is worth porting and not a detection
limit.

**Exploratory at 95%:** KEZ-drop-g and KE-infl minus KEZ; KEZ minus X-nl+K, Xlag-d, and BG2; D9; the
shift-only check; the noise matrix refitted on the whole 2024-25 year.

## Predictions

Written before any fit; each line can fail, and none overrides the decision rule.

- **Gate:** readings 1 and 3 to 5 pass in 80% of draws; reading 2 (heating) is the likeliest to
  fail, with recovery 0.4 to 0.7; power check: PV change 3 or more posterior standard deviations,
  heating 2 to 4.
- **C1:** KEZ minus GK is −0.02 to −0.10 at D+1 and −0.05 to −0.15 over leads under 24 hours, with
  gains at 3 to 4 of 5 sites, largest in November to March.
- **C2:** KE minus GK within ±0.05. **C3:** KEZ minus KE is −0.02 to −0.08 at 0–24 hours and under
  0.03 in size at D+1. **C4:** KEZ-Q0 within ±0.02 of GK. **C5:** KEZ minus KEZ-QI is −0.01 to
  −0.06.
- **Fitted noise matrix:** the level and holiday weights carry the largest relative step per year
  (0.1 to 0.4), heating 0.05 to 0.25, PV 0.02 to 0.15, and profile scales under 0.05.
- **PV weight:** stays within ±0.15 of 1 at 4 of 5 sites, with a rising trend at 3 or more
  (capacity growth). A large drift would mean the vintage lags or the PV shape is wrong.
- **Heating weight:** higher in winter than in the shoulder months (peak-to-peak 0.15 to 0.4 of the
  slope), with the same sign at 4 of 5 sites; over a third of month-to-month moves are under two
  posterior standard deviations.
- **Drop-one and tails:** removing one group moves the score by under 0.02 (holiday in winter
  excepted); coverage is within 3 points of nominal at q05 and q95, and within 1.5 at q01 and q99.

## Decision rule

**The verdict is C1's label, with C5 deciding between `supported` and `partly supported`; C2 to C4
are labelled in the take-home message.** The scored year is the main test, and the replicate months
a second.

- **`supported`:** the gate passed; C1 is below zero in both periods with at least 4 of 5 decision
  sites better in each; the scored-year interval lies wholly below zero; the scored-year estimate
  is at least 0.05 points; and C5's scored-year estimate is below zero (the learned noise matrix
  does the work).
- **`small gain`:** the scored-year interval lies wholly below zero with an estimate under 0.05, and
  the replicate months' estimate is below zero.
- **`partly supported`:** C1's estimate is below zero in both periods, but the scored-year interval
  includes zero or fewer than 4 sites improve in one period; or C1 meets `supported` but C5's
  estimate is not below zero (the gain is the weights' freedom and not their drift); or the scored
  year is `supported` and the replicate months reverse the sign.
- **`refuted`:** C1's interval lies wholly above zero in the scored year or the replicate months.
- **`not supported`:** both periods' C1 intervals lie within ±0.05 of zero.
- **`inconclusive`:** anything else, or a plumbing control or gate reading that cannot be fixed (the
  take-home message names the reading and what it rules out).

A post hoc change to a gate reading is labelled as such, and the first run's result is kept and
never reported as a pass.

## Winter breakdown (pre-fit clarification, added before any fit)

**The best forecast model goes to production first for a winter, so every contrast is also reported
for the winter, beside the annual score.** The clarification below was added to the plan after the
plan was first committed and before any filter was fitted, at the maintainer's request.

- **Winter rows:** the half-hours whose local date is in November to March. Each contrast (C1 to C5)
  is reported on these rows at D+1 and over every lead band, with the same paired month-block
  intervals where the rows allow, and with site counts. Evening peaks within the winter rows are the
  half-hours ending 16:30 to 20:00 local time.
- **Cold snap, defined now:** a local day in November to March whose mean ERA5 temperature at the
  site is below the 10th percentile of such days in the site's ERA5 record from 2019-09 to 2024-06,
  which is a threshold fixed from data before the tuning year. The code prints the threshold and the
  number of cold-snap days per site in the tuning, replicate, and scored periods. Where a period
  holds fewer than 15 cold-snap days at a site, the page says "too few days" and gives no interval.
- **Independent winters:** the evidence holds one winter that was not used to choose anything
  (2025-26, scored) and one winter whose odd months set the noise matrix and the spread (2024-25;
  the even months are the replicate). The page states this beside every winter number.
- **Verdict rule:** a winter loss is a reason to hold an arm back even if the annual mean improves.
  If KEZ's C1 on the winter rows is above zero in the scored year with its interval wholly above
  zero, or if the cold-snap days show KEZ worse than GK by more than 0.05 points, the verdict is
  capped at `partly supported` and the take-home message says so.

## Day-type-aware arm KEZ-DT (pre-fit clarification, added before any fit)

**A stored-forecast check found that experiment 111's GAM-K scores 0.4 to 0.5 quantile-score points
worse on Christmas than on matched ordinary days in all three stored sets, while GAM-0 scores about
the same on both.** The filter may be carrying an ordinary-day level into holidays. The clarification
below was added to the plan at the maintainer's request, after the plan was first committed and
before any filter was fitted. It does not change the verdict, which stays C1's.

- **Holiday days:** the days the calendar marks as a named holiday period (each bank holiday, Easter,
  bridge days, and the Christmas to New Year run) or as a bank-holiday day type.
- **Design:** KEZ-DT is KEZ with two changes. Readings on holiday days do not update the state (the
  filter treats them as missing, so the state still moves by the exact gap discretisation), and the
  noise matrix is refitted on the stream with those readings removed. On a holiday target day the
  level and anomaly corrections are set to 0, so the frozen model's holiday effects stand without an
  ordinary-day level added to them. The other weights still apply.
- **C6, primary for this arm:** KEZ-DT minus KEZ on the holiday days of the replicate months and the
  scored year, resampling whole days. The stored sets hold only 8 to 20 Christmas days, so the page
  says "too few days" for any group of fewer than 15 days, and gives no interval there.
- **C6b, non-inferiority:** KEZ-DT minus KEZ on all other days, passed if the upper end of the
  interval is at most +0.02 points.
- **Also reported:** the same two contrasts against GK, and winter nights (November to March, hour
  band night) with the winter breakdown above.

## Noise-fit likelihood (pre-fit clarification, added after the first tuning-month fit)

**The first fit of the noise matrices, by the one-step likelihood, gave weights that move too fast
to carry a state to the next day, so the noise matrices are refitted by a likelihood that matches
the forecast's lead.** The first fit ran on the tuning months only. No replicate month and no scored
run had been forecast, scored, or opened. The first run's tuning-month scores at D+1, in points
(in-sample for the noise matrix and the spread, so optimistic), were 2.420 for GK and 2.524 for KEZ,
and 2.429 for KEZ with its PV weight held. The level and PV weights sat at the upper bound of
`log q` (2.0), which is below the value 2.25 that GAM-K's own fit reached at one site, so KEZ could
not nest GK.

- **Change 1:** the bounds of `log q` are those of `studies.dp_assimilation` (up to 6).
- **Change 2:** the noise matrix is fitted by the **lead-matched likelihood**: each tuning-month
  reading at D+1 is scored by its predictive density from the state at its day-ahead cutoff (the
  newest half-hour that ended 3.5 hours before 09:00 UTC on the previous day), with the state
  stepped forward with no further reading. A noise matrix that adapts too fast to carry its state
  to the next day is penalised. [de Vilmarest et al. (2024)](https://doi.org/10.1109/TPWRS.2023.3310280)
  work with one daily step, where the one-step likelihood is already a day-ahead likelihood.
- **Change 3:** the first run's KEZ is kept as the arm KEZ-1s and forecast like every other arm, so
  the page reports both likelihoods. Its fitted noise is the first run's, with the old bound.
- **Unchanged:** the arms, the contrasts, the decision rule, the gate's readings and bars, and the
  rule that the replicate months and the scored year are opened once.

## Audit before merge: the seven triggers

The write-up carries "Checks" bullets: median minus mean forecast per site and arm, and the mean
error beside each contrast.

1. **A control fails:** P0 to P4, any gate reading, or GK's reproduction of experiment 111's 2.800.
2. **A contradiction:** C1 against experiment 106's finding that slow adaptation adds little; the PV
   weight's sign against capacity growth.
3. **Quantile score and mean error disagree in sign** on any of C1 to C5.
4. **Leaving one site out flips a sign or a label** of C1 to C5.
5. **Nested arms:** GK beating KEZ (which nests it); KEZ-Q0 beating GK by more than the noise.
6. **Interval width:** C1's interval over twice or under half as wide as experiment 106's C4 (0.023
   at 97.5%), or as the C1 of experiment 115 (W1) if that has merged.
7. **Dependence:** KEZ adopted while the replicate months disagree, or while the PV and heating
   weights are labelled "not interpretable".

## Substation cards (provisional: fitted to past weather, forecast on ENS, screening run)

**Each decision site (D5, D10, D11, D14, D16; D9 beside them) gets one card for KEZ in the scored
year.** A card shows:

- the filtered weight of each group over time with its posterior standard deviation band (PV also as
  the implied capacity, the weight times the pinned vintage, beside the vintage and the hindsight
  `capacity_weekly.parquet` check);
- the fitted relative step per year of every weight;
- a line "a heating slope is known to about 38% of its size at real noise (experiment 113)", with
  any "not interpretable" label;
- a panel "how it does in a cold snap": the cold-snap days of the scored winter at the site, with
  the day count and the rule above, GK and KEZ side by side ("too few days" where the count is under
  15);
- example forecasts on the **second Monday of July, October, January, and April**, a rule fixed
  now, before any score is read (the weeks beginning 2025-07-14, 2025-10-13, 2026-01-12, and
  2026-04-13): D+1 forecasts chained from each 09:00 UTC origin, median with 80% and 90% bands
  against the observed series, GK and KEZ side by side.

One cross-substation figure compares the weights' movements and the fitted noise matrices.

## Privacy

D-labels only, with dates, for D5, D9, D10, D11, D14, and D16. The generator series are excluded.
No topology and no substation name appears.

## What the write-up must contain

The take-home message says how much day-ahead error the weights remove beyond GAM-K's level and
anomaly, whether a learned noise matrix or the weights' freedom carries it, and whether the PV and
heating weights move by more than the spread of recovered weights in S0.

- **Figures, raw data first:** the filtered weights per site with bands; the C1 to C5 headline chart
  with intervals, site counts, and both periods; score by lead band per arm; the gate; the noise
  matrix.
- **Tails table:** coverage and pinball at q01, q05, q95, and q99, every arm and the shift-only
  check, per lead band, with threshold-weighted CRPS (`studies.tail_calibration`); no curtailment
  cost, because experiment 87 found no event that could rank forecasts.
- **Absolute scores** of every arm and reference at D+1 and over 0.5 to 168 hours, on shared rows.
- **Caveats:** the 3.5-hour delay is assumed; hindsight switching masks in scoring; five
  substations; one year of tuning months; collinear weights can offset each other; the posterior
  standard deviation is probably overconfident; the capacity vintage may differ between the tuning
  and scored cuts (dates printed); heating variance grows unobserved through summer, so October and
  April rows are reported. Inherited from experiment 111: the in-sample ENS calibration, the pinned
  capacity reading hindsight weeks, ERA5 in the convex model's last training rows, and a spread
  fitted on the tuning months. The gain of de Vilmarest et al. (2024) is on regional load and is no
  target here.

## Run order, compute, and time

**Compute: about 1.5 hours on 6 CPU cores for the scripts, no GPU.** A filter pass over a year at 18
states takes about 0.5 seconds, and a noise-matrix fit takes about 16 seconds. The gate is 4
scenarios × 20 draws × 5 sites × 2 noise forms × 2 regressor conditions = 1,600 noise-matrix fits,
about 7 hours of one core and 1.2 hours on 6 cores. **Thinning:** if a fit exceeds 2 minutes, the
gate cuts to 10 draws per site and states the wider half-widths. **Cut first:** the drop-one arms.
**Slot:** any free 6-core slot, load checked first; experiment 111's `gamk_tuning_<site>.parquet`
files are read-only.

## Review triage

A Sonnet plan review found 12 issues, all applied (two in part): the trigger 6 reference is the
contrast C4 of experiment 106 and not GAM-K minus GAM-0; the gate needs at least 80% of draws to
pass each reading, and the real amplitude is the gate; the errors-in-variables condition uses ERA5
and CAMS against ENS-control contributions; KE-infl is the fairer arm without an anomaly; the
learned-noise-matrix test is C5; the capacity vintage dates are printed; a group with under 200
active half-hours has its q set by the prior, with a non-blocking holiday scenario; the generator
series are not named on this page; the example weeks follow a stated rule; the shared pieces
(`mixture_quantiles`, the spread factors) are already reviewed code in `studies.ens_members` and
`studies.dp_ens`. The maintainer ruled that the module has one Sonnet and one Opus review (tuning
and gate runs may use the Sonnet-reviewed branch; the module does not merge, and the scored-year
run and the experiment merge do not start, until the Opus review lands), that the 50% heating bar
does not block, and that KEZ-sh is a follow-up.
