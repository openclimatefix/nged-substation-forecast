---
id: '00135'
slug: differentiable-physics-n10-joint-kalman-and-demand-fit
part: N10
theme: differentiable-physics-long-history-and-drift
also_relevant_to:
- differentiable-physics-forecast-engine
- lagged-power-and-error-correction
status: planned
size: 'complex. Stored: private fits, predictions, and scores under data/ml-experiments/00135-differentiable-physics-n10-joint-kalman-and-demand-fit/
  only. Serving path: not touched (the change is in how the convex model is fitted, which is
  training-loop code). Degradation rule: not touched. Designs: more than one is defensible (what
  the loss is, whether the filter is trained or held fixed, how the shapes are tied to the
  two-stage fit), and this plan fixes one of each. Callers: this folder''s scripts only, which
  call the reviewed `studies.dp_gam`, `studies.dp_assimilation`, `studies.dp_filter_grid`, and
  `studies.dp_ens` modules without changing them.'
hypothesis: Fitting the convex demand model (GAM-K) so that its coefficients minimise the day-ahead
  error after the Kalman correction, instead of the plain error before it, lowers the day-ahead
  quantile score at 12 to 36 hours below the two-stage fit (fit the demand model, then filter its
  residuals), without moving the fitted shapes to explain recent-level structure that the filter
  already tracks.
prediction: C1 (J1 minus TS, quantile score at 12 to 36 hours, scored year, five decision
  substations) is -0.05 to +0.02 points, most likely about zero. C2 (J2 minus TS) is -0.03 to
  +0.01 points. J1's fitted shapes move to explain the injected level change and PV-shaped change
  by more than TS's do (the absorption control fails for J1), and J2's do not. The fitted
  shapes of J1 differ from TS's by a root mean square of 5% to 15% of each term's own root mean
  square, and J2's by under 5%.
decision_rule: 'Two planned contrasts (C1 J1 minus TS, C2 J2 minus TS), each read as a two-sided
  percentile interval of the month-resampled paired bootstrap (12 calendar months of the scored
  year, 20,000 resamples) at the 1.25% and 98.75% percentiles, the Bonferroni level for two
  contrasts at 5%. The verdicts apply in this order. (1) The verdict is capped at `inconclusive` if
  a blocking control fails: P1 (reproduction), P2 (the linear filter equals the filter), or P3
  (the solved loss equals the loss of the filter run on the fitted residuals). (2) A joint arm that
  lowers the score and fails the absorption control is reported as an identification failure
  (`not supported`), whatever its score. (3) A joint arm is `supported` when its contrast
  interval lies wholly below zero, at least 4 of the 5 decision substations improve by at least
  0.005 points in both the scored year and the replicate months, the score over all leads 0.5 to
  168 hours has an interval upper bound of at most +0.02, and it passes the absorption control. (4)
  It is `small gain` when the interval lies wholly below zero but a condition of (3) fails. (5) It
  is `refuted` when the interval lies wholly above zero. (6) Otherwise it is `inconclusive`, which
  is the expected outcome for a true effect of a few hundredths of a point, because 12 months
  pooled over 5 substations give wide intervals, and is not evidence of no effect. The threshold
  is zero, because a joint fit replaces the fitting routine inside the training loop and adds no
  model, data source, or serving step. The shape distance is reported, not binding: it feeds the
  substation cards and the discussion of whether to port. The experiment tests only whether the
  coefficients should be fitted through a filter with fixed noise settings, so a null result does
  not rule out learning the noise settings and the coefficients together.'
benchmark: TS, the two-stage GAM-K with the learned-noise filter settings of experiment 133
  (its GRID arm); GAM-K with experiment 111's fixed settings (FIX) as reference; climatology and
  manual_heuristic as in experiment 111 for reference
row_set_id: fold_a_daily_gap_v1
metric: Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87)
  in points of effective capacity, on rows outside the hand-filtered switching events, at the
  five decision substations D5, D10, D11, D14, and D16 (D9 beside them), at leads 12 to 36 hours
  (D+1) with every lead beside it, then winter (November to March), winter nights, cold nights,
  and the Christmas and bank-holiday periods broken out; coverage and pinball loss of the 0.01,
  0.05, 0.95, and 0.99 quantiles
code: ml-experiments/00135-differentiable-physics-n10-joint-kalman-and-demand-fit/
---

# Experiment 135: does fitting the demand model through the Kalman filter beat fitting it first and filtering after?

**Planned; this page is the pre-registration and is rewritten when the experiment finishes.**

**The experiment asks whether the convex demand model forecasts the next day better when its
coefficients are chosen to minimise the error that remains after the Kalman filter has corrected
the forecast.** The demand model is GAM-K, the convex additive model with a Kalman filter on its
residuals from [experiment
111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md). The filter
tracks a slowly moving level and a quickly decaying transient of the model's error in each of four
hour bands. In the two-stage design, the model is fitted to minimise its own squared error, and the
filter then runs on the residuals. In the joint design, the fit minimises the squared error of the
day-ahead forecast after the filter's correction, so the model is not asked to explain what the
filter will explain anyway.

**The filter is linear in the residuals, which makes the joint fit an exact bounded quadratic
problem.** For fixed noise settings, the Kalman gains depend on which readings are present, not on
their values. The correction at a day-ahead case is therefore a fixed linear combination of the
earlier residuals, and the residuals are linear in the coefficients. Running the filter once on
each column of the design matrix (and on the target) gives the corrected design matrix, and the
joint coefficients solve the same bounded least-squares problem that `studies.dp_gam` solves for
the two-stage fit. The GPU runs the filter over a batch of columns with `studies.dp_filter_grid`.
The fit is "differentiable" in the sense that its loss depends on the filter, with no gradient
descent and no learned filter state.

**The earlier failures set the design.** [Experiment
118](../00118-differentiable-physics-kfe-kalman-filter-on-the-gam-effects-on-ens/result.md) and
experiment 128 found that weights on effects and a shared day-level anomaly cannot be told apart
from noise on real residuals. A joint fit can move the model's shapes to absorb recent-level
structure, which would look like a gain and be an identification failure. The filter state is
therefore GAM-K's own (a level and a transient per hour band), no effect weights are added, and an
absorption control and a shape-distance report are pre-registered.

**The experiment re-implements no published method.** The joint training of a demand model and a
state-space correction follows the general idea of fitting through the filter, as in [de Vilmarest
and Goude (2022)](https://arxiv.org/abs/2110.00334), who run a state-space correction on the
residuals of a generalised additive model for electricity load, but this experiment optimises the
coefficients through the filter and they do not. The two-stage baseline is [experiment
111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md)'s GAM-K with
the noise settings that [experiment
133](../00133-differentiable-physics-vk-learned-noise-and-observation-variance-in-gam-k/result.md)
chose by next-day score.

## Pre-registered design

- **Sizing, one line per trigger.** What gets stored: private fits and prediction sets only.
  Serving path: not touched. Degradation rule: not touched. More than one defensible design: yes,
  and the choices below are fixed here. Callers: this folder's scripts only. The size is complex
  because it changes how the demand model is fitted, so it gets the full process, with the Opus
  code review only if the ML manager agrees.
- **Rows and periods.** `fold_a_daily_gap_v1`'s 292 scored daily runs at D5, D10, D11, D14, and
  D16, with D9 beside them; D18 is excluded because it has no readings in the tuning months. The
  tuning, replicate, and scored periods are experiment 111's (odd months of 2024-07 to 2025-06,
  even months of the same year, and 2025-07-01 to 2026-06-30). Every fit uses rows that end no
  later than its monthly cut; the 24-month window, ridge, and pinned capacity are experiment 111's
  `gamk`.
- **The joint loss.** For a fit at cut `c`, the filter runs over the whole training window of
  half-hours (the stream of observed minus fitted power, including rows masked from the loss, as
  the two-stage filter sees them). The loss sums, over the window's days, the squared error of
  the corrected forecast at the day-ahead cases of that day (valid times 12 to 36 hours after
  the 09:00 UTC initialisation, with readings usable up to 3.5 hours before it, as
  `studies.dp_assimilation.usable_cutoff_positions` gives), on rows that the two-stage fit also
  trains on. The first 14 days of the window are left out of the loss, while the filter converges.
- **Noise settings are held fixed at the learned values of experiment 133 (GRID) for each
  substation**, so the experiment changes only the coefficient fit. Learning the noise and the
  coefficients together is a further step, taken only if J1 or J2 shows a gain.
- **Three biases, stated in advance.** First, in training the filter reads in-sample residuals
  from the training weather (ERA5 and CAMS), which are smaller and cleaner than the out-of-sample
  residuals from the analysis proxy that it reads at forecast time, so the joint fit may
  under-weight the correction's noise. A cross-fitted stream would remove this, but it makes the
  loss nonlinear in the coefficients, so the experiment cannot rule the joint design out on this
  point. Second, the noise settings were chosen on TS's own stitched out-of-sample residuals, which
  favours TS: if a joint arm shows a gain, the follow-up is a retune of the settings on that arm's
  own stream. Third, J2's ridge weight is a fourth selection, so J2 is a family, and the chosen
  weight is reported per substation. The replicate months and the scored year, which are out of
  sample for every arm, decide the verdict.
- **The filter's rows and mask.** The filter reads one stream per design column and one for the
  target, all on the same rows: those where the reading and every input are finite, from the
  first training row to the cut (`stream_rows`). Rows the fit masks out of its loss (switching
  events) stay in the filter's stream, as in TS. A training row is a loss case when it is a
  stream row, ends more than 14 days after the first stream row, ends no later than the cut, and
  has a usable reading before its initialisation's cutoff. The cases follow the scoring bands
  (valid times 12 to 36 hours after the 09:00 UTC initialisation, one case per half-hour), not
  `studies.dp_assimilation.day_ahead_initialisation`, whose window is 15.5 to 39 hours.
- **Penalties.** The loss is `A = Zc' Zc + penalty` and `b = Zc' zc - 0.5 * linear`, with `Zc` the
  corrected design matrix and `zc` the corrected target, and `penalty` and `linear` exactly
  those of `fit_gam` at the same strength (scaled by the number of training rows), so TS and the
  joint arms carry the same regularisation.
- **Spread.** Each arm has its own quantile spread, fitted on its own corrected ensemble forecasts
  of the tuning months with the same rule as experiment 111 (centred residual quantiles per lead
  band, hour band, and quarter, and a factor per lead band from {0.9, 1.0, 1.15, 1.3}), so the
  quantile scores compare like with like.

### Arms

| Arm | Fit | Filter at forecast time |
|---|---|---|
| TS (control) | Two-stage: GAM-K at each monthly cut, as experiment 111's `gamk` | Experiment 133's GRID settings on the stitched out-of-sample residuals |
| J1 | Joint fit with the filter in the loss, no tie to TS | The same settings and the same stitching |
| J2 | Joint fit plus a ridge towards TS's coefficients: `lambda x (theta - theta_TS)' G (theta - theta_TS)` with `G` the plain-fit Gram matrix and `lambda` in {0.1, 0.3, 1, 3}, chosen by the tuning months' quantile score | The same |
| FIX (reference) | Experiment 111's `gamk` with its fixed settings | As in experiment 111 |

### Planned contrasts

| Contrast | Meaning | Prediction |
|---|---|---|
| C1 | J1 minus TS | -0.05 to +0.02 points, most likely about zero |
| C2 | J2 minus TS | -0.03 to +0.01 points |

J2 minus J1 is descriptive.

### Controls and the gate

- **P1, reproduction (blocking).** TS's scored-year rows match experiment 133's GRID rows to
  within 1e-4 points in the median forecast. Until experiment 133 has chosen its values, the same
  control runs against experiment 111's stored rows with FIX settings, and is run again once
  GRID's values are in.
- **P2, the linear filter equals the filter (blocking, unit test).** On random streams with
  missing readings, the corrected loss from `studies.dp_joint_fit` equals the loss of
  `run_kalman` and `correction_at` on the residual stream for random coefficients, to a relative
  1e-9 in float64.
- **P3, the solved loss is the true loss (blocking).** For each fit, the loss of the filter run on
  the fitted residuals equals the quadratic form of the solved normal equations to a relative
  1e-8, and random feasible perturbations of the coefficients raise it.
- **P4, known answer.** On a synthetic stream with known shapes and a random-walk level, J1 recovers
  the shapes at least as well as TS, so the joint fit is shown to be able to win when the
  assumptions hold. Failing P4 is reported, and a joint arm that cannot win here has its real
  result read with caution.
- **Absorption control (blocking for `supported`).** The question is whether the fitted curves
  move to explain a change that the filter should track. To each decision substation's real
  training target, add (a) a level change of 2 points and (b) a PV-shaped change of 30% of the
  substation's pinned solar output (as a fall in net power), each over the last 180 days before the
  cut (where the filter has not yet caught up) and over a middle 180 days. Refit TS, J1, and the
  J2 arm with ridge weight 1, with and without the addition, at three cuts (2025-01, 2025-07,
  2026-01). A null refit with the cut moved back by 7 days sets each arm's noise floor of curve
  movement. The movement of a term is the largest absolute change of its fitted curve in points
  over the training rows, for the weekly profile, holiday, tent, harmonics, heating, and lighting
  terms. PV is pinned in this design, so no fitted PV curve exists and the PV-shaped change can
  only move the other terms. A joint arm fails when a term moves by more than the larger of 0.10
  points and twice its own noise floor, and by more than twice the larger of TS's movement and TS's
  noise floor in the same term, because a plain fit should move when the data really change. A term
  whose own noise floor exceeds 1 point is reported as unidentified. A smoke test of the controls on
  one cut of one substation, run before this page was finalised, found that J1's weekly-profile
  curve has a noise floor near 90 points: a constant shift of the profile is absorbed by the
  filter's level, so J1 cannot identify it, whereas J2's ridge removes the freedom (noise floor
  below 0.3 points).
- **Shape distance.** For every monthly cut, the root-mean-square difference between J1's (and
  J2's) fitted curves and TS's, per term, as a share of the root mean square of TS's curve of that
  term (the fitted curve in points over the training rows). Beside it, the condition number and
  effective degrees of freedom of the corrected Gram matrix against the plain one, because the
  slow columns (profile, tent, harmonics) have corrected columns near zero, so their solution is
  set by the ridge. These numbers feed the substation cards, which need stable shapes.
- **Overfitting guard.** The tuning-month gain of each arm beside its replicate gain and its
  scored-year gain, per substation.

### Reporting

- **Headline figure:** quantile score at 12 to 36 hours by arm with 97.5% month-resampled
  intervals, then every lead band beside it, per decision substation.
- **Non-inferiority:** the all-lead score (0.5 to 168 hours) of every contrast, and the lead bands
  of days 2 to 3 and days 4 to 7 separately, because the loss trains on day-ahead cases only.
- **Breakouts, descriptive:** winter (November to March), winter nights, cold nights (hour bands 0
  and 3 with ERA5 temperature at or below 3 degrees C), and the Christmas and bank-holiday
  periods.
- **Tail table:** coverage and pinball loss at 0.01, 0.05, 0.95, and 0.99, and the coverage of
  the 0.95 quantile on cold nights, for TS, J1, J2, and FIX.
- **Cost:** GPU hours and wall-clock time of the joint fit beside the two-stage fit.
- **Checks bullets:** median minus mean forecast per site, the mean-forecast error beside each
  quantile-score contrast, and which audit triggers fired.
- **Retraining schedule:** monthly refits of the convex model, as in experiment 111, so the
  contrast between TS and the joint arms does not depend on the schedule.

## Compute

One GPU job at a time under 24 GB, and `systemd-run --user --scope -p MemoryMax=16G -p
OOMPolicy=continue`. The corrected design matrix needs one filter pass per monthly cut and
substation over about 290 columns. The joint fits of all five joint arms (J1 and the four J2 ridge
weights) share that pass. A smoke test on one cut of one substation took about 8 seconds for three
arms, so the GPU time is expected to be well under 1 hour. The ENS forecasts take about 35 s per
arm and substation, on the CPU in at most 6 processes.
