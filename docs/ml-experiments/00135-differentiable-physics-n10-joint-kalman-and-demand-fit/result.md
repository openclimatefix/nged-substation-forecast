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
decision_rule: 'Two planned contrasts (C1 J1 minus TS, C2 J2 minus TS), each read at the 97.5%
  level of the month-resampled paired bootstrap (12 calendar months of the scored year, 20,000
  resamples), which is the Bonferroni level for two contrasts. A joint arm is `supported` when
  its contrast interval lies wholly below zero, at least 4 of the 5 decision substations
  improve in both the scored year and the replicate months, the score over all leads 0.5 to 168
  hours has an interval upper bound of at most +0.02, AND the arm passes the absorption control
  and the shape-stability bound (below). It is `small gain` when the interval lies wholly below
  zero but a further condition fails. It is `refuted` when the interval lies wholly above zero.
  Otherwise it is `inconclusive`. A joint arm that lowers the score and fails the absorption
  control is reported as an identification failure, not a gain, whatever its score. The
  threshold is zero, because a joint fit replaces the fitting routine inside the training loop
  and adds no model, data source, or serving step. The shape-stability bound is that the
  root-mean-square change of each fitted term (PV, heating, holiday, weekly profile) from TS is
  under 10% of the root mean square of the term itself. The verdict
  is capped at `inconclusive` if the reproduction control P1 fails.'
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
- **The in-sample caveat.** In training, the filter reads in-sample residuals, which are smaller
  than the out-of-sample residuals it reads at forecast time. The joint fit may therefore
  under-weight the correction's noise. This is a stated risk of the design, and the replicate
  months and the scored year, which are out of sample, decide the verdict.

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
- **P2, the linear filter equals the filter.** On random streams, the corrected design matrix
  built by filtering columns gives the same day-ahead corrections as `run_kalman` followed by
  `correction_at` applied to the stream of residuals for a random coefficient vector, to 1e-9.
  This tests the central idea of the design.
- **Absorption control (blocking for `supported`).** To each decision substation's real
  training-window target, add (a) a level change of 2 points on 180 days and (b) a PV-shaped
  change of 30% of the substation's midday PV effect over 180 days. Refit TS, J1, and J2 with
  and without the addition (at the 2025-07 cut), and report the change in each fitted term's
  curve (PV, heating, holiday, weekly profile) in points. An arm fails when its PV, heating, or
  holiday curve moves by more than 0.10 points at the peak AND by more than twice TS's movement;
  a shape that moves is a failure whatever the score does.
- **Shape distance.** For every monthly cut, the root-mean-square difference between J1's (and
  J2's) fitted curves and TS's, per term, as a share of the term's own root mean square. These
  numbers feed the substation cards, which need stable shapes.
- **Overfitting guard.** The tuning-month gain of each arm beside its replicate gain and its
  scored-year gain, per substation.

### Reporting

- **Headline figure:** quantile score at 12 to 36 hours by arm with 97.5% month-resampled
  intervals, then every lead band beside it, per decision substation.
- **Non-inferiority:** the all-lead score (0.5 to 168 hours) of every contrast.
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
substation over about 250 columns, which is a few seconds on the GPU, so the GPU time is expected
to be well under 1 hour. The ENS forecasts take about 35 s per arm and substation, on the CPU in
at most 6 processes.
