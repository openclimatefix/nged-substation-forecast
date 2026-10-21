---
id: "00128"
slug: differentiable-physics-kfd-shared-day-anomaly-and-sparse-q-in-gam-k
part: "KFD"
theme: differentiable-physics-forecast-engine
also_relevant_to:
  - lagged-power-and-error-correction
  - differentiable-physics-long-history-and-drift
  - differentiable-physics-embedded-generation
  - production-reference-and-best-guess
status: planned
size: >-
  complex, by the design trigger alone: more than one design is defensible (how the day-level state
  couples the hour bands, the sparse search for the process noise, the placebo), and the plan
  pre-registers one of each. What gets stored: private files under
  data/ml-experiments/00128-differentiable-physics-kfd-shared-day-anomaly-and-sparse-q-in-gam-k/
  only. Serving path: not touched. Degradation rule: not touched. Callers: this experiment's
  scripts only; packages/studies/ is not changed. The maintainer's brief sets the process (one
  Sonnet plan review, one Opus code review before the first run, Sonnet numbers and prose reviews).
hypothesis: >-
  Adding to experiment 111's level-and-anomaly filter (GAM-K) one day-level error state shared by all
  four hour bands, and estimating the level noise sparsely from zero, makes the filter's midday
  correction track an injected solar-panel-shaped change in real residuals better than GAM-K, and
  lowers the day-ahead score on ENS. Real residuals carry a day-level error shared across the four
  hour bands (correlation of the daily-mean residuals between bands 0.58 in real residuals against
  0.00 under the filter's own AR(1) model), which GAM-K's independent bands cannot represent.
prediction: see "Predictions"
decision_rule: see "Decision rule, smallest gain, and stop rule"
benchmark: >-
  GAM-K as stored by experiment 111 and reproduced by experiment 118 (2.521 points at D+1 over the
  five decision substations in the scored year); GAM-K with this experiment's own spread (GK-sp);
  KEZ (experiment 118) as a reference only; climatology; manual_heuristic
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles) in points of
  effective capacity, on rows outside the hand-filtered switching events, at day D+1 (leads 15.5 to
  39 hours from the 09:00 UTC power_fcst_init_time); every lead band beside it
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
code: ml-experiments/00128-differentiable-physics-kfd-shared-day-anomaly-and-sparse-q-in-gam-k/
---

# Differentiable physics, KFD: a shared day-level error state and a sparse process noise in GAM-K's filter

**This experiment asks whether one extra number per day, shared by every time of day, lets the
correction to a Primary substation's day-ahead demand forecast follow recent errors more closely.**
The forecast comes from a convex model (a physical demand model fitted by least squares) driven by
ECMWF ensemble weather forecasts. After each forecast, a Kalman filter (a method that updates an
estimate of the current forecast error each time a new reading arrives) corrects the next forecast
with the recent errors, separately for four times of day: night, morning, midday, and evening. The
filter treats the four times of day as unrelated. Real errors are not unrelated: at five substations
over one year, the daily averages of the error in two different times of day correlated at 0.58,
against 0.00 in the filter's own model of the error.

**The experiment adds a single day-level error that all four times of day share, and changes how
the filter decides which of its estimates may drift.** The second change keeps an estimate fixed
unless the data show that letting it drift raises the likelihood by a stated amount. We first inject
a known change, shaped like a rise in solar-panel capacity, into real recorded errors and check
whether each version of the filter finds the change. We then score the next-day forecasts of each
version over one year. No code has run: this page is the plan, and a later commit adds the results.

**The filter and the search for its noise re-implement published methods.** [de Vilmarest et al.
(2024)](https://doi.org/10.1109/TPWRS.2023.3310280) put a random walk on the weights of a frozen
additive model of regional electricity load and fitted a sparse, diagonal noise matrix by
likelihood. [de Vilmarest and Goude (2022)](https://doi.org/10.48550/arXiv.2110.00334) describe the
greedy search that starts every noise variance at zero, and the R package
[viking](https://CRAN.R-project.org/package=viking) implements it. The shared day-level state is the
experiment's own addition. The code builds on [experiment
111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md), whose monthly
convex fits, calibration, forecasts, and filter settings it reuses, and on [experiment
118](../00118-differentiable-physics-kfe-kalman-filter-on-the-gam-effects-on-ens/result.md)
(KFE), whose periods, rows, known-answer gate, and breakouts it copies.

## What KFD asks and the evidence it starts from

- **Experiment 118 found that a 14-weight filter did no better than GAM-K, and that its
  known-answer gate failed on real residuals.** The filter recovered an injected solar-panel
  capacity change in 85% to 95% of draws when the noise followed its own model, and in 16% to 21%
  of draws when the noise was resampled real residuals.
- **A follow-up diagnosis (private notes, 2026-10-21) traced the failure to the noise.** Shuffling
  the days of the real residuals failed the gate as badly as keeping their order. Masking the
  switching days did not help. The shared day-level error remained: the daily means of the four
  hour bands correlated at 0.58 in real residuals and 0.00 in the AR(1) noise, so the weights
  absorbed the shared error.
- **The same diagnosis found that estimating the noise sparsely fixes false movement when the model
  is right.** Holding a variance at exactly zero unless a single-weight likelihood gain of 2 nats
  justified it raised the share of clean AR(1) draws with no false movement from 5% to 90%. On
  real month-block residuals it did not help alone.
- **Experiments 106 and 122 credit most of GAM-K's gain to its level state.** The level carries the
  whole day-ahead gain in experiment 106, so any added state has to improve on the level.

## Arms

Every arm runs on experiment 111's stream: observed power minus the monthly convex fits' prediction
on the analysis proxy's weather, with the same frozen convex model, ensemble rows, and calibration.
The filter reads residuals at least 3.5 hours old and has no switching mask. Hyperparameters are
fitted on the odd months of 2024-07 to 2025-06 only, as in experiment 111, and the filter runs warm
through the other months.

| Arm | Filter state | Process noise of the levels | Role |
|---|---|---|---|
| GK | level and AR(1) anomaly per hour band | one shared variance, experiment 111's fit | benchmark, reproduced |
| KD | GK plus one shared day-level state d | one shared variance, refitted | planned |
| KS | GK | one variance per band, sparse from zero | planned |
| KDS | GK plus d | one variance per band, sparse from zero | hypothesis arm |
| KDS-shuf | KDS | KDS's | placebo, must not help |
| KDS-nod | KDS | KDS's | exploratory: d removed from the forecast only |
| KDS-1n, KDS-4n | GK plus d | sparse, thresholds of 1 and 4 nats | sensitivity, exploratory |

The monthly weight reset, the 14-weight state, noise pooling across substations, and any PV or
effect weight are not in the experiment. Experiment 127 stays parked.

### The shared day-level state d

**The state vector has nine entries: the four band levels, the four band anomalies, and d.** A
reading in band b at half-hour t is `r = level_b + z_b + d + eps`, in points of effective capacity.
The levels follow a random walk with variance `q` per day. The anomalies `z_b` are AR(1) with
factor `phi` per half-hour and innovation standard deviation `sigma_z`, as in GAM-K. Observation
noise has standard deviation `sigma_eps`. The joint filter runs over all readings in time order, so
every band's reading updates d, and d's estimate updates the other bands.

**The state d is constant within a local day and follows an AR(1) across days.** The day is the
local calendar day, because the hour bands are local hours, so each day holds all four bands once.
At each local midnight d becomes `phi_d * d + eta`, where `eta` has standard deviation `sigma_d`.
Across a gap that crosses `m` midnights, d decays by `phi_d**m` and its variance moves towards the
stationary value `sigma_d**2 / (1 - phi_d**2)`, with the covariance between d and every other state
scaled by `phi_d**m`.

**Initialisation.** The levels start at 0 with a variance of 100 points squared (GAM-K's value),
each `z_b` at 0 with its stationary variance, and d at 0 with its stationary variance, all
independent. A fitted month with missing readings lets d decay towards its stationary distribution.

**Holding d at forecast time.** A forecast uses the state after the newest reading at least 3.5
hours before the origin. The correction for a target half-hour in band b on local day `k_t`, with
the cutoff on local day `k_c`, is `level_b + phi**h * z_b + phi_d**(k_t - k_c) * d`, where `h` is
the number of half-hours from the cutoff to the target. The same day (`k_t = k_c`) therefore uses
the filtered d without decay, and the next day uses `phi_d * d`.

**Identifiability.** The levels and d both shift all bands at once, so the plan limits d to the
fast part. The bounds `phi_d <= 0.90` (half-life at most 6.6 days) make any shift lasting more than
about two weeks land in the levels, and d cannot carry a permanent level (its stationary standard
deviation is `sigma_d / sqrt(1 - phi_d**2)`). The anomalies `z_b` are band-specific, and d is the
only state that loads on all four bands, so only the cross-band covariance of the residuals
identifies `sigma_d`. With no cross-band covariance the likelihood drives `sigma_d` to its floor
and d becomes inert, which makes KD equal GK. Control P-ident checks this on simulated data.

**Hyperparameters of KD.** Six unconstrained values per substation: log `q`, logit `phi`, log
`sigma_z`, log `sigma_eps` (bounds and normal prior shared across substations exactly as in
experiment 111, one round of fitting), logit `phi_d` (bounds -3 to 2.2, so `phi_d` runs 0.05 to
0.90) and log `sigma_d` (bounds log 0.02 to log 6 points). The two new values have weak normal
priors centred on `phi_d = 0.5` and `sigma_d = 1` point, each with a spread of 1.5. The optimiser
starts at experiment 111's four values and those two. The likelihood is the one-step marginal
likelihood of the tuning months' readings, as for GAM-K.

### Sparse process noise

**KS and KDS give each band its own level variance `q_b`, set to exactly zero unless the data
justify it.** The search starts with every `q_b = 0`. For each band still at zero, the single-weight
gain is the largest increase in the log likelihood over a grid of `q_b = 0.001 * 2**k` points
squared per day (k from 0 to 11, so 0.001 to 2.0, steps of a factor of 2 as in the viking grid)
with the other bands held. The band with the largest gain enters if its gain is at least the
threshold of 2 nats. The gains are then recomputed and the search repeats until no band clears the
threshold. A final sweep re-optimises each active `q_b` on the grid, and removes an active band
whose removal costs less than the threshold.

**The estimator uses a flat prior on `q_b` and the one-step likelihood.** The diagnosis found that
the shared prior on log `q` inflates variances that should be zero. Every other hyperparameter is
held: KS holds `phi`, `sigma_z`, and `sigma_eps` at experiment 111's values, and KDS holds all five
of KD's other values. The thresholds of 1 and 4 nats are sensitivity runs; they report the selected
supports and the gate, and KDS-1n and KDS-4n are scored as exploratory arms. A post hoc refit of the
held values given the chosen support (KS-refit) is allowed and labelled as such.

### The placebo and the exploratory arm

**KDS-shuf keeps KDS's filter and replaces the d term of the forecast with the d of another day.**
For each forecast, the d term is taken from the cutoff state (same clock time, 05:30 UTC) of a day
drawn by a fixed permutation (seed 128) of the local days of the same calendar month. The levels
and anomalies stay those of KDS. The permutation keeps the size of d and removes its link to the
errors of the target day. KDS-shuf must not beat KDS, and must not beat GK by a gain like KDS's.
KDS-nod drops the d term from the forecast and keeps the levels and anomalies that KDS's joint
filter produced. The pair separates two roles of d: tidying the levels, and predicting the next
day. Both arms refit nothing.

## The known-answer gate on real residuals

**The gate runs after the hyperparameters are fitted on the tuning months and before any replicate
or scored forecast.** It copies the construction of experiment 118's gate (`e00118_gate.py`) for the
five decision substations over 2024-07-01 to 2025-06-30, and changes only what the arms need.

**How a solar-panel capacity change shows in GAM-K's state.** GAM-K has no weight on the solar
term. A rise in capacity makes the residual (observed minus forecast) more negative in the daylight
hours, so the change is absorbed by the midday band's level, with the AR(1) anomaly taking the fast
part. The tracked quantity is therefore the midday band's level at the end of each local day, plus
d for the arms that have it, because d is part of the midday correction. For GK and KS it is the
level alone. The level alone is reported beside it for every arm.

**Scenarios.** S0 injects nothing. S1 injects experiment 118's change: the PV weight rises from 1
to 1.30 over 180 days from 2024-07-01, so the stream gains `deviation * pv_regressor` at each
half-hour, with the analysis proxy's PV regressor at weight 1 (the `exact` condition; the arms use
no regressors, so the weather-error condition has nothing to test). S1x3 injects three times the
change, the size of a new solar farm connecting at 90% of the existing capacity. Noise comes in two
forms. The real form is a circular month-block resample of the substation's real tuning-year
residuals, which keeps the cross-band structure and is the decision form. The AR(1) form is
experiment 118's, at each substation's GAM-K `phi`, `sigma_z`, and `sigma_eps`, and is the
reference. Seeds depend on the site, scenario, form, and draw, with the namespace `e128`, so no
draw repeats an injection of experiment 118.

**The truth path** is `I_k = deviation_k * R_k`, where `R_k` is the mean of the PV regressor over
the midday-band half-hours of the 30 days up to local day `k`. Because the injected change starts in
July and the regressor falls in winter, `I_k` peaks in autumn. The read day `k*` is the day between
30 and 200 on which `I_k` peaks, computed per substation from the regressors before any draw runs.

**Draws and fits.** Each draw refits every arm's hyperparameters on the synthetic stream, starting
from the real fit and with the real fit's priors, as experiment 118 did. The real-form cells hold 12
draws per substation for S0, S1, and S1x3 (60 per cell), and the AR(1) cells hold 6 per substation
for S0 and S1 (30 per cell). A cell reports the share of draws that pass, with a 95% binomial
half-width. The bar is experiment 118's: a reading passes when at least 80% of draws pass.

**Readings** (all four must pass for KDS in the real form at the decision amplitude to clear the
gate):

1. **Tracking (S1, S1x3).** The tracked quantity minus the mean of the same arm's S0 draws at the
   same substation recovers at least 70% of `I_{k*}` on day `k*`, and its daily correlation with
   `I_k` over days 1 to 200 is at least 0.70.
2. **No cross-talk (S1, S1x3).** The night band's level and d (arm by arm), each minus its S0 mean,
   stay below 25% of the peak of `I_k` in absolute value after day 60.
3. **No false movement (S0).** The median over draws of the largest change of the tracked quantity
   after day 60, in units of the peak of `I_k`, is at most 1.25 times GAM-K's. Experiment 118's
   absolute bar of 0.10 does not apply to a level, which follows real residuals by design.
4. **Forecast error follows (S1, S1x3).** The arm's mean absolute day-ahead error on the stream
   after day 60 is below GAM-K's in at least 80% of draws.

**Power check, before the gate.** For each substation, the peak of `I_k` is divided by the posterior
standard deviation of the tracked quantity on day `k*` from KD's filter on the real tuning stream.
If it is below 3 at 2 or more of the 5 decision substations at S1, S1 is labelled underpowered (as in
experiment 118), the decision amplitude becomes S1x3, and the page says so. The reading at S1 is
reported regardless.

**Gate outcome.** The gate passes when KDS passes readings 1 to 4 in the real form at the decision
amplitude. A failed gate caps the verdict at `inconclusive`, as in experiment 118, and the replicate
and scored forecasts still run, because the maintainer's brief asks for the day-ahead score either
way. The tracking rates of all four arms in both noise forms are reported beside the gate. Experiment
118's weight-based rates (90% in the AR(1) form, 20% to 25% on real residuals) are a different
quantity, so they are quoted for context and not as a bar. A gate pass that follows a change of the
gate after seeing draws does not count: the revised test must be written down and rerun on new draws.

## Scoring

**Periods and rows are experiment 118's.** The row set is `fold_a_daily_gap_v1`: the tuning months
(odd months of 2024-07 to 2025-06) fit the hyperparameters and tune each arm's spread factors, the
replicate months (even months) give the sign of each contrast, and the scored year is 292 daily
runs (2025-07-02 to 2026-06-29). The decision substations are D5, D10, D11, D14, and D16, with D9
beside them and not counted. Every score is the quantile score at day D+1 over rows outside the
hand-filtered switching events, in points, with a smaller score better. Each arm's spread is the
residual-quantile spread of experiment 118's procedure, fitted on the tuning months and centred on
its median. GK also appears with experiment 111's stored spread (the reproduction row) and with
this procedure (GK-sp).

**Planned contrasts** (99% intervals from resampling whole calendar months, paired across arms;
Bonferroni across the five contrasts; "k of 5" counts decision substations where the first-named arm
scores better):

| Contrast | Question |
|---|---|
| C1 KDS minus GK | does the shared day state plus sparse noise lower the day-ahead score? |
| C2 KD minus GK | does the shared day state alone lower it? |
| C3 KS minus GK | does sparse noise alone lower it? |
| C4 KDS minus KD | does sparse noise add to the day state? |
| C5 KDS minus KS | does the day state add to sparse noise? |

Beside C1 to C5 the page reports the spread-matched contrast KDS minus GK-sp, GK-sp minus GK, the
placebo contrast KDS-shuf minus KDS, KDS-nod minus KDS, and the exploratory KDS-1n and KDS-4n minus
KDS (95%, labelled exploratory).

**Every contrast comes with** its replicate-months value for the sign; its value with each
decision substation left out in turn; the same contrast on the error of the mean forecast (the
absolute error of the mean of the ensemble-mixture forecast); the median forecast minus the mean
forecast per substation; and the breakouts below.

**Breakouts**, each with its own point estimate and an interval only where there are enough days:

- winter (November to March), winter nights (00:00 to 06:00 local), and winter evening peaks (16:00
  to 20:00 local), scored year and replicate months;
- cold-snap days (a November to March day whose mean ERA5 temperature is below the 10th percentile
  of such days from 2019-09 to 2024-06 at the substation) and cold winter nights (22:00 to 08:00
  local after a cold-snap day), as descriptive per-substation scores because there are 2 to 14 such
  days per substation;
- the 17 holiday and Christmas days of the scored year and the day after each, with a
  non-inferiority check on all other days at a margin of 0.02 points, because the shared state could
  carry a holiday's error into the next day;
- all labels are D-labels, and no generator appears on the page.

**Tails.** A table of coverage and pinball loss at q01, q05, q95, and q99 at D+1 for GK, KD, KS,
and KDS, as in experiment 118. For curtailment, the tails on the reverse-flow side matter most: the
q01 and q05 at midday from April to September (where demand is positive and reverse flow is
negative), and the q95 and q99 on winter evening peaks. Both are reported separately. Curtailment
cost is not scored, because the decision substations are demand Primaries with no node-plus-generator
set in the experiment's inputs.

**The substation cards** show, for each decision substation, KD's and KDS's fitted `phi_d`,
`sigma_d`, and `q_b`, the selected support at 1, 2, and 4 nats, the filtered d over the scored year,
and the same four example weeks as experiment 118 (the weeks beginning 2025-07-14, 2025-10-13,
2026-01-12, and 2026-04-13), GK and KDS side by side.

## Predictions

**These predictions are written before any fit runs, and each one can fail.** Scores are in points
at D+1 in the scored year; negative favours the first-named arm.

| Quantity | Prediction |
|---|---|
| C1 KDS minus GK | -0.04 (range -0.09 to 0.00); better at 4 or 5 of 5 substations; same sign in the replicate months |
| C2 KD minus GK | -0.03 (-0.07 to +0.01); better at 3 or more of 5 |
| C3 KS minus GK | within +-0.02 of zero |
| C4 KDS minus KD | -0.01 (-0.03 to +0.01) |
| C5 KDS minus KS | -0.04 (-0.09 to 0.00) |
| Winter nights, KDS minus GK | -0.06 (-0.15 to 0.00), a larger gain than the annual C1 |
| Holiday days, KDS minus GK | not worse than +0.30 on the 17 holiday days |
| Tails | coverage at q01, q05, q95, and q99 within 1 percentage point of GK's |
| Fitted d | `sigma_d` above its floor of 0.02 points at 5 of 5 decision substations; `phi_d` between 0.4 and 0.8; `sigma_z` and `sigma_eps` each lower than GAM-K's by at least 5% |
| Sparse support at 2 nats | at least 3 of the 4 bands keep a nonzero `q_b` at 4 or more of 5 substations |
| Gate, tracking, real form, decision amplitude | KDS 60% of draws (range 40% to 80%), so the 80% bar is likely to be missed; GK 45% (25% to 65%); KDS minus GK +0.15 (+0.05 to +0.30) |
| Gate, tracking, AR(1) form | KDS within 0.10 of GK, both at least 70% |
| Gate, S0 movement | KDS at most 1.25 times GK's |
| Placebo | KDS-shuf minus KDS +0.03 to +0.15, and KDS-shuf minus GK at least -0.01 |

**The mean-forecast error is predicted to agree in sign with every quantile-score contrast.** If a
contrast and its mean-forecast error disagree, audit trigger 3 fires.

## Decision rule, smallest gain, and stop rule

**The verdict is C1's label, capped by the gate and the placebo.**

- `supported`: the scored-year 99% interval of C1 lies wholly below zero, its estimate is at least
  0.03 points below zero, at least 4 of 5 decision substations are better in both the scored year
  and the replicate months, the placebo passes, and the gate shows KDS tracking at least 0.10
  higher than GAM-K's in the real form.
- `small gain`: the C1 interval lies wholly below zero in the predicted direction, but the estimate
  is under 0.03 points in size.
- `not supported`: the C1 interval lies above -0.03 (the data exclude a gain worth porting).
- `refuted`: the C1 interval lies wholly above zero in the scored year and the replicate months.
- `inconclusive`: every other result, and any result that would be `supported` or `small gain` but
  has a failed gate, as in experiment 118. A failed placebo (KDS-shuf minus KDS below +0.01, or
  KDS-shuf better than GK by more than half of C1) also gives `inconclusive` and triggers an audit.

**The smallest gain worth porting is 0.03 points at D+1, about 1.2% of GAM-K's 2.521.** The change
adds one state and two fitted hyperparameters per substation to the recent-level filter that sits
in the serving path, and a port also needs a rule for a band with no reading (the joint filter
skips it, and d keeps its last value and decays). No new data and no new pipeline step are needed.
The 0.03 matches the half-width of a paired scored-year interval at five substations in experiment
118 (C1's interval ran -0.013 to +0.048), so a smaller gain could not be told apart from zero.

**Stop rules.**

- Stop and report, without running the gate, if control P1 (nesting) or P-ident fails, because
  either is a coding fault.
- If KD's `sigma_d` sits at its floor at 4 or more of the 5 decision substations, d is inert and KD
  equals GK. Record the result as `not supported` for the day state, drop KD, KDS, and KDS-shuf, and
  score KS against GK only.
- If KDS tracks worse than GK by more than 0.10 in the AR(1) form, the added state damages tracking
  under the filter's own model. Stop and report before any scored forecast.
- No arm is added after a replicate or scored result is read, except one labelled post hoc.

## Controls

- **P1, nesting.** KD with `sigma_d` set to 0 (outside the fitted bounds, for this control only),
  and KS with every `q_b` set to GAM-K's `q`,
  reproduce GAM-K's states to 1e-9 on the tuning stream, and the quantiles of GK equal experiment
  111's stored files exactly on the rows experiment 118 reproduced.
- **P2, causality.** Changing every reading after the cutoff leaves the joint filter's state and the
  forecasts unchanged.
- **P-ident.** On streams simulated from the KD model with a known `sigma_d` and `phi_d` (30 draws per
  substation), the fit recovers `sigma_d` within a factor of 2 in at least 80% of draws. On streams
  with no cross-band covariance it returns `sigma_d` at its floor. A permanent common level step
  of 3 points, held for 60 days, ends mostly in the levels (d below 25% of the step after 30 days).
- **P-sparse.** On AR(1) streams with true `q_b = 0` (30 draws per substation), a band keeps a
  nonzero `q_b` at 2 nats in at most 10% of draws; on streams with a true `q_b` of 0.05, the band is
  kept in at least 80% of draws.
- **The audit triggers** of the `ml-experiment` skill are checked on the results and listed in the
  write-up's Checks section.

## Earlier advice

**Followed:**

- Keep experiment 111's level-and-anomaly filter as the benchmark and feed it ensemble-driven
  (analysis proxy) residuals ([experiment
  111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md), C4).
- Read residuals up to the telemetry delay of 3.5 hours, and test the speed of a new filter state
  with an injected change before scoring it ([experiment
  115](../00115-differentiable-physics-w1-diurnal-shape-state-on-ens/result.md), C3 and gate test 1).
- Do not let a PV weight move day to day, set no holiday-day correction to zero, and test on
  resampled real residuals before reading a state as a measurement (experiment 118's advice).
- Do not re-run a failed gate on the same injections after changing the test; write the revised test
  first and use new draws ([experiment
  108](../00108-differentiable-physics-shared-anomaly-factor-across-substations/result.md)).
- Run a placebo of the same size beside a new term (experiment 114), score calendar-event days and
  non-inferiority elsewhere (experiment 117), and report winter beside the annual score (experiment
  119).
- Report each contrast beside its leave-one-substation-out contrasts (experiment 122).
- Data processing: the 3.5-hour reading delay, switching masks applied in scoring and not in the
  filter, and spread quantiles centred on their median
  ([data processing](../data-processing.md)).

**Departed from:** experiment 118 advised fitting a filter's noise by a day-ahead likelihood with a
wide bound on log `q`, but its two changes were made together, and the gate diagnosis found the
one-step likelihood with a flat prior suits the sparse search. KD, KS, and KDS use the one-step
likelihood, as GAM-K does, so that the contrasts differ in the state and the noise search only. A
day-ahead likelihood is left for a later arm.

## Compute, storage, and size

- **What gets stored.** Private files under `data/ml-experiments/00128-.../`: fitted hyperparameters,
  gate draws, filter states, and predictions (written with `experiment_store.write_predictions`).
  Figures and the write-up go under `docs/ml-experiments/00128-.../`. No Delta table or contract
  changes.
- **Serving path.** Not touched. The joint filter lives in the experiment folder
  (`e00128_filter.py`), and `packages/studies/` is not changed. A port would touch the recent-level
  filter of the serving path and would be its own issue.
- **Degradation rule.** Not touched. The joint filter skips a missing reading, and d decays across
  the gap, so a port keeps the always-output path.
- **More than one design is defensible.** Yes: the day could be local or UTC, d could couple only
  some bands, the sparse search could refit the held values, and the placebo could shuffle the feed
  instead of the forecast term. The plan fixes one of each.
- **Callers.** The experiment's own scripts only. They import `studies.dp_assimilation` (for
  `FilterHyperparameters` and the 111 filter) and copy experiment 118's stream, gate, and scoring
  helpers into `e00128_*.py` files, because a script may import only its own folder.
- **Resources.** CPU only, at most 6 cores, each command under `systemd-run --user --scope -p
  MemoryMax=16G -p OOMPolicy=continue`, `free -g` first (25 GB available or wait), 6 worker
  processes with 1 thread each. Estimates, to be replaced by one timed filter pass before any other
  step runs: one joint-filter likelihood evaluation over the tuning stream takes 0.3 to 0.8 s, so
  fitting KD at six substations takes 5 to 10 minutes of wall time; KS and KDS take 10 to 20 minutes
  each; the power check and controls take 20 to 40 minutes; the gate (180 synthetic streams and 3
  refits each) takes 3 to 7 hours of wall time and dominates, against 3.4 hours for experiment 118's
  1,600 draws of a heavier fit; tuning the spreads and forecasting the replicate months and scored
  year take 15 to 30 minutes, since experiment 118's tune and forecast steps each took under 3
  minutes per substation group; scoring, figures, and the substation cards take 20 minutes. If one
  gate refit takes more than 4 minutes once timed, the draws fall to 8 per substation in the real
  form and 4 in the AR(1) form, and the write-up says so.

## Order of work

1. Write `e00128_*.py` in the experiment folder: the joint filter and its likelihood, the sparse
   search, the forecast hooks, copied helpers from experiment 118 (streams, gate, scoring), and the
   reports. Time one filter pass.
2. Run the controls P1, P2, P-ident, and P-sparse. Stop on a failure of P1 or P-ident.
3. Fit KD, KS, and KDS on the tuning months, and apply the stop rule on `sigma_d`.
4. Run the power check and the gate. Report it before any replicate or scored forecast.
5. Tune the spreads on the tuning months, then forecast the replicate months and the scored year.
6. Score the contrasts, breakouts, tails, and checks, draw the figures, and write the results.

The maintainer's brief asks for one Sonnet plan review of this page, one Opus review of the code
before step 2, and Sonnet reviews of the numbers and the prose before the pull request.

## Open questions

- **The gate's tracking reading may be underpowered at the real amplitude.** A level sees only the
  band mean of the injected change, and the change peaks in autumn. The plan answers with a power
  check and an S1x3 row, and a different ramp (for example one starting in January) would depart
  from experiment 118's S1.
- **The 80% bar is inherited from a weight-based reading.** Whether it suits a level-based reading
  is untested, and the plan predicts KDS misses it in the real form.
- **The placebo shuffles the d term of the forecast, not the feed of the filter.** A shuffled feed
  would need a second filter run on reordered days that cannot be aligned to the real forecast
  times.
- **KS holds `phi`, `sigma_z`, and `sigma_eps` at GAM-K's values,** which were fitted with a shared
  `q`. The post hoc refit (KS-refit) is the check.
- **Whether `sigma_d` and `phi_d` need the weak priors.** Their spread of 1.5 was chosen without
  data, and the fitted values will show whether the priors bind.
