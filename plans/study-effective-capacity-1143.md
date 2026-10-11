# Plan: is a time-varying effective capacity worth estimating for solar forecasts? (issue #1143)

## Problem

**Every solar forecast in this repository divides power by one fixed number per plant, so nothing
tells the forecast when a plant loses part of its capacity.** The fixed number is the 99th
percentile (P99) of the plant's metered output over its whole history, from the `effective_capacity`
table. The lag-features study ([PR #1139](https://github.com/openclimatefix/nged-substation-forecast/pull/1139))
found one of its 6 solar plants producing two thirds to three quarters of its neighbours' output in
3 of 18 months (December 2024, March 2025, and April 2026; cause unknown). The baseline XGBoost
model learned those months through its day-of-year column and then under-forecast a normal March
2026 by 14.6 points of capacity. The roadmap's answer to that kind of level shift is a two-pass
scheme: estimate each plant's effective capacity at every time step, divide power by that estimate
before training, and multiply the forecast back. Issue #1143 asks two questions. Is an estimator of
time-varying effective capacity worth building for the forecast? And if so, which techniques look
most promising? Nothing in the repository measures either.

## Planned solution

**A two-phase study on the lag-features study's rows: hold one XGBoost forecaster fixed, and change
only the capacity series it is normalised by.** About 20 estimators, from a rolling P99 to a
convex envelope fit and a PyTorch differentiable-physics model, each produce a daily capacity series
per plant in two versions. The causal version, which is the headline, uses only data available
before each forecast's issue time, as a live service would. The hindsight version uses the whole
record and is an upper bound. Every forecast is multiplied back to megawatts and scored against the
same fixed P99 per plant, so a time-varying denominator cannot shrink its own error. Phase 1 is a
broad, shallow screen of every estimator, scored on all 18 months and labelled exploratory. Phase 2
runs five planned paired contrasts, fixed in this plan, on the estimators a nested selection rule
picks, with the second hyperparameter setting, quantile models, a planted-capacity-loss positive
control, and two negative controls. Each estimator is also scored directly on how well it recovers
a planted loss, which is the evidence for the `effective_capacity` delivery table regardless of the
forecast result.

## Verdict, size and departures

**Verdict: worth doing.** The roadmap names this head-to-head as the v0.7 decision
([capacity estimation](https://openclimatefix.github.io/nged-substation-forecast/roadmap/capacity-estimation/)),
and the lag study supplies a real level-shift case to test it on. The study can answer the forecast
question with the data already on disk, and needs no new data download.

**Size: complex.** The five triggers:

- **What gets stored:** a published `docs/studies/` page and files under `data/studies/`; no Patito
  model, Delta table, or asset. Fires under the `study` skill's rule that a published page counts as
  stored.
- **Production serving path:** not touched. Nothing under `src/` or any package other than
  `packages/studies/` changes. The page describes how an estimator would map onto the serving path,
  without building that mapping.
- **Degradation rule:** none touched.
- **More than one defensible design:** yes. The causal timing, the hindsight definition, the
  selection rule, the scoring denominator, and the planted-loss design each admit several designs.
- **Callers not nameable without searching:** yes, narrowly. The plan adds one optional
  keyword-only parameter, `target_scale`, to `studies.cross_validation.out_of_fold_losses`, which
  many study scripts call. The default leaves every caller's behaviour unchanged, and a bit-for-bit
  reproduction check proves the default path unchanged.

**Reviews the size buys:** both plan reviews (the simplicity review, then the correctness review),
not yet run, because this plan was written by a planner told not to dispatch sub-agents. Then the
`study` skill's sequence: a fresh code review of every script before its first run, two Opus
scientific-validity reviews, the diff review, the mutation pass (because `packages/studies/`
changes), and the first-stumble, prose, persona, and evidence reviews of the page and figure text.

**Departures from the issue body and the roadmap:**

1. **The reference arm is the lag study's baseline B0, not a separate "static capacity" arm.** For
   an XGBoost model fitted to one plant, dividing the target by a constant and multiplying the
   forecast back changes the forecast only through XGBoost's regularisation terms. So "static
   capacity" and B0 are the same forecast to within that effect. The plan fits the constant-scale arm
   anyway, as negative control NC1, to measure the regularisation effect.
2. **The convex estimators are solved as linear programs with `scipy.optimize.linprog` (HiGHS), not
   with CVXPY.** A pinball loss plus a fused-lasso penalty plus the clipping hinge is a linear
   program, so `scipy` solves it exactly. `scipy` is already a `packages/studies` dependency, whereas
   `cvxpy` is not in `uv.lock`, and adding `cvxpy` is a dependency change that blocks the `study`
   skill's autonomous merge. The roadmap's secondary goal of hands-on CVXPY experience is therefore
   not served. Open decision 2 asks whether to add `cvxpy` anyway.
3. **SLAC's `solar-data-tools` is left out.** The package is not in `uv.lock` and would bring
   `cvxpy` and other solvers with it. Its capacity-change detection is a convex signal decomposition
   that the fused-lasso estimators approximate.
4. **Active network management (ANM) can separate curtailment from capacity loss at one plant only,
   and that plant is not the plant with the 3 low-output months.** The ANM section below gives the
   evidence. The ANM comparison is therefore exploratory, and no planned contrast depends on it.
5. **The planned "worth it" rule has an equivalence outcome.** The lag study's rule had no outcome
   for "any gain is smaller than the threshold", so that page had to read its intervals post hoc.
   This plan states all three outcomes before any result.

## The facts this plan rests on

Checked against the repository and the data store while writing this plan:

- **Rows.** `data/studies/per_study/lag_features/ens_mean/lag_frame_ens_mean_day1.parquet` holds the
  lag study's lead-day-1 rows with `fold`, `era_code`, the target `power_mw`,
  `effective_capacity_mw`, `cap_mw`, `constrained`, and B0's seven columns. The lag study scores
  29,445 rows, one per plant per daylight hour, over 18 calendar months (2024-12 to 2026-06 without
  2026-01), at 6 plants. B0's error is 8.93 points over the 18 months
  (`absolute_error_capped_fraction_of_capacity`, `METRIC` in `fit_lag_arms.py`). The scored months
  end before the final-test date, 2026-07-01, and `studies.power.scan_power` stops there.
- **The scoring denominator** is `effective_capacity_mw` from the `effective_capacity` table at
  version 1, committed on 2026-09-22 (the matched-lead page's Limitations section), a single P99 per
  plant over the plant's full history.
- **CAMS irradiance** for the 6 plants is on disk at
  `data/studies/downloads/reanalysis/CAMS/site_points/beam_diffuse_cams.parquet`, hourly, from
  2019-09-01 to 2026-09-11 (`studies.pv_dataset.read_cams`). Estimators therefore have up to 5 years
  of warm-up history before the first scored month.
- **The hindsight ceiling.** The lag study's second follow-up round rescaled B0's forecast in each
  plant-month to that month's measured energy, using the scored rows themselves. That rescale lowered
  the error by 0.337 points [−0.473, −0.215] over all 18 months, and by 0.416 points when the scale
  minimised each month's error (`report_followups2_ens_mean.md`, section 2, post hoc). No forecast
  could know that scale in advance, so these numbers bound what a monthly level correction can buy on
  these rows.
- **The low-output plant and its months** are recorded in
  `followups3/derated_plan.json`: the plant's output ratio to its neighbours was 0.68, 0.67, and
  0.71 in the 3 months, against a threshold of 0.8.
- **Existing machinery to reuse:** `studies.cross_validation` (`out_of_fold_losses`,
  `PRIMARY_HYPER_PARAMETERS`, `SENSITIVITY_HYPER_PARAMETERS`, `clamp_to_cap`, `crps`),
  `studies.bootstrap` (`bootstrap_difference_at_level`, `bootstrap_absolute`, `fold_t_interval`,
  `combine_setting_verdicts`), `studies.checkpointed_fits.fit_in_groups` and `choose_device`,
  `studies.pv_physics` (a forward model of a plant from hourly global irradiance),
  `studies.pv_fit.fit_plant` (the five-parameter fit with priors), `studies.baselines`
  (`issue_time`, `hourly_clear_sky`, `clear_sky_index`, `same_clock_hour_window`),
  `studies.pv_dataset` (`pv_sites`, `solar_hourly_power`, `drop_outages_and_spikes`, `read_cams`),
  `studies.commissioning.drop_commissioning_ramp`, `studies.export_cap.with_export_cap`,
  `studies.anonymise.site_labels_for`, and `studies.charts` (`interval_panel`,
  `leaderboard_panel`).
- **PyTorch** (`torch>=2.14.1`) is already in the optional `gpu` dependency group of
  `packages/studies/pyproject.toml`, so using PyTorch is not a dependency change. Installing PyTorch
  needs `uv sync --all-packages --group gpu` in the worktree, which sits on the home partition. CI
  does not install that group, so torch tests must open with `pytest.importorskip("torch")`, as
  `test_battery_state_space.py` does. **But every file that imports torch is listed in a
  `[[tool.ty.overrides]]` block of the root `pyproject.toml`**, so a new torch module means editing
  the root `pyproject.toml`. That file is outside the paths the `study` skill allows for an
  autonomous merge (open decision 1).
- **Hardware:** an RTX A6000 GPU, idle at planning time, and 32 logical CPU cores.

## Active network management: what the set-points can and cannot contribute

**The one generator with an ANM set-point export is one of the 6 study plants.** `pv_sites()` returns
6 plants, and the generator named by the export's filename is among them. NGED has confirmed that
generator is the only one in the trial area connected under ANM
(`data/studies/downloads/observations/NGED-ANM/README.md`), so the other 5 plants ran without an ANM
cap rather than with an unrecorded one.

**The ANM plant is not the plant with the 3 low-output months.** A planning-time check compared the
ANM generator's anonymised label with the label in `derated_plan.json`, and they differ. So ANM
curtailment cannot explain the motivating low months, and the cause of those months stays unknown:
an outage of part of the array, a metering fault, or curtailment of a kind NGED's ANM scheme does
not record. The plan and the page must not say which anonymised label the ANM plant carries,
because pairing a label with the one ANM generator in a published dataset would narrow down the
generator's identity.

**At the ANM plant, the cap moved off the connection limit on a large share of midday hours.** Over
the 19 months from December 2024 to June 2026, the half-hours between 09:00 and 15:00 UTC with the
lowest cap below the connection limit averaged 24% of each month, and exceeded 5% in 10 of the 19
months. That figure includes the 49-day zero-cap period of July to September 2025, whose hours the
multi-day-zero filter already drops. (Measured with a throwaway check at planning time; the build
script must print it into its report.)

**What ANM can contribute, then, is a known-cause test at one plant.** An ANM-blind estimator sees
capped hours as ordinary hours; an ANM-aware estimator either masks them or treats the cap as an
upper bound on export, `min(what the weather allowed, the cap)`, as the roadmap proposes. At that one
plant, the study compares the two versions of each shortlisted estimator:

- at the estimator level, how far the blind version's capacity series dips on days with a moved cap
  (false capacity loss), with the aware version as the reference;
- at the forecast level, the paired difference in error at that plant alone (exploratory, with a
  month-resampled interval at one plant).

**What ANM cannot contribute:** any evidence at the other 5 plants, any explanation of the 3
low-output months, and any planned contrast. The plan stays valid because the forecast rows already
exclude the capped hours from the targets (the lag study's cleaning), training already drops
`constrained` rows (`out_of_fold_losses`), and every estimator's default version fits an upper
envelope, which is the roadmap's structural defence against curtailment nobody logged.

## Design

### Rows, folds, and the forecaster (reused from the lag study)

- **Rows:** the lag study's 29,445 lead-day-1 rows, read from `lag_frame_ens_mean_day1.parquet`. The
  build asserts the row count and records the file's SHA-256 in the run manifest. Reading another
  study's data file is not an import, so the study-boundary rule holds. The same rows make B0's
  8.93 points directly comparable with the lag study.
- **Folds:** the frame's `fold` column, five folds of whole months per plant, cut within the
  matched-lead study's eras. No fold is recomputed.
- **Forecaster:** B0 exactly: seven columns (hour of day, day of year, era code, sun elevation and
  azimuth, and the ENS mean's irradiance and temperature at lead-day 1), the absolute-error
  objective, no column subsampling, three seeds, `PRIMARY_HYPER_PARAMETERS`. Every normalised arm
  has the same seven columns, so the equal-column-count rule holds by construction.
- **Reproduction check:** B0 refitted on the GPU through the new code path with `target_scale=None`
  must reproduce the lag study's B0 per-row losses bit for bit (the lag study showed a GPU refit
  reproduces bit for bit). The script raises if it does not.
- **Final-test cutoff:** every estimator reads power through `studies.power.scan_power`, which stops
  at 2026-07-01, and the hindsight estimators therefore cannot see past that date either. The build
  asserts that no estimator input is at or after the date.

### Estimator inputs

- **Power:** each plant's hourly power from `solar_hourly_power`, cleaned by
  `drop_outages_and_spikes` and `drop_commissioning_ramp`, from the start of each plant's record. The
  export cap is joined with `with_export_cap` and kept as a column, so the ANM-aware versions can use
  the cap while the ANM-blind versions ignore it. NGED's timestamps are already corrected at ingest
  and are not shifted again.
- **Irradiance:** CAMS global horizontal irradiance at each plant (the default for every
  physics-based or ratio-based estimator, as the roadmap recommends), a clear-sky model
  (`hourly_clear_sky`) for the envelope estimators, and, for the shortlist only, the ENS mean's
  lead-day-0 forecast of past hours, which a live service holds without a CAMS feed (exploratory; the
  roadmap's design caveat on whether capacity estimation can depend on ECMWF alone).
- **Orientation:** each plant's tilt and azimuth are fitted once with `pv_fit.fit_plant`, on data
  before 2024-12-01, so the fitted orientation uses no scored month. The ANM plant's settled record
  starts in October 2024, so its orientation rests on about 2 months; the build report says so.
- **Daily resolution:** every estimator outputs one relative capacity value per plant per day (and
  a standard deviation where the method has one). Daily blocks follow the roadmap's piecewise-constant
  prior and keep each solve small.

### Causal and hindsight versions

**The causal version is the headline.** A forecast issued at 09:00 UTC on day D for target day D+1
(`baselines.issue_time`) uses the capacity estimate built from whole days up to and including D−1:
power, CAMS, and neighbours' power alike. The estimate is held constant across the target day. CAMS's
latency was 2 days before February 2026 and 1 day since; the causal version reads CAMS through D−1
throughout, as the lag study's PC arm did, and the shortlisted estimator is re-run with a 2-day CAMS
wait as an exploratory check. Every training row is normalised by the causal estimate at its own
issue day, so training and scoring use the same estimate a live service would have stored as it
went. The causal estimate for a training row can read power from a test month that came earlier.
That does not leak the test month's answer into the XGBoost model, for the lag study's reason: the
model is trained to predict each training row's own power.

**The hindsight version is an upper bound and is labelled so everywhere.** It is a two-sided
estimate fitted on the whole record up to 2026-07-01, with one change: the week holding each scored
target is withheld from the estimator's input, and the estimate for that week is read from the
fit across the gap. Without the withholding, a daily-block estimator would read the target day's own
output and hand the answer to the forecast. The hindsight version uses the same week-withheld
estimate for training rows.

**Fallbacks are counted, never hidden.** Where an estimator has too little history (fewer than 30
valid days, where a valid day holds at least half its clear-sky energy, the rule
`baselines.clear_sky_index` uses), the scale falls back to the static P99. The scale is floored at
20% of the static P99, so a near-zero estimate cannot blow up the normalised target. The build
report prints the share of rows on each fallback, per estimator and plant.

### Fair scoring: one fixed unit for every arm

**Every arm's forecast is multiplied back to megawatts before it is scored, and every error is then
divided by the same fixed P99 per plant.** The arm's estimate never enters the denominator. The
reason is that a time-varying denominator can flatter itself. Suppose an estimator inflates its
capacity on the days its forecast is worst. Scored in normalised units, `|y/ĉ − ŷ/ĉ|`, each of those
days' errors shrinks by the same factor, and the arm looks better without forecasting any better.
In megawatts, `|y − ŷ|`, the inflation cancels, because ŷ is the normalised forecast times the same
ĉ. Dividing every arm's megawatt error by one constant per plant keeps the lag study's units
(percentage points of the plant's static P99), weights the 6 plants equally per unit of capacity,
and leaves every arm on identical denominators row by row. The scale lives in a new column,
`capacity_scale_mw`, and `effective_capacity_mw` stays the static value that `_losses` divides by.
NC2 (below) tests this property: a sham time-varying denominator must not gain.

### Phase 1: the screen of estimators (exploratory)

**Each estimator is one arm: B0's columns and settings, with the target divided by that estimator's
causal scale.** All arms are fitted per plant at the primary setting, three seeds, five folds, point
model only, and scored on all 18 months. The screen ranks the arms by absolute error and labels the
ranking exploratory.

| Group | Code | Estimator |
|---|---|---|
| Cheap | XP | expanding P99 of hourly power |
| Cheap | R30 | rolling 30-day P99 of hourly power (the roadmap's "simple rolling p99") |
| Cheap | RM | running maximum, a ratchet that can only rise (Dantas and Browell's wind method; a known-weak comparator) |
| Cheap | CSQ | rolling 30-day 90th percentile of clear-sky-normalised output (the roadmap's cheap baseline) |
| Cheap | CSE | upper envelope of output on clear hours only (forecast clear-sky index ≥ 0.8, as the lag study's CK arm defines clear hours) |
| Cheap | NR | the plant's 7-day energy over the median of the other 5 plants' capacity-normalised energy, times the static P99 (the rule that found the low-output months) |
| Cheap | PR | 7-day ratio of power to a fixed-orientation `pv_physics` model driven by CAMS (a performance ratio; the lag study's satellite-ratio arm recast as an estimator) |
| Cheap | KF | a Kalman local-level filter on the daily log of PR's ratio, with each day's observation variance set by its clear-sky share; mean and standard deviation native |
| Cheap | PW | `pv_fit.fit_plant` refitted on the trailing 30 days, capacity read off (existing derivative-free physics) |
| Wacky | CP | Bayesian online changepoint detection on PR's daily ratio |
| Wacky | HM | a hidden Markov model whose states are discrete capacity levels (1.0, 0.9, 0.75, 0.5), because strings and inverters fail in whole units |
| Wacky | LR | a low-rank-plus-sparse decomposition of the 6 plants' daily ratios, whose sparse part is plant-specific loss |
| Convex | CX1 | pinball loss at a high quantile plus a fused-lasso penalty on one daily DC trace, CAMS per-unit output at the fixed orientation (linear program) |
| Convex | CX2 | the roadmap's full Candidate A: DC and AC traces, the clipping plateau classifier, and the censoring hinge (linear program, with the AC plateau fitted by absolute rather than squared loss) |
| Convex | CX3 | CX1 at the median (quantile 0.5), which fits the middle of the data, as a least-squares method does; tests what the envelope buys |
| Convex | CX4 | CX1 after dividing out a shared regional bias: the fleet median of the daily clear-sky-normalised residual (the roadmap's plug-in pre-estimate) |
| PyTorch | DP1 | the roadmap's Candidate B: a variational single-plant model (tilt, azimuth, DC and AC capacity as posteriors), with a piecewise-constant weekly DC trace under a Laplace prior on its steps, and an asymmetric (pinball) reconstruction loss |
| PyTorch | DP2 | DP1 fitted jointly across the 6 plants with a shared daily regional irradiance-bias term |

**Arms in the screen that are not estimators of the same kind:**

- **CF, capacity as a feature:** the selected estimator's causal value as one extra column instead
  of a normalisation, with a width-matched null column (a random same-day value from months outside
  the row's fold, the lag study's N2 design).
- **RS, post-model rescale (no fit):** B0's forecast times the ratio of the selected estimator's
  causal value to its mean over the training rows.
- **B0nd:** B0 without the day-of-year column. The lag study found that dropping the column lowered
  B0's error by 0.117 points [−0.308, +0.063]. Dropping one column is the cheapest alternative to an
  estimator, so B0nd is the competitor in P2.

**Left out, with reasons:** SLAC's `solar-data-tools` (dependency change; departure 3); a
precipitation-driven soiling ratio (the differentiable-physics page puts soiling in the v2 forward
model, and its identifiability needs its own synthetic study); a model-residual ratio of observed to
B0-predicted energy (needs fold-indexed stage-1 models, whose leakage traps the lag study had to
engineer around, and the lag study's R1s and S3 arms already tested the idea).

**Tuning happens on synthetic data only.** Each estimator's settings (the window lengths, the pinball
quantile, the fused-lasso weight, the Kalman process noise, the hidden Markov model's switching
probability, the PyTorch priors and learning schedule) are tuned on the tuning draw of the planted
loss (below), never on a forecast error.

### Estimator recovery: the evidence for the capacity series itself

**Every estimator is scored on how well it recovers a planted capacity loss, with no forecast in the
loop.** At each plant separately, power is multiplied by a factor `f = 1 − s` in 3 whole calendar
months drawn by a seeded permutation from that plant's record (2019-09 to 2026-06), with at least
one drawn from the 18 scored months. Each plant gets its own months, so a neighbour-based estimator
can work and a shared-weather change cannot fake a recovery. The loss sizes are s = 10% and s =
30%, which brackets the real low-output months (29% to 33% below the neighbours). Two independent
draws are made: the tuning draw and the selection draw.

The recovery metrics, per estimator, causal and hindsight versions:

- **Relative capacity error:** the mean over daylight days of `|ĉ_planted(d) / ĉ_original(d) − f(d)|`,
  where `ĉ_planted` is the estimate on the planted data and `ĉ_original` on the real data. Taking the
  ratio cancels whatever real capacity changes the real data hold, which are unknown. Reported
  separately for planted days and for the 60 days after each planted period ends (recovery after the
  loss).
- **Detection delay:** days from the start of a planted loss until the estimate crosses halfway to
  the planted level.
- **False movement on real data:** the share of days on which `ĉ_original` moves more than 5% from
  its own trailing 90-day median, at plant-months the neighbour-ratio audit marks as within 5% of
  the neighbours. The ratio in the first metric cancels an estimator's noise on the real data, so
  this metric is what catches an estimator that chases the weather.
- **Interval coverage, where the estimator reports a standard deviation (KF, DP1, DP2, and CX
  estimators by a block bootstrap over days):** the share of planted days whose central 90% interval
  for `ĉ_planted / ĉ_original` covers `f`.

**Two known-answer cases on real data are figures, not metrics:** the commissioning ramp of one
plant (the eight months of plateaus the roadmap describes, before the row set starts), and the 3
low-output months, where the neighbour ratio is the only reference.

### Phase 2: selection, the shortlist, and the rigorous comparison

**The estimator behind each planned contrast is chosen by nested selection, so no planned number is
scored on the months that chose the estimator.** For each fold k, the rule picks, among the
eligible causal estimators, the one with the lowest primary-setting mean absolute error over the
other four folds' months. Fold k's rows are then scored with that estimator's losses. P1 to P5 are
contrasts of that selection procedure over all 18 months. The eligible estimators are every causal
estimator in the screen table; B0, B0nd, the hindsight versions, the controls, CF, and RS are not
eligible. Within-group winners for P3 and P4 are chosen the same way, inside each group.

**The shortlist** is the union of every estimator the nested rule picks in any fold, overall or
within a group, which is expected to hold 3 to 8 estimators. Phase 2 fits the shortlist, B0, and B0nd
at the sensitivity setting, fits their quantile models at both settings, runs their hindsight
versions, runs the positive control on them, and repeats the ANM-blind and ANM-aware comparison at
the ANM plant.

**Leakage note for the selection.** The losses on fold j that the rule reads come from XGBoost models
trained on months that include fold k. Fold k's labels therefore reach the choice indirectly, through
how well each estimator's normalisation of fold k's training rows helped predict fold j. The effect
is second order, and the page says so. A fully nested refit would multiply the fits by five.

### Deciding contrasts (planned, written before any result)

All at lead-day 1, per plant, on all 18 months, with 99% intervals (Bonferroni over five) from
`bootstrap_difference_at_level` (whole months and a seed, 2,000 resamples), at both hyperparameter
settings, and with the month-cluster t-interval (`fold_t_interval` over months) printed beside each
as a check, because a percentile bootstrap over 18 clusters tends to give intervals that are too
narrow. Errors are `absolute_error_capped_fraction_of_capacity` in percentage points of the fixed
P99, and CRPS is `crps_capped_fraction_of_capacity`.

- **P1 (is it worth it?):** the selected causal estimator minus B0, mean absolute error.
- **P2 (is it worth more than dropping one column?):** the selected causal estimator minus B0nd,
  mean absolute error.
- **P3 (does convex machinery beat a cheap estimator?):** the convex group's selected causal
  estimator minus the selected causal estimator from the cheap and wacky groups together.
- **P4 (does PyTorch beat convex, per plant?):** the PyTorch group's selected causal estimator
  minus the convex group's. This settles the roadmap's crossover prediction that the convex
  estimator wins per plant.
- **P5 (does the probabilistic forecast improve?):** the CRPS of the selected causal estimator minus
  that of B0, from nine-quantile models whose quantiles are each multiplied by the causal scale.

**The decision rule for Q1, fixed now.** The threshold is 2% of B0's own error, about 0.18 points of
mean absolute error and 0.12 points of CRPS:

- **Worth building for the forecast:** P1's 99% interval lies wholly below zero at both settings, its
  point estimate is at or below minus the threshold, and P2's 99% interval does not lie wholly above
  zero at either setting (dropping the day-of-year column is not significantly better).
- **Not worth building for the forecast:** at both settings, P1's 99% interval lies wholly above
  minus the threshold, so a gain as large as the threshold is ruled out; or P2's 99% interval lies
  wholly above zero at both settings.
- **Unresolved:** every other outcome. An unresolved P1 is reported with its interval, as in "a gain
  as large as 0.25 points is not excluded".

**Why 2%.** Three reasons, in order of weight. First, the hindsight monthly rescale gained 0.34 to
0.42 points over the same 18 months, so 0.18 points asks an estimator to capture about half of what
perfect knowledge of each plant-month's level buys; the beam/diffuse experiment's site-month oracle, on
the same 6 plants over 2019 to 2026 with past-weather inputs, took a similar 3.9% off its error. Second, a real estimator is costly
to run in production: a new asset, a time-varying `effective_capacity` table with a mean and a
standard deviation, an as-of join in the metrics pipeline, and a new dependency of the serving path,
so a gain smaller than half the attainable ceiling does not repay that cost. Third, the lag study
used the same 2% rule, so the two pages answer on one scale.

**The decision rule for Q2 (which technique), fixed now.** P3 and P4 each read "beats" when the 99%
interval lies wholly below zero at both settings, "loses" when wholly above zero at both, and
"no detectable difference" otherwise. Where a contrast shows no detectable difference, the page
ranks the two groups on the estimator-recovery metrics and on operability (the number of tuned
settings, the runtime, and whether the estimator reports its own uncertainty), which is the
roadmap's judging stance. That ranking is descriptive, not a test.

**The capacity series is judged separately from the forecast.** Whether a time-varying
`effective_capacity` table is worth building for NGED, as a fault signal and as delivery table 4, is
answered by the recovery metrics, not by P1. The page states both verdicts, and the maintainer
decides the second (open decision 9).

### Controls

- **NC1, constant scale (negative control):** the target divided by the static P99. The difference
  from B0 measures XGBoost's regularisation effect alone and should be close to zero.
- **NC2, sham time-varying scale (negative control):** the selected estimator's own causal series,
  with whole months permuted within each plant by a seeded draw. NC2 keeps the estimator's step
  sizes and destroys their timing. NC2 minus B0 shows what a time-varying denominator that carries no
  information does to the error; a gain here would mean the scoring is broken.
- **Positive control:** the selection draw's planted losses, at s = 10% and s = 30%, applied to the
  power the forecasts train on and score against and to the power the estimators read. Arms: B0, the
  oracle (normalised by the true planted factor times the static P99), the shortlist, and NC2. The
  oracle passes the control when its 99% interval against B0 lies wholly below zero at s = 30%; if
  the oracle fails, the instrument cannot see an effect and every null in the study is read as
  "not excluded", never as "no effect". An estimator's recovery share is its gain over B0 on the
  planted data, minus its gain on the real data, divided by the oracle's gain (the lag study's
  definition), with a 95% interval. Because each plant gets its own months, this control separates
  an estimator that tracks one plant from one that tracks a change shared by the fleet, which the lag
  study's month-level control could not.
- **The hindsight monthly rescale** (no fit, from the lag study's method) is printed on this study's
  rows as the ceiling every causal estimator sits under.

### Exploratory additions

- every screen arm against B0, with 95% intervals;
- each shortlisted estimator's hindsight version against its causal version (what causality costs);
- the 2×2 of day of year and normalisation: the selected estimator with and without the day-of-year
  column, beside B0 and B0nd;
- one XGBoost model shared by the 6 plants, at the lag study's fleet-wide folds, normalised by the
  static P99 against normalised by the selected estimator (a shared model needs a per-plant scale at
  all, so normalisation could matter more there);
- the shortlist with the ENS mean's forecast of past hours in place of CAMS, and with a 2-day CAMS
  wait;
- capacity uncertainty propagated into the forecast: for estimators that report a standard
  deviation, 50 draws of the scale, the nine quantiles multiplied by each draw, and the CRPS of the
  mixture, against P5's point-scale version;
- the ANM-blind against ANM-aware comparison at the ANM plant;
- month-by-month and plant-by-plant paired differences, including the 3 low-output plant-months.

### How the estimators would map onto delivery table 4 and the live service

The study builds none of this; the page's Discussion uses it to say what the winning estimator would
need. **Each estimator's daily output becomes rows of the planned time-varying `effective_capacity`
table:** one row per `(time_series_id, time)` half-hour, with `effective_capacity_mw_mean` and
`effective_capacity_mw_std` (issue #247), the daily value held across the day's 48 half-hours. The
table is backward-looking, so the hindsight (smoothed) series is the right content for the historical
table and the metric denominator, while the forecast's normalisation at init time must read the
causal estimate, as the metrics-and-leaderboard page requires. The estimate excludes ANM curtailment
(table 4's spec: a plant capped at 5 MW with 10 MW of physical capability has an effective capacity
of 10 MW), which only the ANM-aware versions satisfy. Issue #246 scales `power_fcst` to [−1, +1] by
the static P99; a time-varying scale would replace that divisor.

**Inherent stability, mapped:** on a day with no power data, a filter-type estimator (KF, DP1, DP2)
holds its mean and widens its standard deviation, which is the in-band degradation signal rule 4
asks for; a missing irradiance input falls back to the clear-sky envelope (rung 4's physical
envelope); an estimator that produces nothing falls back to the static P99 with a degradation flag
on the row, and never raises. The study measures the first behaviour on synthetic gaps for the
shortlist (exploratory): 7 days of power deleted before an issue day, and the estimate's mean shift
and standard-deviation growth reported. The study code itself is R&D, so it fails fast.

### Charts (the story, as figures)

1. **Figure 1, the headline:** P1 to P5 with 99% intervals, both settings as markers, each row's
   absolute errors in its label.
2. **The data:** each anonymised plant's daily output relative to its neighbours over the 18 months,
   showing the 3 low-output months; the ANM plant's share of capped midday hours by month (without
   linking the ANM status to a label, see the ANM section; open decision 7).
3. **The techniques, with known answers:** how one planted loss looks in the power, and each
   estimator family's recovered trace against the truth, with NC2 beside them; the commissioning ramp
   recovered by the shortlist; the estimator-recovery leaderboard (relative capacity error,
   detection delay, and false movement).
4. **The XGBoost models work:** out-of-fold forecasts against measured output for three weeks chosen
   by the lag study's rule, for B0 and the selected estimator; per-plant absolute errors.
5. **The screen:** every arm's error difference from B0, causal and hindsight paired, exploratory.
6. **Results:** the positive control with recovery shares; month-by-month differences with the 3
   low-output plant-months marked; the 2×2 of day of year and normalisation; the shared-model
   comparison; capacity uncertainty and CRPS; the ANM comparison.

Plants are labelled A to F by `site_labels_for`; nothing prints an identifier, name, or coordinate,
and power is shown only as a share of capacity.

## Compute estimate

**XGBoost fits, counted per arm as 6 plants × 5 folds × 3 seeds = 90 fits:**

| Stage | Fits | Notes |
|---|---|---|
| Reproduction check (B0) | 90 | primary, point |
| Screen: 18 estimators, CF and its null, B0nd, NC1, NC2 | 2,070 | primary, point |
| Hindsight versions of the shortlist (up to 8) | 720 | primary, point |
| Sensitivity setting: B0, B0nd, shortlist | 900 | 1,200 rounds, about twice a primary fit |
| Quantile models: B0 and the shortlist at both settings | 1,620 | about four times a point fit |
| Positive control: 2 sizes × (B0, oracle, shortlist, NC2) | 2,000 | primary, point |
| Exploratory extras (2×2, shared model, ENS input, CAMS wait, ANM) | about 900 | primary, point |

**At the lag study's benchmark (1,620 primary point fits in about 30 minutes of single-fit time on 4
CPU threads, and GPU throughput about 2.8 times a single CPU fit), the total is about 4.5 hours of
single-fit time, or about 1.6 hours of GPU wall time with 4 workers.** The request is a 2-hour GPU
slot with 4 CPU cores and about 1 GB of GPU memory, checkpointed with `fit_in_groups`.

**Estimators, before any XGBoost fit:** the cheap and convex estimators run on the CPU: each causal
linear-program solve covers a trailing window of up to 180 days, so about 6 plants × 560 issue days
× 4 convex estimators × 2 planted draws and the real data, roughly 40,000 small solves. The hindsight
versions add about 6 plants × 80 withheld weeks per convex estimator. The estimate is about 1 hour on
8 cores (unverified: benchmark CX2 on one plant first). The PyTorch estimators run on the GPU with
weekly warm-started refits: about 6 plants × 80 weeks × 3 data sets, estimated at 30 to 60 minutes
(unverified: benchmark DP1 on one plant first).

**Slots to request from the Study MAIN COORDINATOR, in order, before any fit:** an 8-core CPU slot
of about 1 hour for the cheap and convex estimators; a GPU slot of about 1 hour for the PyTorch
estimators; then the 2-hour GPU slot for the XGBoost fits. The PyTorch and XGBoost GPU work never
run at once, so no planned contrast mixes GPU contention into its timings, and every XGBoost fit runs
on the GPU, one device for every contrast.

## Solution

### Walk through

**The happy path.** `build_capacity_inputs.py` reads the lag study's lead-day-1 frame and checks its
29,445 rows and its SHA-256. It builds each plant's cleaned hourly power from the start of its
record through `solar_hourly_power`, `drop_outages_and_spikes`, and `drop_commissioning_ramp`, joins
the export cap with `with_export_cap`, joins CAMS from `read_cams` and the clear-sky irradiance from
`hourly_clear_sky`, fits each plant's orientation with `pv_fit.fit_plant` on data before 2024-12-01,
and draws the two planted-loss draws with `studies.capacity_estimators.plant_losses`. The build
writes one hourly input table per data set (the real data and each draw at each size) and a build
report with the ANM capped-hour shares and the row counts.

`estimate_capacity.py` runs every cheap, wacky, and convex estimator on every input table through
`studies.capacity_estimators.causal_daily_estimates`, which hands each estimator only the days
before each issue day, and through `hindsight_daily_estimates`, which withholds each scored week.
`estimate_capacity_torch.py` does the same for DP1 and DP2 with
`studies.capacity_physics_torch`. Both write one daily table per estimator: plant, day, version,
data set, mean, and standard deviation where the estimator has one.

`capacity_recovery.py` reads those tables and computes the recovery metrics on the selection draw,
after tuning each estimator's settings on the tuning draw; that order matters because tuning on the
selection draw would score each estimator on the data that set it. The recovery metrics need no
forecast fit, so they are written before any XGBoost model exists.

`fit_capacity_arms.py` joins each estimator's daily causal value onto the frame as
`capacity_scale_mw` (the value from the day before each row's issue day, with the 30-day and 20%
fallbacks), then calls `out_of_fold_losses` with `target_scale="capacity_scale_mw"` for each arm.
Inside, each fold's training target is divided by the scale, and the point and quantile predictions
are multiplied by the test rows' scale before `_losses` scores them in megawatts against the static
`effective_capacity_mw`. The script first runs the reproduction check, then the screen, then applies
the nested selection rule to the screen's saved losses, writes the per-fold picks and the shortlist
to `shortlist.json`, and only then fits the shortlist's sensitivity, quantile, hindsight, control,
and exploratory arms, checkpointed through `fit_in_groups`.

`report_effective_capacity.py` reads the saved losses and `shortlist.json`, builds P1 to P5 from the
per-fold picks, computes every interval, applies the decision rules, and prints every table the page
quotes, with every arm's column list and every fallback share, into `report.md`.
`effective_capacity_charts.py` writes the figure text first (`--text-only`) for the text reviews,
then renders the figures from the saved tables.

**The main branches off the happy path.**

- An estimator with too little history at an issue day returns no value; the scale falls back to the
  static P99 and the row is counted.
- The ANM-aware version of an estimator treats the cap as an upper bound on the plant's output, at
  the one plant with a cap record and only from the scheme's go-live; at the other plants the
  ANM-aware and ANM-blind versions are identical by construction, which the build asserts.
- A hindsight estimate is read across a withheld week, so the estimators must interpolate across a
  gap: the fused-lasso and filter estimators do so natively, and the window estimators use a centred
  window whose middle week is empty.
- The quantile arms multiply all nine quantiles by the same scale, then sort them, as the lag study
  did.

**The main error-handling paths.** The study is R&D code and fails fast. A row count that differs
from 29,445 raises; a reproduction mismatch raises; an estimator input at or after 2026-07-01
raises; a causal estimate that changes when data after its issue day is altered raises (a probe on
one plant per estimator, run by the build, beside the unit test); an uncovered calendar month raises
(`raise_on_uncovered_months`); a linear program that HiGHS does not solve to optimality raises with
the plant and day named. Fallback rows and dropped rows are counted and printed, never silently
filled.

### What changes, file by file

- **`packages/studies/src/studies/cross_validation.py`:** `out_of_fold_losses` gains
  `target_scale: str | None = None`. With a column name, each fold divides the training target by
  that column, and multiplies the point predictions and every quantile by the test rows' value of
  that column before `_losses`. `_aligned_scoring_frames` and `_out_of_fold_losses_on_many_frames`
  pass the parameter through or refuse it with a clear error (decided at implementation; the study
  does not need the many-frames path).
- **`packages/studies/src/studies/capacity_estimators.py` (new):** the estimators that need no
  PyTorch, as pure functions over a plant's daily or hourly arrays: the window statistics (XP, R30,
  RM, CSQ, CSE), the neighbour and performance ratios (NR, PR), the Kalman local-level filter (KF),
  the online changepoint detector (CP), the hidden Markov model (HM), the low-rank-plus-sparse
  decomposition (LR), and the convex linear programs (CX1 to CX4) on `scipy.optimize.linprog`. Also
  `causal_daily_estimates` and `hindsight_daily_estimates`, the two drivers that enforce what each
  estimate may read, and `plant_losses`, which draws and applies the planted losses.
- **`packages/studies/src/studies/capacity_physics_torch.py` (new):** DP1 and DP2, reusing
  `pv_physics`'s geometry where the arrays allow and writing the transposition in torch. The zenith
  cosine is floored, as the beam/diffuse experiment's lesson requires.
- **`studies/effective_capacity/`** (new folder): `build_capacity_inputs.py`,
  `estimate_capacity.py`, `estimate_capacity_torch.py`, `capacity_recovery.py`,
  `fit_capacity_arms.py`, `report_effective_capacity.py`, `effective_capacity_charts.py`, and
  `README.md`. No basename repeats one elsewhere under `studies/`.
- **`pyproject.toml` (root):** the two torch files added to the `[[tool.ty.overrides]]` include list
  (open decision 1).
- **`docs/studies/forecasts/effective-capacity.md`**, an entry in `docs/studies/index.md`, the nav
  entry in `mkdocs.yml`, and `docs/studies/assets/effective_capacity_figure_<n>.svg`, optimised with
  `svgo`.

### Tests

Each new test states the assertion that fails on `main` today.

- **`test_cross_validation.py`:** with `target_scale` naming a column of unequal per-row values, on
  a synthetic target equal to the scale times a function of the features, the out-of-fold losses
  are near zero and stay in megawatts; on `main`, `target_scale` does not exist. A second case passes
  a constant scale with plants of unequal `effective_capacity_mw` and asserts the fraction-of-capacity
  losses divide by `effective_capacity_mw`, not by the scale (the `study` skill's rule on unequal
  capacities in fixtures). A third asserts that `target_scale=None` returns frames equal to a stored
  call without the parameter.
- **`test_capacity_estimators.py`:** for every estimator, a known-answer step (a synthetic plant with
  a 30% loss for 30 days) is recovered to within a stated tolerance after a stated delay; the causal
  driver's estimate for day d is unchanged when every input after day d is replaced by noise (fails
  if the driver passes one day too many, the off-by-one this study most needs to catch); the
  hindsight driver's estimate for a withheld week is unchanged when that week's power is altered; the
  convex linear program matches a brute-force solution on a 5-day case and gives exactly zero steps
  where the truth is flat; the Kalman filter's standard deviation grows on days with no data;
  `plant_losses` multiplies exactly the drawn months at exactly one plant, with different months per
  plant.
- **`test_capacity_physics_torch.py`** (skipped without torch): DP1 recovers a planted DC capacity
  and tilt from a `pv_physics`-generated plant to within a stated tolerance; DP2's shared bias term
  absorbs a bias applied to every plant's irradiance and leaves the capacities at their truth.
- **Script-level assertions** (study scripts carry no unit tests): the row count, the SHA-256, the
  reproduction check, the final-test cutoff, the causal probe, and the ANM-aware-equals-blind check
  at the 5 plants without a cap.

### Design-philosophy check

**R&D code: fail fast**, as the inherent-stability page requires of training and metrics code.
Nothing runs in production. The study does not choose the production estimator; a follow-up issue
under epic #141 would carry a result into the `effective_capacity` asset (#247) and the metrics
join. The section on delivery table 4 describes how the winner would degrade, without building it.
No hypothesis in `engineering-hypotheses.md` concerns capacity estimation (checked), so the page
cites none.

### Docs to update

The new study page, `docs/studies/index.md`, the `mkdocs.yml` nav, and the study folder's README.
The roadmap's capacity-estimation page is not edited by this study: its head-to-head stays planned
until the maintainer decides what the result means for v0.7 (open decision 9). No roadmap item
completes, so there is no ship-time triage.

### Verification commands

`uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check`, `uv run pytest
--run-studies -n auto packages/studies` (once with the `gpu` group installed, so the torch tests run,
and once without, so they skip), `uv run pymarkdown scan -r docs README.md CLAUDE.md
packages/*/README.md`, pydoclint and the docs-link checker as CI runs them, `uv run mkdocs build
--strict` and reading the rendered page.

## Lessons carried over from the lag-features study

- **Column subsampling off,** and every normalised arm has B0's seven columns.
- **Whole-month folds and whole-month resampling,** with the month-cluster t-interval beside the
  bootstrap.
- **Day-of-year memorisation:** P2 and the 2×2 test whether an estimator does more than drop the
  day-of-year column does.
- **The screen's winner's curse:** no planned number is scored on months that chose its estimator.
- **Withheld-fold leakage:** no feature is built by a model fitted on the scored fold; the hindsight
  estimator withholds each scored week; the causal driver is unit-tested for reading one day too
  many.
- **Reproduction check,** now bit for bit against the lag study's B0 losses.
- **Final-test cutoff,** 2026-07-01, asserted on every estimator input.
- **A positive control that can pass,** with an oracle, at a loss size like the real one, and per
  plant.
- **An equivalence outcome** in the decision rule.

## Risks and open questions

1. **Can the torch estimators be merged autonomously?** Adding `capacity_physics_torch.py` means
   editing the root `pyproject.toml`'s `ty` override list, which the `study` skill's merge conditions
   do not allow. **Recommendation:** approve that one-line configuration change in advance, so the
   PR can still merge autonomously; otherwise the PR stops at a reviewed PR.
2. **CVXPY or `scipy`?** `scipy`'s HiGHS solves every convex estimator here exactly and needs no
   dependency change. Adding `cvxpy` would serve the roadmap's CVXPY-experience goal but blocks the
   autonomous merge. **Recommendation:** `scipy` for this study; leave CVXPY to the production
   estimator, if one is built.
3. **Is 2% of B0's error the right bar, and is the three-outcome rule right?** **Recommendation:**
   yes, for the reasons in "Why 2%". The risk is power: the real level shifts sit in 3 plant-months
   of 108, so even a perfect causal estimator may move the pooled error by less than the threshold.
   The hindsight ceiling of 0.34 to 0.42 points and the positive control show how far the instrument
   can see.
4. **Nested selection, or a fixed estimator named now?** Nested selection uses all 18 months for
   every planned contrast but carries the second-order leakage described above, and its "selected
   estimator" can differ between folds. Naming one estimator now (CX2, the roadmap's predicted
   winner) is simpler and leak-free but may test the wrong estimator. **Recommendation:** nested
   selection, with the per-fold picks printed.
5. **The hindsight definition.** Withholding the scored week stops a daily-block estimator from
   reading the target day's output, but costs one extra solve per withheld week. **Recommendation:**
   keep it; a hindsight version without the withholding is not an upper bound on capacity knowledge
   but on knowledge of the answer.
6. **Ask NGED what happened at the low-output plant in December 2024, March 2025, and April 2026?**
   ANM is ruled out as the cause. A maintenance or outage record would turn those months into labelled
   events for the estimators' event-log check, which the roadmap lists as a diagnostic.
   **Recommendation:** ask; the study does not depend on the answer.
7. **May the page show the ANM plant's capped-hour shares at all?** Showing them under an anonymised
   label pairs that label with the one ANM generator in the trial area. **Recommendation:** show the
   ANM results without the plant's label, as the beam/diffuse page did ("one site is curtailed").
8. **Slots.** Three slots, in order: about 1 hour on 8 CPU cores, about 1 hour of GPU for PyTorch,
   and 2 hours of GPU with 4 cores for XGBoost. The PyTorch and convex timings are unverified until
   benchmarked on one plant.
9. **Who decides what the result means for v0.7?** The study gives a forecast verdict (P1 and P2)
   and an estimator verdict (the recovery metrics). Whether to build the time-varying
   `effective_capacity` table even if P1 shows no forecast gain is the maintainer's decision, and so
   is any edit to the roadmap's capacity-estimation page.
10. **The row set requires the lag study's one-day lag to exist,** which this study does not use. The
    lag study's rows are kept so B0's 8.93 points and every lag result stay comparable; the 674 rows
    that rule drops are not scored here. **Recommendation:** keep the lag study's rows.
11. **Cleaning that looks at later hours.** The multi-day-zero filter decides whether a run of zeros
    lasts 24 hours by reading later hours, so the causal estimators read a slightly cleaner history
    than a live service would. The lag study had the same limitation; the page says so.
12. **The planted loss is multiplicative,** which mimics a lost share of the array (a DC loss). A lost
    inverter lowers the AC clipping limit instead, which CX2 and DP1 model and the others do not. The
    study does not plant an AC loss; the page lists that under Scope.
