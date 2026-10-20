---
id: "00124"
slug: differentiable-physics-h-bsp-closure-audit-and-bsp-hint
theme: differentiable-physics-pooling-across-substations
also_relevant_to: [hierarchy-and-pooling, switching-detectors]
status: done
size: >-
  complex for the new module studies.dp_hierarchy (own PR, two code reviews); medium for the
  scripts. Private files only; serving path and degradation rules untouched; several defensible
  designs, one pre-registered each; callers are this experiment's scripts only.
hypothesis: >-
  A bulk supply point (BSP) reading, fitted jointly with the Primaries beneath it as the sum of
  their demand terms plus the metered generators plus a penalised remainder, with a tuned weight,
  limits the damage that a metering error at one Primary does to that Primary's forecast, without
  raising the Primaries' forecast error when no error is present.
prediction: see "Predictions" below, written before any fit
decision_rule: see "Decision rule" below and "Decision rule for step 0"
benchmark: >-
  W0, W*, W-max, OR, a direct GAM on the BSP, climatology, manual_heuristic; experiment 111's GAM-K
  and BG2 as reference rows
row_set_id: >-
  dp_mw_v1 and dp_mva_dark_v1 (experiment 103); the Primaries in the study; tuning 2023-24, replicate 2024-25, scored 2025-26
metric: >-
  Day-D+1 quantile score in points (C1); damage in points and ratio of summed damages (C2); BSP
  quantile score (C3)
verdict: not supported
production_status: not-for-production
take_home_message: >-
  The bulk supply point reading halved the damage from injected scale errors (41% over all
  injected errors), but it made clean forecasts at least 2.3 points of capacity worse at every
  weight above zero, about 100 times the 0.023-point limit on that worsening, in all three years
  tested, so the tuning rule chooses no hint. The cause is untested and a coding fault is not
  excluded, because the known-answer gate was not run; the closure audit fits the generation
  factor at 0.85 where the joint fit fixes it at 1. As a monitor the reading is weak: at the tuning
  cut a 25% scale error lasting 30 days is seen in 0% of cases when the fit includes the injected
  series, and in 50% when the fit is trained only on earlier rows, which then raises a false alarm
  on 61% of windows in the year after the cut.
advice:
  - >-
    Fit the closure multipliers of a bulk supply point (Primaries, MVA Primaries, generation)
    before fixing them in a joint fit, because the audit gave 0.85 to 1.16 and not 1 (closure
    audit, Figure 2).
  - >-
    Check the rise in clean-forecast error from a hint at the smallest weight before tuning
    anything else; a weight of 0.1 already raised it by 2.4 points here (Figure 1).
  - >-
    Do not read a bulk supply point residual with a fixed 3-standard-deviation threshold in
    service, because a frozen fit drifts and the threshold false-alarmed on 44% to 83% of 30-day
    and 90-day windows, and on 14% to 18% of 24-hour windows, in the year after the cut (closure
    audit).
  - >-
    Report metering-error detection for a fit that is refitted with the injected series included,
    because a running service would make that fit; at the tuning cut a fit that cannot see the
    series detected a 30-day 25% scale error in 50% of injections against 0% for the refit (closure
    audit).
date: <date>
pr: <pr>
---

# Differentiable physics, H: does a bulk supply point reading limit the damage from a Primary's metering error?

**We asked whether the power reading of a bulk supply point (BSP) can protect the forecasts of the
Primary substations beneath it from a metering error at one of them.** A BSP is where the
distribution network takes power from the transmission network. Its reading is roughly the sum of
the Primaries' power minus the power of the metered generators. We fitted each Primary's forecast
together with the BSP reading, with a weight on how closely the fit must match that reading (weight
0 means separate fits). We then injected metering errors into one Primary's history: gaps, stuck
values, and scale errors. We measured the damage, meaning the rise in that Primary's forecast error,
with and without the BSP reading. Errors are in points, where 1 point is 1% of the Primary's
effective capacity. **The BSP reading halved the damage from scale errors, but it made forecasts on
clean data at least 2.3 points worse at every weight we tried above zero (0.1 to 30).** The tuning
rule written into the plan before any fit therefore chooses weight 0. The verdict is `not
supported`, for this joint fit with the multipliers on the Primaries' and generation sums in the BSP
equation fixed at 1.

![Damage from injected errors and the rise in clean-forecast error, against the weight on the BSP reading](figure_1_tradeoff.svg)

**Summary.** Numbers are in `tuning_report.md` and `audit_report.md`; no interval was computed.

- **The BSP reading halves the damage from scale errors.** Over 200 injected errors at two
  Primaries (60 gaps, 60 stuck runs, 80 scale errors) in the tuning year, the summed extra error
  from scale errors fell from 33.0 points at weight 0 to 16.3 at weight 30, 51% less, about 0.2
  points per injection (Figure 1, top). Pooling all 200 injections gives 32.8 to 19.4, a 41% cut.
- **The BSP reading does nothing for gaps and stuck runs.** Neither did measurable damage at
  weight 0 (summed -0.1 points each), and weight 30 added 1.3 and 1.8 points.
- **The BSP reading makes clean forecasts much worse.** Against separate fits, the mean absolute
  error of the clean year rose by 2.4 points at weight 0.1 (2.36 to 2.51 over the tuning,
  replicate, and scored years) and by 4.2 to 5.3 points at weight 30 (Figure 1, bottom). The limit
  was 0.023 points. The tuning rule then chooses weight 0. The gate and the injection runs of the
  replicate and scored years were not run, because with the chosen weight at 0 gate reading 6 (a
  damage cut of at least 0.30) cannot pass. Only the clean-error grid was fitted at the replicate
  and scored cuts. The plan's verdict list does not say which verdict follows when the gate cannot
  pass because the chosen weight is 0; we read the tuning rule's "C2 cannot pass" together with the
  rise at weight 30 (the arm W-max), which is above the margin in all three years, as `not
  supported`.
- **The cause is untested.** The closure audit fits the generation multiplier at 0.85 at the tuning
  cut (the Primaries' multiplier is 1.01 there and 1.16 at the scored cut), where the joint fit
  fixes both at 1 (Figure 2). A coding fault, a remainder that is too rigid, and the unmetered
  Primaries are not excluded, because the known-answer gate was not run.
- **The BSP reading is a weak monitor.** At the tuning cut a 30-day 25% scale error is found in 0%
  of injections when the fit includes the injected series and in 50% when the fit is trained only
  on earlier rows (2% and 29% at the replicate cut). The fit trained only on earlier rows raises a
  false alarm on 1.2% of windows in the data it was fitted to and on 61% of windows in the year
  after the cut.

**The pilot omitted several parts of the plan.** Each is a labelled simplification:

- one BSP (the one that closes), with two MW Primaries and four MVA Primaries, and one remainder
  step penalty;
- a smooth drift in each Primary's demand, without sudden steps, and a fitted solar capacity that
  scoring holds at its last value;
- a fit made once before each scored year, with no Kalman filter re-learning the level (the
  filter-on reading, the production-relevant one, was therefore not produced, and the plan caps a
  pilot at `partly supported`);
- a score that is the mean absolute error of a point forecast, with no quantile score, tails,
  climatology or `manual_heuristic` rows, BSP-forecast arms, or intervals;
- tuning on 10 seeds per cell and Primary (200 injections) and not 30 per kind, with scale errors of
  20% and not the audit's 10%, 25%, and 40%;
- generation series enter only if they hold readings in 90% of the BSP's rows.

## What the closure audit found

![Closure multipliers of the audit, per bulk supply point and cut](figure_2_coefficients.svg)

**One BSP closes in the training years, and the other does not.** At the pilot BSP the in-sample
residual is 2.2 to 3.2 points (1.2 to 1.8 MW) at the three cuts, over six winters with a median
residual within 1 MW. Its fitted multipliers are 1.01 to 1.16 on the Primaries, 0.85 to 0.90 on
the generation, and 1.03 to 1.10 on the Primaries that report in megavolt-amperes. Out of sample,
the frozen fit leaves a residual scale of 5.5 points and a +3.7 MW median offset. The other BSP has
too little generator history, so about half its generation series are left out and its power sits
in the remainder: 11 points in-sample. The full numbers are in `audit_report.md`.

## Checks

- **Mean and median forecasts, and quantile scores, do not apply,** because the pilot scores a
  point forecast by its mean absolute error.
- **Trigger 2 (a contradiction) fired.** The result contradicts the plan's prediction that clean
  error would change by no more than 0.02 points, and the finding of
  [experiment 36](../00036-hierarchy-aware-bsp-gsp-forecasts/result.md) that bottom-up children
  add 0.04 to 0.19 points at the BSPs, and the cause was not tested.
- **Trigger 7 (dependence) also applies,** because a possible follow-on arm depends on this result.
  An Opus audit is requested from the manager before the merge.
- **Predictions that failed:** the chosen weight was predicted at 0.3 to 3 and came out at 0, and
  the extra clean error at weight 30 was predicted within ±0.02 points and came out above 2.3.

## Lessons

- **The hint as built was far over the limit on this BSP.** The audit's generation multiplier is
  0.85 at the tuning cut and the Primaries' multiplier reaches 1.16 at the scored cut. Whether
  fitted multipliers would remove the extra error is untested.
- **One possible next experiment.** Fix the multipliers at the audit fit made before each cut and
  repeat the weight tuning against the same 0.023-point limit. Whether to run it is the
  maintainer's decision.

## Follow-on arm B (planned): the audit-calibrated hint

**This section is the plan for a second arm of the same experiment, written before any fit of
arm B.** The first joint fit above stays as written. Arm B asks whether the same hint stops hurting
clean forecasts once its multipliers are fixed at the closure audit's fit.

- **Hypothesis.** A BSP hint whose multipliers on the MW Primaries, the MVA Primaries, and the
  generation are fixed at the closure audit's fit made on rows before the cut raises the clean
  forecast error of the MW Primaries by no more than 0.023 points at some weight above 0, and at
  that weight it still removes damage from scale errors.
- **Prediction (can fail).** The chosen weight is above 0, and the damage from scale errors falls by
  at least 20% at it. We give this a low chance: a constant multiplier error is absorbed by the
  remainder's level (the module's synthetic test finds a 10% error barely moves a Primary), so the
  multipliers may not be the cause.
- **Design.** The same pilot BSP, cuts, injections (10 seeds per cell and Primary in the tuning
  year), score, and limit as the first joint fit. The multipliers are the audit's values at that
  cut, fitted on rows before it only (no replicate-year or scored-year data). The weight grid adds
  0.003, 0.01, and 0.03 below the first grid's 0.1, because the first grid never saw a weight
  small enough to cost less than 2.3 points.
- **Plumbing check first, blocking.** A synthetic BSP built exactly from the fitted Primaries, with
  the audit's multipliers and a known remainder, must cost under 0.023 points at weight 0.1, which
  excludes a coding fault that the first fit's missing gate did not. If it fails, arm B stops
  as `invalid` until the fault is found.
- **Decision rule.** The chosen weight is the weight with the largest summed damage reduction in the
  tuning year whose clean error rise is at most 0.023 points. If no weight above 0 passes, arm B
  stops at `not supported`, and the BSP reading is recorded as a possible monitoring signal and not
  as a loss term. If one passes, the replicate and scored years and the gate run for that weight,
  under the first plan's rules.
- **Reports.** The curve of damage reduction against clean error rise across the weights, and the
  audit multipliers per BSP with their intervals.
- **Cost.** One GPU job of about an hour for the tuning grid, and about three hours more if a weight
  passes. The code is `BspMember.coefficient` and `BspSeries.generator_coefficient` in
  `studies.dp_hierarchy`.

## The plan as committed, with its clarifications

**Everything below was written before the results and is kept as committed.** The pilot's
simplifications are the clarifications near the end.

### What this experiment asks, and what the maintainer will accept

**The question is robustness, not mean-forecast skill.** The maintainer is content if the hierarchy
gives only explainability and robustness to metering errors, so C2 is the headline and C1 a
non-inferiority check with a gain bonus. Experiment 36 found bottom-up children add 0.04 to 0.19
points at the two BSPs, so no large clean-data gain is predicted.

**The hint is soft:** the BSP rows carry a tunable weight w, and a remainder term absorbs whatever
the Primaries and metered generators do not explain. At w = 0 the fit equals the separate fits (P1).

**There is no GSP fit:** the GSP relation cannot be closed (most of the power at each grid supply point is unmetered).
The crude BSP fit had a daily-mean residual standard deviation of 1.8 to 2.7 MW, above the median
switching transfer (1.0 MW); step 0 tests whether a better fit shrinks it.

**Open assumptions, recorded on the page and not asked of NGED (deferred to January):** nine
unmetered partner Primaries sit under unknown BSPs, and some 11 kV generation may remain inside a
Primary's series. Step 0 measures their footprint through the remainder's weather term, the
dropped-sum sensitivity, and a_G.

### Step 0 (H0): the BSP closure audit

**Fit, per BSP b, on rows before each cut (three cuts), by the same convex solver:**

BSP_b(t) = a_P · Σ_{metered MW Primaries under b} y_i(t) − a_G · Σ_{metered generators at b} g_k(t)
+ c · Σ_{MVA Primaries under b} y_m(t) + q_b · pv_shape(t) + level_b + drift_b(t) + h_b · heat(t)
+ shape_b(hour) [+ κ · loss(t)]

- The Primaries enter as their observed readings (a diagnostic, not the joint model); the two series of one
  Primary enter summed; generators enter as the observed sum, and only the sum and its coefficient are
  ever published.
- BSPs and Primaries are import-positive, generators export-positive. a_P, a_G, c are free in
  [0, 1.5]; q_b ≥ 0 is an unmetered PV capacity on the pinned solar shape; drift_b has monthly hat
  functions with an L1 step penalty tuned on the tuning year by cross-validated residual variance.
- **The unmetered Primaries carry weather-driven demand,** so the remainder has one heating slope h_b
  (on the BSP's area-mean temperature below 15 °C) and one daily shape (48 half-hour effects, ridge
  penalised). A coefficient a_P near 1 is therefore not read as evidence of closure. The audit
  reports the fit with and without these two terms; the loss term stays only if it lowers the
  tuning-year residual variance by 5% or more.
- A sensitivity fit drops that summed Primary, because some generation may remain inside its series.

**Reports, per BSP, in MW and as a share of the BSP's median load:**

1. The coefficients a_P, a_G, c, q_b, h_b with month-block bootstrap intervals.
2. The residual's robust half-hourly scale and its block-mean standard deviation over 6, 24, 72
   hours, 30 days, and 90 days.
3. The residual by hour band and calendar month (does the -9.3 MW midday trough survive?).
4. **Injected visibility, at C2's own window lengths:** the injection harness of experiment 43 adds
   metering errors (a stuck run of 24 hours, and scale errors of 30 and 90 days) and conserved
   cross-BSP transfers to the Primaries' side, at 10%, 25%, and 40% of the injected Primary's median
   load, 100 seeds per cell in each of the tuning and replicate years. A window is **detected** if
   the matched-filter shift of the residual's window mean exceeds 3 times the block-mean standard
   deviation for that window length, with a false-alarm rate (from 1,000 placebo windows outside
   injections) of at most 5%. Gaps need no detection and are not scored here.
5. Detection is also reported with a fit trained on pre-injection rows only; the lower counts.

#### Decision rule for step 0, fixed before any residual is read

Let `recall30` be the detection rate for a 30-day scale error of 25% of the injected Primary's median
load, in the tuning year, and `recall1d` the rate for a 24-hour stuck run of the same size. Per BSP:

- **informative** if `recall30` ≥ 0.5 and the false-alarm rate ≤ 0.05;
- **weak** if `recall30` is 0.2 to 0.5, or the false-alarm rate exceeds 0.05;
- **uninformative** if `recall30` < 0.2.

`recall1d` and the a_P band are reported, not decisive. Then step 1 runs as follows:

- **at least one BSP informative:** H1 runs on every informative BSP's Primaries, full design;
- **none informative:** H1 runs anyway as a **pilot** on the BSP with the highest `recall30`; the
  verdict cannot exceed `partly supported`, and C2 must hold in both years;
- **`not supported` is kept for a pilot whose C2 also fails;** the take-home then says that hierarchy
  gives explainability (the fitted shares, the residual by band) but no robustness.

### Step 1 (H1): the joint fit and the weight

**Model, per BSP b chosen by step 0, with metered Primaries i ∈ I_b:**

- Primary row: y_i(t) = d_i(t; θ_i) + ε, with d_i the convex GAM of the series (experiment 106's
  all-years form: drift with monthly knots, pinned capacity; experiment 111's hand-filtered masks,
  applied identically to every arm).
- BSP row: BSP_b(t) = Σ_{i∈I_b} d_i(t; θ_i) + (MVA Primaries' dark-row demand terms) − Σ_k g_k(t) +
  r_b(t; ρ_b) + ε_b. The **penalised remainder** r_b is the audit's: level, drift with an L1 step
  penalty, unmetered PV capacity q_b ≥ 0, one heating slope, one daily shape, and the loss term if
  step 0 kept it (the sum in it is the observed sum in fitting and the forecast sum in forecasting).
- **MVA Primaries enter the BSP row with demand terms only** (dark-row fits; |S| hides the sign in
  daylight); their embedded PV joins q_b. That Primary's two series enter as one summed term.
- **Joint objective:** Σ_i ‖y_i − d_i‖² / σ_i² + w · Σ_b ‖BSP_b − (…)‖² / σ_b² + the GAM and
  remainder penalties, with σ the robust half-hourly residual scale on pre-cut rows. This is a
  bounded least-squares problem in the stacked coefficients, so it uses `dp_gam`'s Gram accumulation
  and `solve_bounded_quadratic`; the new module stacks the designs and adds the cross-block entries.
- Generators enter as known inputs, sign -1; a missing generator reading drops that BSP row.

**Weight grid:** w in {0, 0.1, 0.3, 1, 3, 10, 30}, relative to a Primary row of equal noise.

**Tuning rule, on the tuning year only, pre-registered:**
- **W\*** is the w with the largest summed damage reduction on the tuning-year injections T (30
  seeds per kind and Primary, disjoint from the C2 seeds), subject to a clean-score cost of at most
  0.023 points against W0 (the Part 4 C4 noise reference, ±0.023 at 97.5% for the mean score). A
  clean-cost limit alone would drift to w = 30.
- **W-max** is w = 30, kept as a sensitivity showing the clean-score cost of a heavy hint.
- If every w > 0 breaks the clean-cost limit, W\* = W0 and C2 cannot pass.

**Years.** Tuning 2023-07-01 to 2024-06-30 (w, injections T, gate systems); replicate 2024-07-01 to
2025-06-30 (C1 and C2 must hold here too); scored 2025-07-01 to 2026-06-30 (the verdict).

Each cut trains on rows before it only, from each site's start (experiment 103's start cuts). **One
frozen fit per cut and arm, with experiment 111's Kalman filter (hyperparameters fitted on the tuning
year); the contrasts are screening-tier and could depend on a monthly refit, which the page says.**
Weather is ERA5 and CAMS (lab conditions); no ENS arm.

**Caveats the page carries:** hindsight capacity vintage (the 2023-07-01 cut reads one about 5 weeks
later); shared hand-filtered masks; filter hyperparameters in-sample for the replicate year; ERA5
is 5 days late, so every number is a ceiling.

### Arms
| Arm | Joint fit | Weight | Filter |
|---|---|---|---|
| W0 | joint code, BSP rows present | 0 | plain Kalman |
| W\* | as above | tuned | plain Kalman |
| W-max | as above | 30 | plain Kalman |
| OR | W\*, with the injected windows masked out of the injected Primary's rows (oracle) | tuned | plain Kalman |
| BD | direct GAM on the BSP (convex, drift, pinned PV) | none | none |
| BU0 | BSP forecast: sum of W0's Primary forecasts + remainder | 0 | none |

**BD and BU0 score C3;** climatology and `manual_heuristic` are scored at both levels. C2 is also
scored with the filter off (W0, W\*, W-max, OR) so that the filter's level re-learning is not
credited to the hint.

### Plumbing controls and the blocking known-answer gate

**Plumbing controls run first; an unfixable failure makes the verdict `inconclusive`:**

- P1: the stacked design reproduces `dp_gam`'s single-site Gram matrix block by block (relative error
  under 1e-9), and the joint solution at w = 0 equals the separate fits (coefficients to 1e-6);
- P2: with a noise-free BSP built exactly as Σ d_i − Σ g + remainder, the joint fit at w = 1
  recovers every d_i's sum to 1e-4 relative;
- P3: no fit reads a row after its cut; every σ and L1 scale uses pre-cut rows; emptying the label
  file changes no arm except the hand-filtered mask arms;
- P4: the audit's a_G estimate lies in 0.5 to 1.5 on the real BSP (-0.8 to -1.2 expected), else step 1
  stops on that BSP.

**Known-answer gate on synthetic systems (blocking: a failed reading stops C1 and C2 and caps the
verdict at `inconclusive`; readings are frozen, and a redefined rerun is forbidden, as experiment
109's post hoc reruns showed it must be).** 100 synthetic systems with the real Primaries' calendars
and weather, truths fitted to each Primary (experiment 113's builder), a BSP built from the fitted
demand terms plus a known generator sum and remainder, and noise as large and autocorrelated as the
real step-0 residual. **Half the systems close exactly; the other half have a_P = 0.9 and a
remainder shape the model lacks (a winter-evening hump),** because a truth in the model's own form
flatters the hint (experiment 113's caveat). Readings, at W\* and at W0:

1. **Identification:** the 80% month-block bootstrap intervals (experiment 113's I1, widened by
   sqrt(m/(m-1))) cover the true heating slopes and latest capacity 70 to 90% of the time. Experiment
   113's verdict was `refuted`: coverage was 84% each at realistic noise, but capacity coverage was
   52% with independent noise, from a bias of about 0.24 points at the drift step of 3,000. The gate
   uses the real-noise cell at the tuned step and reports the independent-noise cell beside it.
   **Fallback:** coverage under 70% still lets C2 run, and the cards carry no intervals.
2. **No harm:** W\*'s median heating-slope interval width is at most W0's, and its median capacity
   bias at most 1.5 times W0's. Remainder: q_b within 30%, level within 0.5 MW of truth.
4. **Error leak control:** with a 25% scale error at one synthetic Primary for 30 days, the siblings'
   median damage at W\* is at most 1.2 times W0's.
5. **Sabotage (the gate can fail):** with one in-sum Primary dropped from the BSP row, the
   remainder's level moves beyond 3 clean-system standard deviations in at least 80% of 100 systems.
6. **Positive control for the claim (also C2's power check):** with a 25% scale error for 30 days at
   one synthetic Primary, W\* cuts that Primary's damage by at least 0.30 at the median system, in both
   halves of the systems. This reading fails if the weight is effectively zero. **Clarified after
   the module was written, before any fit (see "Clarifications"): the damage is scored on the clean
   scored year, outside the corrupted window, with the pilot's one remainder step penalty.**
7. **Power for C1:** the paired half-width (99.17% interval) of W\* minus W0 on the clean synthetic
   systems, at the **real site count** under the chosen BSPs (2 to 3 sites, not 5), is measured and
   printed before any real fit. If it exceeds 0.03 points, the verdict is capped at
   `partly supported`. With 100 systems the 98.3% half-width on a coverage near 0.8 is about ±0.10,
   so the 70 to 90% band separates 80% from 60%.

### Planned contrasts

**Three contrasts, Bonferroni-corrected, each with site counts, absolute scores, mean-forecast error,
and the replicate year's estimate.** Screening tier: one frozen fit per cut. C1 and C3 resample whole
calendar months (`studies.bootstrap`, 3 seeds, paired across arms).

- **C1, the hint costs nothing on clean data:** W\* minus W0 on the Primaries' day-D+1 quantile score,
  outside hand-filtered events, at the Primaries beneath the chosen BSPs (D9 beside them).
  Non-inferiority at margin 0.05 points (about 2% of GAM-K's score; twice the 0.023 noise
  reference): passes if the interval's upper end is at most +0.05 in the scored and replicate years
  and gate reading 7 holds. **Gain bonus** if the interval lies wholly below zero with an estimate
  at most -0.03. W-max minus W0 is reported beside it.
- **C2, robustness to a metering error (the headline):** inject one error at one Primary's readings
  in training history only. **Gap:** 6, 24, or 72 hours missing (the BSP row stays). **Stuck:** the
  last good value repeats for 1, 3, or 7 days, shorter than the cleaning rule's flag threshold (read
  from `nged_data.cleaning` and committed before any score). **Scale:** readings times 0.8 or 1.2 for
  30 or 90 days. Each injection lies in the 12 months before the cut, 7 days apart at least, 10 seeds
  per cell and Primary, drawn independent of residuals. **Damage** = injected minus clean score for
  that Primary's day-D+1 forecast on the clean scored year, in points. **Reduction** = 1 − (summed
  damage at W\*) / (summed damage at W0), a ratio of sums, **read only for a kind and Primary whose
  W0 damage has an interval wholly above zero**; damage is printed beside every reduction, filter on
  and off. **Interval:** two-level bootstrap (Primaries, then injections within each); the Primary
  count (probably 2 to 3) is printed beside it. Passes if, for at least 2 of the 3 kinds, the
  reduction's lower end exceeds 0.10 and its estimate is at least 0.30, **the sign agrees in every
  Primary,** and the siblings' mean damage at W\* is at most 1.2 times that at W0.
- **C3, the BSP forecast by-product:** BU0 and the W\* bottom-up forecast against BD on the BSP's day-D+1
  quantile score; descriptive, with a non-inferiority reading at 0.10 points of the BSP's median
  load. `studies.node_curtailment` is not used: it needs a generator forecast per node, which no arm
  here produces. C3 is reported, never decisive.

**Tails, every arm:** coverage and pinball at q01, q05, q95, q99, on all rows and the 48 hours after
each injection. **Comparators:** every arm, climatology, `manual_heuristic`, and experiment 111's
GAM-K (2.800) and BG2 (2.780) as reference rows (not like for like).

### Predictions

Written before any fit; each line can fail, and none overrides the decision rule.

- **Step 0:** a_P 0.85 to 1.15, a_G -0.9 to -1.2; `recall30` above 0.5 at one BSP, 0.2 to 0.5 at the
  other.
- **Gate:** coverage 70 to 90%; width at W\* 5 to 25% below W0; reading 6 reduction 0.3 to 0.6;
  reading 7 half-width 0.02 to 0.05.
- **W\*:** 0.3 to 3. **C1:** within ±0.02, no gain bonus; W-max worse by 0.05 to 0.15.
- **C2:** gap reduction 0.3 to 0.7; stuck 0.4 to 0.8; scale 0.2 to 0.5; sibling ratio 1.0 to 1.3;
  filter-on damage under 0.05 points for 30-day errors, so the filter-off reading carries most of it.
- **C3:** BU0 and W\* within ±0.05 of BD.

### Decision rule

**Step 0 first, then the gate, then C1, C2, C3; C1 and C2 must hold in the replicate year too.**

- **`supported`:** at least one BSP informative; the gate passes with reading 7 at most 0.03; C1 and
  C2 both pass in the scored and replicate years.
- **`partly supported`:** the gate and C1 pass, and C2 passes in only one year or one kind; or the
  verdict is capped (pilot on an uninformative BSP, or reading 7 above 0.03) and the rest holds.
- **`small gain`:** the gate and C1 pass, and C2's reduction interval lies wholly above zero but
  misses the 0.30 estimate or the 0.10 lower-end condition.
- **`not supported`:** the gate passes and C2's interval lies within ±0.10 of zero (including a
  failed pilot), or C1's upper end exceeds +0.05 in both years.
- **`refuted`:** the gate passes and C2's interval lies wholly below zero (the hint spreads the
  error across the Primaries).
- **`inconclusive`:** a failed plumbing control or gate reading (the take-home names the reading and
  what it rules out), or anything else.

**The verdict rests on injected errors and the scored year's clean scores; real labelled events are
descriptive only.**

### Substation cards and privacy

**Each decision site's card gets a provisional "network" panel with the line "Provisional: fitted to
past weather on a screening run; weakly identified at real noise levels (experiment 113: a heating
slope's 80% month-block interval is about 38% of the slope wide)".** It shows the Primary's own
example forecasts for W0 and W\* on 4 fixed weeks chosen by rule before scoring (the week starting
on the first Monday on or after the 15th of July 2025, October 2025, January 2026, and April 2026),
median with 80% and 90% bands against observed; day counts, no dates, own-capacity units,
`aria=False`. Generator-linked series appear only as G-c in normalised units, with no dates.

**BSP results never sit on a Primary's card.** The BSP coefficients, residuals, and C3 appear in one
table that says "two bulk supply points" and carries no D-label, so no card reveals which Primary
sits under which BSP (two cards sharing a remainder would give away the grouping).

**Privacy.** The page uses D-labels only. Metered generators appear only as one sum labelled "G-c", never beside a D-label, name, or identifier, and with no dates. The sum appears as a coefficient and as a ratio to a BSP's median load (a_G, the generator share of a BSP's load), never as a series. The page may name interconnected D-label pairs (D5-D16, D10-D11) but no substation names and no other topology: no supply-point membership, no unmetered partners, no edge counts, no statement of which Primaries sit under which BSP. No generator's public name appears with its series. A BSP residual that would reveal G-c by subtraction is shown as a ratio.

### Audit before merge: the seven triggers

1. **A control fails:** any plumbing control, gate reading, or P4.
2. **A contradiction:** C2's reduction beyond the oracle's; W\* worse than W0 on clean data with the
   gate passing; C2 passing on a BSP step 0 found uninformative.
3. **Quantile score against mean error:** C1 or C2 with opposite signs in the two.
4. **Leave one out:** dropping a site, injected Primary, or kind flips C1 or C2.
5. **Nested arms:** W\* beating OR on injected windows, or W0 beating W\* on C2.
6. **Interval width:** C1's width over twice or under half experiments 106's or 111's, scaled by the
   square root of the site-count ratio (they use 5 sites).
7. **Dependence:** Flexpectation v2's hierarchy design rests on this, so any positive verdict audits.

"Checks" bullets: median minus mean forecast per site and arm, mean error beside each contrast.

### Module and compute

**Module `studies.dp_hierarchy` (own PR off `main`, two code reviews, green tests, no new dependencies):** stacks per-site `GamLayout`s with a BSP block and wraps `solve_bounded_quadratic` with the weight, remainder, generators, and MVA dark-row mask; tests P1 to P3 under 5 minutes each. No replicate-year or scored-year fit runs before that PR has merged.

**Compute, CPU only, at most 6 cores: about 8 to 18 hours of wall clock at 8 cores, to be re-read at 6.** Single-site Gram accumulation is 12 minutes per site and cut, cached. A joint solve takes 1 to 3 minutes. Counts: tuning grid 42 solves; tuning injections about 300; gate about 400; C2 450 injections x 3 arms plus OR x 2 cuts, about 3,600 solves, each also run through the filter. **Thinning, on the total:** if C2's projected wall clock exceeds 14 hours after 200 solves, C2 drops to 5 seeds per cell and the weight grid to {0, 0.3, 1, 3, 30}.

### Clarifications made after the module was written and before any fit

**These five changes follow a Sonnet code review of the module. They were made before any tuning,
gate, replicate-year, or scored-year fit ran, and are labelled so a reader can tell them from the
original plan.**

- **The remainder's penalties scale with the BSP term.** The BSP term is `(w / sigma_b^2) * (loss +
  penalties)`, so the weight `w` changes how much the BSP reading matters and not how freely the
  remainder can absorb an offset. The original plan added the penalties unscaled.
- **Gate reading 6 is scored out of the corrupted window.** The module's own test showed a BSP
  weight pulls a corrupted Primary back inside the corrupted window (error 11.3 to about 3.3
  points) while the remainder's free level absorbs the constant offset elsewhere. The damage that
  matters in C2 arrives on the clean scored year, so reading 6 scores that year, and also reports
  the in-window error and the BSP residual (the module's own test does this; the gate reports the clean-year damage only, since the pilot has one remainder step penalty).
- **The BSP rows start where the Primaries' training window starts,** so the hint does not
  constrain a Primary's coefficients at times its own block ignores.
- **The remainder's last step knot (at the cut) is dropped,** because its indicator is true for the
  final training row only and cannot be identified.
- **The generator sign is tested with a time-varying series.** A constant would be absorbed by the
  remainder's level.

### Clarification: the closure audit runs first, on its own

**Step 0 runs before the gate and before the joint fits, as a descriptive audit of whether the
Primaries, the metered generators, a level, and unmetered solar close the BSP readings.** The
design session asked for this on 2026-10-20, so it is a labelled clarification made before any
audit result exists.

- **What it reports, by BSP:** the closure residual in points of effective capacity and in MW, the
  residual by hour band and calendar month, and a winter breakdown (November to March, with the
  cold-snap rule fixed before the residual is read).
- **"Cannot tell":** a BSP or winter whose history is too short to give a block-mean standard
  deviation at a window length is reported as "cannot tell", never as a number.
- **Cost:** descriptive only, on CPU, about 2 to 3 hours and under 8 GB.
- **Privacy:** the page names no substation and no topology beyond interconnected D-label pairs,
  and shows the metered generators only as the sum "G-c", without dates.
- **Decisions:** the pre-registered step 0 decision rule is unchanged, and step 1 and the gate
  follow it.

### Step 0 result (first run)

**The closure audit ran before the gate; its numbers are in `audit_report.md`, printed by
`ml-experiments/00124-differentiable-physics-h-bsp-closure-audit-and-bsp-hint/e00124_audit.py`.**
The first-run numbers stay as printed.

- **A coverage rule was added after seeing the history coverage and before any residual was read
  (post hoc).** A series enters a fit only if it holds readings in at least 90% of the rows where
  the BSP does. A series that fails the rule stays inside the remainder, and the report counts it.
- **One BSP closes in the training years and does not hold out of sample.** In-sample residuals are
  2.2 to 3.2 points of effective capacity, with a_P 1.01 to 1.16 and a_G 0.85 to 0.90, and over six
  winters the median residual stays within 1 MW. In the scored year the frozen fit has a 5.5-point
  residual with a +3.7 MW median offset, because its frozen steps do not hold.
- **The other BSP cannot be closed with the generator history available.** About half of its
  generation series have too little history, so their power sits in the remainder: in-sample 11
  points and a_G 0.24. Whether its generators close it is "cannot tell".
- **A fit that may refit hides metering errors from the BSP.** With a refit that includes the
  injected series, a 30-day scale error of 25% is detected in 0% to 3% of injections, and the 24 h
  stuck run in 0% to 1%. A fit trained only on pre-injection rows detects the same 30-day error in
  8% to 50%.
- **By the pre-registered rule neither BSP is informative (`recall30` under 0.2),** so step 1 runs
  as the capped pilot on the BSP with the highest `recall30`, and the verdict cannot exceed
  `partly supported`.

### Clarifications made after a Sonnet code review of the scripts, before the tuning, gate, and scored fits

**The audit was re-run after the review with the fixes below; its first-run report is kept as
`audit_report_first_run.md`, and the first-run numbers in "Step 0 result (first run)" stay as
printed.**

- **The step 0 rule reads the refit variant of detection.** The rule's `recall30` is the detection
  rate when the fit includes the injected series, because that is the fit a running service would
  make. The report also gives the clean fit trained on all pre-cut rows (in-sample, so it flatters
  detection) and a fit trained only on the rows before the injected window's month. In the re-run
  at the tuning cut, the pilot BSP detects a 30-day 25% scale error in 0% (refit), 50% (in-sample
  clean fit), and 50% (pre-injection fit) of injections. The pre-injection fit's false-alarm rate is
  1.2% in-sample but 61% in the year after the cut, because a frozen fit drifts, so a 3-standard-
  deviation detector would not hold up in service. The pilot decision stands.
- **The audit's injected scale error is a share of the Primary's median load, with both signs,**
  and the lower rate of the two signs is reported (before: a share of the instantaneous reading).
  A 90-day threshold with fewer than 8 blocks is reported as "cannot tell".
- **The cold-snap rule is a temperature threshold,** the 5% quantile of winter-day means before the
  tuning cut, applied to every winter (before: a fixed list of past days).
- **The published audit report drops the counts of Primaries and generation series per BSP,**
  which would narrow down which Primaries sit under which BSP.
- **Tuning uses 10 seeds per cell and Primary (plan: 30 per kind),** and the gate adds the missing
  sabotage reading, builds the BSP from the Primaries' noisy series, gives the first MW Primary the
  same winter-evening hump in half the systems, and freezes its thresholds in
  `e00124_gate_readings.py` before the gate runs.
- **The pilot uses the smooth-drift form without L1 steps, fitted capacity, a frozen fit, and the
  mean absolute error as its score,** with no Kalman filter. The filter-on reading, which the
  manager's ruling makes the production-relevant one, is therefore not produced, and the verdict is
  capped at `partly supported`.

### Rulings made before the run

- **A pilot is capped.** A pilot on the better BSP caps the verdict at `partly supported`.
- **Filter-on carries the verdict.** If filter-on damages are under 0.05 points there is no room for the hint to help: the page says so, reports filter-off as mechanism evidence only, and caps the verdict at `partly supported`.
- **W\* is shared across the two bulk supply points,** because per-BSP tuning would overfit.
- **The data owner is not asked about unmetered partners or 11 kV generation** until January.

### Plan review

One Sonnet plan review found 12 issues; 11 were applied as asked and one in part. The part not applied is the saving against `manual_heuristic`: a BSP forecast has no generator forecast here, so `studies.node_curtailment` cannot be computed, as C3 states.
