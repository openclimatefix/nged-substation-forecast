---
id: "00124"
slug: differentiable-physics-h-bsp-closure-audit-and-bsp-hint
theme: differentiable-physics-pooling-across-substations
also_relevant_to: [hierarchy-and-pooling, switching-detectors]
status: planned
size: >-
  complex for the new module studies.dp_hierarchy (own PR, two code reviews); medium for the
  scripts. Private files only; serving path and degradation rules untouched; several defensible
  designs, one pre-registered each; callers are this experiment's scripts only.
hypothesis: >-
  A bulk supply point (BSP) reading, fitted jointly with the Primaries beneath it as the sum of
  their demand terms plus the metered generators plus a penalised remainder, with a tuned weight,
  limits the damage that a metering error at one Primary does to that Primary's forecast, without
  raising the Primaries' scores when no error is present.
prediction: see "Predictions" below, written before any fit
decision_rule: see "Decision rule" below and "Decision rule for step 0"
benchmark: >-
  W0, W*, W-max, OR, a direct GAM on the BSP, climatology, manual_heuristic; experiment 111's GAM-K
  and BG2 as reference rows
row_set_id: >-
  dp_mw_v1 and dp_mva_dark_v1 (experiment 103); Primaries D5, D10, D11, D14, and D16 (the sites
  under the two chosen BSPs); tuning 2023-24, replicate 2024-25, scored 2025-26
metric: >-
  Day-D+1 quantile score in points (C1); damage in points and ratio of summed damages (C2); BSP
  quantile score (C3)
verdict: <supported, partly supported, small gain, not supported, refuted, or inconclusive>
production_status: not-for-production
take_home_message: <written after the results>
advice: <written after the results>
date: <date>
pr: <pr>
---

# Differentiable physics, H: does a bulk supply point reading limit the damage from a Primary's metering error?

**Status: planned. Written before any fit.** This page holds the plan, and the results will replace this notice.

## What this experiment asks, and what the maintainer will accept

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

## Step 0 (H0): the BSP closure audit

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

### Decision rule for step 0, fixed before any residual is read

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

## Step 1 (H1): the joint fit and the weight

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

## Arms
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

## Plumbing controls and the blocking known-answer gate

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
   halves of the systems. This reading fails if the weight is effectively zero.
7. **Power for C1:** the paired half-width (99.17% interval) of W\* minus W0 on the clean synthetic
   systems, at the **real site count** under the chosen BSPs (2 to 3 sites, not 5), is measured and
   printed before any real fit. If it exceeds 0.03 points, the verdict is capped at
   `partly supported`. With 100 systems the 98.3% half-width on a coverage near 0.8 is about ±0.10,
   so the 70 to 90% band separates 80% from 60%.

## Planned contrasts

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

## Predictions

Written before any fit; each line can fail, and none overrides the decision rule.

- **Step 0:** a_P 0.85 to 1.15, a_G -0.9 to -1.2; `recall30` above 0.5 at one BSP, 0.2 to 0.5 at the
  other.
- **Gate:** coverage 70 to 90%; width at W\* 5 to 25% below W0; reading 6 reduction 0.3 to 0.6;
  reading 7 half-width 0.02 to 0.05.
- **W\*:** 0.3 to 3. **C1:** within ±0.02, no gain bonus; W-max worse by 0.05 to 0.15.
- **C2:** gap reduction 0.3 to 0.7; stuck 0.4 to 0.8; scale 0.2 to 0.5; sibling ratio 1.0 to 1.3;
  filter-on damage under 0.05 points for 30-day errors, so the filter-off reading carries most of it.
- **C3:** BU0 and W\* within ±0.05 of BD.

## Decision rule

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

## Substation cards and privacy

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

## Audit before merge: the seven triggers

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

## Module and compute

**Module `studies.dp_hierarchy` (own PR off `main`, two code reviews, green tests, no new dependencies):** stacks per-site `GamLayout`s with a BSP block and wraps `solve_bounded_quadratic` with the weight, remainder, generators, and MVA dark-row mask; tests P1 to P3 under 5 minutes each. No replicate-year or scored-year fit runs before that PR has merged.

**Compute, CPU only, at most 6 cores: about 8 to 18 hours of wall clock at 8 cores, to be re-read at 6.** Single-site Gram accumulation is 12 minutes per site and cut, cached. A joint solve takes 1 to 3 minutes. Counts: tuning grid 42 solves; tuning injections about 300; gate about 400; C2 450 injections x 3 arms plus OR x 2 cuts, about 3,600 solves, each also run through the filter. **Thinning, on the total:** if C2's projected wall clock exceeds 14 hours after 200 solves, C2 drops to 5 seeds per cell and the weight grid to {0, 0.3, 1, 3, 30}.

## Rulings made before the run

- **A pilot is capped.** A pilot on the better BSP caps the verdict at `partly supported`.
- **Filter-on carries the verdict.** If filter-on damages are under 0.05 points there is no room for the hint to help: the page says so, reports filter-off as mechanism evidence only, and caps the verdict at `partly supported`.
- **W\* is shared across the two bulk supply points,** because per-BSP tuning would overfit.
- **The data owner is not asked about unmetered partners or 11 kV generation** until January.

## Plan review

One Sonnet plan review found 12 issues; 11 were applied as asked and one in part. The part not applied is the saving against `manual_heuristic`: a BSP forecast has no generator forecast here, so `studies.node_curtailment` cannot be computed, as C3 states.
