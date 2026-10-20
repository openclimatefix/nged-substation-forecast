---
id: "00123"
slug: differentiable-physics-e7-storage-heating-night-step-estimation
title: >-
  Differentiable physics, E7: how large is the winter night step, with the pattern of storage
  heating, at Primary substations, and how does it grow with cold?
part: n/a
theme: differentiable-physics-demand-calendar-and-weather
also_relevant_to: [differentiable-physics-forecast-engine]
status: planned
size: >-
  medium. Stored: private files under
  data/ml-experiments/00123-differentiable-physics-e7-storage-heating-night-step-estimation/ only.
  Serving path: not touched. Degradation rule: not touched. Designs: more than one defensible
  (window, cold variable, estimator); this plan pre-registers one. Callers: this folder's scripts
  only. Asset graph: not touched.
hypothesis: >-
  At Primary substations, demand steps up at the 01:00 to 03:00 rows on winter nights, the step
  grows with cold, and the pooled step over the six decision series is positive at a daily mean of
  3 C. At D5, D10, and D16, which were chosen because winter 2025-26 showed the step, it is 5 to 10
  points; at the other decision series it is smaller.
prediction: see "Predictions"
decision_rule: see "Decision rule"
benchmark: none (estimation); experiment 121's descriptive 01:00 steps are the reference
row_set_id: >-
  e7_nights_v1 (night-level rows built from the observed power of the series in "Substations");
  the six decision series are also rows of dp_mw_v1 (experiment 103)
metric: >-
  Per series: slope (points per degree below 15.5 C) and step at a daily mean of 3 C (points), each
  with a 90% interval; pooled values; gate: interval coverage over repeated draws
verdict: <supported | partly supported | inconclusive | invalid | failed>
production_status: not-for-production
take_home_message: <written after the results>
advice: <written after the results>
---

**Status: planned. This page is the plan, committed before any code runs.**

**E7 is an estimation experiment. The result is a table of estimates with intervals, not a win-or-
lose verdict, and the experiment forecasts nothing.** The word `calibrated` is used only for the
known-answer gate (interval coverage within a stated band), and `detectable` only for the per-series
reading. The plan follows `briefs/dp_series.md`, then `briefs/common.md`. The Sonnet plan review is
triaged at the end of this page. No data on or after 2026-07-01 is read, nothing in `packages/`
changes, and no dependency is added.

## Pre-fit clarification: what the headline is (written before any fit)

**The headline is the pooled step over all six decision series (D5, D9, D10, D11, D14, and D16),
with a random effect across series, at a daily mean of 3 C.** The extension series, E7-X, are
shown separately. D5, D10, and D16 are reported as the subset that prompted the study, labelled
"selected for showing the step", and no verdict rests on them alone. Every fitted series, including
each with no step, appears in the unpooled table. This replaces the earlier draft, in which the
verdict rested on D5, D10, and D16.

**The selected subset is expected to look larger than the pool, and the gap is the selection
effect.** Experiment 121 read winter 2025-26 and found a step at 01:00 of 4.3, 7.1, and 5.3 points
in the residuals of the day-ahead model at D5, D10, and D16, and 1.9 to 3.4 points at D11, D14, and
D9. The choice of those three series came from that winter, so winter 2025-26 is labelled the
selection winter.

## Pre-fit clarification 2: the unit is points of effective capacity (written before any fit)

**Every size in this plan is in points of effective capacity, and the first draft mislabelled the
unit as megawatts.** One point is 1% of the series' effective capacity. Experiment 121's step
figures (4.3, 7.1, and 5.3 at D5, D10, and D16) are in points: its quantile scores are in points,
and a 7.1 MW step at D10 would exceed the series' 5.4 MW effective capacity. The wrong label was
found in test 1, which showed D10's night difference to be about 0.4 MW (about 8 points). Every
number in the plan (the predictions, the injected shapes `S` of 4 and 7, the bias bound of 0.5, the
trigger thresholds of 4, 8, and 15, and the 1.7 summer step) keeps its value and now reads in
points. Each series' night difference `d` is converted to points before any fit, so the pooled
steps average comparable quantities, and the page also prints each series' step in MW (one point
at D10 is 0.054 MW). Test 1 printed raw night differences for one series in the design winters
and a per-series mean `d` over all nights of winters 2021-22 to 2025-26; no estimate, interval, or
fit was computed before this clarification.

## The question, and what earlier experiments already say

**The night step is real in one winter and unmeasured across winters.** Experiment 121 tried to
detect the step through quantile scores over all 20 night half-hours, where a step of 4 points is
diluted to about one noise standard deviation, and its fourth gate had to be redefined twice.

**E7 changes the instrument.** E7 compares five window half-hours with four baseline half-hours of
the same night, so the dilution disappears and a slow level error cancels. The estimator is a
regression of the night difference on cold, in the spirit of the degree-day regressions used for
load, and the pooling is the random-effects meta-analysis of [DerSimonian and Laird
(1986)](https://doi.org/10.1016/0197-2456(86)90046-2) with the [Hartung and Knapp
(2001)](https://doi.org/10.1002/sim.1009) interval. The block bootstrap follows [Künsch
(1989)](https://doi.org/10.1214/aos/1176347265). The earlier experiments this plan builds on are
[111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md),
[113](../00113-differentiable-physics-n5-convex-identification-at-realistic-noise/result.md),
[114](../00114-differentiable-physics-n3-n4-interactions-and-heating-threshold/result.md),
[120](../00120-differentiable-physics-sh-convex-daily-shape-library/result.md),
[121](../00121-differentiable-physics-mh-multi-hinge-heating-curve-and-night-step/result.md), and
[122](../00122-differentiable-physics-three-winter-backtest-gam-k/result.md). One targeted search
for a DOI on Economy 7 or storage-heating load found none on temperature dependence. The nearest
sources are [Elexon CP1058A](https://assets.elexon.co.uk/wp-
content/uploads/sites/11/2012/02/28171857/cp1058a.pdf), which describes the Economy 7 profile class,
and a [statistical review of UK residential electrical loads](https://arxiv.org/pdf/1306.0802);
neither was checked for a temperature-dependent night step.

## Estimand

- **Window rows:** the five half-hours starting 00:30, 01:00, 01:30, 02:00, and 02:30. Test 1 prints
  the timestamp convention of the observed file against experiment 121's code, and the window
  follows the documented convention only. It is fixed in test 1, before any estimate, and never
  moved. Sensitivity (descriptive): the three-row window 00:30 to 01:30.
- **Baseline rows:** the four half-hours starting 22:00 to 23:30 of the same night, on the same
  clock as the window. A night is labelled by its evening's local date, and its day type (weekday,
  Saturday, Sunday) is that date's.
- **Night difference `d`:** the mean of the window rows minus the mean of the baseline rows, in
  points.
- **Cold `c`:** `max(0, 15.5 - T)` degrees, where `T` is the daily mean of hourly ERA5 temperature
  at the substation's cell over the local calendar day containing the window rows. The daily mean
  is a proxy for the temperature that sets storage-heater charging, and it includes the afternoon
  after the step. Its correlation with the mean temperature from 18:00 to 08:00 is printed, and a
  night-temperature `c` is a descriptive sensitivity.
- **Primary rows (GMT nights):** nights between the October and March clock changes, when the local
  and UTC clocks agree. Clock-change nights, bank-holiday nights, and the Christmas to New Year run
  are dropped from every fit.
- **Summer reference `d_summer`:** the mean of `d` over June to August nights on the local clock.
  Experiment 121's one summer pair found a step of 1.7 points or less; the page states the
  assumption that the summer night has no step.
- **Slope `b`:** the points change in `d` per degree of `c`, from `d = a_w + b*c + day-type terms +
  noise` on GMT nights, with a separate intercept `a_w` for each winter `w` and a common `b` across
  winters. The slope needs no summer assumption.
- **Step at a stated cold night:** `step = a_hat + b*c_ref - d_summer`, with `c_ref = 12.5` degrees
  (daily mean 3 C, inside experiment 122's cold-snap range of -2.0 to 3.1 C), where `a_hat` is the
  mean of the winter intercepts. The step is the full cold-dependent and flat step, with no demand
  model removed. The flat part is `a_hat - d_summer`, so reading the step at `c = 0` is an
  extrapolation, because few GMT nights are warmer than 15.5 C. The flat part is reported separately
  from `b`.
- **Excess over B0 (cross-check R):** the same regression on experiment 114's stored B0 residuals
  measures only the part of the step that B0's free daily profile and its 00:00 to 04:00 heating
  band do not already absorb. R is a different, smaller quantity, expected at 30 to 60% of D, and
  is compared with D only as a measure of the absorption, never as an error.

## Substations and histories

**Fitted series: the six decision series, which are DP Primaries, and the eight E7-X Primaries.**
The rule-based flag of `nged_data.cleaning` is applied to all 14 series, so E7-X needs no
hand-filtered switching list. The six decision series are also run with their hand-filtered list as
a sensitivity. E7-X has its own row set, `e7_nights_v1`, outside `dp_mw_v1`. D12 reports apparent power
in MVA, so its night difference is an apparent-power step, converted to points in the same way.

| Series | Observed from | Winters (2021-22 to 2025-26) | Role |
|---|---|---|---|
| D5, D10, D11 | 2019-09-17 | 5 | decision series; D5 and D10 prompted the study |
| D14 | 2019-10-21 | 5 | decision series |
| D16 | 2020-12-22 | 5 (the history covers 2021-07 onwards) | decision series; prompted the study |
| D9 | 2023-08-23 | 3 (2023-24 to 2025-26) | decision series |
| D2, D3, D4, D8, D12, D19, D21, D22 | 2019-09-17 | 5 | **E7-X**: not chosen for any step |
| D15, D18 | 2025-02-06, 2024-12-11 | n/a | generator-linked, shown only as G-c, not fitted |

**Winters 2021-22 to 2025-26 only.** Winter 2019-20 is not added: its window would leave about 6
months after the lockdown months go out, with no summer and no pinned-capacity vintage.

**Winter roles.** Winter 2023-24 (tuning) and winter 2024-25 (replicate) are used for design checks
(block length). Winters 2021-22 and 2022-23 are not opened for any design choice, and their
estimates are read only after the gate has run. Winter 2025-26 is the selection winter, and the
maintainer ruled that it is fitted only after PR 1397 (experiment 114) has merged. Headline winters
are 2021-22 to 2024-25 (the winters that did not choose the three series), and winter 2025-26 is
reported beside them. There are at most four headline winters and five in all, from one weather
product and one set of substations, so the winters are separate years and not independent
experiments. The page prints the winter count at every estimate.

## Estimators

- **D (primary): the direct per-night regression above.** Intervals come from a 14-day-block
  bootstrap of nights stratified by winter, 1,000 draws, the same count in the real fits and in the
  gate. A bounded (non-negative) `b` is fitted only as a descriptive variant, never in the gate.
- **R (cross-check): the same regression on stored B0 residuals** (`predictions/{tuning,replicate,
  scored}_b0`, median, day D+1), for winters 2023-24 to 2025-26 only, at the six decision series.
  Experiment 114 stores no earlier residuals. R is not injected into the gate, because injected data
  would need B0 refitted per draw. R reads winter 2025-26 only after PR 1397 has merged, and the run
  pins the merge commit's SHA in `decisions.json`.
- **E7-K (descriptive):** R on the stored `*_b0_k` and `*_t_all_k` predictions, which shows how much
  of the step the Kalman filter removes.
- **Cold-snap subset (winter-first rule):** the coldest 10% of days of a winter by the mean ERA5
  temperature averaged across the six decision series (experiment 122's rule). The day count is
  stated, and `too few days` where under 5. On the subset the page prints the raw mean of `d` minus
  `d_summer`. November to March is reported beside November to February.
- **Event-day rule:** not applicable, because E7 has no scored contrast across calendar events.

## Pooling

**The pool is a random effect across series, with limits.** With `k` series the pooled mean has a
Hartung-Knapp interval (t with `k - 1` degrees of freedom) and, beside it, a percentile interval
from resampling whole series. The between-series standard deviation `tau` is printed with a profile-
likelihood interval. The pooled mean is marked `cannot tell` if the two intervals disagree on
whether 0 lies inside. With fewer than 5 series in a set, the page prints the plain mean and range,
with no `tau`. Neighbouring cells share one ERA5 product, so the series are not independent, and the
page says the pooled intervals are optimistic.

**Pooled sets, side by side:**

1. **the six decision series** (headline);
2. D5, D10, and D16 (selected for showing the step);
3. D11, D14, and D9 (not selected);
4. the eight E7-X series;
5. all 14 series.

The gap between sets 2 and 3 shows the selection effect.

## Known-answer test and gate

**Order: test 0, test 1, the rank check, design checks, the gate, and only then the estimates of
winters 2021-22 and 2022-23, and winter 2025-26 after PR 1397 has merged.** The gate can fail by
design. A failed gate is recorded as it stands and is not redefined, and it removes the `detectable`
label, so the raw means carry the page.

0. **Feasibility.** Each fitted series has the rule-based flag, at least two winters, and a night
   table with every window and baseline row present on at least 90% of its nights. Failures are
   named and dropped.
1. **Reproduction (blocking).** The night table's `d` equals a hand-computed `d` on 20 printed
   nights; the timestamp convention and window rows are printed; no row on or after 2026-07-01; R's
   stored residuals reproduce experiment 114's stored B0 quantile score for the tuning year to 1e-4.
2. **Known-answer injection (can fail).** Noise: real nights of the same series with D's fitted
   cold-dependent part removed from the five window rows, resampled in 14-day blocks within winter,
   with the real covariates. Injected shapes, in points at `c_ref`, scaled by `c / 12.5`, for `S` in
   {4, 7} plus a placebo `S = 0`:
    - **matched:** an equal amount on the five window rows (the fitted form);
    - **ramp-plateau:** weights 0, 0.5, 1, 1, 1 on the five window rows, continuing at weight 1
      until 07:00. The truth is the mean weight over the window times `S`;
    - **flat:** `S` on the window rows of every GMT night, not scaled with cold (truth `b = 0`,
      step `= S`);
    - **night-temperature:** the matched shape driven by the 18:00 to 08:00 mean temperature instead
      of the daily mean (descriptive: its bias is reported, and no coverage target is set).

   There are 200 draws per cell per series, at all fitted series.
    - **Target (the label rests on these rows):** coverage of the truth by the 90% interval of the
      step at `c_ref`, for the matched, ramp-plateau, and flat shapes and the placebo, is 85 to 95%
      pooled over series and 80 to 98% for each series. With 200 draws the Monte Carlo standard
      error is about 2.1 points per series and about 0.9 points pooled over 14. Block resampling
      makes the draws less independent, so these errors are lower bounds.
    - **Bias:** the mean estimate lies within 0.5 points of the truth for the matched and flat
      shapes at 4 and 7.
    - **Power (descriptive):** the share of draws with the 90% interval above 0, at 4 and 7 points,
      per series. A series under 80% at 4 points is marked "weak".
    - **Pooled interval:** `S` is drawn per series from {0, 0, 4, 4, 7, 7} across the six decision
      series, and the pooled mean's Hartung-Knapp interval covers its true mean in 85 to 95% of 200
      draws.
3. **Rank check (blocking).** The design matrix (`1`, `c`, day-type terms, winter intercepts) has
   full rank, its smallest singular value and the variance inflation of `c` are printed per series,
   and a series with a variance inflation above 10 is dropped.

**Gate result:** the table says `interval coverage within 85 to 95%` or names the failing cell. The
page does not say the estimate is right, because the gate tests the estimator on shapes this plan
chose.

## Timing (descriptive)

**The gate cannot calibrate timing, because cold British Summer Time (BST) nights are few.** A cold
BST night is a night in BST with a daily mean below 8 C. Per series and per winter-pair, the page
prints the count and says `cannot tell` where it is under 10. Where it is 10 or more, the estimator
is `delta`, the mean over those nights of `d` measured with the local-clock window minus `d`
measured with the UTC-clock window (the baseline on the matching clock). A step fixed to the local
clock makes `delta` positive, and a step fixed to UTC makes it negative. A 90% block-bootstrap
interval is printed and the label (`local`, `UTC`, `unresolved`) is marked uncalibrated.

## Reporting

- **Table 1 (unpooled, first):** per series, `b`, the flat part, and the step at `c_ref`, each with
  90% intervals, the winters used, the raw cold-night mean of `d` minus `d_summer`, R and its ratio
  to D, and the `detectable` flag. Every fitted series appears, including those with no step.
- **Table 2 (winters):** the step at `c_ref` per winter per series, the selection winter marked.
- **Table 3 (pooled):** the five sets above, with both intervals, `tau`, and `cannot tell` where it
  applies. The first set is the headline.
- **Table 4 (timing):** counts, `delta`, and the label.
- **`detectable`:** the 90% interval of the step at `c_ref` lies wholly above 0, and the gate row
  for that series is within its band. No multiplicity correction is made: with 14 series, about 0.7
  false positives are expected under no step (5% one-sided each), and the page says so.
- **Winter breakdown:** every estimate is shown per winter, with the count of winters.

## Predictions

Written before any estimate; each line can fail.

- **Headline pooled step at `c_ref`, six decision series, headline winters:** between 2 and 6
  points, with the Hartung-Knapp interval wholly above 0.
- **Selected subset (D5, D10, D16):** 5 to 10 points, above the pooled step of D11, D14, and D9,
  which is 1 to 3.5 points and may include 0. The selected subset's larger value is the selection
  effect.
- **E7-X:** each series 0 to 3 points, and the E7-X pooled step 0.5 to 2 points, below the pooled
  step of the decision series.
- **Slope `b`:** positive at D5, D10, and D16, between 0.4 and 0.8 points per degree.
- **Flat part:** `a_hat - d_summer` within +-2 points at every series. A series whose 90% interval
  lies wholly outside +-2 points fails the line.
- **R against D:** 30 to 60% at D5, D10, and D16.
- **Raw contrast:** the raw cold-night mean of `d` minus `d_summer` lies within 2 points of D's step
  at the mean `c` of those nights (about 8 to 10 degrees), not at `c_ref`.
- **Timing:** `cannot tell` at most series; if a label is printed, `local`.
- **Gate:** coverage within the bands. The chance of a failing cell is judged about one in three,
  because the ramp-plateau shape changes the window mean.
- **Run time:** under 1.5 hours of wall clock on CPU.

## Decision rule

**There is no win-or-lose verdict on a hypothesis about a model. The verdict word summarises the
estimates and the gate:**

- **`supported`:** the gate rows are within their bands for all six decision series, the headline
  pooled step over the six decision series has a Hartung-Knapp interval wholly above 0 in the
  headline winters, and the percentile interval agrees.
- **`partly supported`:** the gate is within its bands and the pooled step is positive but `cannot
  tell`, or positive only in the selection winter.
- **`inconclusive`:** a gate row for a decision series fails, or fewer than 2 headline winters are
  available at those series. **`invalid`:** a leak (a row on or after 2026-07-01). **`failed`:**
  test 1 or the rank check cannot pass.
- The estimate does not choose between models and does not change the production forecast. A
  `supported` result feeds the substation cards and a calibrated `b` can serve as a prior for
  experiment 121's night arms; neither is part of this plan.

## Audit before merge: the seven triggers

**The seven triggers are those of `briefs/common.md` and `briefs/dp_cal.md`, applied to these
results.**

1. **A control fails:** test 1, the rank check, or the placebo coverage outside its band.
2. **Contradiction:** a pooled step above 15 points, a negative step at a series whose raw cold-
   night mean exceeds 4 points, or an E7-X pooled step above that of the decision series.
3. **Estimate and raw contrast disagree in sign** at a series.
4. **Leaving one series out** moves the pooled mean outside the all-series pooled interval.
5. **A null control moves:** the unbounded and bounded D disagree at a series with `b` far from 0.
6. **Interval width:** D's interval at `c_ref` is wider than 8 points at a decision series.
7. **A later design depends on it:** the cards and experiment 121's night arms use the numbers, so
   trigger 7 fires. The manager runs the Opus review of the estimator and the gate after the quota
   resets (01:00 UTC on 2026-10-21); this session runs one Sonnet code review for leakage (the cut
   at `train_end`) and one Sonnet write-up review.

## Privacy

**D-labels only, with dates, for the Primaries; no substation names, no topology.** D15 and D18 are
generator-linked and appear only as G-c, normalised, with no dates, and are not fitted. Fitted steps
in points are acceptable for Primaries, which are not generators. Per-series slopes are shown only
where the rank check passes and the gate row is within its band, because experiment 121 forbade per-
site slopes from weakly identified fits.

## Substation cards

**Section label: "Provisional: fitted to past weather, estimation only".** One card per fitted
series shows:

- the step at a stated cold night (daily mean 3 C), in points, with its 90% interval, the slope in
  points per degree, and the winter-versus-summer contrast;
- the flat part and the summer reference;
- the **start time**: the first half-hour whose mean (observed minus baseline), on cold GMT nights
  minus the summer mean, exceeds half of its value at 02:30, with a bootstrap interval in half-
  hours;
- the **timing label** (local or UTC, where estimable) or `cannot tell`, with the count of cold BST
  nights;
- R's ratio to D and the Kalman filter's removal (E7-K);
- an example forecast week, where the series has one in `dp_series.md`'s selection rule, drawn from
  stored predictions.

## Comparison with experiments 121 and 120

**Experiment 121 (MH).** Its night-arm estimates are compared with D's step per series, with both
intervals and their overlap. **Experiment 120 (SH)** learns an unsupervised daily-shape library with
a 01:00 shape. When SH reports, the comparison per series is SH's integrated 01:00 jump over the
five window rows, in points at the same cold night, beside D's step. A series where SH shows no jump
and D's interval lies wholly above 0 is a case the learned library misses, and the card says so. The
plan does not decide which is right.

## Run order, reuse, compute, and time

**Scripts import only their own folder, `studies.*`, and reviewed packages.** The scripts read the
series' power through `studies.power.scan_power` (cleaned readings), ERA5 temperature from the
stage 1 pilot's `era5_features.parquet`, and the copy of experiment 114's and experiment 121's
window and block-resampling conventions. D needs no demand model.

**Order:** night table and tests 0, 1, and 3; design checks on winters 2023-24 and 2024-25 (block
length); the gate; then winters 2021-22 and 2022-23; then winter 2025-26 after PR 1397 has merged;
tables, cards, write-up.

**CPU only, 6 cores, with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1` in
every worker (one process per series).** The run pauses when the 1-minute load average is above
about
35. The budget is 1.5 hours of wall clock, no GPU.

## What the write-up must contain

The take-home says how large the full night step is at a daily mean of 3 C, pooled over the six
decision series and per series, whether it grows with cold, what the selection effect is, how much
of the step B0 already absorbs, and that the clock is not identified where cold BST nights are few.
The page is figure-led: raw first (the mean of `d` against `c` with the fitted line and the summer
reference), then the step per series with intervals, the same per winter with the selection winter
marked, known-answer recovery per shape, the pooled forest plot with the five sets, and cold BST
night counts.

**Caveats:** the three large series were chosen on the selection winter; the five winters share one
weather product; the summer reference may hold a step of up to 1.7 points; the flat part is an
extrapolation; the step has the pattern of storage heating and NGED has not confirmed the cause
(daylight, lighting, and switching schedules also change at night); the daily mean temperature is a
proxy for the temperature that sets charging; the gate tests only the shapes this plan chose.

## Size: the five triggers

1. **Stored:** private estimates and figures under `data/ml-experiments/`, and the write-up under
   `docs/ml-experiments/`; nothing in Delta. 2. **Serving path:** not touched. 3. **Degradation
   rule:** not touched. 4. **More than one defensible design:** yes (window, cold variable,
   estimator); this plan pre-registers D. 5. **Callers not nameable without searching:** none.
   Size: medium.

## Review triage

**The plan was reviewed once by a Sonnet agent (30 findings), and the manager and maintainer rulings
were applied: D replaces the earlier joint estimator; the bound is removed from the gated estimator;
the gate tests the used estimator on misspecified shapes; the pooled estimate is reported after the
unpooled table; E7-X runs on the rule-based flag; no winter 2019-20.** The full review is private. A
second change, the pre-fit clarification at the top of this page, makes the six decision series the
headline.
