# Plan: is a time-varying effective capacity worth estimating for solar forecasts? (issue #1143)

## Problem

**Every solar forecast in this repository divides power by one fixed number per plant, so nothing
tells the forecast when a plant loses part of its capacity.** The fixed number is the 99th
percentile (P99) of the plant's metered output over its whole history. The lag-features study
([PR #1139](https://github.com/openclimatefix/nged-substation-forecast/pull/1139)) found one of its
6 solar plants producing two thirds to three quarters of its neighbours' output in 3 of 18 months
(cause unknown). The baseline XGBoost model learned those months through its day-of-year column and
then under-forecast a normal March 2026 by 14.6 points of capacity. The roadmap's answer is to
divide power by a time-varying capacity estimate before training and multiply the forecast back.
Issue #1143 asks whether such an estimator is worth building for the forecast, and which techniques
look most promising.

## Planned solution

**Stage A runs 9 capacity estimators on synthetic planted losses, with no forecast in the loop, and
answers "which technique?" from how well each estimator recovers the planted loss.** Stage B
normalises the lag study's baseline XGBoost forecaster by the one causal estimator a fixed rule
picks from Stage A, and answers "is it worth it?" with a single planned contrast against that
baseline, scored in megawatts against the fixed P99.

## Verdict, size and departures

**Verdict: worth doing.** The roadmap names this head-to-head as the v0.7 decision ([capacity
estimation](https://openclimatefix.github.io/nged-substation-forecast/roadmap/capacity-estimation/)),
and every input is already on disk.

**"Unresolved" is a likely answer to the forecast question, and the page says so up front.** The
lag study's hindsight monthly rescale lowered the error by 0.337 points [−0.473, −0.215], but it
matched each plant-month's own energy and so also removed the month's weather-forecast bias. That
figure bounds a monthly energy correction, not a capacity correction. The capacity-only bound in
Stage B is likely smaller, so the 2% threshold (0.18 points) is a larger share of it. Stage A's
verdict does not depend on that power.

**Size: complex.** What gets stored: a published study page. Production serving path: not touched.
Degradation rule: none touched. More than one defensible design: yes (causal timing, hindsight
definition, selection rule, scoring, planted-loss design). Callers not nameable without searching:
no, because no existing function changes. **Reviews:** the simplicity and correctness plan reviews
have run; next is the `study` skill's full sequence, including the mutation pass for the new tests.

**Departures from the issue body and the roadmap:**

1. **The roadmap's order is inverted.** The roadmap ([capacity
   estimation](https://openclimatefix.github.io/nged-substation-forecast/roadmap/capacity-estimation/#the-head-to-head-protocol))
   lets downstream forecast skill decide, with fault injection as a diagnostic. This plan chooses
   on planted-loss recovery, because 9 forecast arms on 3 real low plant-months would leave every
   contrast unresolved, so E2 tests "convex wins per site" on recovery only.
2. **CX2 departs from the roadmap's Candidate A in two ways.** Candidate A fits the AC trace by
   least squares and grid-searches the orientation; CX2 fits the AC trace by absolute error (L1) so
   the whole program stays a linear program, and fixes the orientation from `pv_fit.fit_plant`.
3. **The reference arm is the lag study's baseline B0, not a separate "static capacity" arm.** Under
   `reg:absoluteerror` (gradient the residual's sign, hessian 1), dividing the target by a constant
   only scales the leaves, so a static-capacity arm is B0 to float noise.
4. **The convex estimators are linear programs solved by `scipy.optimize.linprog` (HiGHS), not by
   CVXPY,** because adding `cvxpy` (or SLAC's `solar-data-tools`) to `uv.lock` would block the
   `study` skill's autonomous merge. The roadmap's goal of hands-on CVXPY experience is not served.
5. **Active network management (ANM) data covers one plant, which is not the low-output plant**, so
   the ANM comparison is exploratory.
6. **The forecast decision rule has an equivalence outcome**, which the lag study's rule lacked.

## Literature

**The project's [energy-forecasting review](https://openclimatefix.github.io/nged-substation-forecast/background/energy-forecasting-review/#3-estimating-the-effective-capacity-of-metered-generators) names the published methods, and each estimator here corresponds to one.** CX1 and CX2 are the linear-programme counterpart of the quadratic optimisation of [Viotti et al. (2026)](https://doi.org/10.1002/we.70136), who fit a capacity time series against a capacity factor simulated from weather and a power curve. DP1 follows the differentiable-physics route of [Pierrot and Pinson (2024)](https://doi.org/10.1080/00401706.2024.2350421) (capacity as a fitted bound with uncertainty) and [Gijón et al. (2025)](https://arxiv.org/abs/2502.07344) (a physical model fitted to metered power). KF is a sequential Bayesian update like the one the honourable-mention team used in [Authen et al. (2026)](https://doi.org/10.5617/nmi.13106). PR divides power by irradiance as [RdTools](https://doi.org/10.5281/zenodo.1210316) does. CSQ is a quantile of clear-sky-normalised output, the idea behind [Meyers et al. (2020)](https://doi.org/10.1109/JPHOTOV.2019.2957646). R30 is the rolling relative of the running maximum of [Dantas and Browell (2026)](https://doi.org/10.1002/we.70079). NR and HM have no published precedent that the review found.

**Four published methods are not tested.** The running maximum itself is the reference that Viotti et al. quote their numbers against, and a ratchet cannot follow a loss downwards. The Solar Data Tools pipeline and StatisticalClearSky would add dependencies to `uv.lock` (departure 4). Joint estimation with the forecast ([Pierrot and Pinson (2024)](https://doi.org/10.1080/00401706.2024.2350421)) needs a different forecaster, and this study keeps the lag study's. Per-park power curves ([Authen et al. (2026)](https://doi.org/10.5617/nmi.13106)) are for wind.

**Departures from the review's recommendation, and results our controls respect.**

- The review proposes running Viotti et al. against Solar Data Tools on the project's sites. This study runs a linear programme instead of either (departure 4).
- Viotti et al. score capacity recovery and forecast error separately, and find that the variant that recovers capacity best is not the variant that forecasts best. Stage A scores recovery and Stage B scores the forecast, and the page reports both.
- Viotti et al. validate on synthetic capacity steps planted in real data, and warn that turbine outages near a step confound the score and that about a quarter of their cases show an erroneous step at the record's end. The planted-loss design here does the same, the ratio of planted to original estimate cancels real outages, and the footprint rule keeps events away from the record's edges.
- Their de-rating case (a loss in the 30 days after a step) is the one their paper answers worst, and Stage A's ramp and intermittent losses are that case.
- [Pierrot and Pinson (2024)](https://doi.org/10.1080/00401706.2024.2350421) found that tracking a varying bound on its own gained 2.43%, which they call not significant. The 2% decision threshold sits at the same order, so "unresolved" is a likely forecast verdict.
- [Dantas and Browell (2026)](https://doi.org/10.1002/we.70079) drop curtailed half-hours using published bid volumes. NGED's active-network-management log is a noisy label, so this study removes the hours where the export cap binds (as the lag study does) and shows the ANM-blind estimate beside the ANM-aware one.
- No ground truth exists for real effective capacity, so Stage A's evidence is planted losses only, and the three real low months are descriptive.
- [de Vilmarest et al. (2024)](https://doi.org/10.1109/TPWRS.2023.3310280) found an adaptive model absorbing a removed regional capacity, a caution that normalisation may not earn its place.

## The facts this plan rests on

- **Rows:** `data/studies/per_study/lag_features/ens_mean/lag_frame_ens_mean_day1.parquet`, 29,445
  lead-day-1 daylight hours at 6 plants over 18 months (2024-12 to 2026-06 without 2026-01), with
  `fold`, `power_mw` and `effective_capacity_mw` (both Float32), `cap_mw`, `constrained`, and B0's
  seven columns. B0's error is 8.93 points. The months end before the final-test date, 2026-07-01.
- **`cap_mw` is non-null at the ANM plant only:** 3,696 rows, 102 of them constrained.
- **The lag frame's `pc_power_to_cams_7d` is unusable as PR:** it is a same-clock-hour window mean,
  so it varies within the target day, it is null on 17% of rows, and the frame's CAMS starts on
  2024-03-26. `rp_7d` is the plant's 7-day capacity-normalised energy over the daily mean of the
  other 5 plants' (about 1 at a normal plant).
- **Reusable losses:** B0 at both settings, with quantile models (`losses_ens_mean.parquet`), and
  B0-noDOY, B0 without day of year, at the primary setting (`followups2/`).
- **The low-output plant** is in `followups3/derated_plan.json` (ratios to its neighbours of 0.68,
  0.67, and 0.71 against a threshold of 0.8). The file lists 16 of the 18 scored months; 2025-08 and
  2025-09 are absent. `followups3/output_ratios_ens_mean.parquet` holds every plant-month's ratio to
  its neighbours.
- **CAMS** is hourly from 2019-09-01; its publication latency was 2 days before February 2026.
  **PyTorch** is in the optional `gpu` group, which CI does not install, so torch tests open with
  `pytest.importorskip("torch")`.
- **The ANM plant's commissioning ramp ends on 2024-10-06** (`studies.commissioning`).

**ANM.** The ANM plant is not the low-output plant, so ANM does not explain the low months. The
beam/diffuse page already publishes its anonymised label and connection limit in megawatts, and
`studies.commissioning` pairs the label with the scheme, so this page exposes nothing new; whether
it repeats the label is open decision 2.

## Design

### Forecaster, rows, and causal timing

**The forecaster is B0 exactly, on the lag study's rows and folds:** seven columns (hour of day, day
of year, era code, sun elevation and azimuth, and the ENS mean's irradiance and temperature at
lead-day 1), the absolute-error objective, no column subsampling, three seeds, and five folds of
whole months per plant. Every normalised arm has the same seven columns. The build records the lag
frame's SHA-256. A 90-fit refit of B0 on the GPU must reproduce the lag study's B0 losses bit for
bit, or the script raises.

**One causal quantity, `ĉ_issue(T)`, is used everywhere: the estimate built from whole days up to
and including T−2 for target day T.** That is what a forecast issued at 09:00 UTC on day T−1 can
read (power, CAMS, and neighbours' power), held constant across day T. Stage A's primary metric,
detection delay, and false movement, and Stage B's join, all use `ĉ_issue`. Every training row is
normalised by `ĉ_issue` of its own target day, which may read an earlier test month's power without
leaking, because each training row predicts only its own power.

**The hindsight reference is a forward and backward smoother, run for the 4 shortlisted estimators
only.** For each target week, it combines the forward estimate from whole days through the day
before the week with a causal estimate run on the time-reversed record from the day after the week:
a plain mean, or inverse-variance weighted for KF and DP1. The target week's own output is never
read. The page calls it the "hindsight reference", not an upper bound, because CX1, CX2, and DP1
could fit both sides jointly and do better.

**Fallbacks are counted, never hidden.** Any null estimate falls back to the static P99: fewer than
30 valid days of history (a valid day's observed hours cover at least half its clear-sky energy,
`MIN_OBSERVED_CLEAR_SKY_SHARE` = 0.5), or a null from an estimator's own minimum-count rule. The
scale is floored at 20% of the static P99. The report prints the share of rows on each fallback, per
estimator and plant.

### Fair scoring with no code change

**Every arm is scored on its megawatt error over the same fixed P99 per plant, so an estimator
cannot shrink its errors by inflating its capacity on its worst days.** Dividing `power_mw`,
`cap_mw`, and `effective_capacity_mw` by the row's scale s before `out_of_fold_losses` gives exactly
that score. Checked against `_losses`: the training target becomes y/s; the fraction error becomes
`|y/s − ŷ| / (P99/s) = |y − s·ŷ| / P99`; clamping at `cap/s` then multiplying by s equals clamping
`s·ŷ` at the cap, because s > 0; and CRPS is positively homogeneous of degree 1, so the CRPS
fraction reduces the same way. The scored column is `absolute_error_capped_fraction_of_capacity`;
the scaled `_mw` loss columns are dropped. Float64 scaling and XGBoost's Float32 labels make a
constant scale reproduce B0 to about 1e-7 relative, so the constant-scale assertion uses a tolerance
of 1e-4 points.

### The estimators

| Family | Code | Estimator |
|---|---|---|
| Cheap | R30 | rolling 30-day P99 of hourly power (the roadmap's "simple rolling p99") |
| Cheap | CSQ | rolling 30-day 90th percentile of clear-sky-normalised output |
| Cheap | NR | the plant's 7-day energy ratio to the daily mean of the other 5 plants, times the static P99 |
| Cheap | PR | sum of power over sum of CAMS global horizontal irradiance across days T−8 to T−2, over its median in the year before 2024-12-01, times the static P99 |
| Cheap | KF | Kalman local-level filter on the daily log ratio of power to a `pv_physics` model at the fitted orientation, driven by CAMS |
| Wacky | HM | hidden Markov model over discrete capacity levels (1.0, 0.9, 0.75, 0.5) |
| Convex | CX1 | pinball loss at a high quantile plus a fused-lasso penalty on one daily DC trace |
| Convex | CX2 | DC and AC traces, the clipping classifier, and the censoring hinge (Candidate A, with the departures above) |
| PyTorch | DP1 | the roadmap's Candidate B: variational single-plant model with a piecewise-constant weekly DC trace and a pinball reconstruction loss |

**HM is kept over the online changepoint detector because only HM encodes that strings and
inverters fail in whole units.**

**NR and PR are computed fresh by one code path in both stages, from one CAMS source.** The import
rule forbids importing `studies/lag_features/` helpers, so this study re-implements NR and asserts
that it equals the lag frame's `rp_7d` on the overlapping rows, which also checks the causal driver.
PR uses horizontal irradiance and so drifts with season through the plant's tilt; the false-movement
check catches that drift. Each plant's orientation is fitted once with `pv_fit.fit_plant` on data
before 2024-12-01 (for the ANM plant, about 2 months, which the report states).

### Stage A: recovery, the technique contrasts, and the selection rule

**Each draw plants three events at every plant: a step, a ramp, and an intermittent loss, each with
its own size s drawn uniformly from 5% to 40%.** A step multiplies power by `f = 1 − s` for a month;
a ramp moves f linearly from 1 to `1 − s` across a month, then restores it; an intermittent loss
applies `f = 1 − s` on a seeded half of a month's days. Months are drawn from 2019-09 to 2026-06.

**Events never overlap.** An event's footprint is the 30 days before it (the longest estimator
look-back), its month, and the 60 days after. Within a plant, footprints are disjoint; across plants,
no planted month falls inside another plant's footprint, so a neighbour-based estimator works. The
ANM plant's events start after its commissioning ramp. The build raises if a draw cannot be placed.
There is 1 tuning draw and there are 10 selection draws. The convex and PyTorch estimators are
evaluated only on footprints and on the false-movement windows.

The metrics, per causal estimator:

- **Relative capacity error (primary):** the mean over daylight target days T in each planted month
  and the 60 days after it of `|ĉ_issue,planted(T) / ĉ_issue,original(T) − f(T)|`. The ratio cancels
  whatever real capacity changes the real data hold. An intermittent month is scored against its
  window mean, `f = 1 − s/2`, because no daily estimate can know which days are hit.
- **Detection delay:** days from the start of a planted step until `ĉ_issue` crosses halfway.
- **False movement:** the share of days on which the real-data `ĉ_issue` moves more than 5% from its
  trailing 90-day median, at up to 4 seeded plant-months per plant whose ratio to the neighbours
  lies within 0.95 to 1.05. This catches an estimator chasing the weather or the season, which the
  primary metric's ratio cancels. **The windows are chosen with NR's own statistic, so the guard
  favours NR**, which the page states.
- **Interval coverage, for KF and DP1:** the share of planted days whose 90% interval covers f.

**A descriptive table, used in no selection, shows each estimator's monthly ratio at the low-output
plant against the audit's ratios (0.68, 0.67, and 0.71).**

**Two planned estimator-level contrasts answer "which technique?",** each a difference in the
primary metric with a 97.5% t-interval over the 10 selection draws (Bonferroni over two). The
intervals are conditional on these 6 plants and this record.

- **E1:** the convex representative minus the representative of the cheap and wacky families;
- **E2:** DP1 minus the convex representative, testing the roadmap's prediction that convex wins
  per plant.

**The 10% false-movement gate is applied first; each family's representative is then its
lowest-error surviving estimator on the tuning draw,** so neither contrast is scored on the draw
that chose it. A contrast reads "beats" when its interval lies wholly below zero, "loses" when wholly
above, and "no detectable difference" otherwise; in the last case, the page ranks the families
descriptively on detection delay, false movement, and operability.

**Each estimator is tuned on the tuning draw only, never on a forecast error, and the selection
rule is fixed now:** after the gate, the lowest primary metric over the 10 selection draws, averaged
over all events and all 6 plants. If nothing passes, the page says every estimator chases the weather,
and P1 runs on the lowest-error estimator, labelled exploratory. The best surviving estimator of
each other family joins the selected one in a shortlist of 4.

**CAMS latency:** the backtest reads CAMS through T−2, a day fresher than the live service could
before February 2026. Stage A's primary metric is recomputed for PR, KF, and CX2 with CAMS read only
through T−3, as a sensitivity.

**Whether a time-varying `effective_capacity` table is worth building as a fault signal rests on
Stage A, not on P1** (open decision 5).

### Stage B: the forecast contrast and the decision rule

**P1 (planned): the selected causal estimator's arm minus B0, mean absolute error pooled over the 6
plants,** on all 18 months, at both hyperparameter settings, with a 95% interval from
`bootstrap_difference_at_level` (whole months and a seed, 2,000 resamples) and the month-cluster
t-interval beside it. The decision uses the pooled P1; per-plant figures are exploratory. A single
planned forecast contrast needs no Bonferroni correction.

**P1's power benchmark is a no-fit capacity-only bound:** B0's forecast multiplied by each
plant-month's ratio to its neighbours (from `output_ratios_ens_mean.parquet`) over that ratio's mean
on the fold's training rows, scored the same way.

**The decision rule, fixed now, with a threshold of 2% of B0's error (about 0.18 points):**

- **Worth building:** P1's interval lies wholly below zero at both settings, and its point estimate
  is at or below minus the threshold.
- **Not worth building:** at both settings, P1's interval lies wholly above minus the threshold.
- **Unresolved:** every other outcome, reported with its interval ("a gain as large as 0.25 points
  is not excluded"), and whenever NC2's interval lies wholly below zero.

**Why 2%:** a production estimator needs a new asset, a time-varying table, and a new serving-path
dependency, and the lag study used the same threshold.

### Controls

- **Constant-scale assertion:** at the ANM plant, on a fold holding constrained rows, B0 with the
  three `_mw` columns divided by s = 0.37 must reproduce B0 within 1e-4 points on both
  `absolute_error_fraction_of_capacity` and `absolute_error_capped_fraction_of_capacity`, or the
  script raises before any other fit.
- **NC2 (negative control):** the selected estimator's causal series with whole months deranged
  within each plant, so that no month keeps its own scale and no low month lands on another low
  month. A gain over B0 would mean the scoring is broken, and turns P1's verdict to "unresolved".
- **Positive control:** a control draw plants a 30% step in 3 scored months at every plant, in the
  power the forecasts train on, score against, and the estimators read. Planted months do not
  overlap across plants and exclude the low-output plant's 3 real low months, which would otherwise
  leave NR blind. Arms: B0, the oracle (normalised by the true factor times the static P99), and the
  selected estimator. If the oracle's 95% interval against B0 does not lie wholly below zero, every
  null is read as "not excluded". The recovery share is the selected estimator's planted-data gain
  minus its real-data gain, over the oracle's gain. **This control (18 planted plant-months) is far
  stronger than the real case (3)**, so passing shows the instrument works, not that P1 has power.

### Explainability

**The maintainer wants the capacity estimator to be explainable, so every estimator is graded on a
fixed rubric before any result exists.** Each estimator gets a grade (yes, partly, or no) on five
criteria, by reading its code and outputs. The grades are in `studies.capacity_explain` and the
report script prints them with a reason for each:

1. An operator can be told in one sentence why the capacity fell on a given day.
2. The fitted quantities are physical quantities with units.
3. A change can be attributed to a cause: soiling, a lost string, curtailment, or a weather-forecast
   bias.
4. The estimator returns a spread, not only a value.
5. The estimator can be differentiated through, for joint fitting with other models.

KF and DP1 report their fitted physical quantities (tilt and azimuth in degrees, the direct-current
to alternating-current ratio, an availability factor over time with its standard deviation). The
report script prints one explanation sentence per estimator for a planted event and for each of the
three real low months of the low-output plant, with no plant label beside those dates. The rubric
adds no estimator and no fit, and it takes no part in the selection rule.

### Exploratory additions

- the other 3 shortlisted arms, and each hindsight-reference arm against its causal arm;
- RS: B0's forecast times the selected `ĉ_issue` over its training-row mean (no fit);
- the 2×2 of day of year and normalisation at the primary setting, reusing B0-noDOY;
- the selected arm's CRPS from one primary quantile fit, against B0's reused CRPS;
- at the ANM plant, each shortlisted estimator's dip on days with a moved cap, ANM-blind against
  ANM-aware (the cap as an upper bound on export; no fits);
- month-by-month and plant-by-plant differences, including the 3 low-output plant-months.

### Delivery table 4, for the page's Discussion

**The study builds none of the production mapping.** The page says how the selected estimator would
fill the planned `effective_capacity` table (mean and standard deviation, issue #247): the
hindsight reference for history, `ĉ_issue` for the forecast. Table 4 excludes ANM curtailment.

### Charts

In order: the headline (E1, E2, and P1 with intervals); each plant's output relative to its
neighbours, marking the low months; the ANM plant's capped share by month; one planted event of each
shape with each family's recovered trace, and the recovery leaderboard; B0 and the selected arm
against measured output for three weeks; then the controls, month-by-month differences, the 2×2,
and the ANM dips. Plants are labelled A to F by `site_labels_for`.

## Compute estimate

**XGBoost, in primary-equivalent fits** (an arm is 6 plants × 5 folds × 3 seeds = 90 fits; from the
lag study's benchmark, a sensitivity point fit counts as about 1 and a primary quantile fit as about
6):

| Stage | Fits |
|---|---|
| Reproduction check and constant-scale assertion | 93 |
| P1's arm: primary point (90), primary quantile (90 × 6), sensitivity point (90) | 720 |
| NC2 | 90 |
| The other 3 shortlisted arms | 270 |
| Hindsight-reference arms of the 4 shortlisted estimators | 360 |
| The selected arm without day of year | 90 |
| Positive control: B0, oracle, selected | 270 |
| **Total** | **about 1,900** |

**At the lag study's benchmark (1,620 primary fits in about 30 minutes), 1,900 fits are about 35
minutes of single-fit time, or about 13 minutes on the GPU,** so a 1-hour GPU slot is ample.

**Estimators (unverified until benchmarked on one plant):** CX1 and CX2 each need about 15,000
causal solves on the original data (6 plants × about 2,500 days, cached per day), plus about 2,700
per planted data set (3 events × about 150 days × 6 plants) over 12 planted data sets (11 draws and
the positive control), plus tuning and the reversed passes: on the order of 100,000 small linear
programs, an estimated 1 to 2 hours on 8 cores. DP1 batches every plant and draw into one tensor
over about 80 warm-started weekly refits, an estimated 30 to 60 minutes of GPU; its `ĉ_issue` comes
from the latest refit ending by T−2, so it can be up to 7 days stale.

## Solution

### Walk-through, files, and error paths

**Tested code lives in `packages/studies/src/studies/`, following `battery_state_space.py`.**
`capacity_estimators.py` holds the 8 non-torch estimators, `issue_time_estimates`,
`hindsight_reference_estimates`, `plant_losses`, the fallback, the selection rule, and the
forecast-row join; `capacity_physics_torch.py` holds DP1. Tests import `studies.*`, so no
`extra-paths` or `pythonpath` entry is needed. The scripts live in `studies/effective_capacity/` and
write to `data/studies/per_study/effective_capacity/`.

1. **`build_capacity_inputs.py`** builds each plant's cleaned hourly power with the export cap,
   CAMS, and clear-sky irradiance joined, fits orientations, computes the neighbour ratios, and
   makes the draws. Raises on a row count other than 29,445, an input at or after 2026-07-01, an
   unplaceable event, or an ANM-aware input that differs from the ANM-blind input at a plant without
   a cap record.
2. **`estimate_capacity.py` and `estimate_capacity_torch.py`** write one daily table per estimator.
   Raise on a linear program not solved to optimality (naming plant and day), a causal probe whose
   estimate changes when later inputs are altered, or a recomputed NR that differs from `rp_7d`.
3. **`capacity_recovery.py`** tunes, scores the selection draws, and writes `selection.json` before
   any XGBoost fit.
4. **`fit_capacity_arms.py`** runs the two checks, divides the three `_mw` columns by each arm's
   `capacity_scale_mw`, calls `out_of_fold_losses` through `fit_in_groups`, and drops the scaled
   `_mw` loss columns. Raises on a reproduction mismatch, a failed constant-scale assertion, a
   failed join probe (below), or an uncovered month.
5. **`report_effective_capacity.py`** prints every quoted number, every arm's column list, and every
   fallback share into `report.md`; **`effective_capacity_charts.py`** writes the figure text first.

Outside the study folder and the package: the page `docs/studies/forecasts/effective-capacity.md`
with its index entry, the `mkdocs.yml` nav entry and the `ty` torch override (open decision 1), and
`svgo`-optimised figures.

### Tests

- **Known-answer step:** every estimator recovers a 30% loss lasting 30 days within a stated
  tolerance and delay.
- **Causal off-by-one:** the estimate for day d is unchanged when every input after d is noise.
- **Join probe (also a raise in `fit_capacity_arms.py`):** for sampled lag-frame rows, the scale is
  unchanged when every input on or after the issue day (power, CAMS, neighbours) is noise.
- **Hindsight reference:** a week's estimate is unchanged when that week's power is altered.
- **Scoring:** scaled scoring equals the multiplied-back megawatt error over the P99, with a
  time-varying s, non-null caps, and quantiles (so CRPS is covered).
- **Fallback and floor:** a null and a short history fall back and are counted; the floor holds.
- **Metrics and selection:** primary metric and detection delay on a hand-built trace; the gate,
  averaging, and `selection.json` write.
- **Smaller cases:** the convex program matches brute force on 5 days; KF's standard deviation grows
  without data; `plant_losses` changes exactly the drawn months; DP1 recovers a planted DC capacity
  (skipped without torch).

### Design philosophy and verification

R&D code fails fast; nothing runs in production; no engineering hypothesis concerns capacity
estimation (checked). Verification: the CLAUDE.md set, `pytest --run-studies` with and without the
`gpu` group, pydoclint, the docs-link checker, and `mkdocs build --strict`.

**Limitations the page states:** the multi-day-zero filter reads later hours, so causal estimators
see a slightly cleaner history than a live service; the planted loss multiplies all power, clipping
plateau included, so it is a proportional block loss of DC and AC together. A pure DC loss at a
clipping plant, where CX2 and DP1 would differ from the rest, is the unplanted case.

## Discussion hooks

**Groundwork for version 2: what DP1 needs to become a component of a larger differentiable
model.** The final page's "Groundwork for version 2" paragraph can draw on these notes.

- **Inputs:** satellite-driven (later, forecast) global horizontal irradiance per hour, the plant's
  coordinates, and a window of measured power. The tilt, azimuth, and alternating-current limit are
  fixed from an orientation fit, not learned jointly.
- **Outputs:** a weekly direct-current capacity trace with a normal posterior (a mean and a standard
  deviation per week). A larger model would take the trace and the physics output as its inputs.
- **Differentiable now:** the physics chain from irradiance to power (Erbs split is outside the
  graph, see below), the soft clip, the pinball reconstruction loss, the Kullback-Leibler term, and
  the total-variation penalty, all in PyTorch.
- **Not differentiable or not yet joint:** the Erbs split of irradiance and the solar position come
  from `pvlib` in NumPy, so gradients stop at the irradiance; the orientation fit is a separate
  SciPy fit; the refit schedule is a loop of independent windows, not one joint fit over the record.
- **Needed for joint fitting:** the physics inputs as tensors (a port of the Erbs split and the
  plane-of-array chain), one objective over the whole record, and a prior that ties the weekly trace
  across plants in the same weather.

- **Literature hook for the Introduction and Discussion:** cite the works in the Literature section with the link on the author-year label, and say which estimator stands for which published method.

## Open decisions for the maintainer

1. **Approve two edits outside the `study` skill's autonomous-merge paths in advance?** The `ty`
   override in the root `pyproject.toml` gains `capacity_physics_torch.py` and its test, and
   `mkdocs.yml` gains the nav entry. Recommended: approve.
2. **May the page show the ANM plant's capped-hour shares, and should it repeat the plant's
   label?** Recommended: show the shares without the label. Out of scope but worth knowing: the
   beam/diffuse page already publishes this plant's label with its connection limit in megawatts.
3. **Ask NGED what happened at the low-output plant in its 3 low months?** Recommended: ask; the
   study does not depend on the answer.
4. **Slots, in order:** 8 CPU cores for about 2 hours, a GPU for about 1 hour (DP1), then a GPU with
   4 cores for 1 hour (XGBoost), never overlapping.
5. **What does the result mean for v0.7?** Whether to build the time-varying table without a
   forecast gain, and any roadmap edit, are the maintainer's decisions.
6. **Keep the lag study's row set,** which leaves out 674 rows lacking the lag study's one-day lag?
   Recommended: keep it, so B0's 8.93 points stay comparable.

## Review triage

Review 1 (simplicity), all accepted: recovery answers "which technique?"; 9 estimators; quantiles
exploratory; constant-scale assertion for NC1; smaller positive control; forward-backward smoother;
P1 alone decides; scoring with no code change; variants and prose cut; compute fixed.

Review 2 (correctness), all accepted:

- M1: one `ĉ_issue(T)` everywhere; join probe test and raise.
- M2: PR a fresh daily ratio; NR re-implemented, equality to `rp_7d` asserted.
- M3: every null falls back and is counted; tested.
- M4: assertion at the ANM plant, s = 0.37, both fraction columns; scoring test widened.
- M5: 0.337 reframed; capacity-only bound is P1's power benchmark.
- M6: positive-control months disjoint, real low months excluded.
- S1: s uniform 5% to 40%. S2: footprint rule; ANM draws after 2024-10-06.
- S3: NR bias stated; audit-ratio table. S4: gate before representatives.
- S5: intermittent scored against 1 − s/2. S6: "hindsight reference". S7: limitation reworded.
- S8: CX2 departures. S9: order inversion is departure 1.
- S10: derangement; NC2 failure means "unresolved". S11: pooled P1 decides.
- S12: ANM rationale rewritten; existing exposure in open decision 2.
- S13: T−3 CAMS sensitivity. S14: re-implement, equality as guard.
- S15: tested code in `packages/studies`. S16: tests added.
- Minor: applied, with two corrections. The re-weighted fit total is 1,893 ("about 1,900"). With
  continuous sizes there are 12 planted data sets, not 22, so the convex solves stay near 100,000
  even with the original-data solves added.
