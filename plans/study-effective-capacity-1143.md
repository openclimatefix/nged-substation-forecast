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
lag study's hindsight monthly rescale, which knew each plant-month's true level, lowered the error
by 0.337 points [−0.473, −0.215] at 95%. The 2% threshold below (0.18 points) is about half of that
ceiling, so even a good causal estimator could leave an interval spanning the threshold. Stage A's
verdict on the techniques does not depend on that power.

**Size: complex.** What gets stored: a published study page. Production serving path: not touched.
Degradation rule: none touched. More than one defensible design: yes (causal timing, hindsight
definition, selection rule, scoring, planted-loss design). Callers not nameable without searching:
no, because no existing function changes. **Reviews:** the simplicity plan review has run and the
correctness review is next, then the `study` skill's full sequence, including the mutation pass for
the new tests.

**Departures from the issue body and the roadmap:**

1. **The reference arm is the lag study's baseline B0, not a separate "static capacity" arm.** Under
   `reg:absoluteerror`, the gradient is the sign of the residual and the hessian is 1, so dividing
   the target by a constant leaves every split unchanged and scales every leaf value. A
   static-capacity arm is therefore B0 to float noise, which a one-plant, one-fold assertion checks.
2. **The convex estimators are linear programs solved by `scipy.optimize.linprog` (HiGHS), not by
   CVXPY.** Adding `cvxpy` to `uv.lock` would block the `study` skill's autonomous merge, so the
   roadmap's goal of hands-on CVXPY experience is not served. SLAC's `solar-data-tools` is left out
   for the same reason.
3. **Active network management (ANM) data covers one plant, which is not the low-output plant**, so
   the ANM comparison is exploratory.
4. **The forecast decision rule has an equivalence outcome**, which the lag study's rule lacked.

## The facts this plan rests on

- **Rows:** `data/studies/per_study/lag_features/ens_mean/lag_frame_ens_mean_day1.parquet`, 29,445
  lead-day-1 daylight hours at 6 plants over 18 months (2024-12 to 2026-06 without 2026-01), with
  `fold`, `power_mw` and `effective_capacity_mw` (both Float32), `cap_mw`, `constrained`, and B0's
  seven columns. B0's error is 8.93 points. The months end before the final-test date, 2026-07-01.
- **Reusable lag-frame columns:** `rp_7d`, the plant's 7-day capacity-normalised energy over the
  mean of the other 5 plants' (about 1 at a normal plant), and `pc_power_to_cams_7d`, the ratio of
  power to CAMS global horizontal irradiance over the 7 days before the issue day. The `ck_*` and
  `tf_ratio` columns duplicate estimators below and are unused.
- **Reusable losses:** B0 at both settings, with quantile models (`losses_ens_mean.parquet`), and
  B0-noDOY, B0 without day of year, at the primary setting (`followups2/`).
- **The low-output plant** is in `followups3/derated_plan.json` (ratios to its neighbours of 0.68,
  0.67, and 0.71 against a threshold of 0.8). The file lists 16 of the 18 scored months; 2025-08 and
  2025-09 are absent.
- **CAMS** is hourly from 2019-09-01. **PyTorch** is in the optional `gpu` group, which CI does not
  install, so torch tests open with `pytest.importorskip("torch")`.

**ANM.** The one generator with an ANM set-point export is one of the 6 plants, and not the
low-output plant, so ANM does not explain the low months. Neither plan nor page names the ANM
plant's label, because the pairing would narrow down the generator's identity.

## Design

### Forecaster, rows, and causal timing

**The forecaster is B0 exactly, on the lag study's rows and folds:** seven columns (hour of day, day
of year, era code, sun elevation and azimuth, and the ENS mean's irradiance and temperature at
lead-day 1), the absolute-error objective, no column subsampling, three seeds, and five folds of
whole months per plant. Every normalised arm has the same seven columns. The build records the lag
frame's SHA-256. A 90-fit refit of B0 on the GPU must reproduce the lag study's B0 losses bit for
bit, or the script raises.

**The causal version is the headline.** A forecast issued at 09:00 UTC on day D for day D+1 uses the
capacity estimate built from whole days up to and including D−1 (power, CAMS, and neighbours'
power), held constant across the target day. Every training row is normalised by the causal estimate
at its own issue day. That estimate can read power from an earlier test month without leaking the
answer, because the model is trained to predict each training row's own power.

**The hindsight version is a forward and backward smoother, run for the 4 shortlisted estimators
only, and labelled an upper bound.** For each target week, it combines the forward causal estimate
as of the week's first day with a causal estimate run on the time-reversed record as of the week's
last day: a plain mean, or inverse-variance weighted for KF and DP1. The target week's own output is
never read. For CX1, CX2, and DP1, which could fit both sides jointly, the page calls the smoother a
slightly weaker upper bound.

**Fallbacks are counted, never hidden.** With fewer than 30 valid days of history (a valid day holds
at least half its clear-sky energy), the scale falls back to the static P99, and the scale is
floored at 20% of the static P99. The report prints the share of rows on each fallback, per
estimator and plant.

### Fair scoring with no code change

**Every arm is scored on its megawatt error over the same fixed P99 per plant, so an estimator
cannot shrink its errors by inflating its capacity on its worst days.** Dividing `power_mw`,
`cap_mw`, and `effective_capacity_mw` by the row's scale s before `out_of_fold_losses` gives exactly
that score. Checked against `_losses`: the training target becomes y/s; the fraction error becomes
`|y/s − ŷ| / (P99/s) = |y − s·ŷ| / P99`; clamping at `cap/s` then multiplying by s equals clamping
`s·ŷ` at the cap, because s > 0; and CRPS is positively homogeneous of degree 1, so the CRPS
fraction reduces the same way. The scaled `_mw` loss columns are dropped. **Float caveat:** dividing
the Float32 columns by a Float64 scale forms the errors in Float64, and XGBoost rounds labels to
Float32, so a constant scale reproduces B0 to about 1e-7 relative, not bit for bit. The
constant-scale assertion therefore uses a tolerance of 1e-4 points; B0's reproduction check divides
by nothing and stays bit for bit.

### The estimators

| Family | Code | Estimator |
|---|---|---|
| Cheap | R30 | rolling 30-day P99 of hourly power (the roadmap's "simple rolling p99") |
| Cheap | CSQ | rolling 30-day 90th percentile of clear-sky-normalised output |
| Cheap | NR | the lag frame's `rp_7d` times the static P99 |
| Cheap | PR | the lag frame's `pc_power_to_cams_7d` over its median in the year before 2024-12-01 |
| Cheap | KF | Kalman local-level filter on the daily log ratio of power to a `pv_physics` model at the fitted orientation, driven by CAMS |
| Wacky | HM | hidden Markov model over discrete capacity levels (1.0, 0.9, 0.75, 0.5) |
| Convex | CX1 | pinball loss at a high quantile plus a fused-lasso penalty on one daily DC trace |
| Convex | CX2 | the roadmap's Candidate A: DC and AC traces, the clipping classifier, and the censoring hinge |
| PyTorch | DP1 | the roadmap's Candidate B: variational single-plant model with a piecewise-constant weekly DC trace and a pinball reconstruction loss |

**HM is kept over the online changepoint detector because only HM encodes that strings and
inverters fail in whole units;** the changepoint detector's piecewise-constant level is already
CX1's and DP1's assumption. No cut estimator is reinstated.

**NR and PR reuse the lag frame's reviewed columns on the forecast rows, which removes one place to
get the causal timing wrong.** Stage A needs both on planted data, so the build recomputes them and
asserts that the recomputation on real data equals the lag-frame columns, which also checks this
study's causal driver. PR uses horizontal irradiance and so drifts with season through the plant's
tilt; the false-movement check is what catches that drift, and KF carries the orientation-corrected
ratio. Each plant's orientation is fitted once with `pv_fit.fit_plant` on data before 2024-12-01
(for the ANM plant, about 2 months, which the report states).

### Stage A: recovery, the technique contrasts, and the selection rule

**Each draw plants three events at every plant, in months drawn separately per plant from 2019-09 to
2026-06:** a step (power times `f = 1 − s` for a month), a ramp (f falling linearly from 1 to
`1 − s` across a month, then restored), and an intermittent loss (`f = 1 − s` on a seeded half of a
month's days). Every draw runs at s = 10% and s = 30%, which brackets the real low months. There is
1 tuning draw and there are 10 selection draws. Separate months per plant let a neighbour-based
estimator work. The convex and PyTorch estimators are evaluated only on each event's month plus 60
days either side, and on the false-movement windows.

The metrics, per causal estimator:

- **Relative capacity error (primary):** the mean over daylight days of
  `|ĉ_planted(d) / ĉ_original(d) − f(d)|` over each planted month and the 60 days after it. The
  ratio cancels whatever real capacity changes the real data hold.
- **Detection delay:** days from the start of a planted step until the estimate crosses halfway.
- **False movement:** the share of days on which the real-data estimate moves more than 5% from its
  trailing 90-day median, at up to 4 seeded plant-months per plant whose ratio to the neighbours
  lies within 0.95 to 1.05. This metric catches an estimator chasing the weather or the season,
  which the primary metric's ratio cancels.
- **Interval coverage, for KF and DP1:** the share of planted days whose 90% interval covers f.

**Two planned estimator-level contrasts answer "which technique?",** each a difference in the
primary metric with a 97.5% t-interval over the 10 selection draws (Bonferroni over two):

- **E1:** the convex representative minus the representative of the cheap and wacky families;
- **E2:** DP1 minus the convex representative, testing the roadmap's prediction that convex wins
  per plant.

Each family's representative is its lowest-error estimator on the tuning draw, so neither contrast
is scored on the draw that chose it. A contrast reads "beats" when its interval lies wholly below
zero, "loses" when wholly above, and "no detectable difference" otherwise; in the last case, the
page ranks the families descriptively on detection delay, false movement, and operability.

**Each estimator is tuned on the tuning draw only, never on a forecast error, and the selection
rule is fixed now.** The rule discards any causal estimator whose false movement exceeds 10% of
days, then picks the one with the lowest primary metric over the 10 selection draws, averaged over
both sizes, all three shapes, and all 6 plants. The real-data guard stops the rule over-fitting to
the planted shapes. If nothing passes, the page says every estimator chases the
weather, and P1 runs on the lowest-error estimator, labelled exploratory. The best estimator of each
other family joins the selected one in a shortlist of 4.

**The capacity series is judged separately from the forecast.** Whether a time-varying
`effective_capacity` table is worth building as a fault signal and as delivery table 4 rests on
Stage A, not on P1; the maintainer decides (open decision 5).

### Stage B: the forecast contrast and the decision rule

**P1 (planned): the selected causal estimator's arm minus B0, mean absolute error,** per plant, on
all 18 months, at both hyperparameter settings, with a 95% interval from
`bootstrap_difference_at_level` (whole months and a seed, 2,000 resamples) and the month-cluster
t-interval beside it. A single planned forecast contrast needs no Bonferroni correction. A second
contrast holding day of year equal would inform no decision once P1 decides, so the 2×2 of day of
year and normalisation is exploratory.

**The decision rule, fixed now, with a threshold of 2% of B0's error (about 0.18 points):**

- **Worth building:** P1's interval lies wholly below zero at both settings, and its point estimate
  is at or below minus the threshold.
- **Not worth building:** at both settings, P1's interval lies wholly above minus the threshold.
- **Unresolved:** every other outcome, reported with its interval ("a gain as large as 0.25 points
  is not excluded").

**Why 2%:** the threshold asks for about half the hindsight ceiling, a production estimator needs a
new asset, a time-varying table, and a new serving-path dependency, and the lag study used the same
rule.

### Controls

- **Constant-scale assertion:** at one plant and one fold, B0 with the three `_mw` columns divided
  by a constant must reproduce B0 within 1e-4 points, or the script raises before any other fit.
- **NC2 (negative control):** the selected estimator's causal series with whole months permuted
  within each plant, keeping the step sizes and destroying their timing. A gain over B0 would mean
  the scoring is broken.
- **Positive control:** a control draw plants a 30% step in 3 scored months at every plant, in the
  power the forecasts train on, score against, and the estimators read. Arms: B0, the oracle
  (normalised by the true factor times the static P99), and the selected estimator. If the oracle's
  95% interval against B0 does not lie wholly below zero, every null is read as "not excluded". The
  selected estimator's recovery share is its planted-data gain minus its real-data gain, over the
  oracle's gain. **The page states that this control (18 planted plant-months) is far stronger than
  the real case (3)**, so passing shows the instrument works, not that P1 has power; P1's power
  benchmark is the hindsight ceiling. The 10% size is tested at the estimator level only.

### Exploratory additions

- the other 3 shortlisted arms against B0;
- each shortlisted hindsight arm against its causal arm;
- RS: B0's forecast times the selected estimator's causal value over its training-row mean (no fit);
- the 2×2 of day of year and normalisation at the primary setting, reusing B0-noDOY;
- one quantile fit of the selected arm at the primary setting, its CRPS against B0's reused CRPS;
- at the ANM plant, how far each shortlisted estimator's ANM-blind series dips on days with a moved
  cap, against an ANM-aware version that treats the cap as an upper bound on export (no fits);
- month-by-month and plant-by-plant differences, including the 3 low-output plant-months.

### Delivery table 4, for the page's Discussion

**The study builds none of the production mapping; the page says what the selected estimator would
need.** Its daily value would fill the planned time-varying `effective_capacity` table (mean and
standard deviation, issue #247): the hindsight series for history and the metric denominator, the
causal series for the forecast's normalisation. Table 4 excludes ANM curtailment, which only an
ANM-aware estimator satisfies. KF and DP1 widen their standard deviation on a day without data, the
in-band degradation signal the inherent-stability rules ask for.

### Charts

In order: the headline (E1, E2, and P1 with intervals); each anonymised plant's output relative to
its neighbours, marking the 3 low months and the 2 the audit lacks; the ANM plant's capped share by
month, unlabelled; one planted event of each shape with each family's recovered trace, the
commissioning ramp, and the recovery leaderboard; B0 and the selected arm against measured output
for three weeks; then the controls, month-by-month differences, the 2×2, and the ANM dips. Plants
are labelled A to F by `site_labels_for`.

## Compute estimate

**XGBoost, in primary-equivalent fits** (an arm is 6 plants × 5 folds × 3 seeds = 90 fits; a
sensitivity point fit counts as 2, a primary quantile fit as 4):

| Stage | Fits |
|---|---|
| Reproduction check and constant-scale assertion | 90 |
| P1's arm: primary with quantiles (90 × 5), sensitivity point (90 × 2) | 630 |
| NC2 | 90 |
| The other 3 shortlisted arms | 270 |
| Hindsight arms of the 4 shortlisted estimators | 360 |
| The selected arm without day of year | 90 |
| Positive control: B0, oracle, selected | 270 |
| **Total** | **about 1,800** |

**At the lag study's benchmark (1,620 primary fits in about 30 minutes of single-fit time), 1,800
fits are about 35 minutes of single-fit time, or about 12 minutes on the GPU at 2.8 times a single
CPU fit's throughput.** A 1-hour GPU slot leaves room for the overhead the benchmark omits.

**Estimators (unverified until benchmarked on one plant):** CX1 and CX2 need about 2,700 causal
solves per data set (3 events × about 150 days × 6 plants) over 22 planted data sets, plus tuning
and the reversed passes: on the order of 150,000 small linear programs, an estimated 1 to 2 hours
on 8 cores. DP1 batches every plant and draw into one tensor and steps through about 80
warm-started weekly refits, an estimated 30 to 60 minutes of GPU; its causal value comes from the
latest refit ending by D−1, so it can be up to 7 days stale. The three slots are open decision 4.

## Solution

### Walk-through, files, and error paths

All scripts live in `studies/effective_capacity/`.

1. **`build_capacity_inputs.py`** builds each plant's cleaned hourly power with the export cap,
   CAMS, and clear-sky irradiance joined, fits orientations, computes the neighbour ratios, and
   makes the draws. Raises on a row count other than 29,445, an input at or after 2026-07-01, or an
   ANM-aware input that differs from the ANM-blind input at a plant without a cap record.
2. **`capacity_estimators.py`** holds the 8 non-torch estimators, `causal_daily_estimates`,
   `hindsight_daily_estimates`, and `plant_losses`; **`capacity_physics_torch.py`** holds DP1.
3. **`estimate_capacity.py` and `estimate_capacity_torch.py`** write one daily table per estimator.
   Raise on a linear program not solved to optimality (naming plant and day), a causal probe whose
   estimate changes when later inputs are altered, or a recomputed NR or PR that differs from the
   lag frame.
4. **`capacity_recovery.py`** tunes, scores the selection draws, and writes `selection.json` before
   any XGBoost fit.
5. **`fit_capacity_arms.py`** runs the two checks, divides the three `_mw` columns by each arm's
   `capacity_scale_mw`, calls `out_of_fold_losses` through `fit_in_groups`, and drops the scaled
   `_mw` loss columns. Raises on a reproduction mismatch, a failed assertion, or an uncovered month.
6. **`report_effective_capacity.py`** prints every quoted number, every arm's column list, and every
   fallback share into `report.md`; **`effective_capacity_charts.py`** writes the figure text first.

Outside the folder: tests under `packages/studies/tests/effective_capacity/`, the root
`pyproject.toml` (open decision 1), the study page with its index and nav entries, and
`svgo`-optimised figures.

### Tests

- **Known-answer step:** every estimator recovers a 30% loss lasting 30 days within a stated
  tolerance and delay.
- **Causal off-by-one:** the causal driver's estimate for day d is unchanged when every input after
  day d is replaced by noise.
- **Hindsight:** a week's estimate is unchanged when that week's power is altered.
- **Smaller cases:** the convex program matches brute force on 5 days; KF's standard deviation grows
  without data; `plant_losses` changes exactly the drawn months; DP1 recovers a planted DC capacity
  (skipped without torch); and the scaled scoring equals the multiplied-back megawatt error over the
  unscaled capacity, with unequal capacities.

### Design philosophy and verification

R&D code fails fast; nothing runs in production, and no engineering hypothesis concerns capacity
estimation (checked). The roadmap page is not edited (open decision 5). Verification: the CLAUDE.md
set, with `pytest --run-studies` run with and without the `gpu` group, plus pydoclint, the
docs-link checker, and `mkdocs build --strict`.

**Limitations the page states:** the multi-day-zero filter reads later hours, so causal estimators
see a slightly cleaner history than a live service; the planted loss is a DC loss, and a lost
inverter's lower AC limit, which only CX2 and DP1 model, is not planted.

## Open decisions for the maintainer

1. **Approve the root `pyproject.toml` edit in advance?** The study folder joins `extra-paths` and
   `pythonpath`, and the torch files join the `ty` override; the file is outside the `study` skill's
   autonomous-merge paths. Recommended: approve.
2. **May the page show the ANM plant's capped-hour shares?** Recommended: yes, without the plant's
   label, as the beam/diffuse page did.
3. **Ask NGED what happened at the low-output plant in its 3 low months?** Recommended: ask; the
   study does not depend on the answer.
4. **Slots, in order:** 8 CPU cores for about 2 hours, a GPU for about 1 hour (DP1), then a GPU with
   4 cores for 1 hour (XGBoost), never overlapping.
5. **What does the result mean for v0.7?** Whether to build the time-varying table without a
   forecast gain, and any roadmap edit, are the maintainer's decisions.
6. **Keep the lag study's row set,** which leaves out 674 rows lacking the lag study's one-day lag?
   Recommended: keep it, so B0's 8.93 points stay comparable.

## Review triage

Review 1 (simplicity), all accepted:

1. Answer "which technique?" on planted-loss recovery: nested selection, P3, and P4 deleted.
2. Cut to about 9 estimators: 9 kept, HM over the changepoint detector.
3. Reuse lag-frame columns for NR and PR: done, with an equality assertion.
4. Quantile models and capacity uncertainty exploratory: one quantile fit; the mixture cut.
5. Replace NC1 with a constant-scale assertion: done; NC2 kept.
6. Shrink the positive control: 30% at forecast level, 10% at estimator level.
7. Simplify the hindsight version: a forward and backward smoother, shortlist only.
8. Rest the decision on P1 alone: done; P2 dropped as informing no decision.
9. Score with no code change: verified against `_losses`.
10. Cut six exploratory variants: done.
11. Prose cuts: done.
12. Compute arithmetic and the 16-month audit: fixed.
