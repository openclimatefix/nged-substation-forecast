# Plan: do lagged power features improve a solar power forecast? (issue #1138)

## Problem

Issue #1138 theorises that an XGBoost model forecasting solar power should gain from lagged power
features. The lags could calibrate a forecast to soiling, panel degradation, a failed string, part
of a plant offline, aerosol episodes, and changes in the weather forecast product. In a model
trained across all plants, the lags could also tell the model each plant's quirks (such as its
DC:AC ratio) without a physical model of the plant. The maintainer also asked whether lags help the
forecast quantify its uncertainty. Nothing in the repository tests any of this. The persistence
baselines read lagged power, but no study gives an XGBoost model lagged power next to weather.

## Planned solution

One study on the 6 NGED utility-scale solar plants, driven by ECMWF IFS HRES (Open-Meteo Single
Runs). It reads the matched-lead study's inputs that are already on disk, so the no-lag baseline
is the published "IFS HRES (9 km, Open-Meteo)" arm and its error can be checked against the
published figure. The study adds five lag arms to that baseline and compares them. The point
forecast and a probabilistic forecast (nine quantiles) are both scored. Per-plant XGBoost models
carry the full comparison at the day-ahead lead. A single XGBoost model across all plants, and the
other lead days, carry only the baseline and the one-lag arm. The page is figure-led.

## Verdict, size and departures

**Verdict: worth doing, with the changes below to the issue's wording.**

**Size: complex.** The five triggers:

- **What gets stored:** a published `docs/studies/` page and files under `data/studies/`; no Patito
  model, Delta table, or asset. Fires under the `study` skill's rule that a published page counts as
  what gets stored.
- **Production serving path:** not touched. Nothing under `src/` or any package except
  `packages/studies/` changes.
- **Degradation rule:** none touched.
- **More than one defensible design:** yes. The lag definition, the two-step leak control, the
  control arm, and the global model's inputs each admit several designs.
- **Callers not nameable without searching:** none; the plan adds one function to
  `packages/studies/` and edits no existing one.

**Reviews:** simplicity plan review (done, findings triaged below), correctness plan review, two
Opus scientific-validity reviews, the diff review, the mutation pass (the package changes, by one
function), the prose reviews and personas the `study` skill sets. The study skill's merge
conditions apply; the diff stays inside `studies/`, `packages/studies/`, and `docs/studies/`.

**Departures from the issue body:**

1. **"24-hour lag" becomes "the same clock hour on the latest fully observed day".** A forecast
   issued for a target hour after the run starts cannot read power 24 h before the target at every
   hour. The lag of a lead-day-`N` target is `24 (N + 1)` hours, exactly the lag
   `baselines.diurnal_persistence` reads (24, 48 and 72 h for lead-days 0, 1 and 2). The page
   defines "lag" this way once.
2. **Lead days 0, 1 and 2 for the baseline and one-lag arms only.** Lead-day 1, the day-ahead lead
   the live service uses, carries the full comparison. Lead-day 0 is not a forecast anyone could
   have used for its early hours, because the 00 UTC run is not published until later; the page
   says so.
3. **The issue's "XGBoost with no lagged features" is padded to the same column count** as the
   widest lag arm (see Arms).
4. **No new PV dataset** (see the next section).

## Do we need the UK PV dataset?

**Recommendation: no, for this study.** The target use is utility-scale plants; the
`openclimatefix/uk_pv` data is domestic rooftop PV, whose failures and soiling average out across
many small systems. It would also need a second weather download. Whether its time coverage even
overlaps the IFS archive (which starts 2024-03-14) is unchecked. The 6 plants give about 2.5 years,
two summers but one winter of some months, and sit in two grid cells and one 25 km box, so the
effective sample size is weather episodes (months), not plants. An effect is detected only if it is
large; a null is read as "an effect as large as X is not excluded", and a positive control sized to
a realistic soiling loss measures the detection limit. The six similar plants are weak evidence
for the global-model "plant quirks" hypothesis, and the page scopes that result narrowly. The Scope
section says domestic PV and other regions are untested.

## Design

### Rows and inputs (reused from the matched-lead study)

- **Rows and weather:** the matched-lead inputs on disk under
  `data/studies/per_study/nwp_forecast_comparison/`: `original/solar_forecast_inputs.parquet`
  (power, `cap_mw`, `constrained`, `effective_capacity_mw`, the sun's position, `hour_of_day`,
  `day_of_year`, the persistence columns) joined on `(site, time)` to
  `leads_day10d/solar_extra_lead_inputs.parquet` (`ifs_single_day{0,1,2}_{ghi,temp}`). The
  baseline arm therefore has seven columns: hour of day, day of year, an era code, the sun's
  elevation and azimuth, and the product's global horizontal irradiance and 2 m air temperature at
  its own lead. Direct radiation and cloud cover are not added; they are irrelevant to the lag
  question and would widen every arm.
- **Origin convention:** the existing one (`baselines.issue_time`): the 00 UTC run is issued at its
  init time for lead-day 0 and at 09:00 UTC on the run's own day for later lead days. The latest
  whole observed day is the run's own day minus one, so the lag of a lead-day-`N` target is
  `24 (N + 1)` hours.
- **Target:** hourly power divided by the plant's `effective_capacity_mw`, in percentage points of
  capacity, with curtailed hours left out of training and kept for scoring, and predictions held
  to the export cap (`clamp_to_cap`), as the matched-lead study does. NGED timestamps are already
  corrected at ingest, and are not shifted again.
- **The lag is read strictly.** A new function reads the exact lag hour from the hourly power
  table, with no tolerance, unlike `diurnal_persistence`, which falls back up to 7 days. A row whose
  lag hour is absent, or was dropped as an outage, a commissioning-ramp hour, or an export-cap hour,
  is dropped for every arm, B0 included. A target hour whose IFS run is absent (8 run days are
  missing) is dropped for every arm, and radiation is clipped at zero. The number of rows lost
  is printed.

### Arms

Each arm is an XGBoost model with the absolute-error objective, `colsample_bytree=1`, three
fitting seeds, and the primary and sensitivity hyperparameter settings. All arms have the same
column count: the baseline's 7 plus 4. Narrower arms are padded with duplicates of their own columns.

| Arm | Extra columns beyond the baseline's 7 |
|---|---|
| **B0 baseline** | none (4 duplicates of its own columns) |
| **L1 one lag** | observed power at the lag hour (1, plus 3 duplicates) |
| **L2 lag with lagged weather** | L1's column, plus the forecast global horizontal irradiance and 2 m temperature for the lag hour (3, plus 1 duplicate) |
| **S2 two-step** | stage 1's prediction for the target hour, stage 1's prediction for the lag hour, the observed power at the lag hour, and observed minus predicted at the lag hour (4) |
| **W7 weekly statistics** | the minimum, maximum, and mean of observed power at the target's clock hour over the 7 latest whole observed days, plus L1's lag (4) |
| **N1 level-only control** | one lag column from a random other day at the same clock hour, plant, and year-month (1, plus 3 duplicates) |

**N1 is a control for what kind of information the lag carries, not a null.** A lag from another
day of the same month keeps the plant's slow level (soiling, degradation, the month's clearness) and
drops yesterday's own weather. L1 minus N1 therefore measures the value of day-specific
information beyond the month's level. In the global scope, the month's level also identifies the
plant, which is part of what N1 measures there. The duplicate-padded B0 is the column-count null;
the report prints each arm's difference from B0 so a gain of the size the column count alone gives
(about 0.4% of error even at `colsample_bytree=1`) is visible.

**Lagged weather for L2 and S2** is the lead-day-0 forecast of the lag hour (the run issued that
day at its init time), a forecast of the past, not a measurement.

**S2's leak control** reuses the repository's withheld-pair pattern
(`run_hybrid_experiment._add_physics_predictions`, `*_fold{fold}` columns that `out_of_fold_losses`
resolves): for each scored fold, stage 1 is fitted with the scored fold and the row's own fold
withheld, so no stage-1 prediction comes from a model that saw the row. Stage 1 is the primary
setting with one seed, reused for every stage-2 seed and setting. Its prediction for the lag hour
uses the lag hour's own row with its lead-day-0 weather.

### Model scopes

- **Per-plant:** one XGBoost model per plant. Carries all six arms at lead-day 1, and B0 and L1 at
  lead-days 0 and 2.
- **Global:** one XGBoost model across the 6 plants, with no plant identifier, coordinates, or
  capacity column, so that only the lags can tell the model about a plant. B0 and L1 only, at
  lead-day 1.

### Folds, leakage and purging

The shared rows' folds and `era_code` (as `fit_extra_leads.py` uses them), with no extra cut at
IFS Cycle 49r1 (2024-11-12) or Cycle 50r1 (2026-05-12). The matched-lead and ENS-horizons pages
cross those cycles with no cut, an era cut at Cycle 49r1 gave the same solar results there, and
the Cycle 50r1 era is only about three months, too short for five folds. A cut would also remove
the situation the issue names, a lag calibrating to a change in the weather forecast product.
`raise_on_uncovered_months` checks that every calendar month is covered by training, and the
fold of a target hour is the same at every lead day (asserted). The page reports the before and
after Cycle 50r1 split of the one-lag gain as exploratory. Open-Meteo labels runs before
2024-11-12 as Cycle 49r1 hindcasts, which is a source change; the exploratory split at that date is
reported too. **Purge:** a training row is dropped if any of its lag inputs (3 days for L1, 8
for W7) falls inside a test month. The purge is a filter in the fit script.

### Metrics and intervals

- **Point forecast:** mean absolute error in percentage points of capacity, each row normalised by
  its own plant's capacity before any mean or difference.
- **Probabilistic forecast:** every arm also fits the quantile model `fit_one_fold` offers
  (`with_quantiles=True`, the nine levels 0.1 to 0.9), on the same columns and rows. Quantiles are
  sorted to remove crossing and held to the export cap. Scores: the continuous ranked probability
  score (CRPS) from the nine quantiles (`studies.cross_validation.crps`); the share of rows inside
  the 10% to 90% interval (nominal 80%) and its mean width; the share of rows below each of the
  nine quantiles (reliability); and the interval's width in clear, mixed, and cloudy hours, to show
  where lags change the uncertainty.
- **Intervals:** `bootstrap_difference` (paired, whole calendar months, one of three seeds per
  resample, 2,000 resamples) and `bootstrap_absolute`; CRPS differences use the same bootstrap.
- **No leaderboard number.** The study reports error in points of capacity, not a leaderboard
  skill, so `score_study.py` is not used. The page says so.
- **Compute:** CPU through `arm_runner.run_all`, which suits about 9,000 daylight rows per plant.
  Fits need a slot from the Study MAIN COORDINATOR and checkpoint after every few arm-settings,
  because a reboot is planned.

### Controls

- **Positive control:** the same pipeline on a synthetic target in which a level shift of a
  realistic soiling or string-loss size (a few percent of capacity, at a known date, per plant) is
  applied to power, with weather unchanged. B0 and L1 only, per-plant, lead-day 1, primary setting,
  one seed. It gives a measured detection limit: the smallest shift that L1 recovers.
- **Negative control:** the duplicate-padded B0 (column-count null) and N1 (level-only control).

### Deciding contrasts (planned, written before any result)

All at lead-day 1, at both hyperparameter settings.

- **P1:** L1 minus B0, mean absolute error, per-plant (does a single lag help the point forecast?).
- **P2:** L2 minus L1, per-plant (does lagged weather add to the lag?).
- **P3:** S2 minus the better of L1 and L2, per-plant, the better chosen by the primary-setting point
  estimate and written to the report before the contrast is computed (does the two-step design
  help?).
- **P4:** CRPS of L1 minus CRPS of B0, per-plant (do lags improve the probabilistic forecast?).
- **P5:** L1 minus B0, mean absolute error, global (do lags help one XGBoost model across plants,
  without a plant identifier?).

Five planned contrasts, so the page makes no multiple-comparison correction and says so (six would
need a Bonferroni 99.17% interval). **Decision rule, fixed before any result:** a lag "helps" if, at
both hyperparameter settings, the 95% interval of the error difference lies wholly below zero and
the point estimate is at least 0.2 percentage points of capacity (the smallest effect worth acting
on; a smaller gain is reported as statistically detectable but not worth the complexity). Every
other number is exploratory and labelled so: W7 minus L1, N1 comparisons, CRPS for every other
contrast, the lead-day 0 and 2 reads, the global-against-per-plant difference of P1, coverage and
sharpness (descriptive), the before and after Cycle 50r1 split, and any analysis added after the
first run (post hoc).

### Charts (the story, as figures)

1. **Figure 1, the headline:** P1 to P5 with their 95% intervals, primary and sensitivity setting as
   a marker, under the bottom line.
2. **The data:** forecast irradiance against measured power for each anonymised plant over one
   stated week per rule (the clearest, the most variable, the dullest), showing steps, gaps and
   outages; the lag hour's relation to the target hour as a scatter per lead-day.
3. **The techniques with known answers:** how each lag column is built for one example day; the
   synthetic level shift and the arms that do and do not recover it; N1 beside L1 and what the
   difference means; the two-step design's withheld folds drawn as months.
4. **The XGBoost models work:** out-of-fold predictions against measured output for each anonymised
   plant over the same weeks; each arm's absolute error per plant; B0's error beside the published
   matched-lead figure.
5. **Results:** error against lead-day for B0 and L1; monthly paired differences (the gain is not
   one episode); per-plant against global; reliability, coverage and width; feature importance
   (descriptive only) summed by feature group.

Plants are anonymised A to F (`studies.anonymise.site_labels_for`); nothing prints an identifier,
name, or coordinate, and power is shown only as a share of capacity.

## Walk through

**The happy path.** `build_lag_frame.py` loads the matched-lead rows and the IFS day columns,
reads the hourly power (`studies.power`), and adds, with strict joins, the lag-hour power, the
lag-hour forecast, the weekly statistics, and the N1 column. It drops rows with an absent lag hour,
or a missing run, for every arm, and writes the frame under
`data/studies/lag_features/`. `fit_lag_arms.py` assigns the shared rows' folds, purges training
rows, fits stage 1 of S2 with the withheld-pair pattern, and fits every arm through
`arm_runner.run_all` and `out_of_fold_losses` at both settings, with quantiles, writing per-row
losses and out-of-fold predictions. `report_lag_features.py` computes the intervals
(`bootstrap_difference`, `bootstrap_absolute`) and prints every table the page quotes, plus each
arm's column list, into `report.md`. `lag_features_charts.py` draws the figures from the saved
files and the report. The page `docs/studies/lag-features.md` quotes the report.

**Branches.** The per-plant run splits `dataset` by site; the global run fits once per fold on all
plants. The sensitivity setting repeats every planned contrast's fits. The positive control reuses
the pipeline with a modified target column. The lead-day 0 and 2 runs differ only in the lag and
the IFS day columns.

**Error handling.** A study fails fast: an arm whose column list has a different length from the
others raises before any fit; an uncovered calendar month raises (`raise_on_uncovered_months`); a
shared-row check that does not hold raises. Dropped rows are counted and printed. No production
degradation path is touched.

## What changes, file by file

- `studies/lag_features/build_lag_frame.py` — the strict lag, the lag-hour forecast, N1, the row
  drops; column lists come from one function per arm type returning a fixed-length tuple.
- `studies/lag_features/fit_lag_arms.py` — folds, purge, stage 1 of S2, the fits, the positive
  control, saved losses and predictions.
- `studies/lag_features/report_lag_features.py` — intervals, tables, the printed arm column lists,
  the published-figure check, `report.md`.
- `studies/lag_features/lag_features_charts.py` — the figures, reading every shared number from the
  report; figure text also written to one file for the figure-text reviews.
- `studies/lag_features/README.md` — scripts, files, and the commands that reproduce the page.
- `packages/studies/src/studies/baselines.py` — one added function, the weekly statistics
  (minimum, maximum and mean of the same clock hour over the 7 latest whole days before the issue
  day), with a test in `packages/studies/tests/test_baselines.py`. It is the one piece with a
  silent-failure risk that no existing function covers. The strict lag stays in the build script.
- `docs/studies/lag-features.md`, an entry in `docs/studies/index.md` and `mkdocs.yml`, and
  `docs/studies/assets/lag_features_*.svg` (optimised with `svgo`).

## Tests

- **Weekly statistics:** min, max and mean use only the 7 latest whole days at the same clock hour,
  never the issue day itself; the fixture gives each day a distinct value, so an off-by-one day
  changes all three outputs. Fails on the plausible bug of including the issue day.
- **Capacity normalisation and equal column counts** are asserted in the scripts and checked by the
  report, because study scripts have no unit tests; the report prints the column lists.

## Design-philosophy check

R&D code: fail fast, as the inherent-stability page requires of training and metrics code. Nothing
runs in production. The study does not choose the production model; a follow-up issue would carry
any result into `ml_core` feature code. The page cites no hypothesis label unless one on
persistence or lag features exists in `engineering-hypotheses.md` (checked at write-up).

## Docs to update

The new study page, the studies index and nav, the `studies/README.md` row, and the family README.
No roadmap item completes, so there is no ship-time triage.

## Verification

`uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check`, `uv run pytest
--run-studies -n auto packages/studies`, `uv run pymarkdown scan -r docs README.md CLAUDE.md
packages/*/README.md`, `uv run mkdocs build --strict` and reading the rendered page, pydoclint and
the docs-link checker.

## Risks and open questions

1. **Probabilistic contrast:** making the CRPS contrast planned (P4) forced the global-versus-
   per-plant claim to become exploratory, because six planned contrasts need a 99.17% interval.
   Recommendation: keep this. Alternative: six planned contrasts with the stricter interval.
2. **No era cut at IFS cycle changes:** this follows the matched-lead precedent and keeps the
   issue's "calibrate to NWP changes" situation, but a sibling IFS plan cuts at both cycles. If a
   reviewer insists, the before and after split reported as exploratory becomes the sensitivity
   check.
3. **Sample:** 6 plants, 2.5 years, two grid cells. No new PV data; a null is bounded, not called
   "no effect".
4. **Origin:** the existing convention reads power to the run's own day minus one. A live service
   with fresher power could do better; the page says so.

## Triage of the simplicity review

Accepted: the era scheme (cycle-change cuts dropped, shared folds used); reuse of the matched-lead
inputs, with B0 reproducing the published arm; the existing origin convention; no new package
module except one tested function; the leave-one-plant-out fit cut; the smaller grid (global and
other lead days for B0 and L1 only); a positive control sized to a realistic shift; S2 stage 1 at
one setting and seed with the withheld-pair pattern; the CPU-only plan.

Modified: N1 stays, relabelled as a level-only control rather than a null, because its
disagreement with L1 is the scientific question of calibration against day-specific information;
the duplicate-padded B0 is the null. The lag is read strictly rather than through
`diurnal_persistence`'s 7-day tolerance, because a sibling study found a lag taken from a dropped
hour is wrong data and every arm must share the same rows.

Rejected: none outright.
