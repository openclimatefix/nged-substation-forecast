# Plan: do lagged power features improve a solar power forecast? (issue #1138)

## Problem

Issue #1138 theorises that an XGBoost model forecasting solar power should gain from lagged power
features. The lags could calibrate a forecast to soiling, panel degradation, a failed string, part
of a plant offline, aerosol episodes, and changes in the weather forecast product. In a model
trained across all plants, the lags could also tell the model each plant's quirks (such as its
DC:AC ratio) without a physical model of the plant. Nothing in the repository tests this. The
existing persistence baselines read lagged power, but no study gives an XGBoost model lagged
power next to weather.

## Planned solution

One study on the 6 NGED utility-scale solar plants, driven by ECMWF IFS HRES (Open-Meteo Single
Runs, already on disk). Every XGBoost model forecasts from a 00 UTC run, and sees power only up to
that run's start. The study compares a no-lag baseline with five ways of adding lagged power, in
two model scopes (one XGBoost model per plant, and one XGBoost model across all plants with no
plant identifier), at three lead days. The deciding contrasts are written below, before any
result exists. The page is figure-led, as the `study` skill requires.

## Verdict, size and departures

**Verdict: worth doing, with the changes below to the issue's wording.**

**Size: complex.** The five triggers:

- **What gets stored:** a published `docs/studies/` page and files under `data/studies/`; no Patito
  model, Delta table, or asset. Fires under the study skill's rule that a published page counts as
  what gets stored.
- **Production serving path:** not touched. Nothing under `src/` or any package except
  `packages/studies/` changes.
- **Degradation rule:** none touched.
- **More than one defensible design:** yes. The lag definition under a day-ahead origin, the
  two-step leak control, the equal-column-count rule, and the global model's inputs each admit
  several designs.
- **Callers not nameable without searching:** no new shared code is expected to change existing
  callers. If a function in `packages/studies/` has to change, its callers are found by search.

**Reviews:** both plan reviews, both Opus science reviews, the diff review, the mutation pass
only if `packages/studies/` changes, and the prose reviews and personas the study skill sets. The
study skill's merge conditions apply; the diff stays inside `studies/`, `packages/studies/`, and
`docs/studies/`.

**Departures from the issue body:**

1. **"24-hour lag" becomes "the same clock hour on the latest fully observed day".** A forecast
   issued at 00 UTC for a target hour more than 0 h ahead cannot read power 24 h before the target,
   because that hour has not happened. Lead-day 0 (targets 0 to 23 h after the run) reads the
   previous day, a 24 h lag; lead-day 1 reads 48 h before the target; lead-day 2 reads 72 h before.
   The page defines "lag" this way once.
2. **Lead days 0, 1 and 2, not one lead.** The value of a lag falls as the lag lengthens, so a
   single lead would hide the answer.
3. **No new PV dataset.** The recommended answer to the issue's question is below, in "Do we need
   the UK PV dataset?".

## Do we need the UK PV dataset?

**Recommendation: no, for this study.** The target use is utility-scale plants; the
`openclimatefix/uk_pv` data is domestic rooftop PV, whose lag behaviour differs (many small
systems average out failures and soiling). It would also need a second NWP download. The 6 plants
give 933 IFS runs from 2024-03-14 to 2026-10, about 2.5 years, which holds two summers but a
single year of most other months. The 6 plants sit in two grid cells and one 25 km box, so the
effective sample size is weather episodes (months), not plants (the `study` skill's design rule).
An effect would be detected only if it is large; a null is read as "an effect as large as X is not
excluded". The Scope section says domestic PV and other regions are untested. An Opus reviewer is
asked, in the plan review, to challenge this recommendation.

## Design

### Rows and targets

- **Rows:** `ens_forecast_horizons.base_frame(domain="solar")`, the hourly solar rows the
  matched-lead page scores (commissioning ramp, zero half-hours, outages, curtailed hours handled as
  there). Target: hourly power divided by the plant's `effective_capacity_mw`, in percentage points
  of capacity. The forecast is held to the export cap as `clamp_to_cap` does.
- **Origin:** every forecast is made at 00 UTC of day D from the IFS HRES 00 UTC run of day D
  (`studies.ifs_single_runs.served_init_time`). Observed power is available up to 00 UTC of day D.
  This is a conservative origin: the live service reads NGED data later in the day, which a later
  study could test. A run missing from the archive leaves its target hours out, for every arm.
- **Lead days:** 0, 1, 2. Day `N` serves targets from the run issued `N` days earlier at leads
  `24N + 1` to `24N + 24` (`served_lead_hours`). Lead-day 1 is the day-ahead lead the live service
  uses and is the headline.
- **Latest fully observed day:** day D-1. The lag of a day-`N` target is `24 (N + 1)` hours, at the
  same clock hour. For solar the target hour is the hour ending at its label; the lag hour is the
  same label `24 (N + 1)` hours earlier.

### Weather inputs (the same for every arm)

IFS HRES global horizontal irradiance (`shortwave_radiation`), `direct_radiation`, 2 m temperature,
and total cloud cover at the target hour and the run's lead, plus the sun's zenith and azimuth,
extraterrestrial horizontal irradiance, hour of day, day of year, and an IFS-cycle era code (IFS
Cycle 49r1, 12 November 2024, and Cycle 50r1, 12 May 2026; Open-Meteo labels runs before
2024-11-12 as Cycle 49r1 hindcasts, which is a source change, so that date is an era boundary
whatever the labels say). Radiation
is clipped at zero (`clip_radiation`). The exact column list is fixed in the plan's file-by-file
section below and printed into the report (the "arm can silently lose a column" rule).

### Arms

Each arm is an XGBoost model with the absolute-error objective, `colsample_bytree=1`, three
fitting seeds (0, 1, 2), and the primary and sensitivity hyperparameter settings
(`PRIMARY_HYPER_PARAMETERS`, `SENSITIVITY_HYPER_PARAMETERS`). All arms have the same number of
columns, `W + 6`, where `W` is the weather-and-geometry column count and 6 is the widest lag arm's
extra columns. Narrower arms are padded with duplicates of their own columns (the `study` skill's
rule); the negative control below is separate.

| Arm | Extra columns beyond weather and geometry |
|---|---|
| **B0 baseline** | none (padded) |
| **L1 one lag** | observed power at the lag hour (1) |
| **L2 lag with lagged weather** | L1's column, plus the weather forecast for the lag hour: global horizontal irradiance, direct irradiance, 2 m temperature, total cloud cover (5) |
| **S2 two-step** | stage 1's prediction for the target hour, stage 1's prediction for the lag hour, the observed power at the lag hour, and observed minus predicted at the lag hour (4; padded) |
| **W7 weekly statistics** | the minimum, maximum, and mean of observed power at the target's clock hour over the 7 latest fully observed days, plus L1's lag (4; padded) |
| **N1 negative control** | one lag column, taken from a randomly chosen other day at the same clock hour and plant, inside the same year-month and fold (the same marginal distribution as L1, no information) (1; padded) |

**Lagged weather for L2 and S2:** the forecast for the lag hour comes from the 00 UTC run issued
on the lag hour's own day, at that hour's lead (the freshest run available at the origin that
covers the hour, which is the lead-day-0 reading of that hour). The page says this is a forecast
of the weather in the past, not a measurement.

**S2's leakage control:** stage 1 is the baseline weather-only XGBoost model. Stage 2 must not see
stage-1 predictions that stage 1 has fitted on the same rows. Within each outer training set, stage
1's predictions on the training rows are out-of-fold over inner month-block folds (3 inner folds),
and stage 1's predictions on the test fold come from stage 1 fitted on the whole outer training
set. Stage 1's prediction for the lag hour uses the same stage-1 model, shown the lag hour's
lagged-weather columns. Both predictions of every training row are out-of-inner-fold.

### Model scopes

- **Per-plant** (the issue's first set): one XGBoost model per plant, as the matched-lead studies
  do.
- **Global**: one XGBoost model across all 6 plants, with no plant identifier, no coordinates, and
  no capacity column, so that only the lags can tell the model about a plant. Folds are
  contiguous month blocks withheld at every plant at the same time (`assign_folds` with the
  month-block rule), because neighbouring plants share their weather.
- **Exploratory only, global with an unseen plant:** a leave-one-plant-out global XGBoost model with
  the scored months withheld at all plants, for arms B0 and L1 only. It tests whether lags help
  the model on a plant it has never seen.

### Folds, leakage and purging

Month-block folds cut inside each era with `cut_eras`, rotated with `search_fold_offsets`, and
checked with `raise_on_uncovered_months`, because plain `assign_folds` leaves whole calendar months
out of a fold's training rows once eras are cut (the sibling IFS plan found July and August absent
from one fold's training data). The part-months straddling the two era boundaries (2024-11 and
2026-05) are dropped. The era after 2026-05 is too short for folds, so those rows are dropped.
The fold of a target hour is the same at every lead day, and the script asserts it. **Purge:** a training row is dropped if any of its lag inputs (up to 8 days
back) fall inside a test month, because that row would let the test month's observed power enter
training as a feature. Every arm is purged identically, so all arms share the same training rows,
and all arms are scored on the same test rows (a row needs the lag, so rows with an unobserved
lag hour, such as the first 8 days after a gap, are dropped for every arm, B0 included). The
number of rows lost to this is printed in the report.

### Probabilistic output (added at the maintainer's request)

Every arm also fits the quantile model that `fit_one_fold` already offers (`with_quantiles=True`,
the nine levels 0.1 to 0.9 in `QUANTILE_LEVELS`), on the same columns and the same rows as its
point model. Quantiles are sorted to remove crossing and held to the export cap. The scores,
all per row and normalised by the row's own capacity:

- **CRPS** from the nine quantiles (`studies.cross_validation.crps`), the headline probabilistic
  score, in percentage points of capacity, smaller is better.
- **Coverage and width of the 10% to 90% interval:** the share of rows inside it (nominal 80%),
  and its mean width. A lag could help by narrowing the interval where the lag shows the weather
  forecast is behaving, or by widening it where the lag disagrees with the forecast; coverage shows
  whether the narrower interval is still honest.
- **Reliability:** the share of rows below each of the nine quantiles, per arm.
- **Sharpness by condition:** interval width in clear, mixed, and cloudy hours (by the IFS cloud
  cover forecast), to show where the lags change the uncertainty.

Intervals on CRPS differences use the same paired month-and-seed bootstrap
(`bootstrap_difference` on the CRPS column).

### Metric, intervals, controls

- **Metric:** mean absolute error in percentage points of capacity, each row normalised by its own
  plant's capacity before any mean or difference.
- **Intervals:** `studies.bootstrap.bootstrap_difference` (paired, whole calendar months, one of
  three seeds per resample, 2,000 resamples) and `bootstrap_absolute` for each arm's own error.
- **No leaderboard number.** The study reports mean absolute error, not a leaderboard skill, so
  `score_study.py` is not used. The page says so.
- **Controls:** N1 is the negative control (the size of difference produced from nothing, plus the
  column-count effect). A positive control (a synthetic target in which the lag must help, built
  by scaling observed power by a smooth per-plant multiplier that changes at a known date while
  weather is unchanged) shows the instrument can detect a soiling-like level shift: if L1 and S2
  do not recover it, a null in the real data is not read as "no effect".

### Deciding contrasts (planned, written before any result)

Each is an error difference at the primary and the sensitivity setting, for the per-plant and the
global scope, at lead-day 1 (the headline lead), with lead-days 0 and 2 as planned secondary
reads.

- **P1:** L1 minus B0 (does a single lag help?).
- **P2:** L2 minus L1 (does lagged weather add to the lag?).
- **P3:** S2 minus the better of L1 and L2, chosen by the primary-setting point estimate, with the
  choice written to the report before the contrast is computed (does the two-step design help?).
- **P4:** L1 minus N1 (is any gain more than the lag column count produces from noise?).
- **P5:** CRPS of L1 minus CRPS of B0 (do lags improve the probabilistic forecast?), per-plant and
  global, at lead-day 1.

Five planned contrasts, so no multiple-comparison correction is made and the page says so (six
would need a Bonferroni 99.17% interval). **Decision rule, fixed before any result:** a lag "helps"
if, at both hyperparameter settings, the 95% interval of the error difference lies wholly below
zero and the point estimate is at least 0.2 percentage points of capacity (the smallest effect
worth acting on; a smaller gain is reported as statistically detectable but not worth the
complexity). Every planned gain is also read against N1's own difference from B0, which the report
prints, because a wider arm keeps a small edge (about 0.4% of error on synthetic data) even at
`colsample_bytree=1`.

A verdict needs the primary and sensitivity settings to agree. Every other number is exploratory
and labelled so (the claim that lags help the global XGBoost model more than the per-plant one,
judged by the difference of P1 between the two scopes; CRPS for every other contrast; coverage and
sharpness, which are descriptive; W7 minus L1, since the issue asks about weekly statistics but the arm is not one of the
five; day-0 and day-2 reads of the contrasts, the unseen-plant fit, the seasonal split
into summer and winter, importances, any analysis added after the first run is post hoc).

### Charts (the story, as figures)

1. **Figure 1, the headline:** the paired contrasts P1 to P5 at lead-day 1 with their 95%
   intervals, per-plant and global in two panels, primary and sensitivity setting as a marker.
2. **The data:** weather-forecast irradiance against measured power for each anonymised plant over
   one stated week (the clearest, the most variable, the dullest), showing steps, gaps, and
   outages; the lag hour's relation to the target hour as a scatter per lead-day.
3. **The techniques with known answers:** how each lag column is built for one example day; the
   synthetic level shift and the arms that do and do not recover it (the positive control); N1
   beside L1; the two-step design's inner folds drawn as a diagram of months.
4. **The XGBoost models work:** out-of-fold predictions against measured output for each
   anonymised plant over the same weeks; each arm's absolute error per plant.
5. **Results:** error against lead-day for each arm; how much each arm gains at each month (the
   monthly paired differences, to show that the gain, if any, is not one episode); the global
   model against the per-plant model; feature importance (descriptive only) summed by feature
   group.

Plants are anonymised A to F (`studies.anonymise.site_labels_for`); nothing prints an identifier,
name, or coordinate, and power is shown only as a share of capacity.

## Walk through

**The happy path.** `build_lag_frame.py` loads the hourly solar rows (`base_frame`), reads the
IFS HRES runs the target hours need (`served_init_time`, `served_lead_hours`), and adds the lag
hour's observed power, the lag hour's weather forecast, and the weekly statistics from the same
hourly power table, all computed from power strictly before 00 UTC of the origin day. It writes the
frame under `data/studies/lag_features/`. `fit_lag_arms.py` assigns folds (`assign_folds`, era
cuts), purges training rows, and, per outer fold and seed, fits every arm
(`fit_one_fold`, plus the stage-1/stage-2 chain for S2), then writes per-row losses and
out-of-fold predictions. `report_lag_features.py` computes the intervals
(`bootstrap_difference`, `bootstrap_absolute`) and prints every table the page quotes into
`report.md`. `lag_features_charts.py` draws the figures from the saved files and the report. The
page `docs/studies/lag-features.md` quotes the report.

**Branches.** The per-plant run splits `dataset` by site; the global run fits once per fold on all
plants. The sensitivity setting repeats every planned contrast's fits. The positive control reuses
the same pipeline with a modified target column. A missing IFS run leaves the target hours out of
every arm.

**Error handling.** A study fails fast: an arm whose column list differs in length from the
others raises before any fit; a training or test row shared-row check that does not hold raises;
a missing lag hour drops the row for every arm and the count is printed. No production
degradation path is touched.

## What changes, file by file

All new files; nothing existing is edited, except as noted.

- `studies/lag_features/build_lag_frame.py` — builds the rows, the lag and weekly-statistics
  columns, the lagged-weather columns, the purge mask, and the N1 and positive-control columns.
  The column lists come from one function per arm type returning a fixed-length tuple.
- `studies/lag_features/fit_lag_arms.py` — the per-plant and global fits of every arm at both
  hyperparameter settings; S2's nested inner folds; saves losses and out-of-fold predictions.
- `studies/lag_features/report_lag_features.py` — intervals, tables, controls, the printed arm
  column lists, `report.md`.
- `studies/lag_features/lag_features_charts.py` — the figures, with every shared number read from
  the report; the figure text is also written to one file for the figure-text reviews.
- `studies/lag_features/README.md` — scripts, files, the commands that reproduce the page.
- `packages/studies/src/studies/lag_features.py` plus tests in
  `packages/studies/tests/test_lag_features.py` — the lag-hour arithmetic (which instant is the lag
  hour of a target at each lead-day), the observed-before-origin availability mask, the weekly
  statistics, and the nested-fold generator for S2. Each is a silent-failure function (an off-by-one
  hour is plausible-looking), so it is tested; the study skill requires a mutation pass when this
  package changes.
- `docs/studies/lag-features.md` and an entry in `docs/studies/index.md` and `mkdocs.yml` nav.
- `docs/studies/assets/lag_features_*.svg` (optimised with `svgo`).

## Tests

Tests live in `packages/studies/tests/test_lag_features.py`. Each assertion would fail on a
plausible bug:

- **`lag_instant`:** for each lead-day, the lag hour is `24 (N + 1)` hours before the target label;
  fails if a test uses 24 h for every day (the tempting mistake that reads unobserved power).
- **`observed_before_origin` mask:** a lag hour at or after 00 UTC of the origin day is rejected;
  fixture with a lag hour exactly at 00 UTC and one an hour later.
- **`weekly_statistics`:** min, max, and mean use only the 7 latest fully observed days at the same
  clock hour and ignore the origin day itself; fixture with distinct values per day.
- **Nested folds for S2:** no training row's stage-1 prediction comes from a model that saw that
  row's month; fixture checks the month sets are disjoint.
- **Purge:** a training row whose lag window touches a test month is dropped; a row 9 days after the
  block is kept.
- **Equal column counts:** a function asserting every arm has the same number of columns; fixture
  with one arm one column short must raise.
- **Capacity normalisation:** fixtures with unequal capacities.

## Design-philosophy check

R&D code: fail fast, as the inherent-stability page requires of training and metrics code. Nothing
runs in production. The study does not decide the production model; a follow-up issue would carry
any result into `ml_core` feature code. `H` hypotheses: the page cites none unless one on
persistence or lag features exists in `engineering-hypotheses.md` (checked at write-up).

## Docs to update

The new study page, the studies index and nav, `studies/README.md` table row, and the family
README. No roadmap item is completed; no ship-time triage. If a result changes the roadmap's
feature plan, a follow-up issue records it rather than this PR.

## Verification

`uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check`, `uv run pytest
--run-studies -n auto packages/studies`, `uv run pymarkdown scan -r docs README.md CLAUDE.md
packages/*/README.md`, `uv run mkdocs build --strict` and reading the rendered page, pydoclint and
the docs-link checker (the CI steps the skill's set omits).

## Risks and open questions

1. **Origin:** a 00 UTC origin with data to 00 UTC is conservative. The live service reads at 09:00
   UTC with later observed power. Recommendation: keep 00 UTC for this study, and say so.
2. **Sample:** 6 plants, 2.5 years, two grid cells. Recommendation: no new PV data; the results are
   read with month-resampled intervals, and a null is bounded rather than called "no effect".
3. **IFS archive:** 8 run days are missing (2025-08-05 to 09, 2026-06-11, 2026-06-23, 2026-09-19);
   radiation is null at lead 0 and exactly -1 W m-2 appears at leads 73 to 90 h (clipped to zero).
   Target hours on missing days, and rows whose lag hour is absent or was dropped as an outage, a
   commissioning-ramp hour or an export-cap hour, are dropped for every arm, B0 included. The
   weather columns come from the 20-variable product
   `data/studies/downloads/NWP/ECMWF-IFS-SINGLE-RUNS-SOLAR/`, which allows cloud-layer columns.
   Leads here stay at or below 72 h, so interpolated hourly values beyond 90 h do not enter.
4. **Compute slot and checkpoints:** fits need a slot from the Study MAIN COORDINATOR, and the fit
   script checkpoints after every few arm-settings (as `era5_ladder_fit.py` does) because a reboot
   is planned.
4b. **GPU:** XGBoost fits on the GPU if `nvidia-smi` shows one, one device per planned contrast, and
   one refit of an arm on both devices gives the noise floor.
5. **Probabilistic contrast:** making the CRPS contrast planned (P5) forced the global-vs-per-plant
   claim to become exploratory, because six planned contrasts would need a 99.17% interval.
   Recommendation: keep this arrangement. The alternative is six planned contrasts with the
   stricter interval.
6. **Cost:** about 6 arms x 2 scopes x 2 settings x 3 seeds x 3 lead days x folds, with S2 about
   4x; a rough estimate is a few hundred GPU-minutes. No money is spent; no data is ordered.

## Reviews

(Recorded below after each review.)
