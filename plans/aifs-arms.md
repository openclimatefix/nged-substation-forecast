# Plan: add ECMWF AIFS Single and AIFS ENS to the matched-lead comparison (issue #923)

**The matched-lead page compares weather forecast products for solar and wind power at six solar
farms and three wind farms, and it does not mention ECMWF's machine-learned weather model, AIFS.**
The data is now on disk under `data/studies/weather/ECMWF-AIFS/` (AIFS Single) and
`data/studies/weather/ECMWF-AIFS-ENS/` (AIFS ENS): four runs a day, leads 0 to 360 h in 6-hour
steps, a 3 by 3 crop of 0.25° cells. The two products are harder to compare fairly than the products
already on the page. Each serves 6-hourly steps where ENS serves 3-hourly steps to 144 h, so a
product with finer time steps is favoured by resolution alone. AIFS Single carries radiation and
100 m wind only from the 2025-02-24 06 UTC run, and AIFS ENS starts on 2025-07-02, so neither
product covers the page's 35,263 solar and 37,407 wind shared rows. Both products change model
version inside the record, and the store carries no version marker.

**The plan builds AIFS's inputs with the functions that already build ENS's and GEFS's, scores AIFS
on its own row set, and adds paragraphs to the page without any claim that AIFS improves.** A new
build script reads each AIFS run at the same lead as ENS (the 00 UTC run, day `d` after the run's
day, so lead `24d + h`), and upsamples 6-hourly steps to hourly targets with the ENS-horizons page's
own rules. A like-for-like control is ENS's mean and control member emulated at 6-hourly steps
(`band_steps(six_hourly=True)`), and IFS 0.25° sampled at 6-hourly valid times. Every reference is
refitted on the AIFS rows on the same graphics processing unit (GPU) as the AIFS arms, at the
primary hyperparameter setting only. Folds are cut inside the AIFS version eras, whose edges are
whole months, and every AIFS number is labelled exploratory and post hoc. The output is a new
write-once folder, `data/studies/nwp_forecast_comparison_aifs/`, a `report.md` that prints every
number the page quotes, and one new results section on the page. The page's answer to "is AIFS
improving faster than conventional weather models" is a marked placeholder until the coordinator's
literature summary arrives.

## Verdict, size, and departures

**Verdict: worth doing as described, with the departures below.** No pull request or branch is open
against #923 (`gh pr list --search 923` and `git branch -a --list '*923*'` found none). The issue's
premises match the data on disk: I re-derived them from the two `README.md` files, the two
`lineage.json` files, and the parquet files (see "What the data holds").

**Size: complex, by the five triggers in `plan-issue` step 3.**

1. *What gets stored:* fires. A new folder under `data/studies/`, and a published page, count as
   stored (the `study` skill sizes every study as complex). No Patito contract, Delta table, or
   Dagster asset changes.
2. *Production serving path:* does not fire. Nothing under `src/`, `packages/ml_core/`, or
   `packages/xgboost_forecaster/` changes.
3. *Degradation rule:* does not fire. This is R&D code, which fails fast on purpose
   (`docs/design-philosophy/inherent-stability.md`).
4. *More than one defensible design:* fires. The lead convention, the era edges, the 6-hourly
   control, and the row set each admit alternatives, which "Risks and open questions" lists.
5. *Callers not nameable without a search:* does not fire. The two shared helpers this plan touches,
   `band_steps` and `_ens_member_arms`, have callers I named by grep: `ens_forecast_horizons.py`
   (lines 952, 1505, 1521) and `build_forecast_inputs.py` (lines 344, 387, 846, 911).

Two triggers fire, so the issue is complex: **all four reviews** (two plan reviews, two diff
reviews). The `study` skill adds two Opus scientific-validity reviews before any number reaches
`docs/`, and a mutation pass because `packages/studies/` changes.

**Departures from the agreed design input (`/tmp/claude-1000/aifs-design.md`):**

- **Era edges are whole months, not init-time instants.** The design keys eras on init time. A cut
  at an instant inside a month needs a new era cutter in `packages/studies`. Whole months reuse
  `studies.cross_validation.cut_eras` unchanged, the same way the published study drops January 2026
  around the UKV upgrade. A guard proves every kept row's run lies in the era its month gave (see
  `model_versions.py` below), so "keyed on init time" still holds. The cost is three dropped months
  out of 19 (2025-02, 2025-08, and 2026-05), and the AIFS ENS set also loses 2025-07.
- **The bracket method replaces "IFS 0.25° at day 1" alone.** The archive's lead for IFS 0.25° is
  `24d + (h mod n)` with the run cycle `n` unmeasured for IFS 0.25° (the V1b run-switch check reports
  "no clear signature"). The page's own method, which needs no knowledge of `n`, brackets a Previous
  Runs product between the reference at day `d − 1` and day `d`. AIFS therefore needs day 0 as well
  as days 1 and 2.
- **The Single arm reads its `control` reduction, not `mean`.** The mean-wind-vector direction in
  `reduce_members` divides by the vector norm, which is 0 for a calm hour when there is one member.
- **The 2026-05-12 boundary reads 06 UTC, not 00 UTC.** The design says AIFS Single v2 starts
  "2026-05-12 00 UTC". Both `README.md` files and `docs/roadmap/data-sources.md` (rows for
  2026-05-12) say 06 UTC. It does not matter to the rows, because 2026-05 is dropped whole, but the
  era table in code uses the later, better-sourced time.

## What the data holds (read-only checks, run while writing this plan)

- **AIFS Single:** 3,629 runs from 2024-04-01 00 UTC to 2026-09-25 00 UTC, 61 leads (0 to 360 h in
  6-hour steps), 9 cells, 549 rows per run for every 00 UTC run. Every 00 UTC run from 2025-02-26
  to 2026-09-25 is present. The 06, 12, and 18 UTC runs are present except 2026-09-25's, which had
  not run when the download finished. Shortwave, longwave, and 100 m wind are `NaN` (not null)
  before the 2025-02-24 06 UTC run. Shortwave and longwave are `NaN` at lead 0 in every run.
  There is no `ensemble_member` column.
- **AIFS ENS:** 1,801 runs from 2025-07-02 00 UTC to 2026-09-25 00 UTC, 51 members (0 to 50, member 0
  the control), 61 leads, 9 cells, all fields present beyond radiation's lead 0. No 00 UTC run is
  missing.
- **AIFS Single's 2025-01-21 06 UTC run** is `NaN` at four leads. It pre-dates the row set and is a
  06 UTC run, which the 00 UTC arms never read.
- **Coverage of the published shared rows** (35,263 solar, 37,407 wind) by a complete 00 UTC run at
  the site's nearest 0.25° cell, at day 1: Single covers 31,815 solar and 31,386 wind rows in 19
  valid months (from init 2025-02-25); AIFS ENS covers 22,333 solar and 22,797 wind rows in 14
  valid months (from init 2025-07-02). Days 2 and 3 differ from day 1 by under 0.5%; the report
  prints day 0.
- **Wind at 10 m only:** the store holds 10 m wind and temperature from 2024-04-01, so wind alone
  could be scored on 6,220 shared rows in 2024-12, 2025-01, and 2025-02, from runs of unknown model
  version (before AIFS Single v1.0 went operational on 2025-02-25 06 UTC). Solar has no equivalent,
  because radiation is `NaN` there. The plan does not score them (see open question 3).

## The row sets, eras, and folds

**The AIFS row set is the published shared rows restricted to the hours whose AIFS run is complete
and inside a version era.** Two nested sets follow, and each arm is fitted on the sets it belongs to.

| Set | Arms | Run rule | Valid months kept | Solar rows (day 1) | Wind rows (day 1) |
|---|---|---|---|---|---|
| `single` | AIFS Single, every control | 00 UTC init from 2025-02-26 (the first run after v1.0 went operational) | 16: 2025-03 to 2025-07, 2025-09 to 2025-12, 2026-02 to 2026-04, 2026-06 to 2026-09 | 28,570 | 28,873 |
| `ens` | those, plus AIFS ENS | as `single`, and AIFS ENS's own start (init 2025-07-02) | 11: 2025-09 to 2025-12, 2026-02 to 2026-04, 2026-06 to 2026-09 | 16,911 | 18,555 |

Day 2 and day 3 counts equal day 1's on both sets, and the report prints day 0's.
The published shared rows already drop 2026-01 (the UKV upgrade) and much of spring 2026, so
AIFS inherits those gaps: 2026-04 and 2026-05 hold only 272 and 224 wind rows.

**Three eras for `single` and two for `ens`, each a version of AIFS.** The dated table is from
ECMWF's release pages as recorded in `docs/roadmap/data-sources.md` and both `README.md` files, and
is not verified in the data.

| Era | AIFS Single | AIFS ENS | Months in era (`single` / `ens`) |
|---|---|---|---|
| 0 | v1.0, init 2025-02-25 06 UTC to 2025-08-26 (the 2025-07-31 06 UTC to 2025-08-01 18 UTC window is excluded) | not yet running (v1 starts 2025-07-01 06 UTC) | 5 / not present |
| 1 | v1.1, init 2025-08-27 06 UTC to 2026-05-11 | v1, init 2025-07-01 06 UTC to 2026-05-11 | 7 / 7 |
| 2 | v2, init from 2026-05-12 06 UTC | v2, init from 2026-05-12 06 UTC | 4 / 4 |

The `ens` set starts in 2025-09, and not with ENS's first month (2025-07), because AIFS Single
changes version from v1.0 to v1.1 inside 2025-07 to 2025-08. Keeping 2025-07 would put one month of
Single v1.0 into an era whose other 7 months are v1.1, or make a one-month era that the model could
never train on when that month is held out. Both sets therefore use an era code that is a version of
every arm on the set.
The month cut is `cut_eras(first_months=("2025-09", "2026-06"))` on `single` and
`("2026-06",)` on `ens`. Era 0's last month is 2025-07, because 2025-08 holds the reverted v1.1
attempt and the v1.0 to v1.1 switch, and it is dropped whole. Era 2 starts in 2026-06, because
2026-05 holds the v2 switch.
Era 2 is short (4 months) and the IFS Cycle 50r1 change of the same date is a confound the page
states: a v2 change cannot be told apart from a change in IFS itself, and the IFS 0.25° and ENS
references sit in the same era.

**Every arm on a set gets the same era code, and the code is the AIFS era.** `arm_columns` puts
`era_code` into every arm's calendar columns. The build overwrites the published `era_code` with the
AIFS era on the AIFS rows, so the references refitted here see the same era feature as the AIFS
arms, and every arm keeps seven columns.

**Fold offsets are found before any fit and re-checked on every run.** I ran
`studies.cross_validation.search_fold_offsets` read-only on both sets and both technologies. The
identity rotation covers every calendar month, and it is the first result: 13 rotations cover
`single` and 4 cover `ens`, all found from which hours exist, with no error read. The plan therefore
sets `AIFS_FOLD_OFFSETS = {0: 0, 1: 0, 2: 0}` for `single` and `{0: 0, 1: 0}` for `ens`.
`nwp_forecast_comparison.coverage_table` re-checks them on every run and raises on an uncovered
month, and the fit script reruns `search_fold_offsets` and stops if the constant is no longer in its
result.

## How the AIFS build reuses the published build

**AIFS is read as a whole-run archive, at the same lead as ENS.** For a target hour on UTC day `D`
the band for day `d` reads the 00 UTC run of day `D − d`, so the lead is `24d + h`. That is
`ens_forecast_horizons.band_steps`'s rule, and it is what the published ENS and GEFS arms do. Only
the 00 UTC run is read, so the three fresher AIFS runs a day are unused; a live service could read
them, and the page says the study did not test that. The 00 UTC AIFS run is assumed readable by
09:00 UTC, like ENS's, and its publication time is not measured.

**Each AIFS product enters through the same four functions as ENS and GEFS.** The new script
`studies/nwp_forecast_comparison/build_aifs_inputs.py` builds an extract frame in the shape
`ens_forecast_horizons.members` returns (`site`, `init_time`, `ensemble_member`, `lead_hours`,
`ghi_w_m2`, `temp_c`, `speed_100m`, `direction_100m`, `speed_10m`, `direction_10m`), then calls
`ens_member_arms` (today `_ens_member_arms`), which calls `band_steps`, `upsampled_fields`,
`combine`, `reduce_members`, and `prefixed`. The chosen upsampling combination is read from
`UPSAMPLING_METHODS` (`clear_sky` for solar, `speed_components` for wind), never re-chosen. So
radiation is rebuilt through the clear-sky index, temperature is interpolated linearly, wind speed is
interpolated as vector components, and each hourly target is read from one run, with no
interpolation across runs.

**The extract is built the way `_gefs_members_frame` builds GEFS's, with three differences.**

- *NaN becomes null first.* `band_steps` drops incomplete runs with `drop_nulls`, which does not see
  `NaN`. A `NaN` radiation or 100 m wind from before 2025-02-24 would otherwise pass through the
  interpolation and reach XGBoost. The function `dynamical_members` in the new
  `packages/studies/src/studies/dynamical_members.py` converts every float column's `NaN` to null,
  and the build then calls `studies.guards.check_no_missing` on every arm's columns.
- *No window inversion.* AIFS radiation is already a 6-hour window mean ending at the lead
  (`README.md`), so it needs no `gefs_step_means`. `verify_aifs_steps.py` checks that reading against
  ERA5 (see below), as `verify_extra_leads.py` checks GEFS's.
- *All steps are 6-hourly.* `band_steps` gets `fine_step_last_lead=0`, which gives every step width 6.
  Single gets `ensemble_member = 0` and `ensemble_size=1`. AIFS ENS is read lazily
  (`pl.scan_parquet`, filtered to 00 UTC and the sites' cells before collecting), because the file
  holds 50 million rows.

**Site to cell uses the nearest 0.25° cell, exactly as GEFS does.** `nearest_cell_by_site` (today
`_gefs_cell_selection`) matches each site to a cell id `lat_index * 10 + lon_index`, through
`studies.grid_sampling.nearest_cells`. The cell id, the roster's coordinates, and
`_grid_cells.parquet`'s contents are never printed or written. ENS is an area-weighted mean over an
H3 resolution-5 cell and IFS 0.25° is a point read, so the three spatial representations differ. The
page says so.

**The control is ECMWF's own products at the same lead and the same 6-hourly steps.**

- *ENS mean and control member, 6-hourly.* `ens_member_arms(..., six_hourly=True)` passes
  `six_hourly` through to `band_steps`, which calls `coarsen_to_six_hourly`: it keeps ENS's steps at
  multiples of 6 h, and averages each pair of 3-hourly radiation steps into their exact 6-hour mean.
  Arm names are `ens_mean6_day<d>` and `ens_control6_day<d>`. The mean and control at full 3-hourly
  resolution are the published columns `ens_mean_day<d>` and `ens_control_day<d>`, refitted, so the
  page shows what the 6-hourly steps cost ENS as well.
- *IFS 0.25°, sampled at 6-hourly valid times.* The published columns `ifs025_day<d>` are hourly
  values served from Previous Runs. `studies.stamps.six_hourly_stamps` (new) takes the hourly
  `previous_day<d>` series, keeps each instantaneous field at valid times 00, 06, 12, and 18 UTC,
  and takes the radiation stamp as the mean of the six hourly values ending at the stamp (a stamp
  missing any of the six is dropped, never averaged over five). The build lays those stamps into an
  extract in the same shape as above, with one pseudo-run per (site, day) whose lead 0 is midnight
  at the start of day `D − d`, and passes it to `ens_member_arms` with one member and the `control`
  reduction. IFS 0.25° wind direction at 10 m comes from `wind_direction_10m_previous_day<d>` in
  `ECMWF-IFS-025/previous_runs/combined.parquet`, because the published inputs carry direction at
  100 m only. Arm names are `ifs025_6h_day<d>`. The construction takes the served value at every
  stamp, so the stamps of consecutive pseudo-runs mix Open-Meteo runs; the page says so.
- *A negative control.* `aifs_single_day1_permuted` shows AIFS Single's day-1 columns shuffled among
  hours that share a site, a year-month, and an hour of day, through `studies.blending.
  climatology_permutation`, the guard the published blends use. It shows what the pipeline produces
  from AIFS's time series without its information.

**Bands are days 0, 1, and 2 (`AIFS_DAYS = (0, 1, 2)`).** Day 1 is the day-ahead product and the
deciding band. Days 0 and 1 bracket IFS 0.25° day 1 and days 1 and 2 bracket IFS 0.25° day 2,
following the page's method. ENS's steps are 3-hourly up to lead 144 h, so a day-2 band needs no
6-hourly beyond-144 h handling.

**Two problems in code already on disk are fixed or guarded, in scope because the build depends on
them.**

- *`band_steps` drops lead 0 (and the first stamp) of a wind band in 6-hourly mode.* It passes
  `period_means=frozenset({"ghi_w_m2"})` for both technologies, and `coarsen_to_six_hourly` then
  demands a preceding 3-hour step for every kept lead, though a wind field has no period mean. For
  day 1 the dropped stamp is lead 18 h, which no target reads, so the published day-1 arm is
  unchanged. For day 0 the dropped stamp is lead 0 h, so wind targets from 0 to 5 h would fall
  outside the stamps, and the 6-hourly ENS control would be treated differently from AIFS at those
  hours. The fix passes `period_means` only for solar. The report prints the day-1 wind
  `ens_mean6_day1` columns before and after the fix and their maximum absolute difference, which must
  be 0.
- *The published build's helpers are private.* The `study` skill says not to copy the pattern of
  importing private helpers. `_ens_member_arms` becomes `ens_member_arms` with a new `six_hourly:
  bool = False` argument, and `_gefs_cell_selection` becomes `nearest_cell_by_site`, in
  `build_forecast_inputs.py`, with their three call sites updated. The reconciliation table in the
  report rebuilds `ens_mean_day1` with the default `six_hourly=False` and compares it with the
  published column, maximum absolute difference 0.

**Output is a new write-once folder, and nothing is written to the published folder.**
`build_aifs_inputs.py --published-dir DIR --output-dir DIR2` reads the published inputs' `(site,
time)` keys only, as `build_extra_leads` does, and writes `<domain>_aifs_inputs.parquet`. It refuses
an output folder equal to the published folder, and calls `studies.guards.refuse_to_overwrite`.
Every written column is checked against a schema that holds no coordinate, cell id, or generator
name; the only site column is the `A` to `F` and `W1` to `W3` label.

## Fit design

**The fit script `fit_aifs.py` follows `fit_extra_leads.py`, with resume added.** One row set and
one technology at a time, on the GPU, at the primary setting, exploratory, and one fit job at a time.

- *Arms per set.* `single`: AIFS Single days 0, 1, 2; controls `ens_mean6`, `ens_control6`,
  `ens_mean`, `ens_control` at days 0, 1, 2; `ifs025_6h` and `ifs025` at days 1 and 2; the negative
  control at day 1. `ens`: those, plus AIFS ENS `aifs_ens_mean` and `aifs_ens_control` at days 0, 1,
  2. Roughly 44 arm fits per technology at each of 3 to 6 sites: about 400 fits across the two
  technologies.
- *Rows and folds.* `aifs_rows(input_dir, aifs_dir, domain, row_set)` loads
  `nwp_forecast_comparison.rows` on the published inputs, joins the AIFS columns, drops rows whose
  run is outside the set's rule or whose valid month is dropped, requires every arm's columns on
  every row (`check_no_missing`), overwrites `era_code` with the AIFS era, recuts folds with
  `cut_eras(first_months=..., fold_offsets=AIFS_FOLD_OFFSETS[row_set])`, and calls `coverage_table`.
  The era guard `model_versions.check_month_eras_match_versions` then runs.
- *Column counts.* Every arm is `arm_columns(domain, prefixes=(prefix,))` with seven columns. The
  report prints every arm's column list (the `study` skill's "an arm can silently lose a column").
- *Resume.* Each (technology, row set, arm) writes `_arm_cache/<domain>_<set>_<arm>.parquet` when
  its fits finish, and the run skips arms already cached. A crash costs one arm. At the end the
  script concatenates the cache into `<domain>_<set>_losses.parquet` and
  `<domain>_<set>_predictions.parquet` (through `predictions_from_losses`).
- *Determinism.* `--check` fits `aifs_single_day1` at one wind site twice on the GPU and stops unless
  `fingerprint` agrees, and it is run before the full fit, as `fit_extra_leads.py --check` is. Before
  either run, check CPU load (`uptime`; one spinning core oversubscribes an all-core XGBoost fit).
- *Device noise floor.* A third set, `published`, refits the reference arms `ens_mean_day1`,
  `ens_control_day1`, and `ifs025_day1` on the published shared rows, folds, and eras at the same GPU,
  and reports each arm's GPU fit minus its published CPU fit. The report also compares those GPU fits
  with the two arms the extra-lead run already refitted (`ens_mean_day1`, `ifs025_day1` in
  `data/studies/nwp_forecast_comparison_leads/`), which shows whether a GPU rerun reproduces itself
  across sessions. No contrast ever mixes a GPU fit with a CPU fit.
- *Second setting.* The sensitivity setting is fitted for an arm pair only when the primary result is
  near the 5% line (an interval bound within 20% of its width from zero), through
  `--setting sensitivity --arms A,B`, and the report says which pairs got it. A verdict needs both
  settings to agree, and a disagreement is stated.

**The contrasts are listed here before any fit, and every one is exploratory and post hoc.** No AIFS
contrast was written into the published plan, so none is "planned". Each line of the report carries
its row count and month count (from `bootstrap_difference`) and the label.

- *Single against ENS on `single`:* `aifs_single_d1` minus each of `ens_mean6_d1`,
  `ens_control6_d1`, `ens_mean_d1`, and `ens_control_d1`; the same at day 2, and at day 0 as the
  bracket's lower side.
- *Single against IFS 0.25° on `single`:* `aifs_single_d0` and `aifs_single_d1` each minus
  `ifs025_6h_d1` and minus `ifs025_d1`; `aifs_single_d1` and `aifs_single_d2` each minus
  `ifs025_6h_d2` and minus `ifs025_d2`.
- *What the 6-hourly steps cost:* `ens_mean6_d1 − ens_mean_d1`, `ens_control6_d1 − ens_control_d1`,
  `ifs025_6h_d1 − ifs025_d1`.
- *AIFS ENS on `ens`:* `aifs_ens_mean_d1 − ens_mean6_d1`, `aifs_ens_control_d1 − ens_control6_d1`,
  `aifs_ens_mean_d1 − aifs_single_d1`, `aifs_ens_mean_d1 − aifs_ens_control_d1`, and the same at
  days 0 and 2.
- *Each version era on its own:* the day-1 absolute error of every arm and the two contrasts above
  (`aifs_single_d1 − ens_mean6_d1` and `aifs_ens_mean_d1 − ens_mean6_d1`) in each era, with its rows
  and months. The v2 era holds 4 months and is reported with that caveat.
- *Hours on an AIFS step and hours between steps:* the same contrasts with the rows split by whether
  the hour's label falls on a multiple of 6 (`hour % 6 == 0`) or not.
- *Negative control:* `aifs_single_d1 − aifs_single_d1_permuted`.
- *Absolute error of every arm,* with its 95% interval, rows, and months, for each set. This is the
  report's first table, because which input is best can matter more than any contrast
  (`report-absolute-skill-not-only-contrasts`).

About 60 exploratory intervals will print, so about 3 will exclude zero by chance at the 5% level.
The report says so in its header, as the extra-lead report does.

## What changes, file by file

**New: `packages/studies/src/studies/dynamical_members.py`.** `dynamical_members(*, raw, cell_by_site,
domain, ensemble_size)` reshapes a Dynamical.org store frame (`init_time`, `lead_time`, `lat_index`,
`lon_index`, `ensemble_member` if present, the value columns) to the extract shape. Steps: convert
every float column's `NaN` to null, map each site to its cell, compute speed and direction from `u`
and `v` (`direction = arctan2(-u, -v) % 360`, the from-direction), convert `lead_time` to whole
hours, drop lead 0's radiation, set `ensemble_member = 0` where the store has none, and return
`site`, `init_time` (UTC), `ensemble_member`, `lead_hours`, and the six value columns.

**New: `packages/studies/src/studies/stamps.py`.** `six_hourly_stamps(*, hourly, radiation_columns,
instantaneous_columns)` implements the IFS 0.25° sampling above.

**New: `packages/studies/src/studies/model_versions.py`.** `AIFS_SINGLE_VERSIONS` and
`AIFS_ENS_VERSIONS` (dated tables with the excluded window), `version_of(init_time, table)` (returns
the version or `None` inside the excluded window or before the first version), and
`check_month_eras_match_versions(frame, first_months, table)`, which raises unless every row's
init-time version equals the version its month-cut era stands for and no row falls in the excluded
window.

**Edit: `studies/beam_diffuse_split/ens_forecast_horizons.py`.** `band_steps` passes
`period_means=frozenset({"ghi_w_m2"}) if domain == "solar" else frozenset()`.

**Edit: `studies/nwp_forecast_comparison/build_forecast_inputs.py`.** Rename `_ens_member_arms` to
`ens_member_arms` with a `six_hourly: bool = False` argument passed to `band_steps`, and rename
`_gefs_cell_selection` to `nearest_cell_by_site`. Update the module docstring, and the callers on
lines 387, 703, 846, and 911.

**New: `studies/nwp_forecast_comparison/build_aifs_inputs.py`** (build, above),
**`studies/nwp_forecast_comparison/verify_aifs_steps.py`** (read-only checks, below), and
**`studies/nwp_forecast_comparison/fit_aifs.py`** (fits and report, above). `fit_aifs.py` imports
`METRIC`, `SETTINGS`, `TARGET`, `arm_columns`, `assert_equal_rows`, `coverage_table`, `difference`,
`fingerprint`, `leaderboard`, `predictions_from_losses`, and `rows` from `nwp_forecast_comparison`,
and `out_of_fold_losses`, `cut_eras`, and `search_fold_offsets` from `studies.cross_validation`. It
does not copy `fit_extra_leads.py`'s report code; the shared table formatters (`interval_text`,
`error_text`, `contrast_line`, `by_hour_modulo`) move to `nwp_forecast_comparison.py` as public
functions only if a review of the diff finds two copies would be needed, which the diff review
decides.

**Edit: `studies/nwp_forecast_comparison/nwp_forecast_charts.py`.** One new function draws, per
technology, AIFS Single, AIFS ENS, and the controls' absolute errors on their own rows (day 1, both
row sets, with intervals), with the row and month counts in the axis title so the marks cannot be
read against Figure 1. Loaded with the `dataviz` skill, and it refuses any site label that is not
an anonymised label, like the other charts. Output `docs/studies/assets/nwp_forecast_{solar,wind}_aifs.svg`,
optimised with `svgo`.

**Edit: `studies/nwp_forecast_comparison/README.md`.** Add the three scripts, the output files, and
what each holds (the `study` skill's reproducibility rule).

**Edit: `docs/studies/nwp-forecasts-at-matched-leads.md`** (see "Docs to update").

**Not touched, and reported:** `studies/README.md` line 65 still says "In progress — no fit has run
yet" for this study. That row is stale for reasons unrelated to AIFS, so it belongs in its own pull
request off `main`.

## The verification script

`verify_aifs_steps.py` reads only and writes `verification/` in the new folder:

- **Radiation window.** For each candidate window end offset from -6 to +6 h, the mean absolute
  difference between AIFS Single's radiation at lead `L` (day 1 and day 2 bands, shortwave) and the
  mean of ERA5's hourly `ghi_era5` over the six hours ending at `L + offset` at the same sites. The
  minimum must fall at offset 0, or the script fails. This is the same shape as V3 in
  `verify_previous_runs_leads.py` and `verify_extra_leads.py`'s GEFS check.
- **Units.** AIFS temperature in °C (a mean between -30 and 50), wind in m/s (10 m speed mean under
  15, 100 m above 10 m), radiation between 0 and 1,100 W/m².
- **The `NaN` boundary.** The share of `NaN` shortwave and 100 m wind per month, showing the store's
  first non-`NaN` run is 2025-02-24 06 UTC, and no `NaN` after it beyond lead 0.
- **Runs complete.** Every 00 UTC run from 2025-02-26 (Single) and 2025-07-02 (ENS) to the last row's
  date holds 61 leads, all 9 cells, and, for ENS, 51 members.
- **Grid orientation.** For each site the chosen cell's 2 m temperature correlates with the site's
  shared-row `temp_c` (solar sites) better than any other of the 9 cells' does, which catches a
  swapped `lat_index` and `lon_index`. The output prints a correlation and no cell id.

## Design-philosophy check

**This is R&D code, and it fails fast.** No asset, serving path, or degradation rule changes, so
`inherent-stability.md` does not apply beyond its own carve-out: "R&D is the opposite: the CV,
training and metrics assets fail fast". The build raises on a missing run, a missing value in an
arm's column, an uncovered month, an era mismatch, and an existing output file. None of the
hypotheses `H1` to `T5.1` is delivered or affected. The change trades away no principle in
`design-principles.md`: the shared helpers gain a public name and a defaulted argument, and each new
`packages/studies` module has one job and tests.

## Tests

The three new `packages/studies` modules carry tests, and each test names the assertion that fails on
a wrong implementation. The scripts have no unit tests, and their check is their printed report
(`study` skill). The mutation pass runs on the three modules.

- **`test_dynamical_members.py`**
    - A raw frame with `NaN` shortwave in one run and real values in another: the extract holds
      null, not `NaN`, for the first run (fails if the `NaN` conversion is dropped, since `is_nan()`
      would then be true and `is_null()` false).
    - `u = 0, v = -5` (wind from the north) gives direction 0 and speed 5; `u = -5, v = 0` gives 90
      (fails on `arctan2(u, v)` and on a swapped sign).
    - Two sites mapped to different cells each get their own cell's values (fails if the cell join
      ignored the site).
    - A frame with no `ensemble_member` column gives member 0 on every row.
    - Lead 0's radiation is dropped and its temperature and wind are kept.
- **`test_stamps.py`**
    - Six hourly values 1 to 6 ending at the stamp give a radiation stamp of 3.5 (fails on a window
      that is off by one hour in either direction).
    - A stamp whose six-hour window is missing one hour is dropped, not averaged over five.
    - An instantaneous field's stamp equals its hourly value at exactly 00, 06, 12, and 18 UTC and
      no other hour.
    - Two sites with unequal values do not mix (fails if the window crossed sites).
- **`test_model_versions.py`**
    - `version_of` at `2025-08-01 18:00` returns `None` (excluded window), at `2025-08-02 00:00`
      returns v1.0, at `2025-08-27 06:00` returns v1.1, and at `2026-05-12 06:00` returns v2 (fails on
      an off-by-one at each edge, and on 00 UTC used for the v2 edge).
    - `check_month_eras_match_versions` raises for a frame with a run initialised on `2025-08-30`
      assigned to era 0, raises for a row in the excluded window, and passes on a correct frame.

**Read-only comparisons the reports print, in place of tests for the scripts:**

- The `band_steps` fix leaves every day-1 6-hourly wind and solar column unchanged (maximum
  absolute difference 0), and the day-0 wind band keeps its lead-0 stamp.
- `ens_member_arms(six_hourly=False)` reproduces the published `ens_mean_day1` and `ens_control_day1`
  columns (maximum absolute difference 0).
- Every AIFS extract holds no `NaN`.
- Every contrast passes `assert_equal_rows`.

## Docs to update

**`docs/studies/nwp-forecasts-at-matched-leads.md`.** Present-tense, no history.

- *Title block and first summary paragraph:* add one sentence saying the page also reads ECMWF's
  AIFS Single and AIFS ENS, on rows of their own, and that those results are exploratory. The
  headline sentence is unchanged, because the AIFS comparison cannot alter a published verdict.
- *Key findings:* one bullet, labelled exploratory and post hoc, that states each AIFS number with
  its interval, rows, and months and says AIFS numbers do not share an axis with Figures 1 and 2.
  No bullet says AIFS improves.
- *Data and methods:* a new subsection, "How AIFS is read", after "How the leads are matched". It
  covers the 6-hourly steps and how each is mapped to hourly targets, the like-for-like control, the
  own-row-set rule, and the version eras. The existing sentence "Scoring is on the same hours for
  every arm" gains "except the AIFS arms, which have their own row set". The "Folds" paragraph gains
  the AIFS era edges in two sentences.
- *Results:* a new section "AIFS on its own rows (exploratory)", after "Among the other products,
  IFS 0.25° comes closest to ENS", with the two new charts and the tables' figures, each number
  followed by a `<!-- report: ... -->` marker naming its report section.
- *Discussion: what to use:* one paragraph on what the AIFS numbers do and do not change. The scoped
  answer to "is AIFS improving faster than conventional weather models" is a marked placeholder
  (`<!-- AIFS-IMPROVING-PLACEHOLDER: awaiting the coordinator's literature summary -->`), and the
  page does not merge with the marker still in it. The paragraph around it states that this study's
  data cannot measure a rate of improvement: it holds one version at a time.
- *Limitations:* the 3 by 3 crop and three spatial representations; the 6-hourly steps; the version
  blend; the IFS Cycle 50r1 confound; the unmeasured IFS 0.25° run cycle; the unmeasured AIFS
  publication time; the short v2 era; the 6,220 wind rows at 10 m that are not scored; and the GPU
  noise floor.
- *Scope, Data and code availability, and Reproducing this page:* the AIFS inputs, their start dates
  (2024-04-01 for Single, 2025-07-02 for ENS), the new commands in order, the new folder's write-once
  rule, and the commit hash of the last code change (the page's last-change line is updated at the
  end, as the existing hash is).

**`docs/background/weather-products-survey.md`.** The two rows for AIFS Single and AIFS-ENS say
"Not scored". They change to "Scored, exploratory, on the matched-lead page" with a link to the
published page's new section.

**`studies/nwp_forecast_comparison/README.md`** (above). **`docs/studies/index.md`:** the
matched-lead entry, if it lists what the page compares, gains AIFS.

No ship-time roadmap triage applies: this issue completes no roadmap item.

## Verification commands

Run every CI step locally, including the two the `implement-issue` set omits (`pydoclint`, the docs
link checker):

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest packages/studies
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md studies/nwp_forecast_comparison/README.md
uv run --isolated --no-project --with pydoclint==0.9.1 pydoclint --quiet --style=google .
uv run python scripts/lint/check_docs_links.py
uv run mkdocs build --strict     # then read the rendered new sections
```

Study runs, in order, one fit job at a time, each a write-once output:

```bash
uv run python studies/nwp_forecast_comparison/verify_aifs_steps.py --published-dir data/studies/nwp_forecast_comparison --output-dir data/studies/nwp_forecast_comparison_aifs
uv run python studies/nwp_forecast_comparison/build_aifs_inputs.py --published-dir data/studies/nwp_forecast_comparison --output-dir data/studies/nwp_forecast_comparison_aifs
uv run python studies/nwp_forecast_comparison/fit_aifs.py --check --published-dir data/studies/nwp_forecast_comparison --output-dir data/studies/nwp_forecast_comparison_aifs
uv run python studies/nwp_forecast_comparison/fit_aifs.py --published-dir data/studies/nwp_forecast_comparison --output-dir data/studies/nwp_forecast_comparison_aifs --workers 1
uv run python studies/nwp_forecast_comparison/nwp_forecast_charts.py --input-dir data/studies/nwp_forecast_comparison --extra-dir data/studies/nwp_forecast_comparison_leads --aifs-dir data/studies/nwp_forecast_comparison_aifs --output-dir docs/studies/assets
```

Only one agent may run these against the shared `data/` folder at a time (`worktrees-share-one-data-folder`).
Every published number is checked against the printed `report.md`, and the page's marker comments name
the report section. The published folder's checksums (`sha256sum data/studies/nwp_forecast_comparison/*.parquet`)
are recorded before the run and compared after it, to prove nothing was written there.

## Review schedule

Both plan reviews (simplicity, then correctness and testability), then the study's two Opus
scientific-validity reviews after the first fit, then the diff review and the mutation pass on the
three `packages/studies` modules, then a `prose-review` of the page. Per the `study` skill, the
maintainer's authority is needed to merge.

## Risks and open questions

1. **Which bands: days 0, 1, and 2, or more?** Days 0 to 2 give the page's bracket for IFS 0.25°
   day 1 and day 2. Adding day 3 adds about 25% more fits and no contrast the page can use, so I
   recommend stopping at day 2. *Recommendation: 0, 1, 2.*
2. **Is the `band_steps` fix acceptable in this pull request?** It changes a shared function for
   wind at day 0 in 6-hourly mode only, and the report proves the published day-1 arm unchanged. The
   alternative is to drop day 0 for the wind control, which loses the bracket's lower side for wind.
   *Recommendation: make the fix, with the proof.*
3. **The 6,220 wind rows at 10 m only (2024-12 to 2025-02).** Scoring them needs a 10 m-only column
   set for every arm (10 m speed, sine, and cosine, plus the calendar columns), a fourth era for
   runs of unknown version, and its own folds, for 3 months of one technology. *Recommendation: do
   not score them; the page states they are excluded and why.*
4. **Is the IFS 0.25° 6-hourly control worth its construction?** It gives the control a stamp at
   each of 4 valid times a day, but its stamps mix Open-Meteo runs, and its lead at those stamps
   rests on the unmeasured run cycle. The ENS 6-hourly controls are exact within a run. Dropping
   `ifs025_6h_*` removes one function (`studies.stamps`), one test module, and 4 arms. *Recommendation:
   keep it for the first run, since the design names IFS 0.25° as a control, and drop it if the
   simplicity review finds it adds nothing the ENS controls do not.*
5. **Whole-month era edges versus mid-month cuts.** Whole months lose 3 of 19 valid months and need
   no new machinery. A mid-month cut keeps them and needs a new era cutter, `search_fold_offsets`
   variant, and tests in `packages/studies`. *Recommendation: whole months.*
6. **The 2026-05-12 boundary time.** The design says 00 UTC and the READMEs and roadmap say 06 UTC.
   The rows do not depend on it, because 2026-05 is dropped whole. The table in code uses 06 UTC and
   cites `docs/roadmap/data-sources.md`. *Question: is the design's 00 UTC from a different source
   the maintainer wants cited?*
7. **The coordinator's literature summary.** The page's answer to "is AIFS improving" needs it. The
   page cannot merge with the placeholder in it. *Question: should the pull request wait for it, or
   ship without the sentence and add it in a follow-up?*
8. **A 6-hour and 3-hour asymmetry remains.** ENS at full resolution, the published reference, keeps
   3-hourly steps. The report shows AIFS against both the full-resolution and the 6-hourly ENS, and
   the page's headline comparison uses the 6-hourly one. A reader could read the full-resolution row
   as the fair one. The page's lead states which comparison is which.
9. **Spatial representation differs.** AIFS reads the nearest 0.25° cell, ENS an H3 resolution-5
   area mean, and IFS 0.25° a point. Averaging AIFS's 9 cells would mimic ENS but is not what a live
   service reading the product point-wise does. *Recommendation: nearest cell, and say so.*
10. **Multiplicity.** About 60 exploratory intervals print. The page quotes only those the text
    discusses, labels each exploratory and post hoc, and does not build a claim on a single interval
    near zero.
11. **Device.** The wind device noise floor in the extra-lead run (-0.087 to -0.040 points) is larger
    than some contrasts this study can resolve. Every contrast uses two GPU fits, and the floor
    appears beside them.
12. **Sensitivity setting.** Fitted only for near-line results, after the primary run. The near-line
    rule is applied mechanically by the script, not by eye.
13. **Dynamical.org's access ends on 2026-09-30.** The AIFS inputs are on disk, and the plan reads
    nothing that cannot be re-run from disk. `README.md` in the new folder records that.
14. **Stale row in `studies/README.md`** (not part of this issue): see "What changes, file by file".
