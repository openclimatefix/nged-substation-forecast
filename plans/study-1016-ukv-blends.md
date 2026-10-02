# Plan: does UKV-CEDA lower the error of the ECMWF ENS mean at lead days 1 to 4? (issue #1016)

**The problem.** The [blends page](../docs/studies/forecasts/blends-with-ens.md) tests UKV (the Met
Office's 2 km UK weather model) only at day 1, from Open-Meteo's archive of live UKV, which holds
one previous day. CEDA's archive (the Centre for Environmental Data Analysis) holds whole UKV runs,
and the `UKV-CEDA-T120` store holds the 03 and 15 UTC runs out to 120 hours. No study yet says
whether those runs lower the power-forecast error of the ECMWF ENS mean at days 2 to 4, or whether
the day-1 gain survives a second source. The maintainer has lifted the hold on #1016 and authorised
the study to run autonomously.

**The plan.** For solar and for wind separately, and at lead days 1 to 4, fit an XGBoost model given
the ENS mean's weather plus UKV-CEDA's weather, and compare its out-of-fold error with an XGBoost
model given the ENS mean's weather alone (padded to the blend's column count) and with a
shuffled-UKV control. The rows, eras, and folds are the matched-lead page's, and the control,
settings, and reading rule are the blends page's: month-block folds, paired month-resampled
intervals, both hyperparameter settings, GPU fits. Day 5
is not fitted: UKV-CEDA's 120-hour reach cannot supply day 5 under the matched-lead rule.

## Verdict, size, and departures

**Verdict: worth doing.** The question is open, the inputs are on disk or arriving, and the
machinery exists.

**Size: complex, by the five triggers in `plan-issue` step 3.**

- **What gets stored:** fires. A new write-once folder `data/studies/ukv_ceda_blends/` and a
  published page (a page counts as stored).
- **Production serving path:** does not fire. The diff touches `studies/`, `packages/studies/`, and
  `docs/studies/` only.
- **Degradation rule:** does not fire. The code is R&D and fails fast.
- **More than one defensible design:** fires. The lead definition, the wind columns, the row set,
  and day 5 each admit more than one design.
- **Callers nameable without searching:** does not fire. The changes to shared code are one
  parameter, one dictionary entry, and one column-name branch, each with named callers.

**Reviews.** The simplicity review has run, and this revision applies it. Next the correctness plan
review, then two diff reviews (correctness and cut-down, then mutation, which runs only if
`packages/studies/` still changes), the study skill's two Opus scientific-validity reviews, and a
fresh review of each script before its first run. The page then gets the `prose-review` sweep and
the builder, user, and evidence persona reviews. All reviewers are Opus.

**Departures from the issue body.**

- **Days 1 to 4, not 1 to 5.** At day 5 the ENS run, read at 09:00 UTC, leads the valid hour by 120
  to 143 hours. A UKV run readable at that time reaches only the first 4 valid hours of day 5.
- **The hold is lifted.** The issue's "Status: deferred until probabilistic forecasts" no longer
  applies, so the metric is the blends page's mean absolute error in percent of capacity.
- **ENS alone is padded to the blend's column count**, because the study skill requires equal
  column counts and the blend has two or four more columns than ENS alone.

## The question and the arms

**An arm is one XGBoost model's input set.** Per technology and per lead day `N` in 1 to 4, the
planned arms are below.

| Arm | Solar columns | Wind columns |
|---|---|---|
| ENS reference (`blend_ukv_ceda_dayN_pad`) | calendar 5 + ENS 2 + 2 exact copies = 9 | calendar 3 + ENS 4 + 4 copies = 11 |
| Blend (`blend_ukv_ceda_dayN`) | 5 + ENS 2 + UKV-CEDA 2 = 9 | 3 + ENS 4 + UKV-CEDA 4 = 11 |
| Control (`blend_ukv_ceda_dayN_control`) | 5 + ENS 2 + 2 shuffled UKV-CEDA = 9 | 3 + 4 + 4 shuffled = 11 |
| Second control (`..._control_b`, seed 1000) | as the control | as the control |

- **Calendar columns:** `hour_of_day`, `day_of_year`, `era_code`, plus for solar
  `solar_elevation_deg` and `solar_azimuth_deg` (`nwp_forecast_comparison.arm_columns`).
- **ENS columns:** solar `ens_mean_dayN_ghi` and `ens_mean_dayN_temp`; wind
  `ens_mean_dayN_speed_100m`,
  `_sin_100m`, `_cos_100m`, and `_speed_10m`.
- **UKV-CEDA solar columns:** `ukv_ceda_dayN_ghi` and `ukv_ceda_dayN_temp`.
- **UKV-CEDA wind columns:** `ukv_ceda_dayN_speed_10m`, `_sin_10m`, `_cos_10m`, and
  `_speed_925hpa`. These are UKV-CEDA's native 10 m and 925 hPa winds, and the page says they are
  not the 100 m wind of ENS. The count matches ENS's four columns.
- **Control:** the prefix `ukv_ceda_dayN_permuted`, shuffled within site, year-month, and hour of
  day with `studies.blending.climatology_permutation`, the shuffle the blends page uses. Two
  shuffle seeds are always fitted, 0 and 1000, as the arms `..._control` and `..._control_b`, at
  both settings. P2 holds only if both seeds' upper bounds are below zero, as
  `test_the_day_14_rule_needs_two_shuffle_seeds_to_agree` does for day 14.
  The shuffle is applied after the row cut, as `aifs_rows` does, so no null or out-of-window value
  enters the control. The existing shuffle moves wind speed and direction independently, which
  matches the blends page and is out of scope.
- **Padding:** the copies carry the prefix `ens_mean_dayN_copy`, so they hold no information the ENS
  columns lack. With `colsample_bytree` 1 the study skill expects the padded arm to score exactly as
  the unpadded one. A one-off `--check` fits the unpadded and padded ENS arms once on one stage and
  prints whether their per-row losses are identical. If they are, the page says so. If they differ,
  the padded arm stays the only ENS reference and the unpadded arm is not reported.

**Reuse of `fit_aifs.py`.** The arms register as blends of a new product, `ukv_ceda`:

- One entry, `"ukv_ceda": "ukv_ceda_day{day}"`, in `fit_aifs.BLEND_AIFS_PREFIXES`. The regex
  `_BLEND_ARM` is built from the dictionary's keys, so `ukv` and `ukv_ceda` both match under
  `fullmatch`. Then `arm_prefixes`, `expected_column_count`, `source_columns`, `control_shuffles`,
  `add_shuffled_columns`, and `fit_jobs` (which raises when an arm's column count differs from its
  kind's) work unchanged.
- Two new roles, `_pad` and `_control_b`, in `BlendRoleType`, in `_BLEND_ARM` (which becomes
  `(_control|_control_b|_mirror|_pad)`), and in `arm_prefixes`. `_pad` returns ENS's prefix and the
  copy prefix `ens_mean_dayN_copy`, and `_control_b` returns the second seed's shuffled prefix.
  `control_shuffles` builds the `""` variant for `_control` and the `_b` variant for `_control_b`.
  The fit script adds the copy columns before `source_columns` or `check_no_missing` run.
- One branch in `nwp_forecast_comparison._wind_weather_fields`: a prefix that starts with
  `ukv_ceda` returns the four native wind column names above. The code checked this: the function
  hard-codes the suffixes `_speed_100m`, `_sin_100m`, `_cos_100m`, and `_speed_10m`, and a native
  925 hPa speed must not sit under a `_speed_100m` name. The branch returns
  `(speed_10m, sin_10m, cos_10m, speed_925hpa)` in that order, because `add_shuffled_columns`
  groups wind fields by position (`[(f0,), (f1, f2), (f3,)]`). The arm-column test asserts the
  order. Every existing arm's columns stay unchanged, which a one-off check proves: for every saved
  `*_losses.json` stamp under `data/studies/nwp_forecast_comparison_*`, its `columns` field must
  equal the new `arm_features` output for that arm. The check needs no fit.
  (`fit_product_blends.py --dry-run` cannot show this, because `SAME_BUILD_KEYS` excludes
  `columns`.)
- One test that `arm_features` lists the expected columns, in order, for the four roles at both
  technologies, and that the existing arms' names still resolve. It also asserts
  `arm_columns(domain="wind", prefixes=(p,))` is unchanged for every prefix in `PLANNED_PREFIXES`,
  for `ukv_day1`, and for a `_permuted` prefix.

**ENS is the Dynamical.org 51-member mean of the 00 UTC run**, averaged over the cells overlapping
each generator's H3 resolution-5 hexagon. Day `N` reads the run `N` days before the valid hour's own
day, at lead `24N + h`, where `h` is the hour of day. ENS days 1 to 3 come from the published inputs
(`nwp_forecast_comparison/<domain>_forecast_inputs.parquet`) and day 4 from
`nwp_forecast_comparison_day4_shared/<domain>_extra_lead_inputs.parquet`. The ENS history starts on
2024-04-01. The study reads it from 2024-12-01, as the matched-lead page does.

## Lead matching

**There is one UKV-CEDA lead: the 03 UTC run of the ENS run's own day, which a 09:00 UTC service
could read.** For a valid hour `h` on day `D`, ENS day `N` reads the 00 UTC run of day `D-N`, at
lead
`24N + h`. The UKV-CEDA arm reads the 03 UTC run of day `D-N`, at lead `24N + h - 3`.
[The matched-lead page](../docs/studies/forecasts/matched-lead.md) assumes a UKV run is readable
about 4 hours after it starts, so the 03 UTC run is readable by about 07:00 UTC, before 09:00 UTC.
The assumption was not measured for CEDA, and the page says so. Lead `24N + h - 3` runs from 21 to
117 hours over days 1 to 4 (a solar hour labelled 00:00 at day 4 reads `24N + 21`), inside the
120-hour reach. The code reads the T120 store only. The 15 UTC
run, the 6-hourly stores, and `UKV-CEDA-part3` are not read.

**UKV-CEDA's lead is 3 hours fresher than ENS's at every hour, and that favours the blend.** The 3
hours are 12% to 12.5% of ENS's shortest day-1 lead (24 hours for wind, 25 for solar) and 3% of
its day-4 lead (3 of 96), so the shift matters mostly at
day 1. The page states this beside every result and in its limitations, and reports the lead gap
without bracketing it.

**Day 5 is not coverable.** The lead at day 5 is `120 + h - 3`, which is at most 120 only for `h` of
0 to 3. Those hours are 4 of 24 and have no solar daylight. No other run readable at 09:00 UTC (the
15 UTC run of the day before leads by 129 hours or more) reaches further. The build's log prints the
day-5 share.

**The run and lead come from `studies.ifs_single_runs`, with a `run_hour` parameter.**
`served_init_time` and `served_lead_hours` already return the run `N` days before the hour's own
day, with the solar label convention (an instant one hour before the label) handled. A parameter
`run_hour: int = 0` adds the run's start hour to the init time and takes it off the lead. The two
functions' existing callers pass nothing and are unchanged. Two tests cover `run_hour=3` at `h` of
0,
2, 3, and 23.

**Steps beyond 48 hours are 3-hourly.** The store holds hourly steps to lead 48, then 51 and 54,
then 57 to 120 in steps of 3. Days 2 to 4 need temporal upsampling. The plan does not reuse
`ens_forecast_horizons.upsampled_fields`: the code checked it, and it treats every radiation step as
a mean over the step (step midpoints at `x - widths / 2`, and a rescale to step means). UKV-CEDA's
`shortwave_down` is an instantaneous snapshot, so that function would place each 3-hourly value 1.5
hours early. Its wind branch also expects `speed_100m`, `direction_100m`, and an ENS `Steps` object.
The build calls the tested primitives in `studies.resample` directly:

- **Radiation:** `clear_sky_index_resample`, under five conditions:
    1. Clear sky is instantaneous: `studies.baselines.haurwitz_w_m2(zenith(...))` at each snapshot
       instant and each target instant. `hourly_clear_sky` is never used, because it is a mean over
       the hour ending at its label and would sit 30 minutes early.
    2. The positions are the instants themselves: the snapshot instants as `step_midpoints` and the
       instants at leads `L-1` and `L` as `target_midpoints`, not ENS's `x - widths/2` and
       `targets - 0.5`.
    3. `morning` comes from the UTC hour of the instant, not from `lead mod 24`, which equals UTC
       only for a 00 UTC run.
    4. Only leads the store lacks are resampled. Native leads (up to 48, then 51 and 54, and every
       3-hourly step) pass through unchanged, because below the 50 W/m2 floor the function returns
       a neighbour's index times the clear sky, not the native value.
    5. The two anchors bracketing each target are checked finite before resampling, because a NaN
       gives a NaN index that `hold_flat_outside_daylight` fills silently from a neighbour. The
       gap rule's "needed steps" are those anchors.
- **Temperature:** `interpolate_linear`.
- **Wind:** `wind_components`, `interpolate_linear`, and `wind_polar`, for the 10 m vector and the
  925 hPa vector.

**Timestamp conventions.**

- **Radiation is an instantaneous snapshot in UKV-CEDA** (`shortwave_down`, GRIB template 0). The
  column is the mean of the snapshots at leads `L-1` and `L`, where `L` is the lead of the hour's
  label. The code is `studies.hourly_means.hourly_from_snapshots` with
  `slot_offsets_minutes=(-60, 0)`, which `build_forecast_inputs.py` already calls for UKV. The
  function drops an hour missing either snapshot, which is the convention wanted here. The key
  column holds the site and the run's init time.
- **Solar temperature** is `temperature_1p5m` in degrees Celsius, the mean of the same two
  snapshots.
- **Wind is instantaneous at the label.** The target's wind power is the shared target, centred on
  the label, so no UKV-specific shift is applied.
- **Pre-fit alignment check:** the radiation peak-offset check (`studies.timestamp_checks`) on the
  rebuilt column is a gate at day 1: `timestamp_checks.check_hour_ending` must find an offset within
  15 minutes
  of -30 (the matched-lead page found about -43 minutes for the UKV rebuild, `matched-lead.md`).
  Days 2 to 4 are printed, and the page says the check cannot test the lead there, because the
  clear-sky multiplication sets the diurnal shape whatever run the anchor came from. A wind
  power-hour offset scan is printed. Neither changes the shared target.

## Data and the row set

**Stores.** The arms read `data/studies/weather/UKV-CEDA-T120/store` only.

**Row base, window, and eras.** The study uses the matched-lead page's window and eras
(`nwp_forecast_comparison`), not the blends page's, whose `single` rows start in March 2025 under
AIFS run eras. The matched-lead window is the only one with ENS days 1 to 4 inputs on disk
(day 4 does not exist before 2024-12), so the plan chooses it. The row base is
`nwp_forecast_comparison.candidate_rows()` (`ROW_SET_START`, `DROPPED_MONTHS`), filtered by this
study's three requirements below, then `assign_folds_with_eras`. It is not `rows()`, which also
requires Open-Meteo UKV day 1, ICON-EU, IFS 0.25 degrees, and GEFS columns that this question does
not need. The build prints how many rows `rows()` would have kept, for comparison.

**Window.** Valid hours from 2024-12-01 to the last power hour (2026-09-10): the first whole month
after ECMWF's IFS Cycle 49r1, with 2026-01 dropped for the Met Office's UKV
upgrade on 21 January 2026. That is 21 months. A longer window from 2024-04 would add 8 months but
needs a new ENS era boundary, rebuilt ENS rows at days 1 to 3, and a day-4 build that does not exist
before 2024-12, so the plan keeps the shared window. Day 4 at valid hour 2024-12-01 reads the 03 UTC
run of 2024-11-27, so the store must reach
2024-11-27 03Z, expected about 5 or 6 October 2026. The build reads about 650 runs (03 UTC only).
The fetcher also downloads 15 UTC runs, which the study does not read.

**Rows.** The frame for one technology and one lead day holds each `(site, time)` of the published
inputs inside the window with a non-null target, non-null ENS columns at that day, and non-null
UKV-CEDA columns at that day. Every arm of that day is trained and scored on that frame. Rows are
dropped for missing coverage of an input, never for an input's value, and the target decides the
hours (the shared cleaning is already in the published `power_mw`). Each day has its own frame, so
contrasts are made within a day only.

**Gap rule.** A T120 slot has status complete, partial, or missing (`status` array). A run counts as
covering an hour only if its status is complete, or partial with every needed variable finite at the
needed steps at the site's cell. A missing or partial-and-short run removes the hour from every arm
of that day, including its training rows. The build prints, per technology and day, the rows lost to
each cause: target absent, ENS absent, UKV-CEDA run missing, UKV-CEDA run partial, run not listed by
CEDA, lead beyond the
store. Every row's run `init_time` is asserted equal to `served_init_time(..., run_hour=3)` for
day N, as `fit_aifs.check_runs` does for AIFS.

**Eras and folds.** The shared design: `nwp_forecast_comparison.assign_folds_with_eras`, with era
starts 2025-10 and 2026-02, fold offsets `{0: 0, 1: 0, 2: 3}`, five folds of whole months within
each era, and `era_code` as a column of every arm. The part-month 2026-01 is dropped. A
`coverage_table` call after each frame is cut raises if any `(site, fold, calendar month)` is
uncovered, and `studies.cross_validation.search_fold_offsets` picks another rotation if it does.
Two source changes inside the window are not era boundaries of their own and are listed under
risks: ENS's IFS cycle 50r1 (12 May 2026) and any step in UKV-CEDA the step check finds.

## The planned contrasts

**Two contrasts are planned, written before any result, each per technology and lead day at both
hyperparameter settings.** Differences are the first arm's error minus the second's, in percentage
points of capacity, so a negative difference means the first arm is better.

- **P1 (planned): blend minus padded ENS**, days 1 to 4. The question the issue asks.
- **P2 (planned): blend minus control.** The information test: the control carries the blend's
  column count and UKV-CEDA's marginal distribution without its timing.

**Reading rule, fixed now.** The blend lowers the error at a technology and day only if the upper
bound of P1 and P2 is below zero at both settings. The page reports the Bonferroni-adjusted level
over the 8 P1 intervals per setting (99.375%), and says P2 is not adjusted.

**Exploratory, and the page labels it so.**

- E4: P1 by era, with "before the upgrade" as eras 0 and 1 pooled and "after" as era 2. Era 0 is
  2024-12 to 2025-09 (10 months), era 1 is 2025-10 to 2025-12 (3 months, under
  `MIN_MONTHS_FOR_INTERVAL`, so no interval alone), and era 2 is 2026-02 onwards (8 months).
- E5: P1 per generator, with the anonymised labels A to F and W1 to W3.

Both come from the saved losses with no new fit.

**A null is read through its interval bound.** The page does not run a planted-effect control. Every
null P1 is reported as "an effect as large as X points is not excluded" or "an effect larger than X
points is excluded", where X is the interval's bound towards a gain. The 21 months and the paired
month resampling give the bound its width.

## Folds, bootstrap, seeds, settings, metric, device

- **Metric:** `absolute_error_capped_fraction_of_capacity`, printed as percentage points. Each row's
  error is divided by its own generator's effective capacity before any mean or difference.
- **Fitting seeds:** the three of `studies.cross_validation.SEEDS`.
- **Intervals:** `studies.bootstrap.bootstrap_difference` and `bootstrap_difference_at_level`, 2,000
  resamples of whole calendar months, paired across arms, with one fitting seed per resample. A row
  set of fewer than 6 months (`MIN_MONTHS_FOR_INTERVAL`) gets no interval.
- **Settings:** `PRIMARY_HYPER_PARAMETERS` and `SENSITIVITY_HYPER_PARAMETERS`, both on every planned
  arm. A planned verdict stands only if both settings give it. A result with a bound within 20% of
  the interval's width from zero is called near the line, and the page says where the sensitivity
  setting changes sign or significance.
- **XGBoost:** `colsample_bytree` 1, row `subsample` 0.8 as the settings set it, and
  `device="cuda"` (an RTX A6000 is present). One device inside each planned contrast. One arm (the
  blend at day 1, wind site W1) is refitted on the CPU to report the GPU-CPU difference as a noise
  floor.
- **Load rule:** run `uptime` and `nvidia-smi` before each fit. Start only below a load average of
  about 24, with `workers` at most 2 (`THREADS_PER_FIT` is fixed at 4). The CPU noise-floor refit
  calls `out_of_fold_losses(device="cpu")` directly, because `fit_jobs` uses the module constant
  `DEVICE`.

## Scripts and files

**A new directory, `studies/ukv_ceda_blends/`, holds the study.** The row set, the output folder,
and
the inputs are the study's own, and the 400-line README of `studies/nwp_forecast_comparison/` is
already too long to extend. The scripts import `fit_aifs` and `nwp_forecast_comparison` by adding
the
sibling directory to `sys.path`, as `build_forecast_inputs.py` does for `beam_diffuse_split`. One
agent writes each published script, and a fresh Opus reviewer reads it before its first run.

| Script | What it does |
|---|---|
| `build_ukv_ceda_inputs.py` | Writes `<domain>_ukv_ceda_inputs.parquet` on the published `(site, time)` keys: every UKV-CEDA column at every day and the run's `init_time`. Reads each site's nearest 2 km cell (private roster, labels only in output). Refuses to build unless the fetcher has passed the window (the oldest archived slot is at or before 2024-11-27 03Z) and every status-0 slot inside the window ("not yet attempted") has been retried once, by re-running the fetch with `--start` and `--end` over those days. A slot still unlisted after the retry (CEDA lists no runs for 2026-06-26 to 2026-07-02) counts as missing under its own cause, "run not listed by CEDA", and does not block the build. Writes the Icechunk snapshot ID it read into the stamp and the README, so the inputs can be rebuilt exactly. Prints the rows-lost-by-cause table and the day-5 share into its log and the folder's README. `--dry-run` times one month and prints the same table on the slots that exist. |
| `verify_ukv_ceda_inputs.py` | Reads only. Recomputes a sample of built values from the store in plain Python, stratified to hit leads up to 48, the step from 48 to 51, leads of 57 and above, and wind hours 0 to 2. Runs the alignment checks and gates each day's skill: the correlation of `ukv_ceda_dayN_ghi` with `ghi_cams`, and of `_speed_10m` with ERA5's 10 m speed, printed beside Open-Meteo UKV day 1's. Day 1 must come close to Open-Meteo UKV's own correlation, and the correlation must fall as the day rises, or the check fails. Also a monthly step check (each month's mean UKV-CEDA ghi and 10 m speed divided by ENS's, flagging a month-to-month change of 15% or more, as `check_input_steps.py` does). Exits non-zero on a failed check. |
| `fit_ukv_ceda_blends.py` | The fit and the report. Imports `fit_aifs` (`fit_jobs`, `add_shuffled_columns`, `control_shuffles`, `build_stamp`, `time_two_fits`, `blend_arm_name`), `nwp_forecast_comparison` (`difference`, `coverage_table`), `studies.cross_validation`, `studies.bootstrap` (`combine_setting_verdicts`, as `fit_product_blends.py` does), and `studies.guards`. |
| `ukv_ceda_blends_charts.py` | Draws every figure from saved losses and predictions, with no refit. |

**Changes in shared code.** One parameter on `studies.ifs_single_runs` (`run_hour`; its module
docstrings, which say "00 UTC run", are updated), one dictionary entry and two roles in
`fit_aifs.py`, and one branch in `nwp_forecast_comparison.py`. No new module
in `packages/studies/`. Tests: the `run_hour` tests in `test_ifs_single_runs.py` (both technologies,
hours 0, 2, 3, and 23, asserting `served_init_time` as well as the lead), and the arm-column
test in `test_fit_aifs_blends.py`. The shuffle's within-group property is already covered by
`test_blending.py`, and the fit script gets a test only if it adds a pure helper (the copy
columns). A mutation pass runs only if the diff to `packages/studies/` is more than the `run_hour`
parameter, and otherwise the diff reviewer mutation-tests that parameter by hand.

**Outputs.** All under `data/studies/ukv_ceda_blends/`, a new folder, written once.

- `<domain>_ukv_ceda_inputs.parquet`: the built columns.
- `<domain>_day<N>_<group>_losses.parquet`, `_predictions.parquet`, and `.json`: per-row losses
  and out-of-fold predictions (actual value, anonymised site label, time, fold, seed, arm, setting),
  and the stamp (inputs' SHA-256, the Icechunk snapshot ID, device, GPU, XGBoost version, settings,
  seeds, columns).
- `intervals.parquet`: every interval the report prints.
- `report.md`: every table a page quotes, every arm's column list and absolute error, the row counts
  by cause, and the contrasts at both settings.
- `README.md`: what each file holds. No generator name, identifier, or coordinate in any file.

**Fit-script behaviour.**

- `--dry-run` builds every frame, runs the column-count checks, and lists the fits, writing nothing.
- `--check` fits one arm at one wind site twice on the GPU, stops unless the two fingerprints agree,
  prints the time estimate, and runs the one-off padded-against-unpadded ENS comparison above.
- A stage (technology, day, group) whose losses file exists is not refitted. `--only-missing` fits
  the `(arm, setting)` pairs that no saved file holds. Both refuse to overwrite: a stage's files are
  written to a temporary name and renamed, `studies.guards.refuse_to_overwrite` guards the report,
  and a changed input hash stops the run.
- A stage runs only if the build's stamp says the build passed its coverage guard.
- `--report-only` writes the report from saved losses, to a new file name.

## Compute estimate

**About 288 `(arm, site, setting)` fits, about 1 hour of GPU time with 2 workers.** The
count is 4 arms (padded ENS, blend, and two controls) times 4 days, 9 sites, and 2 settings, which
is
288, plus one CPU refit and the `--check` fits. The blends study's README reports 8 to 10 seconds
per
`(arm, site)` fit at the primary setting and about 1 hour for 315 fits with 2 workers, which scales
to
about 55 minutes for 288. The sensitivity setting runs 1,200 rounds at depth 4 and may take 1.5 to 2
times as long, so the total is near 1 hour or somewhat above, and `--check` measures it. The input
build reads about 650 runs (03 UTC only) of the T120 store and is not yet timed, so `--dry-run`
measures it. The load rule above applies to every run.

## The write-up

**The page is `docs/studies/forecasts/ukv-ceda-blends.md`, listed in `mkdocs.yml` and
`docs/studies/index.md` directly after `blends-with-ens.md`.** It follows the study skill's twelve
sections. The title states the finding once results exist, scoped to the 9 generators and 21 months.

- **Headline figure:** per technology, one panel per lead day (1 to 4) showing P1 and P2 as dots
  with
  95% intervals, the primary setting as the dot and the sensitivity setting as a marker. The page
  carries no leaderboard, because the question is whether one input helps. A table of every arm's
  absolute error follows.
- **Results sections:** "The XGBoost models work" first (out-of-fold forecasts against measured
  power for A to F and W1 to W3 over three weeks chosen by a stated rule, then each arm's absolute
  error per generator), then one section per finding.
- **Limitations:** CEDA only, 21 months, nearest-cell reads, native 10 m and 925 hPa wind instead of
  100 m wind, the unmeasured delivery time, the 3-hour fresher UKV lead (which favours the blend),
  the shared effective-capacity table, and multiplicity.
- **Licence line:** the page states the licence after the check under risks, and says results are
  published, not data.
- **Reproducing the figures:** the command list, each step after the last, with the `uptime` check.

**Cross-references in this PR stay inside `docs/studies/` and `mkdocs.yml`, so the PR keeps the
study skill's self-merge conditions.** One sentence each, written to describe the present state:

- `blends-with-ens.md`, "Limitations and scope": link the new page where it says UKV from the CEDA
  archive and days 3 and 5 are not covered.
- `matched-lead-extra-products.md`, the UKV row of the table at line 42: add a link for UKV at days
  2 to 4 from CEDA.
- `docs/studies/index.md`: the new page's entry.
- `mkdocs.yml`: the nav entry.

**A separate follow-up PR, needing the maintainer's merge, makes two edits outside
`docs/studies/`.**
`docs/roadmap/xgboost-improvements.md` (the paragraph at line 1398, which says the plan is to defer
further blend studies) and `docs/background/weather-products-survey.md` (the CEDA NWP-UKV row, where
"Not scored" becomes a statement that blends at days 1 to 4 are scored, and the licence fixed
once checked). This PR's body says so. The follow-up branches off `main`.

## Risks and open questions

- **CEDA lists no runs for 2026-06-26 to 2026-07-02.** About 8 consecutive 03 UTC runs
  (2026-06-26 to 2026-07-03) are absent, so each lead day loses about 8 valid days, roughly 1.3% of
  rows. The build counts them under "run not listed by CEDA".
- **925 hPa NaN** marks a level below the ground, which depends on the weather's value. The build
  prints the count at the generators' cells. It should be zero, and if it is not, dropping those
  rows selects on a value and the plan is revisited.
- **CEDA gaps.** The three stores hold 12 missing and 4 partial runs of 2,461 slots (the first part)
  and 2 of 2,448 (the third part), and `UKV-CEDA-T120` holds 2 missing so far. A gap costs hours on
  every arm, never one arm. If the loss tops 3% of rows at a day, the page says so.
  *Recommendation:* go on, and report the loss.
- **The January 2026 upgrade and CEDA.** The study drops 2026-01 and cuts eras. Whether CEDA's
  archive carries the upgrade on the same date is not known. *Recommendation:* the step check
  decides, and E4 reports the contrast by era.
- **CEDA's UKV differs statistically from live UKV.** The study cannot say how live UKV would do,
  and
  no XGBoost model trained on CEDA may be used on live UKV. The page says CEDA only.
- **Day-1 comparison with the blends page.** The blends page's UKV day 1 is from Open-Meteo, of
  unknown lead. The page does not compare the two sources, and links the blends page's result
  without a contrast.
- **Native wind columns.** The 10 m and 925 hPa winds are not the 100 m wind ENS carries, so the
  wind
  blend's gain may differ from a blend of two 100 m winds. The columns are fixed before any result.
- **Spatial support.** ENS is a hexagon mean and UKV-CEDA is a nearest 2 km cell. A hexagon-mean UKV
  read would match ENS. *Recommendation:* not fitted first. If a reviewer asks, it adds one arm per
  day to the build and fit.
- **ENS's IFS cycle 50r1 (12 May 2026)** sits inside the window and adds no era feature in the
  shared design. Every arm shares it, so it cancels in contrasts, though it can change what a gain
  means. The page says so.
- **Multiplicity.** There are 16 planned intervals per setting (P1 and P2, 4 days, 2 technologies),
  of which about 1 would reach the 5% level by chance if none of the effects were real. The reading
  rule needs both contrasts at both settings, and the Bonferroni level is reported for P1.
- **Delivery time.** The 09:00 UTC reading of the 03 UTC run is an assumption. Newer CEDA runs are
  published later than live feeds. The page says so and does not bracket it.
- **Licence disagreement.** `fetch_ukv_ceda.py` records CC BY-NC-SA 4.0 in the store's attributes,
  and the survey page says "Met Office licence via CEDA". Check the CEDA catalogue record before
  writing the page, state the verified licence on the page, and fix the survey row in the follow-up
  PR. A non-commercial licence limits how the data feeds a service.

## Considered and rejected

Points from the correctness review that this revision did not apply: none. Every finding
(M1 to M3, S1 to S8, and the notes) was checked against the code and applied, with these
scoping choices: fold rotations are fixed by rule (if `coverage_table` raises on a day's frame, take
the first rotation `search_fold_offsets` returns, before any fit, and each day may differ); the
`ukv_ceda` blend reads 10 m and 925 hPa winds only; and the delivery-time assumption stays
unmeasured. Padding is fair but probably inert (XGBoost's `hist` splitter breaks gain ties towards
the lower feature index), which `--check` tests on the GPU. P2 is the column-count-controlled
comparison.

Points from the simplicity review that this revision did not apply:

- **Review item 1, option (b), fitting only the unpadded ENS arm.** Rejected: it breaches the
  equal-column rule without the identity check, so option (a), padded ENS as the single reference,
  is used and the unpadded arm needs the one-off `--check`.
- **Review item 4, a "column naming that is honest about the heights" without touching
  `_wind_weather_fields`.** Not rejected but costed: the code check found no per-prefix suffix
  option, so the plan adds one branch there, with a bit-identical check on existing arms.
- **Review item 8, "drop `ukv_ceda.py` and its tests".** Applied, but the plan keeps a hand
  mutation check on `run_hour` rather than a full mutation pass.
- **Review item 9, the build's dry run replaces the coverage report.** Applied as written.
- **Dropping the sensitivity setting, the shuffled control, the era cut, or pooling days.** The
  review itself rejects these by the study skill, and the plan keeps all of them.

## Docs to update

The study's README (`studies/ukv_ceda_blends/README.md`) records each script, each file, and the run
order. The plan file is deleted at ship time into the PR body.

## Verification commands

Run from the worktree before every push, then once more after the last fit:

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest packages/studies
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md studies/ukv_ceda_blends
uv run --isolated --no-project --with pydoclint==0.9.1 pydoclint --quiet --style=google .
uv run --no-sync mkdocs build --strict --site-dir /tmp/ci-site
uv run --no-sync python scripts/lint/check_docs_links.py
```

Study-specific checks before the first fit: the unchanged-columns check against the saved
`*_losses.json` stamps passes, `build_ukv_ceda_inputs.py` exits 0 on its coverage guard,
`verify_ukv_ceda_inputs.py` exits 0, and `fit_ukv_ceda_blends.py --dry-run` then `--check`
pass. After the fits, `sha256sum` the published inputs, the day-4 folder, and the earlier blend
folders before and after, and confirm no file in them changed.
