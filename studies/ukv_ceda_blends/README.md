# Does UKV-CEDA lower the error of the ECMWF ENS mean at lead days 1 to 4?

Study for [issue #1016](https://github.com/openclimatefix/nged-substation-forecast/issues/1016). The
question is whether adding the Met Office's UKV (its 2 km UK variable-resolution weather model),
read from the CEDA archive (the Centre for Environmental Data Analysis), to the mean of the European
Centre for Medium-Range Weather Forecasts (ECMWF) ensemble forecast (ENS) lowers the power-forecast
error at lead days 1 to 4, for solar and for wind separately. The rows, eras, folds, hyperparameter
settings, seeds, metric, and paired month-resampled intervals follow [Which weather forecast is best
at day-ahead lead?](../../docs/studies/forecasts/matched-lead.md). The shuffled control and the
reading rule follow [ENS plus one weather product](../../docs/studies/forecasts/blends-with-ens.md).

**The planned blend reads UKV-CEDA's 03 UTC run (UTC is Coordinated Universal Time).** An hour on
day `D` at lead day `N` reads the 03 UTC run of day `D - N` from the `UKV-CEDA-T120` store, at lead
`24 * N + h - 3` hours, where `h` is the hour of day. A service running at 09:00 UTC could read that
run. The lead is 21 to 117 hours over days 1 to 4, inside the store's 120 hours. Day 5 cannot be
built, because its lead exceeds 120 hours for every hour after 03:00 UTC. The UKV-CEDA lead is 3
hours shorter than ENS's lead at every hour, which favours the blend.

**Four arms are fitted per technology and lead day, each at both hyperparameter settings.** An arm
is one XGBoost model per generator, and the arms hold equal column counts (9 for solar, 11 for
wind).

| Arm | Columns after the calendar columns |
|---|---|
| `blend_ukv_ceda_dayN_pad` | ENS's mean, then exact copies of the same columns |
| `blend_ukv_ceda_dayN` | ENS's mean, then UKV-CEDA |
| `blend_ukv_ceda_dayN_control` | ENS's mean, then UKV-CEDA shuffled within generator, year-month, and hour of day (seed 0) |
| `blend_ukv_ceda_dayN_control_b` | the same shuffle under seed 1000 |

UKV-CEDA's wind columns are its native 10 m speed, the sine and cosine of its 10 m direction, and
its 925 hPa speed. None of these columns is the 100 m wind that ENS carries. Two contrasts are
planned before any fit. P1 is the blend minus the padded ENS arm. P2 is the blend minus each
control. The blend lowers the error only if the upper 95% bound of P1 and of both P2 contrasts is
below zero at both settings.

## Run order

Each step after the first needs the step before it to have exited 0. Run `uptime` and `nvidia-smi`
before the build and before every fit, and start only below a load average of about 24.

1. `uv run python studies/ukv_ceda_blends/check_arm_columns_unchanged.py` reads every
   `*_losses.json` stamp under `data/studies/nwp_forecast_comparison_*` and exits 0 if each arm's
   recorded columns still equal `fit_aifs.arm_features`. The check proves that the one branch added
   to `nwp_forecast_comparison._wind_weather_fields` leaves every earlier arm alone.
2. `uv run python studies/ukv_ceda_blends/build_ukv_ceda_inputs.py --dry-run` builds one month
   (`--dry-run-month`, default 2026-03), prints the rows lost to each cause, and writes nothing.
3. `uv run python studies/ukv_ceda_blends/build_ukv_ceda_inputs.py` writes
   `<domain>_ukv_ceda_inputs.parquet`, `build.json`, and `README.md` into the write-once folder
   `data/studies/ukv_ceda_blends/`. The build refuses to run until the download covers the window
   and every run slot the store marks as never archived has been fetched again once. Name the days
   CEDA still does not list in `--unlisted-days`.
4. `uv run python studies/ukv_ceda_blends/verify_ukv_ceda_inputs.py` recomputes a stratified sample
   of built values in plain Python, gates the radiation timestamp at day 1 (see "The radiation
   timestamp" below), compares each lead day's correlation with the Copernicus Atmosphere Monitoring
   Service (CAMS) and ECMWF's fifth reanalysis (ERA5) against Open-Meteo UKV day 1 on the rows all
   four lead days hold, and screens for month-to-month steps. The script writes `verify.json`,
   holding whether every gating check passed and the SHA-256 of each inputs file, into the build's
   folder.
5. `uv run python studies/ukv_ceda_blends/fit_ukv_ceda_blends.py --dry-run` builds every frame,
   checks every arm's columns, and lists the fits.
6. `uv run python studies/ukv_ceda_blends/fit_ukv_ceda_blends.py --check` fits one arm at one wind
   site twice on the graphics processing unit (GPU), stops unless the two fingerprints agree, prints
   a time estimate, and compares ENS's mean alone with its padded copy on the wind day-1 rows.
7. `uv run python studies/ukv_ceda_blends/fit_ukv_ceda_blends.py --verified` fits every stage,
   writes the losses once, refits the wind day-1 blend at one site on the central processing unit
   (CPU), and writes `report.md` and `intervals.parquet`. Add `--only-missing` to fit the pairs no
   saved file holds into a new `_added_<k>` file, and `--report-only --report-name NAME` to write
   the report again from the saved losses into new files.
8. `uv run python studies/ukv_ceda_blends/fit_ukv_ceda_blends.py --post-hoc-stale --dry-run` lists
   the post hoc stale-blend fits (see "Post hoc stale blend" below) and the rows each stage loses to
   the blend's extra columns.
9. `uv run python studies/ukv_ceda_blends/fit_ukv_ceda_blends.py --post-hoc-stale --report-name
   report_2` fits the stale blend into new `_added_<k>` files, never touching the planned fits, and
   writes `report_2.md` and `report_2_intervals.parquet`.
10. `uv run python studies/ukv_ceda_blends/ukv_ceda_blends_charts.py --figures-dir DIR
    --intervals-name report_2_intervals.parquet` draws every figure from the saved intervals and
    losses without fitting.

11. `uv run python studies/ukv_ceda_blends/fit_ukv_ceda_blends.py --post-hoc-permutation --dry-run`
    lists the post hoc permutation test's fits (see "Post hoc permutation test" below), and the same
    command with `--report-name report_3` instead of `--dry-run` fits the controls into new
    `_added_<k>` files and writes `report_3.md` and `report_3_intervals.parquet`.
12. `uv run python studies/ukv_ceda_blends/build_ukv_ceda_inputs.py --older-run --dry-run`, then
    without `--dry-run` and with the same `--unlisted-days`, builds the older-run inputs into the
    write-once folder `data/studies/ukv_ceda_blends_run15/`. `verify_ukv_ceda_inputs.py --older-run`
    verifies them. `fit_ukv_ceda_blends.py --post-hoc-older-run --report-name report_4` fits the
    older-run blend and writes `report_4.md` (see "Post hoc older run" below).
13. `uv run python studies/ukv_ceda_blends/ukv_ceda_blends_charts.py --post-hoc-only --figures-dir
    DIR --intervals-name report_4_intervals.parquet` draws the permutation and older-run figures.
14. `uv run python studies/ukv_ceda_blends/fit_ukv_ceda_blends.py --report-only --post-hoc-older-run
    --report-name report_5` writes `report_5.md` and `report_5_intervals.parquet` from the saved
    losses, fitting nothing. The report adds the day-1 older-run split at lead 48 hours, the share
    of the planned gain each post hoc blend keeps, and each arm's error at each generator.
    `ukv_ceda_blends_charts.py --figures-dir DIR --intervals-name report_5_intervals.parquet` then
    draws every figure of the page.

## Outputs the report adds after the first science review

- **Three readings.** A technology and lead day reads `lowers the error`, `no detectable
  difference`, or `unresolved: lower than padded ENS, control test not passed`. The third reading
  applies where P1 is below zero at both settings and some P2 bound is not.
- **Bonferroni at both settings.** The report prints P1's 99.375% interval at the primary and the
  sensitivity setting, and the readings table says whether P1 stays below zero at both. The planned
  rule is unchanged: the correction is an extra column, not part of the rule.
- **Largest gain left open.** For every stage the readings table gives the largest gain P1's lower
  bound does not exclude, at each setting. `no detectable difference` never means no gain.
- **Control gap.** The report prints seed-0 control minus seed-1000 control at both settings. The
  two controls carry the same (no) information, so the gap is the size of difference the pipeline
  produces from nothing. The interval resamples months and a fitting seed, not shuffle seeds.
- **Leave-one-month-out.** Post hoc and exploratory, the report recomputes every planned P1 with
  each calendar month dropped in turn from the saved losses, and prints the lowest and highest point
  estimate, the month that moves it most, and the highest 95% upper bound.
- **Padding check.** `--check` runs ENS's mean alone against its padded copy at wind day 1 and solar
  day 1, and writes `padding_check.json` once. The report prints the result.
- **`intervals.parquet` rows.** Beside the planned and exploratory rows, `scope` can be `Bonferroni`
  (P1 at `level` 99.375), `control gap` (`contrast` `control_gap`), `E6 most influential month
  dropped`, `generator error <site>: <arm>` (contrast `generator_error`, the arm's mean absolute
  error at one generator, with no interval), `post hoc stale` (contrasts `stale_p1`, `stale_p2`,
  `stale_p2b`, `stale_vs_fresh`, `fresh_p1_same_rows`, and `training_rows`), and the `error`
  contrast, whose `difference` is an arm's own mean absolute error and whose `scope` is the arm. The
  charts read every figure number from these rows.

## Post hoc stale blend

**The stale blend moves UKV-CEDA's run 24 hours earlier, which changes its lead, its start against
ENS, and at days 1 and 2 its time resolution together.** The arm `blend_ukv_ceda_stale_dayN` is ENS
day `N` plus UKV-CEDA day `N + 1` for `N` of 1 to 3, with the same column counts as the planned
arms. UKV-CEDA's run is the 03 UTC run one day before ENS's run, 21 hours staler than ENS's run,
where the planned blend's run is 3 hours fresher. The arm is fitted for both technologies at both
settings with its own padded ENS reference (`blend_ukv_ceda_stale_dayN_pad`) and both shuffled
controls, all trained on the stage's rows where UKV-CEDA's day `N + 1` is present, about 0.4% fewer
rows than the planned arms. Every contrast scores those rows. The report prints the planned padded
reference minus the stale-rows padded reference as the measured tilt from the planned reference's
extra training rows. The fit count is 216 (arm, site) fits: 3 lead days, 2 settings, 4 arms, and 9
generators. The stale blend is a post hoc, exploratory analysis added after the first science
review.

## Post hoc permutation test

**The permutation test asks whether the solar blend's gain is larger than shuffled controls give by
chance.** For each solar lead day, `--post-hoc-permutation` fits 15 further shuffled controls
(`blend_ukv_ceda_dayN_control_s<seed>`, seeds 2010 to 2150 in steps of 10) at the primary setting
only. Each further control shuffles UKV-CEDA's columns within generator, year-month, and hour of
day, with the same groups as the planned control. The report prints the planned blend's P1 against
the 17 values of (shuffled control minus padded ENS) from the planned two controls and the 15 extra
controls, the rank of P1 among the 18 values, and the one-sided permutation p-value `(1 + controls
at or below P1) / 18`, whose smallest possible value is 0.056. The test is exploratory and post hoc.
The test fits 6 generators x 4 lead days x 15 seeds, 360 fits.

## Post hoc older run

**The older-run blend tests how the gain falls as the UKV-CEDA run gets older.** The arm
`blend_ukv_ceda_run15_dayN` is ENS day `N` plus UKV-CEDA columns built from the 15 UTC run of the
day before ENS's run, for `N` of 1 to 3. For an hour on day `D` at lead day `N`, the run starts at
15:00 UTC on day `D - N - 1` and the lead is `24 * N + h + 9` hours (`h` the hour of the hour's
instant, plus 1 for a solar label). That run starts 9 hours before ENS's 00 UTC run of day `D - N`,
where the planned blend's run starts 3 hours after ENS's run, and leads 12 hours longer than the
planned run. Day 4 cannot be built: its lead reaches 129 hours, beyond the store's 120, for every
hour after 14:00 UTC. The two changes, a longer lead and an earlier start against ENS, move
together, so the arm cannot separate the effect of the lead from the effect of the timing. The store
is hourly only to lead 48 hours, so the older run also has more hours rebuilt from 3-hourly steps
than the planned run has at days 1 and 2. At day 1 the older run is read at a lead of 33 to 56
hours. The report splits the day-1 rows at lead 48 hours, because both runs are hourly on the rows
at or before that lead. The arm has its own padded ENS reference refitted on its rows, one shuffled
control (seed 0), and equal column counts (9 for solar, 11 for wind), at both settings, for both
technologies. The report prints the older-run blend minus its padded ENS (P1), minus its control
(P2), minus the planned blend, and the planned P1 on the same rows. The older-run inputs are built
by `build_ukv_ceda_inputs.py --older-run` with the same coverage guard, stamp checks, and init-time
assertions as the main build, into `data/studies/ukv_ceda_blends_run15/`.

## Scripts

- `build_ukv_ceda_inputs.py` reads the 03 UTC runs of the `UKV-CEDA-T120` Icechunk store, takes each
  generator's nearest 2 km cell, and rebuilds the leads the store lacks. The store holds hourly
  steps to lead 48 hours, then 51 and 54, then every third hour to 120. Radiation is rebuilt through
  the clear-sky index at each instant, temperature by a straight line, and wind through its eastward
  and northward components. A rebuilt value is missing unless both steps either side of it are
  finite. Solar radiation is the mean of the two snapshots at leads `L - 1` and `L`.
- `verify_ukv_ceda_inputs.py` only reads, and exits non-zero if a gating check fails.
- `fit_ukv_ceda_blends.py` imports `fit_aifs` and `nwp_forecast_comparison` from the sibling
  `studies/nwp_forecast_comparison/` directory and changes neither. A stage whose losses exist is
  not refitted, and every output is written once.
- `ukv_ceda_blends_charts.py` draws the headline figure (both settings' intervals, a title and panel
  titles that state the reading), the per-generator figure, the per-arm absolute-error figure with
  no intervals, the per-generator absolute-error figure, the permutation and older-run figures, and
  the week figures from the saved intervals, losses, and predictions.
- `check_arm_columns_unchanged.py` is described in step 1. The check cannot cover the main
  matched-lead fit, which wrote no stamp. No arm prefix there starts with `ukv_ceda`, so the new
  branch of `_wind_weather_fields` is never reached by that fit.

## Changes to shared code

- `studies.ifs_single_runs.served_init_time` and `served_lead_hours` take `run_hour`, the UTC hour
  at which the run starts (default 0).
- `fit_aifs.BLEND_AIFS_PREFIXES` has `ukv_ceda` and `ukv_ceda_stale` entries, and `BlendRoleType`
  has the roles `_pad` and `_control_b`.
- `nwp_forecast_comparison._wind_weather_fields` returns the 10 m and 925 hPa column names for a
  prefix that starts with `ukv_ceda`.

## The radiation timestamp

The day-1 gate in `verify_ukv_ceda_inputs.py` is the median over the solar generators of two peak
offsets, each the offset at which the radiation correlates best with the cosine of the solar zenith
angle. (a) The raw native day-1 snapshots, which are instants, must peak within 15 minutes of 0. (b)
The rebuilt column, a mean of the snapshots at `L - 1` and `L`, must peak 30 plus or minus 10
minutes before (a). The snapshots are not reweighted to land on -30 minutes.

## Reading the results

- UKV-CEDA's radiation is centred about 20 minutes later than the power hour. On the 262 complete 03
  UTC runs of 2026-01-02 to 2026-09-28, the raw snapshots peaked about 10 minutes after their stamp
  (15 minutes at one generator), so the mean of two snapshots peaks about 20 minutes before its
  label instead of 30. Open-Meteo's UKV peaked 13 minutes before its stamp. The offset is a property
  of the archive that no construction choice can remove. The study does not measure how the offset
  changes the error.

- UKV-CEDA is the archive of the Met Office's UKV, which is statistically different from the live
  UKV feed. An XGBoost model trained on UKV-CEDA must not be run on live UKV.
- The 09:00 UTC delivery time of the 03 UTC run is an assumption that has not been measured for
  CEDA.
- The page that reports the results is `docs/studies/forecasts/ukv-ceda-blends.md`.
