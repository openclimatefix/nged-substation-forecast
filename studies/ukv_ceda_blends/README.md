# Does UKV-CEDA lower the error of the ECMWF ENS mean at lead days 1 to 4?

Study for [issue #1016](https://github.com/openclimatefix/nged-substation-forecast/issues/1016).
The question is whether adding the Met Office's UKV (a 2 km UK weather model), read from the CEDA
archive (the Centre for Environmental Data Analysis), to the ECMWF ENS mean lowers the power-forecast
error at lead days 1 to 4, for solar and for wind separately. The rows, eras, folds, hyperparameter
settings, seeds, metric, and paired month-resampled intervals are those of
[Which weather forecast is best at day-ahead lead?](../../docs/studies/forecasts/matched-lead.md).
The shuffled control and the reading rule are those of
[ENS plus one weather product](../../docs/studies/forecasts/blends-with-ens.md).

**The 03 UTC run is the only UKV-CEDA lead.** An hour on day `D` at lead day `N` reads the 03 UTC
run of day `D - N` from the `UKV-CEDA-T120` store, at lead `24 * N + h - 3` hours, where `h` is the
hour of day. A service running at 09:00 UTC could read that run. The lead is 21 to 117 hours over
days 1 to 4, inside the store's 120 hours. Day 5 cannot be built, because its lead exceeds 120 hours
for every hour after 03:00 UTC. The 3 hours are shorter than ENS's lead at every hour, which favours
the blend.

**Four arms are fitted per technology and lead day, each at both hyperparameter settings.** An arm is
one XGBoost model per generator, and the arms hold equal column counts (9 for solar, 11 for wind).

| Arm | Columns after the calendar columns |
|---|---|
| `blend_ukv_ceda_dayN_pad` | ENS's mean, then exact copies of the same columns |
| `blend_ukv_ceda_dayN` | ENS's mean, then UKV-CEDA |
| `blend_ukv_ceda_dayN_control` | ENS's mean, then UKV-CEDA shuffled within generator, year-month, and hour of day (seed 0) |
| `blend_ukv_ceda_dayN_control_b` | the same shuffle under seed 1000 |

UKV-CEDA's wind columns are its native 10 m speed, the sine and cosine of its 10 m direction, and its
925 hPa speed. They are not the 100 m wind that ENS carries. Two contrasts are planned before any
fit. P1 is the blend minus the padded ENS arm. P2 is the blend minus each control. The blend lowers
the error only if the upper 95% bound of P1 and of both P2 contrasts is below zero at both settings.

## Run order

Each step after the first needs the step before it to have exited 0. Run `uptime` and `nvidia-smi`
before the build and before every fit, and start only below a load average of about 24.

1. `uv run python studies/ukv_ceda_blends/check_arm_columns_unchanged.py` reads every
   `*_losses.json` stamp under `data/studies/nwp_forecast_comparison_*` and exits 0 if each arm's
   recorded columns still equal `fit_aifs.arm_features`. It proves that the one branch added to
   `nwp_forecast_comparison._wind_weather_fields` leaves every earlier arm alone.
2. `uv run python studies/ukv_ceda_blends/build_ukv_ceda_inputs.py --dry-run` builds one month
   (`--dry-run-month`, default 2026-03), prints the rows lost to each cause, and writes nothing.
3. `uv run python studies/ukv_ceda_blends/build_ukv_ceda_inputs.py` writes
   `<domain>_ukv_ceda_inputs.parquet`, `build.json`, and `README.md` into the write-once folder
   `data/studies/ukv_ceda_blends/`. It refuses to run until the download covers the window and every
   run slot the store marks as never archived has been fetched again once. Name the days CEDA still
   does not list in `--unlisted-days`.
4. `uv run python studies/ukv_ceda_blends/verify_ukv_ceda_inputs.py` recomputes a stratified sample of
   built values in plain Python, gates the radiation timestamp at day 1 (see "The radiation
   timestamp" below), compares each lead day's correlation with CAMS and ERA5 against Open-Meteo UKV
   day 1 on the rows all four lead days hold, and screens for month-to-month steps. It writes
   `verify.json`, holding whether every gating check passed and the SHA-256 of each inputs file, into
   the build's folder.
5. `uv run python studies/ukv_ceda_blends/fit_ukv_ceda_blends.py --dry-run` builds every frame,
   checks every arm's columns, and lists the fits.
6. `uv run python studies/ukv_ceda_blends/fit_ukv_ceda_blends.py --check` fits one arm at one wind
   site twice on the GPU, stops unless the two fingerprints agree, prints a time estimate, and
   compares ENS's mean alone with its padded copy on the wind day-1 rows.
7. `uv run python studies/ukv_ceda_blends/fit_ukv_ceda_blends.py --verified` fits every stage, writes
   the losses once, refits the wind day-1 blend at one site on the CPU, and writes `report.md` and
   `intervals.parquet`. Add `--only-missing` to fit the pairs no saved file holds into a new
   `_added_<k>` file, and `--report-only --report-name NAME` to write the report again from the saved
   losses into new files.
8. `uv run python studies/ukv_ceda_blends/fit_ukv_ceda_blends.py --post-hoc-stale --dry-run` lists
   the post hoc stale-blend fits (see "Post hoc stale blend" below) and the rows each stage loses to
   the blend's extra columns.
9. `uv run python studies/ukv_ceda_blends/fit_ukv_ceda_blends.py --post-hoc-stale --report-name
   report_2` fits the stale blend into new `_added_<k>` files, never touching the planned fits, and
   writes `report_2.md` and `report_2_intervals.parquet`.
10. `uv run python studies/ukv_ceda_blends/ukv_ceda_blends_charts.py --figures-dir DIR
    --intervals-name report_2_intervals.parquet` draws every figure from the saved intervals and
    losses without fitting.

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
  each calendar month dropped in turn from the saved losses, and prints the lowest and highest
  point estimate, the month that moves it most, and the highest 95% upper bound.
- **Padding check.** `--check` runs ENS's mean alone against its padded copy at wind day 1 and solar
  day 1, and writes `padding_check.json` once. The report prints the result.
- **`intervals.parquet` rows.** Beside the planned and exploratory rows, `scope` can be
  `Bonferroni` (P1 at `level` 99.375), `control gap` (`contrast` `control_gap`),
  `E6 most influential month dropped`, `post hoc stale` (contrasts `stale_p1`, `stale_p2`,
  `stale_p2b`, `stale_vs_fresh`, `fresh_p1_same_rows`, and `training_rows`), and the `error`
  contrast, whose `difference` is an arm's own mean absolute error and whose `scope` is the arm.
  The charts read every figure number from these rows.

## Post hoc stale blend

**The stale blend tests whether UKV-CEDA's 3-hour lead advantage explains the gain.** The arm
`blend_ukv_ceda_stale_dayN` is ENS day `N` plus UKV-CEDA day `N + 1` for `N` of 1 to 3, with the
same column counts as the planned arms. UKV-CEDA's run is the 03 UTC run one day before ENS's run,
21 hours staler than ENS's run, where the planned blend's run is 3 hours fresher. The arm is fitted
for both technologies at both settings with its own padded ENS reference
(`blend_ukv_ceda_stale_dayN_pad`) and both shuffled controls, all trained on the stage's rows where
UKV-CEDA's day `N + 1` is present, about 0.4% fewer rows than the planned arms. Every contrast
scores those rows. The report prints the planned padded reference minus the stale-rows padded
reference as the measured tilt from the planned reference's extra training rows. The fit count is
216 (arm, site) fits: 3 lead days, 2 settings, 4 arms, and 9 generators. The stale blend is a post
hoc, exploratory analysis added after the first science review.

## Scripts

- `build_ukv_ceda_inputs.py` reads the 03 UTC runs of the `UKV-CEDA-T120` Icechunk store, takes each
  generator's nearest 2 km cell, and rebuilds the leads the store lacks. The store holds hourly steps
  to lead 48 hours, then 51 and 54, then every third hour to 120. Radiation is rebuilt through the
  clear-sky index at each instant, temperature by a straight line, and wind through its eastward and
  northward components. A rebuilt value is missing unless both steps either side of it are finite.
  Solar radiation is the mean of the two snapshots at leads `L - 1` and `L`.
- `verify_ukv_ceda_inputs.py` only reads, and exits non-zero if a gating check fails.
- `fit_ukv_ceda_blends.py` imports `fit_aifs` and `nwp_forecast_comparison` from the sibling
  `studies/nwp_forecast_comparison/` directory and changes neither. A stage whose losses exist is
  not refitted, and every output is written once.
- `ukv_ceda_blends_charts.py` draws the headline figure (both settings' intervals, a title and panel
  titles that state the reading), the per-generator figure, the per-arm absolute-error figure for
  every lead day, and the week figures from the saved intervals, losses, and predictions.
- `check_arm_columns_unchanged.py` is described in step 1. It cannot cover the main matched-lead
  fit, which wrote no stamp. No arm prefix there starts with `ukv_ceda`, so the new branch of
  `_wind_weather_fields` is never reached by that fit.

## Changes to shared code

- `studies.ifs_single_runs.served_init_time` and `served_lead_hours` take `run_hour`, the UTC hour at
  which the run starts (default 0).
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

- UKV-CEDA's radiation is centred about 20 minutes later than the power hour. On the 262 complete
  03 UTC runs of 2026-01-02 to 2026-09-28, the raw snapshots peaked about 10 minutes after their
  stamp (15 minutes at one generator), so the mean of two snapshots peaks about 20 minutes before
  its label instead of 30. Open-Meteo's UKV peaked 13 minutes before its stamp. The offset is a
  property of the archive that no construction choice can remove. It handicaps the blend and does
  not favour it, because every UKV-CEDA value comes from a run issued at least 21 hours ahead.

- UKV-CEDA is the archive of the Met Office's UKV, which is statistically different from the live
  UKV feed. A model trained on UKV-CEDA must not be run on live UKV.
- The 09:00 UTC delivery time of the 03 UTC run is an assumption that has not been measured for CEDA.
- The page that reports the results is `docs/studies/forecasts/ukv-ceda-blends.md`.
