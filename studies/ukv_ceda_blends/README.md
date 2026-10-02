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
   built values in plain Python, checks the radiation timestamp at day 1, compares each lead day's
   correlation with CAMS and ERA5 against Open-Meteo UKV day 1, and screens for month-to-month steps.
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
8. `uv run python studies/ukv_ceda_blends/ukv_ceda_blends_charts.py` draws every figure from the saved
   losses without fitting.

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
- `ukv_ceda_blends_charts.py` draws the headline figure and the per-generator figure from the saved
  losses.
- `check_arm_columns_unchanged.py` is described in step 1.

## Changes to shared code

- `studies.ifs_single_runs.served_init_time` and `served_lead_hours` take `run_hour`, the UTC hour at
  which the run starts (default 0).
- `fit_aifs.BLEND_AIFS_PREFIXES` has a `ukv_ceda` entry, and `BlendRoleType` has the roles `_pad` and
  `_control_b`.
- `nwp_forecast_comparison._wind_weather_fields` returns the 10 m and 925 hPa column names for a
  prefix that starts with `ukv_ceda`.

## Reading the results

- UKV-CEDA is the archive of the Met Office's UKV, which is statistically different from the live
  UKV feed. A model trained on UKV-CEDA must not be run on live UKV.
- The 09:00 UTC delivery time of the 03 UTC run is an assumption that has not been measured for CEDA.
- The page that reports the results is `docs/studies/forecasts/ukv-ceda-blends.md`.
