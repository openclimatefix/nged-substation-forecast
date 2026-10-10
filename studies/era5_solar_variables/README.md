# Which ERA5 variables help predict solar power?

Everything in this directory is a one-off. It exists to answer the question planned in [PR
1097](https://github.com/openclimatefix/nged-substation-forecast/pull/1097) and recorded in its
issue, and the answer, not the code, is what gets kept.

**The question: if an XGBoost model is given every ERA5 variable that could plausibly matter, does it
predict the output of NGED's six solar farms better than an XGBoost model given the minimal ERA5
set?** The same question is asked of a second target, CAMS satellite clearness index at the same
farms. The minimal set is ERA5's downward solar radiation (`ssrd`) and 2 m temperature (`t2m`), with
solar geometry and the clearness index. The decision the study feeds is which ECMWF IFS variables to
use in a solar power forecast, and whether it is worth fetching the variables that only ECMWF's full
archive (MARS) serves.

**ERA5 is an upper-bound screen.** ERA5 was produced by a frozen 2016 version of the IFS and its
fields come from short forecasts, so a variable that helps an XGBoost model trained on ERA5 may add
only noise to an XGBoost model fed day-3 IFS forecasts. The plan explains why, and what follow-up the
result calls for.

## The scripts, in the order they run

| Script | What it does |
|---|---|
| `era5_ladder_arms.py` | The arms, targets, planned contrasts, and paths that every other script imports |
| `era5_ladder_build_dataset.py` | Joins output, CAMS, and every downloaded ERA5 variable into one hourly frame holding all six farms. Run it with `uv run --with netcdf4`. `--through-rung g2` builds from the variables downloaded so far, `--keep-zero-hours-with-snow` builds the snow variant, and the full build adds the EAC4 aerosol columns whenever the aerosol download exists |
| `era5_ladder_fit.py` | Fits every arm out of fold for both targets. `--view aerosol` fits the aerosol view, and `--view extra_sensitivity --sensitivity-arms ...` adds the second hyperparameter setting for arms whose contrasts lie near the 5% line. It checkpoints every 4 arm-settings into a `losses_<...>.parts` directory and resumes from it, and a checkpoint is matched by arm and setting names only, so move or delete a `.parts` directory whenever the dataset, an arm's columns, or `QUANTILE_ARMS` change. After a restart that follows a finished target, move that target's `losses_` and `arms_` files to `superseded/` first, because the script refuses to overwrite them |
| `era5_ladder_importance.py` | Refits `g0`, `g2`, `g9`, and `g9` with shuffled copies of the `g3` to `g9` columns, and saves each column's share of XGBoost's total gain (`importance_<variant>_through_<rung>.parquet`). It needs a GPU slot from the study coordinator |
| `era5_ladder_report.py` | Reads the saved losses and writes `report.md` and the interval tables, each named for the variant and the highest rung |
| `era5_ladder_charts.py` | Draws the page's figures into `docs/studies/assets/` |

The ERA5 variables come from two sources. `studies/weather_downloads/fetch_era5_solar_variables.py
--tier tier1a` fetches the cloud covers from the Climate Data Store, and
`studies/weather_downloads/fetch_era5_solar_arco.py --tier tier1b`, `--tier tier2`, and `--tier
tier3` fetch the other variables from Google's ARCO-ERA5 copy. `validate_era5_solar_variables.py`
checks the Climate Data Store tables. The aerosol comes from `fetch_cams_eac4_aod.py`. The tested machinery is `studies.era5_ladder` and
`studies.correlation`.

## The arms

- **`g0` to `g9`** are the ladder: each rung adds one physical idea to every rung below it, from the
  minimal set (`g0`) to every ERA5 variable in the plan (`g9`). `studies.era5_ladder.RUNG_ADDITIONS`
  is the one list.
- **`negative_control`** is `g2` plus a permuted copy of every later column, so a difference against
  it shows what the pipeline produces from nothing.
- **`positive_control`** is `g2` plus CAMS global irradiance, on the output target only. It must help.
- **`known_answer_ssrd_only`** is the CAMS target given `ssrd` and the sun position only.
- **`g9_without_mars_only`** is `g9` without the 12 variables found only in MARS (and the clear-sky
  index built from `ssrdc`). Planned contrast P4 compares `g9` with it.
- **`drop_g1` to `drop_g9`** are `g9` without one rung's variables.
- **`g10` and `g9_aerosol_rows`** are `g9` with and without CAMS EAC4 aerosol optical depth, on the
  rows that EAC4 covers, with the folds cut again on that span.

Every arm of one fit is scored on exactly the same rows, and `era5_ladder_fit.py` raises if two arms
of one fit hold different rows. The aerosol view has its own rows, which EAC4 covers.

## What each output holds

Everything is under `data/studies/per_study/era5_solar_variables/`.

| File | What it holds |
|---|---|
| `inputs/dataset_<variant>_through_<rung>.parquet` | The kept hourly rows. Each row carries the farm label, the time, the output, the capacity, the export cap, CAMS irradiance and top-of-atmosphere flux, and the ERA5 variables up to the rung |
| `inputs/checks_<variant>_through_<rung>.md` | Row counts, the final-ERA5 span, and each missing-under-clear-sky variable's missing share |
| `results/losses_<variant>_through_<rung>_<target>_<view>.parquet` | One row per (arm, setting, farm, time, seed) with the losses. The prediction is the measured target plus `signed_error_capped_mw` |
| `results/arms_<...>.json` | Each arm's columns, the device, the settings, and the row count |
| `results/report_<variant>_through_<rung>.md`, and the `leaderboard`, `contrasts`, `splits`, `worst_days`, `aerosol_conditions`, and `probabilistic` parquet files (`aerosol_conditions` holds `g10` against `g9_aerosol_rows` in clear, clear-and-dusty, dusty, and clear-and-clean hours) named in the same way | The numbers the page quotes, and the tables the charts read |

A re-run refuses to overwrite a result. Move the old one into a `superseded/` folder first.

## Rules this study follows

- **The rows are set by the clock, the place, the targets, and the span, and never by an ERA5
  value.** The build raises if any ERA5 variable except `cbh` and `cin` is missing on a kept row.
- **No column is subsampled.** XGBoost runs with `colsample_bytree=1`, so an arm with more columns
  has no advantage from the count.
- **A farm's output appears only as a fraction of the farm's own capacity, under an anonymous label,
  with no calendar date on an axis.**
