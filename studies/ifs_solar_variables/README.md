# Which IFS forecast variables help predict solar power?

Everything in this directory is a one-off. It exists to answer the question planned in [PR
1137](https://github.com/openclimatefix/nged-substation-forecast/pull/1137), and the answer, not the
code, is what gets kept.

**The question: if an XGBoost model is given the extra variables that Open-Meteo serves from ECMWF's
high-resolution forecast (IFS), does it predict the output of NGED's six solar farms better than an
XGBoost model given only IFS's radiation and air temperature, at lead days 1 to 9?** The study
repeats, on real past forecasts, the question that the [ERA5 study](../era5_solar_variables/README.md)
asks of a reanalysis. It feeds three decisions: which variables to ask Dynamical.org to add, whether
the production forecast needs a second IFS feed, and whether to ingest the variables that only
ECMWF's MARS archive serves or another forecast.

**The forecasts are the 00 UTC run of every day, from March 2024.** Lead day `L` of a valid hour is
read from the run issued `L` days before the valid day. The study drops the two months that
straddle a change of IFS cycle (November 2024 and May 2026) and the months after the second change,
adds an era column, and cuts the folds inside each of the two eras that remain
(`studies.ifs_lead_days`).

## The scripts, in the order they run

| Script | What it does |
|---|---|
| `ifs_ladder_arms.py` | The arms, targets, planned contrasts, lead days, and paths that every other script imports. `arm_features` is the one function that says which columns each arm is shown |
| `ifs_ladder_build_dataset.py` | Takes the ERA5 study's rows, targets, and geometry, joins the IFS forecast at each lead day, and writes one frame per lead day, one frame for the blend rows, and `checks.md`. Run it with `uv run python studies/ifs_solar_variables/ifs_ladder_build_dataset.py` |
| `ifs_ladder_fit.py` | Fits every arm out of fold. `--lead-days`, `--arms`, `--view {ladder,blend}`, `--target`, `--sensitivity-arms`, `--max-workers`, `--device`, and `--ignore-load` choose what runs. It checkpoints every 4 arm-settings into a `losses_<...>.parts` directory and resumes from it. A checkpoint is matched by arm and setting names only, so move or delete a `.parts` directory whenever the frame or an arm's columns change |
| `ifs_ladder_report.py` | Reads the saved losses and writes `report.md` and the interval tables. It applies the decision rules in `studies.ifs_decisions` |
| `ifs_ladder_charts.py` | Draws the page's figures into `docs/studies/assets/` |

The IFS runs come from `studies/weather_downloads/fetch_open_meteo_ifs_solar_variables.py`. The
second forecast, AIFS Single, comes from the site-level columns that the NWP forecast comparison
built. The tested machinery is `studies.ifs_lead_days`, `studies.ifs_ladder`,
`studies.ifs_decisions`, and `studies.checkpointed_fits`.

## The arms

- **`f0` to `f6`** are the ladder: F0 is IFS radiation and air temperature, and each later rung adds
  one physical idea to every rung below it, up to all 20 IFS variables (`f6`).
  `studies.ifs_ladder.RUNG_ADDITIONS` is the one list.
- **`fp`** is the production reference: `f0` plus the four variables the production ensemble feed
  carries.
- **`fb` and `f6b`** are `f0` and `f6` plus AIFS Single's radiation and temperature, on the blend
  rows.
- **`control_*`** are the negative controls. Each has the columns of the arm it pads, with the new
  variables replaced by copies shuffled among the rows that share a farm, a month, and an hour of
  day.
- **`positive_control`** is `f2` plus CAMS global irradiance. It must help.
- **`drop_f1` to `drop_f6`** are `f6` without one rung's variables, at lead days 1 to 3.
- **`era5_g0`, `era5_g1`, `era5_g2`, and `era5_g9`** are the ERA5 arms refitted on the lead-day-1
  rows, with the same shared columns as the IFS arms.

Every arm of one fit is scored on exactly the same rows, and `ifs_ladder_fit.py` raises if two arms
of one setting hold different rows.

## What each output holds

Everything is under `data/studies/per_study/ifs_solar_variables/`.

| File | What it holds |
|---|---|
| `inputs/dataset_lead_day_<L>.parquet` | The kept hourly rows at lead day `L`, with the targets, capacity, export cap, geometry, CAMS columns, folds, and the 20 IFS variables |
| `inputs/dataset_blend.parquet` | The blend rows of lead days 1 to 3, with a `lead_day` column, folds cut again inside one era, and the partner's two columns |
| `inputs/checks.md` | Row counts per farm and lead day, missing-value shares, and the era and fold table |
| `results/losses_<view>_lead_day_<L>_<target>.parquet` | One row per (arm, setting, farm, time, seed) with the losses. The prediction is the measured target plus `signed_error_capped_mw` |
| `results/arms_<...>.json` | Each arm's columns, the settings it was fitted at, the device, and the row count |
| `results/report.md` | The numbers the page quotes |
| `results/leaderboard.parquet`, `contrasts.parquet`, `splits.parquet`, `farms.parquet`, `planned_verdicts.parquet`, `priority_list.parquet`, `decisions.parquet` | The tables the charts read |

A re-run refuses to overwrite a result. Move the old one into a `superseded/` folder first.

## Rules this study follows

- **The rows are set by the ERA5 study's rows, the clock, and IFS's presence, and never by an IFS
  value's size.** Convective inhibition is missing by design and kept as missing.
- **No column is subsampled.** XGBoost runs with `colsample_bytree=1`, so an arm with more columns
  has no advantage from the count.
- **Lead days 1 to 3 are pooled in the planned contrasts** by giving each lead day's rows a farm key
  of its own (`A-L1`, `A-L2`, ...), so that one month resample serves all three.
- **A farm's output appears only as a fraction of the farm's own capacity, under an anonymous label,
  with no calendar date on an axis.** No script prints a coordinate.
