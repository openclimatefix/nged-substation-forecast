# Lagged power features for solar forecasts

Study for [issue #1138](https://github.com/openclimatefix/nged-substation-forecast/issues/1138).
The question is whether an XGBoost solar power forecast improves when it is also shown lagged power.
The design is in `plans/study-lag-features-1138.md` (deleted on merge) and, once published, on its
page under `docs/studies/`.

## The scripts

**Run the four scripts in this order, once per weather product.** Each script takes
`--weather-product ens_mean` (the default and the primary product) or `ifs_single` (an exploratory
replicate of B0 and L1), and `--output-root` (default `data/studies/per_study/lag_features/`). Every
output path carries the product name, and no script overwrites a finished output.

| Script | What it does |
|---|---|
| `build_lag_frame.py` | Rebuilds the 35,263 shared rows, folds and `era_code`, builds each lead-day's frame and the positive-control frames, asserts the lag arithmetic, and writes `build_report_<product>.md`. |
| `lag_arm_columns.py` | Imported by the builder: the columns of IM, TF, AN, PC, CK, RP and DT, and the anchor assertions. |
| `fit_lag_arms.py` | Fits stage 1, every arm, the shortlist rule, phase 2, the global and leave-one-plant-out fits, the positive controls and the longer leads, with checkpoints. |
| `report_lag_features.py` | Prints every table the page quotes into `report_<product>.md` and saves `tables_<product>/*.parquet`. |
| `lag_features_charts.py` | Draws the figures from those tables, and writes `figure_text_<product>.txt` for the text reviews first (`--text-only`). |

## Commands that run the study end to end

```bash
uv run python studies/lag_features/build_lag_frame.py
uv run python studies/lag_features/fit_lag_arms.py --reproduction-check --device cuda --max-workers 4
uv run python studies/lag_features/fit_lag_arms.py --device cuda --max-workers 4
uv run python studies/lag_features/report_lag_features.py
uv run python studies/lag_features/lag_features_charts.py --text-only
uv run python studies/lag_features/lag_features_charts.py
# The exploratory replicate with IFS HRES:
uv run python studies/lag_features/build_lag_frame.py --weather-product ifs_single
uv run python studies/lag_features/fit_lag_arms.py --weather-product ifs_single --device cuda --max-workers 4
uv run python studies/lag_features/report_lag_features.py --weather-product ifs_single
```

The fit script needs a compute slot. `--smoke` fits two arms on a few hundred rows with 20 boosting
rounds, to check the plumbing, and is not a result.

## Choices the plan did not settle

- **The lag source** is NGED's hourly power with the multi-day zero runs, meter spikes, commissioning
  ramp and export-capped hours removed. The target is the shared rows' `power_mw`.
- **Power after `final_test_start` (2026-07-01) is never read**, because `studies.power.scan_power`
  stops there, so the rows from that date are dropped and the page's months are the 18 before it.
- **CAMS's publication delay.** The study assumes CAMS irradiance reaches a forecast two whole days
  late, so PC's windows cover days 2 to 8 (power to satellite) and days 2 to 31 (satellite to
  forecast), counted back from the latest whole day. The assumption comes from the CAMS
  documentation, which says the radiation service provides data with up to 2 days of delay
  ([Copernicus radiation service in a nutshell](https://atmosphere.copernicus.eu/sites/default/files/2020-03/Copernicus_radiation_service_in_nutshell_v11.pdf),
  [CAMS solar radiation time-series](https://www.ecmwf.int/node/37329)). Some CAMS pages describe
  the service as available to the previous day. The 2-day gap is therefore an assumption, and
  the shorter delay would let PC read one more day.
- **Power-data latency is ignored** by design: PV power can be had within minutes, so IM reads every
  hour that ended by 09:00 UTC on the issue day.
- **IM's forecast irradiance** is the same run's lead-day 0 value of the issue-morning hours.
- **TF and PC reuse `same_clock_hour_window`** with a table of irradiance in place of power, so that
  a ratio of window means is a ratio of sums over the same days.
- **A day is valid** (daily features, DT, RP, CK) when its observed hours hold at least half the day's
  clear-sky energy, the rule `studies.baselines.clear_sky_index` applies.
- **CK's clear hours** have a forecast clear-sky index of at least 0.8 and a clear-sky irradiance of
  at least 300 W m⁻², and are near the ceiling at 95% of the expanding 99.5th percentile.
- **DT** is the mean over 30 days of each valid day's power-weighted hour minus its clear-sky-weighted
  hour, and of its share of energy in hours below 30% of the day's peak clear-sky irradiance.
- **R5's sky classes** use the terciles of the forecast clear-sky index over every hour with a stage-1
  prediction.
- **N2** draws 16 independent same-clock-hour lags from months outside the row's fold; N2-k reads the
  first `k` of them. A month in no fold is allowed.
- **The sweep arms with `{fold}` columns** are the two-step arms S2 and S3 only; stage 1 uses the
  primary setting and seed 0.
- **Positive control**: the library fits all three seeds, so the plan's "one seed" is three.
- **The interval figure** (`intervals__B0`, `intervals__L1` checkpoints) uses seed 0 only.
- **Feature importance is not computed**: `out_of_fold_losses` does not return the boosters.

## What each file holds

- `lag_frame_<product>_day<N>.parquet`: one lead-day's rows, with `fold`, the target, capacity, B0's
  columns and every arm's columns (`{fold}` columns are added by the fit script).
- `stage1_hours_<product>.parquet`: every daylight hour the stage-1 models predict.
- `positive_control_<product>_s<percent>.parquet`: the frames with power scaled after 2025-06-01.
- `losses_<product>.parquet`: per-row losses of every fit, with `actual` and `prediction` in the
  target's units (fractions of capacity in the `global` and `lopo` scopes).
- `checkpoints/`: one file per (scope, setting, arm), the stage-1 columns and predictions, the
  shortlist rule's X, and the saved 10% and 90% quantile predictions.
