# What is on disk

**Everything an experiment reads lives under one `data/` folder, which every git worktree shares.**
This page lists each folder, who writes it, and how to read it. Nothing under `data/` is committed.
The folder is private: the NGED hierarchy files and the experiments' predictions must not leave
the machine.

## One shared folder

- **`studies.sources.REPO_DATA_DIR` points at the main checkout's `data/` from any worktree.**
  Set the `DATA_PATH_INTERNAL` environment variable to override it. A remote URI is not supported
  in these scripts (disk).
- **Only one agent may re-run a published study script at a time.** All worktrees share the folder,
  so a second agent's re-run overwrites the first agent's evidence (disk).
- **Name the path through the constants in `studies.sources`**: `STUDIES_DATA_DIR`,
  `ML_EXPERIMENTS_DATA_DIR`, `DOWNLOADS_DIR`, and the four download folders below (disk).

## Production outputs

These are written by the Dagster application, not by experiments.

| Folder | Holds | How to read |
|---|---|---|
| `data/NGED/power_time_series.delta` | Raw power telemetry, appended from NGED's S3 bucket | See [NGED power telemetry](nged-power-telemetry.md#raw-and-cleaned-tables) |
| `data/NGED/cleaned_power_time_series.delta` | The raw rows plus a `drop_reason` column | `studies.power.scan_power` |
| `data/NGED/metadata.parquet` | The `TimeSeriesMetadata` table | `pl.read_parquet` |
| `data/NGED/list_of_downloaded_files.parquet` | Which NGED files the ingest has downloaded | Ingest bookkeeping |
| `data/NWP/` | The ECMWF ENS Delta table, from 2024-04-01 | `delta_store.nwp`; see [weather inputs](weather-inputs-by-product.md#ecmwf-ens) |
| `data/h3_grid_weights.parquet` | Fractional H3 cell overlap with the Great Britain boundary | Spatial NWP aggregation |
| `data/effective_capacity/` | The effective capacity per series (99th percentile of absolute power) | `EffectiveCapacity` contract |
| `data/eligible_time_series/` | Which series are eligible per cross-validation fold | Delta, partitioned by `fold_id` |
| `data/power_forecasts/`, `data/forecast_metrics/` | Stored cross-validation forecasts and metrics for `manual_heuristic`, `climatology`, and `xgboost_baseline` | Delta, partitioned by `experiment_name` |
| `data/production_model/` | The promoted production model | `BaseForecaster.load` |

- **The hierarchy files sit in `data/NGED/substation_hierarchy_2026-10-15/`.** The folder holds
  `Substations.csv`, `Primary Substation Interconnections.csv`, `Monitor Direction.csv`,
  `Peak Loads.xlsx`, `Switching Logs.xlsx`, and a `README.md`. They are private. The folder's
  README says the switching logs and peak loads were not examined when it was written (disk).

## Studies

`data/studies/` holds the one-off studies' downloads and built frames, in four download folders under
`data/studies/downloads/`.

| Folder | Holds |
|---|---|
| `NWP/` | ECMWF AIFS and AIFS-ENS, IFS single runs, GEFS, GFS, UKV from the Centre for Environmental Data Analysis (CEDA) and from AWS, WeatherNext 3, Open-Meteo previous runs and ensemble means |
| `reanalysis/` | CAMS, CAMS aerosol, CERRA, ERA5 (and a separate ERA5 wind folder), ICON-DREAM-EU, NORA3 |
| `observations/` | MIDAS Open weather stations, NGED active network management (ANM) curtailment, SARAH-3 satellite irradiance |
| `market/` | About 30 folders of Elexon, NESO, Octopus Agile, carbon-intensity, and PV_Live data |

- **Check `data/studies/downloads/` before fetching a product again.** A re-fetch or an egress that
  costs over £30 needs the maintainer's approval first (disk).
- **Studies write their built frames under `data/studies/<study>/`**, and private tables under
  `data/studies/_private/`. Neither goes to git (disk).

## Experiments

- **Each experiment writes to `data/ml-experiments/<id>-<slug>/`**, created by
  `experiment_store.reserve_experiment`. About 200 folders exist (disk).
- **Predictions live at `predictions/<name>/` inside that folder**, as `predictions.parquet` and a
  `manifest.json`, written by `experiment_store.write_predictions` and read by
  `experiment_store.read_predictions`. The reader raises when the row set differs (disk).
- **Other subfolders vary by experiment** (`work/`, `logs/`, `briefs/`, tables). The DP data look
  of experiment 103 stores `rows.parquet`, `sites.parquet`, `series_table.parquet`,
  `recommended_cut_dates.csv`, and `decisions.json` there (disk).
- **`data/ml-experiments/briefs/` holds the manager's private briefs** and is not part of any
  experiment (disk).

## Disk space

- **`data/` sits on the second SSD:** the main checkout's `data/` is a symbolic link to
  `/mnt/data/nged-substation-forecast/data` (disk)

Last checked against experiment 183.
