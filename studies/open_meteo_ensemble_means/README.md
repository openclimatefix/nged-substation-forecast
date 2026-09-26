# Open-Meteo ensemble means compared for solar and wind power

Study for [issue #841](https://github.com/openclimatefix/nged-substation-forecast/issues/841). The
question is how well the ensemble-mean products that Open-Meteo serves for the MOGREPS-UK,
ICON-D2-EPS, ICON-EU-EPS, and ECMWF IFS ENS ensembles predict solar and wind power, compared with
CAMS (solar) and ERA5 (wind). The comparison is descriptive, over one 88-day summer window, and puts
no interval on any difference.

## Scripts

- `ensemble_means_mae.py` builds one row set for the six solar generators and one for the three
  wind generators, fits an XGBoost model per generator and arm with leave-one-week-out folds at two
  hyperparameter settings, and writes the tables and `report.md` under
  `data/studies/open_meteo_ensemble_means/`. Its module docstring states the design: the arms, the
  shared features, the rows, the folds, and the two wind designs. `--report-only` rebuilds the tables
  from the saved frames and losses.

Fits run on the CPU. The local ECMWF ENS mean takes its 00:00 UTC valid time from lead 24 of the
previous day's run, because the 00 UTC run's own leads start at 3 hours.

## Files written

- `frame_solar.parquet`, `frame_wind.parquet`: every arm's inputs and the power target, one row per
  anonymised generator-hour, with the fold.
- `losses_solar.parquet`, `losses_wind_10m.parquet`, `losses_wind_hub.parquet`: one row per scored
  generator-hour, seed, arm, and setting, with the signed and absolute error in megawatts and as a
  fraction of capacity.
- `mae_by_arm.parquet`, `mae_by_site.parquet`, `weather_error.parquet`: the tables the report prints.
- `run_facts.json`: the row funnel and the count of overlap rows on which an original download and
  its refresh differ.
