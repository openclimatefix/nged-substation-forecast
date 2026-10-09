# Embedded-battery forecast

Scripts for the study of how well the distribution of an embedded battery's next-day output can be
predicted. The plan is `plans/embedded-battery-forecast.md`. Every output sits under
`data/studies/per_study/embedded_battery_forecast/` (the folder `studies.sources` names), never in
the repository.

## Scripts, in the order they run

| Script | What it does | Writes |
|---|---|---|
| `census.py`, `census_report.py` | Matches NGED's Embedded Capacity Register to the Elexon BMU register and classifies each connected battery | `census_matches.csv`, `census_classes.parquet`, `census_report.md` |
| `forecast_inputs.py` | Builds, for each battery and issue time, the wide frame of target, prices, persistence value, climatology, and Physical Notification columns | `inputs/<issue>__<battery>.parquet`, `inputs/inputs_report.md` |
| `forecast_synthetic.py` | Rung A0: 40 synthetic batteries that check the instrument finds a price effect and stays silent without one | `inputs/synthetic/`, `fits_as_written/<setting>/A0_DA-late/`, `a0_report_<setting>.md` |
| `forecast_price_model.py` | The price model: a forecast of the N2EX price available at 06:00 UTC | `price_model_by_fold.parquet`, `price_model_report.md` |
| `forecast_bmus.py` | Part A, rungs A1 to A5: the 35 testbed battery BMUs | `fits_as_written/<setting>/<issue>/<battery>__<arm>.parquet` |
| `forecast_nged_battery_a.py` | Part B, rungs B1 and B2: NGED battery A | the same, under the battery name `nged_battery_a` |
| `forecast_report.py` | Prints every table the page quotes | `forecast_report.md`, `report_tables/*.parquet` (and, for the other variants, the same names with `_idle_dropped` or `_pre_review` added) |

`forecast_fit.py` holds the out-of-fold fit and score loop, `forecast_arms.py` the list of arms,
`forecast_runner.py` the per-battery job, and `forecast_results.py` the loader that reads the
saved fits back.

## Three sets of fits

Each set is a folder of the same layout, chosen by the `FIT_VARIANT` environment variable.

| `FIT_VARIANT` | Folder | What it holds |
|---|---|---|
| `pre_review` | `fits/` | The fits made before the first science review. Their price features treated the whole UTC day as published. The report reads them only to show the size of that error, and nothing writes to them. |
| unset (`as_written`) | `fits_as_written/` | The plan's rule, every half-hour with its inputs, with price features that use only the prices public at each issue time. Files for arms the change leaves as they were are links to `fits/`. |
| `idle_dropped` | `fits_idle_dropped/` | `as_written` with each battery's idle lead-in dropped from training and scoring. Only the batteries with a lead-in are refitted; every other file is a link to `fits_as_written/`. |

## What a saved fit holds

Each file `fits*/<setting>/<issue>/<battery>__<arm>.parquet` has one row per scored half-hour and
fitting seed, with the half-hour's start (`time`), `month`, `fold`, `seed`, `truth_mw`, `p99_mw`
(the battery's 99th-percentile absolute output), the 13 forecast quantiles `q0.01` to `q0.99` in
megawatts, and the per-row losses `crps_pct`, `pinball_pct`, and `median_abs_error_pct` as a
percentage of `p99_mw`. NGED battery A is stored as a fraction of its own 99th-percentile absolute
output, so its `p99_mw` is about 1 and the file carries no megawatts. The arms that hold no random
draw (`clim`, `persistence_conformal`, and `rank_conformal__*`) repeat their single result under
all three seed labels.

## Commands

```bash
export OMP_NUM_THREADS=2
uv run python studies/embedded_battery_forecast/forecast_inputs.py
uv run python studies/embedded_battery_forecast/forecast_synthetic.py primary
uv run python studies/embedded_battery_forecast/forecast_price_model.py
uv run python studies/embedded_battery_forecast/forecast_bmus.py primary
uv run python studies/embedded_battery_forecast/forecast_bmus.py sensitivity
uv run python studies/embedded_battery_forecast/forecast_nged_battery_a.py primary
uv run python studies/embedded_battery_forecast/forecast_nged_battery_a.py sensitivity
FIT_VARIANT=idle_dropped uv run python studies/embedded_battery_forecast/forecast_bmus.py primary
FIT_VARIANT=idle_dropped uv run python studies/embedded_battery_forecast/forecast_bmus.py sensitivity
FIT_VARIANT=pre_review uv run python studies/embedded_battery_forecast/forecast_report.py
FIT_VARIANT=idle_dropped uv run python studies/embedded_battery_forecast/forecast_report.py
uv run python studies/embedded_battery_forecast/forecast_report.py
```

The three report runs go in that order, because the unsuffixed report prints the planned
contrasts of the other two beside its own. Rerun `forecast_inputs.py` only when no fit is
running, because it rewrites the input frames the fits read.
