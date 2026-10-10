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

The fit script needs a compute slot. `--smoke` runs the real path on 20 seeded random rows of each
plant's month with 20 boosting rounds, writes `checkpoints_smoke/` and
`losses_<product>_smoke.parquet`, and is not a result. `report_lag_features.py --smoke` and
`lag_features_charts.py --smoke` read its output.

## Choices the plan did not settle

- **The lag source** is NGED's hourly power with the multi-day zero runs, meter spikes, commissioning
  ramp and export-capped hours removed. The target is the shared rows' `power_mw`.
- **Only the reproduction check reads rows at or after `final_test_start` (2026-07-01).** It refits
  the ENS-mean B0 at lead-day 1 on the study's device and all 35,263 shared rows, including the
  5,144 from that date, because the published number includes them, and asserts a mean absolute
  error within 0.03 points of the published 8.771%. B0 reads no power. **Nothing else at or after
  the date reaches a fit.** `studies.power.scan_power` stops there, the builder drops the shared
  rows and the stage-1 hours from that date, and the build and fit scripts assert that no training
  row is at or after it. The study has 18 usable
  months.
  Phase 1 screens 2024-12 to 2025-09 (10 months). P2 is scored on 2025-10 to 2026-06 without
  2026-01 (8 months).
- **CAMS's latency.** Since February 2026 CAMS's point service has a latency of one day (see
  [CAMS: use the point
  API](https://openclimatefix.github.io/nged-substation-forecast/roadmap/data-sources/#cams-use-the-point-api-not-the-gridded-product)),
  so at 09:00 UTC the whole previous day is available. PC's windows therefore cover days 1 to 7
  (power to satellite) and days 1 to 30 (satellite to forecast) before the issue day, and the leak
  probe fails a window that reads the issue day. Before February 2026 the real latency was longer,
  so the backtest reads CAMS slightly fresher than the service then offered.
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
- **The arms with `{fold}` columns** are S2, S3 and KS; stage 1 uses the
  primary setting and seed 0.
- **Positive control**: power is scaled by one minus the shift in exactly 9 of the 18 scored months,
  chosen by a seeded permutation, and in a seeded random half of the other calendar months (2019-09
  to 2026-06), so no tree can learn the shift from the date. The library fits all three seeds.
- **The interval figure and R4** use the nine saved quantiles of B0 and L1 (`intervals__B0`,
  `intervals__L1` checkpoints), from seed 0 only, sorted within each row.
- **Feature importance** comes from a separate refit (`importance__<arm>` checkpoints) of B0, L1, L2,
  S2 and X at the primary setting with seed 0, per plant and fold, point model. It is descriptive and
  enters no planned contrast.
- **The global X** skips arms whose columns carry a `{fold}` placeholder (S3 and KS), because stage-1
  columns withhold the scored fold at one plant only.
- **X's quantile run refits X's point model**, because `fit_one_fold` always fits both; the
  quantile run's point losses replace the sweep's for that arm.
- **Per-plant jobs are batched**: up to four arms share one `run_all` call, so the four workers stay
  busy across arms, and each arm is still checkpointed alone.

## What each file holds

- `lag_frame_<product>_day<N>.parquet`: one lead-day's rows, with `fold`, the target, capacity, B0's
  columns and every arm's columns (`{fold}` columns are added by the fit script).
- `stage1_hours_<product>.parquet`: every daylight hour the stage-1 models predict.
- `positive_control_<product>_s<percent>.parquet`: the frames with power scaled by one minus the
  shift in 9 of the 18 scored months and a random half of the other months.
- `losses_<product>.parquet`: per-row losses of every fit, with `actual` and `prediction` in the
  target's units (fractions of capacity in the `global` and `lopo` scopes).
- `checkpoints/`: one file per (scope, setting, arm), the stage-1 columns and predictions, the
  shortlist rule's X, the nine saved quantiles of B0 and L1, the importance refit's gain shares, the
  stage-1 anchor probe's result, and `run_manifest.json` (the device, the mode and a SHA-256 of
  every built frame, checked on a resume).
