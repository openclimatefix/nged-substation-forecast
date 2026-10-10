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
| `lag_features_charts.py` | Draws the figures from those tables, and writes `figure_text_<product>.txt` for the text reviews first (`--text-only`). Each title states a finding taken from the saved tables, and the script raises if a table no longer supports it. A full render also writes svgo-optimised SVGs to `docs/studies/assets/lag_features_figure_<n>.svg` (`lag_features_followup_figure_<letter>.svg` for `--followups`). |

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

## The post hoc follow-ups

**The first science review asked for these re-runs and analyses, and every one is post hoc.** They
add to the first run and never change it: all outputs sit under
`<output-root>/ens_mean/followups/`, with their own checkpoints, manifest and report. Run the
scripts in this order, after the first run's four scripts.

```bash
uv run python studies/lag_features/followup_frames.py
uv run python studies/lag_features/fit_followups.py --dry-run
uv run python studies/lag_features/fit_followups.py --device cuda --max-workers 4
uv run python studies/lag_features/report_followups.py
uv run python studies/lag_features/lag_features_charts.py --followups --text-only
uv run python studies/lag_features/lag_features_charts.py --followups
```

`--smoke` on `fit_followups.py`, `report_followups.py` and `lag_features_charts.py --followups` runs
the real path on 20 seeded random rows of each plant's month with 20 boosting rounds, and writes
`_smoke` files.

| Script | What it does |
|---|---|
| `followup_frames.py` | Builds the control frames, the long-lead frames with the climatology columns, and the PC2 frame, with the anchor assertions and leak probes. |
| `fit_followups.py` | Fits the follow-up arms (4,800 fits: `--dry-run` prints the count by group) and scores the no-fit climatology blends. |
| `report_followups.py` | Prints the follow-up tables and the no-fit analyses into `report_followups_ens_mean.md`. |

**What the follow-ups add.**

- **Positive controls that can be passed.** The oracle O (B0 plus the true shift factor) and the
  arms L1, W7, Q30, TF, AN and PC in the month-level control at 5% and 10%, and a second control with
  plant-specific persistent steps (8 steps of 4 to 12 weeks per plant, from three months before the
  scored period) for O, L1, W7, Q30, TF and AN. "Recovers" is a 99% interval wholly below zero and
  the share of the oracle's gain recovered, not the 2% rule. B0 is fitted on every control frame
  too, which the 2,160 fits the review counted leave out (360 more fits). The month-level control
  also fits L1 (180 more).
- **Long leads** (lead-days 7, 10, 14): CL (B0 plus the out-of-fold climatology), W7+CL, N2 (refit),
  N2-3, the sensitivity setting at days 10 and 14, and the no-fit 50/50 blends of B0 and W7 with climatology.
- **PC2**: PC with a 2-day CAMS latency (windows days 2 to 8 and 2 to 31) at lead-day 1.
- **Fingerprint decomposition** (global scope, the first run's fleet-wide folds): G-ID+TF,
  G-FPnoCK (G-FP without CK), leave-one-plant-out of G-FPnoCK, and per-plant B0 on the fleet-wide
  folds. A pooled arm is 15 fits, so this group is 210 fits.
- **No-fit analyses**: every arm's gain over B0 on the 8 unselected months, split by era;
  per-quantile hit rates of B0 and L1; the count of exploratory intervals in each report; the
  hindsight per-plant-month scaling bound on the real data and the controls; and the month-cluster
  t-interval beside the bootstrap interval for P1 to P5.

**Notes on the long-lead nulls.** The first run's N2 came from a build whose draw order was not
fixed and cannot be regenerated, so the follow-ups refit N2 on the fixed seeded draw and the report
prints both realisations. N2 at long leads is not a pure null: it samples the plant's own same-hour
power from other months, later ones included, so it acts as a weak climatology. W7 against N2-3,
which has W7's width, is the fair test. CL and W7+CL read later months too, which a live service
would not have, so they are labelled "(uses later months)".

**The follow-up smoke runs need the first run's smoke outputs.** `fit_followups.py --smoke` and
`report_followups.py --smoke` read the first run's smoke losses, report and checkpoints, so first
run `build_lag_frame.py`, `followup_frames.py` and `fit_lag_arms.py --smoke` and its
`report_lag_features.py --smoke` into a scratch `--output-root`. A missing file raises with that
instruction.

**Choices the brief did not settle.**

- **R1s** needs no fit and is not scored on the controls, in the plan or here.
- **The climatology column** is a `{fold}` column: for a scored fold, a training row reads the
  median over the folds outside both its own and the scored fold. It uses other folds' months,
  including later ones, as the published climatology baseline does.
- **Overlapping steps** do not compound: the factor is one minus the shift while any step covers
  the hour.
- **The hindsight scale** is each plant-month's measured energy over its export-capped predicted
  energy (`actual + signed_error_capped_mw`), and the rescaled forecast is capped again.
- **The leak probe** runs on the 10% control of each kind, on lead-day 14 and on PC2, because the
  other frames share their code. The probe cuts PC2's CAMS at two days before the issue day.
- **The first report's interval count** reads the difference tables outside the planned contrasts
  and the controls.

**Files under `followups/`.** `frames/<name>_ens_mean.parquet`: the follow-up frames.
`build_report_followups_ens_mean.md`: their build report, with the realised shifted share per plant.
`checkpoints/`: one file per (scope, setting, arm) and `run_manifest.json`.
`losses_followups_ens_mean.parquet`: the per-row losses, including the blends.
`report_followups_ens_mean.md` and `tables_followups_ens_mean/`: the report and the tables the
figures read. `figures_followups_ens_mean/` and `figure_text_followups_ens_mean.txt`: the figures
and their text.

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
