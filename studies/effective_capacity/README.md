# Is a time-varying effective capacity worth estimating for solar forecasts?

Study for [issue #1143](https://github.com/openclimatefix/nged-substation-forecast/issues/1143).
Stage A plants known capacity losses in six solar plants' measured power and asks nine estimators to
find them, with no forecast in the loop. Stage B divides the lag-features study's baseline forecast
(B0) by the one causal estimator that a rule fixed in advance selects, and scores it in megawatts
over each plant's fixed 99th-percentile power. The design is in `plans/study-effective-capacity-1143.md`
and the results will be on the study's page under `docs/studies/forecasts/`.

**Every script takes `--smoke`, which reads and writes under `<study directory>/_smoke/` and is not a
result.** `capacity_paths.claim_root` marks each root `real` or `smoke` and raises if a run of the
other kind points at it, so a smoke checkpoint cannot be mistaken for a real one. No script
overwrites a finished output.

## The scripts

| Script | What it does |
|---|---|
| `capacity_paths.py`, `capacity_scores.py` | Imported by the others: paths, loaders, the tuning grids, and the conversion of saved estimates to issue-time scales. |
| `build_capacity_inputs.py` | Builds the day-by-hour record (cleaned power, CAMS and clear-sky irradiance, fitted orientations and the modelled unit output), the 11 draws of planted losses, the positive control's months, and the false-movement windows. |
| `estimate_capacity.py` | CPU. Runs R30, CSQ, NR, PR, KF, HM, CX1, and CX2 in the phases `tuning`, `main`, and `hindsight`. Checks that NR equals the lag frame's `rp_7d` and that every estimator is causal before the first estimate. |
| `estimate_capacity_torch.py` | GPU. Runs DP1 at both settings on every record. |
| `score_capacity_recovery.py` | Chooses each estimator's setting on the tuning draw (`--phase tuning`), then scores the selection draws, applies the false-movement gate, and writes `selection.json` (`--phase selection`). Fits no XGBoost model. |
| `fit_capacity_arms.py` | GPU. Runs the constant-scale assertion, the 90-fit reproduction of B0, the forecast-row join probe, and then every Stage B arm, with checkpoints. |
| `report_effective_capacity.py` | Prints every table and number the page quotes into `report.md`, with the explainability rubric and examples. |

## Run order

```bash
# 1. Inputs (a few minutes, CPU).
uv run python studies/effective_capacity/build_capacity_inputs.py
# 2. CPU slot: the estimators, tuning then main. At most 8 processes.
uv run python studies/effective_capacity/estimate_capacity.py --phase tuning --workers 8
uv run python studies/effective_capacity/score_capacity_recovery.py --phase tuning
uv run python studies/effective_capacity/estimate_capacity.py --phase main --workers 8
# 3. GPU slot 1: DP1, then the selection, then the hindsight estimates of the shortlist.
uv run python studies/effective_capacity/estimate_capacity_torch.py --device cuda
uv run python studies/effective_capacity/score_capacity_recovery.py --phase tuning --estimators DP1
uv run python studies/effective_capacity/score_capacity_recovery.py --phase selection
uv run python studies/effective_capacity/estimate_capacity.py --phase hindsight --workers 2
# 4. GPU slot 2 (with 4 cores): the forecast arms, then the report.
uv run python studies/effective_capacity/fit_capacity_arms.py --device cuda --max-workers 4
uv run python studies/effective_capacity/report_effective_capacity.py
```

## What the files hold

All under `data/studies/per_study/effective_capacity/`:

- `built/record.npz`: the record (power, the power with export-capped hours left in, irradiance,
  modelled unit output, static power). `built/orientation.json`: each plant's fitted tilt, azimuth,
  and capacities. `built/events.json`: the draws, the control, the real low months, and the
  false-movement windows. `built/manifest.json`: the lag frame's SHA-256.
- `estimates/<estimator>/<setting>/<record>.npz`: raw daily levels (and standard deviations) of each
  estimator on each record. A linear program estimates only the days a scoring needs; other days
  are NaN.
- `tuned/tuned_<estimator>.json`, `selection.json`, `tables/stage_a_*.parquet`: Stage A.
- `scales_stage_b.npz`, `arm_columns.json`, `losses_stage_b.parquet`, `checkpoints/`: Stage B.
  `losses_stage_b.parquet` holds each arm's per-row losses in fractions of the static power, the
  scale, and the scaled actual.
- `report.md`: every number the page quotes.
