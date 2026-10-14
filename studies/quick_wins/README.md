# Quick wins for the production XGBoost forecast

Study for [issue #1150](https://github.com/openclimatefix/nged-substation-forecast/issues/1150). The
question is which cheap changes to the production XGBoost forecast lower its error on the
leaderboard fold (trained on April 2024 to June 2025 and scored on July 2025 to June 2026), using
only data the production system already holds (weather forecasts from the European Centre for
Medium-Range Weather Forecasts (ECMWF) ensemble (ENS), and power from National Grid Electricity
Distribution (NGED)). The results are on the study's page under
`docs/studies/forecasts/quick-wins.md`. The study changes no production code: every arm is a copy of
the production recipe with one named difference, fitted by the scripts here.

## The scripts

**Run the scripts in the order of the Commands block below.** The table also lists the modules those
scripts import. Outputs go to `data/studies/per_study/quick_wins/`, and a script skips any arm whose
scores already exist.

| Script | What it does |
|---|---|
| `quick_wins_inputs.py` | The folds, the 31 series, effective capacity, the production configuration, and a copy of the production feature loader (studies may not import the Dagster `src/` tree). |
| `build_frames.py` | Builds and caches the training frame (control member) and one validation frame per sampled run (all members) for a fold. |
| `fit_production_arm.py`, `report_production_arm.py` | Refit production with the production code path on CPU or GPU and score it against the stored `xgboost_baseline`, `manual_heuristic` and `climatology` on the sampled runs. This is the reproduction gate. |
| `cleaning_rules.py`, `extra_features.py`, `arms.py` | Round 0 cleaning rules, the extra feature columns, and the definition of every arm. |
| `fit_arms.py` | Fits named arms on a fold and seed, predicts every member of the sampled runs, and writes per-row scores. |
| `calibrate.py` | The post hoc spread rescaling (B1), using only outcomes known at each run's initialisation time. |
| `blend_heuristic.py` | A post hoc diagnostic: scores the XGBoost members mixed with the stored `manual_heuristic` members. |
| `report_quick_wins.py` | Month-resampled intervals and paired contrasts for every arm, printed into `report_*.md` and saved as parquet files. |
| `report_summary.py`, `report_compute.py` | Print every table the page quotes into `summary.md`, and the training-time table into `compute.md`. |
| `charts_summary.py`, `charts_examples.py`, `charts_diagnostics.py`, `charts_features.py` | The page's figures 1 to 18. |

## Commands

```bash
uv run python studies/quick_wins/build_frames.py
uv run python studies/quick_wins/fit_production_arm.py --device cuda
uv run python studies/quick_wins/report_production_arm.py
uv run python studies/quick_wins/fit_arms.py --device cuda --seed 0 --arms production_cleaned c4_absolute_error
uv run python studies/quick_wins/calibrate.py --arms production_cleaned
uv run python studies/quick_wins/report_quick_wins.py --seeds 0 1 2
```

Fold B (the backwards-in-time robustness fold) takes `--fold backwards_2025_07_to_2026_06` on
`build_frames.py`, `fit_arms.py`, `calibrate.py` and `report_quick_wins.py`.

## What the scores are

Every arm is scored on 53 ECMWF runs started every 7 days from the start of the validation year,
with `ml_core.metrics`' fair CRPS and the absolute error of the ensemble mean, and each row's error
is divided by its series' effective capacity. These are the study's own scores. They are not
leaderboard numbers, because the study did not submit to `score_study.py`. The script needs the row
keys of the reference experiment, which holds a run for every day of the validation year, and Quick
wins forecasts one run in 7 days. The script also cannot hold Quick wins' quantile arms, whose
`ensemble_member` codes run from 0 to 458, outside the Int8 range that `PowerForecast` uses.

Generators are never named: the scripts print series by type, and charts normalise power by
capacity.

## Diagnostic scripts and control arms

- `lead_band_contrasts.py` splits any arm's contrast against production by lead band.
- `quantile_check.py` writes `report/quantile_check.md`: coverage of the 459-value sample (51
  members times 9 quantile levels) and the CRPS of the control member's 9 quantiles alone. The
  script needs the samples saved by `fit_arms.py --save-quantile-samples`, which write about 2 GB.
- Arms `a1b_lags_removed_only` (only the four power lags removed) and `c1p_shuffled_copy_control`
  (production plus a shuffled copy of `temperature_2m`) are the lag-only ablation and the padded
  control for the lead-time column.
