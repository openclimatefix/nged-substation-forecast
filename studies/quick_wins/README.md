# Quick wins for the production XGBoost forecast

Study for [issue #1150](https://github.com/openclimatefix/nged-substation-forecast/issues/1150).
The question is which cheap changes to the production XGBoost forecast lower its error on the
leaderboard fold, using only data the production system already holds (ECMWF ENS weather forecasts
and NGED power). The results are on the study's page under `docs/studies/forecasts/quick-wins.md`.
The study changes no production code: every arm is a copy of the production recipe with one named
difference, fitted by the scripts here.

## The scripts

**Run the scripts in this order.** Outputs go to `data/studies/per_study/quick_wins/`, and a script
skips any arm whose scores already exist.

| Script | What it does |
|---|---|
| `quick_wins_inputs.py` | The folds, the 31 series, effective capacity, the production configuration, and a copy of the production feature loader (studies may not import the Dagster `src/` tree). |
| `build_frames.py` | Builds and caches the training frame (control member) and one validation frame per sampled run (all members) for a fold. |
| `fit_production_arm.py`, `report_production_arm.py` | Refit production with the production code path on CPU or GPU and score it against the stored `xgboost_baseline`, `manual_heuristic` and `climatology` on the sampled runs. This is the reproduction gate. |
| `cleaning_rules.py`, `extra_features.py`, `arms.py` | Round 0 cleaning rules, the extra feature columns, and the definition of every arm. |
| `fit_arms.py` | Fits named arms on a fold and seed, predicts every member of the sampled runs, and writes per-row scores. |
| `calibrate.py` | The post hoc spread rescaling (B1), using only outcomes known at each run's initialisation time. |
| `report_quick_wins.py` | Month-resampled intervals and paired contrasts for every arm, printed into `report_*.md` and saved as parquet files. |
| `charts_summary.py`, `charts_examples.py` | The page's figures. |

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
leaderboard numbers: the study did not submit to `score_study.py`.

Generators are never named: the scripts print series by type, and charts normalise power by
capacity.
