# Codebase Review Findings — Index

A full review of the codebase (2026-07-01) covering architecture, MLops, scientific
correctness, and the AWS deployment plan (issue #206). Each finding below has its own
implementation plan in this directory, intended to be worked through one PR at a time
(per `plans/README.md`, delete each file when its work lands).

## Overall verdict

Unusually well-engineered for its stage: ~7k SLOC of source, disciplined lazy-Polars memory
management, careful and *verified-correct* lookahead-bias prevention, and honest docs. The main
risks are scientific (ensemble calibration, leaderboard hygiene), not engineering — plus one
glaring tooling gap (no CI) and one real hole in the AWS plan (the model-artifact story for
ephemeral containers).

Verified during review (not just sub-agent claims):

- `_nullify_leaky_lags` and the dual-strategy weather-lag join are leak-free. Past target times
  can only be covered by NWP runs initialised at or before them, so the freshest-run join
  cannot see the future. The cross-mode equivalence test pins the right invariant.
- Training uses only ensemble member 0 (`cv_assets.py:373`); metrics average all members into a
  deterministic mean (`metrics.py:84-90`); only the `"all"` horizon slice is computed
  (`metrics.py:51`).
- No baseline forecaster exists anywhere (only docstring mentions).
- No git SHA or data version is stamped on MLflow runs.
- `.env` is **not** tracked by git (gitignored, no history) — no secrets exposure.
- `uv run pytest` from the repo root fails collection (duplicate `test_metrics.py` basenames).

## The plans, in priority order

| Plan | Finding |
|---|---|
| [01_ci.md](01_ci.md) | No CI runs lint/types/tests; root pytest collection is broken |
| [02_baseline_forecasters.md](02_baseline_forecasters.md) | Leaderboard has no persistence/climatology baseline, so scores aren't interpretable |
| [03_probabilistic_evaluation.md](03_probabilistic_evaluation.md) | Ensemble is likely underdispersed and nothing measures it; only "all" horizon slice scored |
| [04_reproducibility_stamping.md](04_reproducibility_stamping.md) | Runs carry no git SHA or Delta table versions |
| [05_production_model_artifacts.md](05_production_model_artifacts.md) | AWS plan (issue #206) has no MLflow/model-artifact story for ephemeral Fargate |
| [06_nwp_clip_logging.md](06_nwp_clip_logging.md) | NWP Int16 quantisation clips out-of-range values silently |
| [07_drop_hydra.md](07_drop_hydra.md) | Hydra is a fourth config system whose value (composition/sweeps) is unused |
| [08_leaderboard_fold_hygiene.md](08_leaderboard_fold_hygiene.md) | The single fold serves both model selection and reported skill — leaderboard overfitting risk |

## Merged from the Dagster ML-assets plan (phases 7–9)

The pre-review `dagster_plan.md` (phases 0–6.7 complete, PRs #182–#214) was absorbed into this
set and deleted; its permanent design ideas now live in
`docs/architecture/ml-orchestration.md` and
`docs/ml_experimentation/evaluating-new-data-sources.md`. Its three remaining phases:

| Plan | Work |
|---|---|
| [09_live_forecasts.md](09_live_forecasts.md) | Production inference asset (live/replay NWP availability, 51 members, `fold_id="live"`) — the asset plan 05's container runs |
| [10_production_monitoring.md](10_production_monitoring.md) | `production_monitoring` metrics scope + `monitoring_sensor` + `retire_experiment_job` (after 09) |
| [11_rigor_tests_and_cleanup.md](11_rigor_tests_and_cleanup.md) | CV-windowing no-lookahead, leaderboard-fairness, and determinism tests; split `cv_assets.py`; docs freshness pass (after 09+10) |

## Final step: forecast-skill quick wins

| Plan | Work |
|---|---|
| [12_xgboost_quick_wins.md](12_xgboost_quick_wins.md) | The issue #145 sub-issues ordered by skill-per-effort — starting with a review discovery: the model never sees `nwp_lead_time_hours` despite the `horizon_as_feature` config tag. Best run after 02 and 03 Phase A so wins are measurable |

## Findings with no plan (accepted / watch)

- **AWS deployment Level 1 is the right call.** Delta already provides the atomic-write property
  the plan worries about (make the forecast Delta commit the last write). Dagster partition
  bookkeeping evaporates with throwaway SQLite, so "which `ecmwf_ens` partitions to materialise"
  must be derived from Delta contents vs Dynamical availability — covered in plan 05's notes.
- **Aggregate `mae__all`/`rmse__all`** are unweighted means across series spanning ~2 orders of
  magnitude of scale (GSPs dominate). NMAE is the headline metric; worth a sentence in
  `docs/roadmap/metrics-and-leaderboard.md` when next edited.
- **Global models (v0.5) will require target normalisation.** Raw-MW targets only work because
  models are per-series today. Note in the roadmap when v0.5 planning starts.
- **Patito friction budget.** CLAUDE.md documents four Patito-vs-Polars gotchas. No action now,
  but if a fifth workaround appears, consider validating only at I/O boundaries (typed
  annotations everywhere, `.validate()` only at persistence edges) or evaluate `dataframely`.
- **`cv_assets.py` (898 lines)** is the complexity hotspot; consider splitting when next
  touched, but the design itself is a defensible price for training-serving symmetry.
