# Codebase Review Findings — Index

A full review of the codebase (2026-07-01) covering architecture, MLops, scientific
correctness, and the AWS deployment plan
([#206](https://github.com/openclimatefix/nged-substation-forecast/issues/206)). Each finding
has its own implementation plan in this directory, worked through one PR at a time (per
`plans/README.md`, delete each file when its work lands). The old `dagster_plan.md` (phases
0–6.7 complete, PRs #182–#214) was absorbed into this set: its remaining phases 7–9 became
plans 02, 05, and 13, and its permanent design ideas moved to
`docs/architecture/ml-orchestration.md` and
`docs/ml_experimentation/evaluating-new-data-sources.md`.

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
- `XGBoostForecaster` trains only on `selected_features`, which omits `nwp_lead_time_hours` —
  the model never sees the forecast horizon despite the `horizon_as_feature` config tag.
- `.env` is **not** tracked by git (gitignored, no history) — no secrets exposure.
- `uv run pytest` from the repo root fails collection (duplicate `test_metrics.py` basenames).

## The plans, in implementation order

**Top priority is getting *any* v0.1 forecast running on AWS**
([#137](https://github.com/openclimatefix/nged-substation-forecast/issues/137), decided
2026-07-01); the scientific-improvement work (06 onwards) waits until it is live. The file
numbering *is* the implementation order:

| Plan | Work |
|---|---|
| [01_ci.md](01_ci.md) | CI safety net: ruff + ty + pytest on every PR (root pytest collection is currently broken by duplicate test basenames) |
| [02_live_forecasts.md](02_live_forecasts.md) | Production inference asset (live/replay NWP availability, 51 members, `fold_id="live"`) — built before its container so the container has something to run ([#208](https://github.com/openclimatefix/nged-substation-forecast/issues/208)) |
| [03_production_model_artifacts.md](03_production_model_artifacts.md) | Container + champion model baked into the image (the MLflow-in-ephemeral-Fargate hole in [#206](https://github.com/openclimatefix/nged-substation-forecast/issues/206)) |
| [04_aws_deployment.md](04_aws_deployment.md) | **Ship v0.1 on AWS** ([#137](https://github.com/openclimatefix/nged-substation-forecast/issues/137) / [#206](https://github.com/openclimatefix/nged-substation-forecast/issues/206)): S3-capable data paths ([#121](https://github.com/openclimatefix/nged-substation-forecast/issues/121), [#50](https://github.com/openclimatefix/nged-substation-forecast/issues/50)), production job with freshness check, Fargate runs + IAM-role infra, alerting; five architecture options costed 2026-07-02, leaning Option B (small always-on control-plane box + `EcsRunLauncher`) |
| [05_production_monitoring.md](05_production_monitoring.md) | `production_monitoring` metrics scope + `monitoring_sensor` + `retire_experiment_job` — once live forecasts are accumulating |
| [06_baseline_forecasters.md](06_baseline_forecasters.md) | Persistence + climatology baselines — without them leaderboard scores aren't interpretable |
| [07_probabilistic_evaluation.md](07_probabilistic_evaluation.md) | Horizon-sliced metrics, PICP/spread-skill/CRPS, then cheap ensemble calibration — the ensemble is likely underdispersed and nothing measures it |
| [08_reproducibility_stamping.md](08_reproducibility_stamping.md) | Git SHA + Delta table versions on every MLflow run — cheap, and improves every later experiment |
| [09_xgboost_quick_wins.md](09_xgboost_quick_wins.md) | Skill quick wins: [#145](https://github.com/openclimatefix/nged-substation-forecast/issues/145) sub-issues merged with review suggestions (init-time-anchored features, holidays, effective temperature, clear-sky-index upsampling, early stopping, …), best bang-for-the-buck first — led by the `nwp_lead_time_hours` discovery |
| [10_nwp_clip_logging.md](10_nwp_clip_logging.md) | NWP Int16 quantisation clips silently — count and surface clipped values ([#161](https://github.com/openclimatefix/nged-substation-forecast/issues/161)) |
| [11_leaderboard_fold_hygiene.md](11_leaderboard_fold_hygiene.md) | The single fold serves both model selection and reported skill — document the caveat, reserve a final-test window at the next epoch |
| [12_drop_hydra.md](12_drop_hydra.md) | Hydra's value (composition/sweeps) is unused — replace with importlib + pyyaml + pydantic |
| [13_rigor_tests_and_cleanup.md](13_rigor_tests_and_cleanup.md) | CV-windowing no-lookahead, leaderboard-fairness, and determinism tests; split `cv_assets.py`; docs freshness pass — the final phase |

## Findings with no plan (accepted / watch)

- **AWS deployment architecture.** The review initially endorsed #206's Level 1
  ("nothing always-on"), but a 2026-07-02 pressure-test found #206's always-on cost estimate
  inflated ~4× and the decision moved to leaning **Option B** (small EC2 control-plane box +
  `EcsRunLauncher`) — the full five-option cost analysis is in plan 04. Two findings stand
  regardless of option: Delta already provides the atomic-write property the freshness logic
  needs (make the forecast Delta commit the last write), and under any ephemeral/one-shot
  execution "which `ecmwf_ens` partitions to materialise" must be derived from Delta contents
  vs Dynamical availability, not Dagster's materialisation records.
- **Aggregate `mae__all`/`rmse__all`** are unweighted means across series spanning ~2 orders of
  magnitude of scale (GSPs dominate). NMAE is the headline metric; worth a sentence in
  `docs/roadmap/metrics-and-leaderboard.md` when next edited.
- **Global models (v0.5) will require target normalisation.** Raw-MW targets only work because
  models are per-series today. Covered as a prerequisite inside plan 09, item 16.
- **Patito friction budget.** CLAUDE.md documents four Patito-vs-Polars gotchas. No action now,
  but if a fifth workaround appears, consider validating only at I/O boundaries (typed
  annotations everywhere, `.validate()` only at persistence edges) or evaluate `dataframely`.
- **`cv_assets.py` (898 lines)** is the complexity hotspot; the split is scheduled in plans 02
  and 13.
