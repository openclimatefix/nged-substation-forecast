# Leaderboard fold hygiene: a held-out final-test window

## Finding

The single leaderboard fold (`mid_2025_to_mid_2026` in `conf/cv/default.yaml`: train 2024-04 →
2025-06, validate 2025-07 → 2026-06) serves as **both** the model-selection set and the
reported skill number. Every hyperparameter choice, feature ablation, and model comparison is
adjudicated on the same 12 months that the leaderboard reports. With hundreds of planned
experiments (the roadmap mentions LLM-driven auto-experimentation in v0.5), the winner's
reported skill will be optimistically biased — classic leaderboard overfitting. The docs'
epoch mechanism handles *data* changes but not *adaptive selection* on a fixed fold.

## Implementation

Two parts: a cheap documentation fix now, and a structural fix at the next leaderboard epoch.

### 1. Document the caveat (immediately)

In `docs/ml_experimentation/cross-validation-folds.md` and the leaderboard section of
`docs/roadmap/metrics-and-leaderboard.md`: a short "Selection bias" subsection stating that
with a single fold, leaderboard metrics are selection metrics; differences smaller than
fold-level noise should not drive decisions; and the number of experiments per epoch is itself
a relevant statistic (visible as the MLflow experiment count).

### 2. Reserve a final-test window (next leaderboard epoch)

Found a new epoch in `conf/cv/default.yaml` (the epoch mechanism exists for exactly this):

- Shrink the leaderboard fold's validation window to `2025-07-01 → 2026-03-31`.
- Add a `final_test` fold `2026-04-01 → 2026-06-30` with a new per-fold flag
  `final_test: true` (extend `CvConfig` / the fold schema in
  `packages/contracts/src/contracts/hydra_schemas.py`; it is neither a leaderboard fold nor a
  dev fold — `_fold_ids_for_run_mode` in `defs/jobs.py:96` must *not* include it in any
  run mode, so no experiment trains or scores on it in the normal flow).
- Scoring against the final-test window is a deliberate, rare act — only for champion
  candidates immediately before promotion — via the `metrics` asset with
  `evaluation_scope="ad_hoc"` and the window's `valid_time` bounds in the existing
  `PopulationFilter`. No new asset needed; the discipline is procedural. Note: the model
  trained for the leaderboard fold is reused as-is (train window unchanged), so final-test
  scoring needs a `cv_power_forecasts` run over the reserved window — check whether the
  existing asset can forecast a window disjoint from the fold's `val_start/val_end`, and add a
  window override to its config if not.
- Rule, documented alongside: final-test results are never used to *choose between*
  candidates (that re-creates the problem); they exist to report honest skill for the chosen
  champion and to detect gross overfitting (final-test NMAE ≫ validation NMAE).

### 3. Trade-off note (for the PR description)

This costs 3 of the 12 validation months, on a dataset that is already short. The alternative
— accepting documented bias until Dynamical.org backfills enable multiple yearly folds — is
defensible; if the backfill is expected within a couple of months, do part 1 now and fold
part 2 into the multi-fold epoch instead of spending a separate epoch on it. Decide based on
the backfill outlook at implementation time.

## Verification

1. `register_experiment_job` in all three run modes never creates a partition for the
   `final_test` fold (extend `tests/test_register_experiment_job.py`).
2. Eligibility for the leaderboard fold is unchanged by the shrunk validation window (the
   eligibility rule keys off `val_start`/`val_end` — re-materialise `eligible_time_series` and
   diff).
3. End-to-end: score one existing experiment against the reserved window via the `ad_hoc`
   metrics path and confirm rows land in `forecast_metrics.delta` with the window label, and
   nothing is logged to the leaderboard MLflow runs.
