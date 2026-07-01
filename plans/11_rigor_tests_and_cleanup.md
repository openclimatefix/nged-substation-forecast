# Remaining rigor tests and cleanup

*Merged from the Dagster ML-assets plan, Phase 9 — the final phase. Runs after plans 09 and 10.*

## Context

The feature-level no-lookahead tests, cross-mode equivalence test, idempotency tests, and the
full-stack cross-process MLflow test all exist. Three "not cheating" guardrail tests from the
original testing strategy remain unwritten, plus general cleanup.

## Part 1 — scientific-rigor tests

- **CV-windowing no-lookahead** (complements the feature-level tests, which cover lag leakage
  but not window construction): assert no *training* row has `valid_time >= val_start` for its
  fold — i.e. the training window built by `training_window(fold)` and applied in
  `trained_cv_model` never bleeds into validation.
- **Leaderboard fairness**: two different experiments over the same fold are scored on the
  **identical** `(time_series_id, fold)` population — a regression guard on the
  experiment-independence of `eligible_time_series`.
- **Determinism**: training a fold twice with a fixed `random_seed` yields identical
  predictions. This underpins idempotent retries and a stable leaderboard.

## Part 2 — cleanup

- Remove any remaining dead code/imports from the phased build-out.
- **Split `defs/cv_assets.py`** (898 lines — the complexity hotspot flagged in
  `00_review_findings.md`) into `cv_assets.py` / `production_assets.py` /
  `metric_assets.py`. Plan 09 already starts `production_assets.py`; move the `metrics` asset
  and its helpers into `metric_assets.py` here. Pure logic stays in `ml_core._cv_helpers`.

## Part 3 — docs freshness pass

The permanent-docs migration from the old `dagster_plan.md` is **already done** (July 2026):
`docs/architecture/ml-orchestration.md` (design decisions, rejected alternatives, cadence
limitation) and `docs/ml_experimentation/evaluating-new-data-sources.md` now capture its
important ideas, and the plan file has been deleted. What remains:

- Check `docs/` against the code after plans 09/10 land — in particular extend
  `docs/ml_experimentation/dagster-workflow.md` with the live-forecast and monitoring flows,
  and update the "Known limitation" and MLflow-logging notes in
  `docs/architecture/ml-orchestration.md` if the implementations diverged from the plans.
- Move any `docs/roadmap/` content that plans 09/10 implemented into the permanent docs —
  `docs/roadmap/` should only contain ideas not yet implemented in code (e.g. flip the
  relevant 🚧 statuses in `docs/roadmap/metrics-and-leaderboard.md` once monitoring lands).

## Verification

Full `uv run pytest` green from the repo root, **including the full-stack cross-process
integration test**; `uv run pymarkdown scan` (per CLAUDE.md) green on the touched docs;
`grep -ri 'dagster_plan' docs/ src/ packages/` returns nothing.
