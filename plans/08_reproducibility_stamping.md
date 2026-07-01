# Reproducibility: stamp git SHA and Delta table versions on MLflow runs

## Finding

Experiments log config, class targets, hyperparams, and window bounds to MLflow — but no git
commit and no data versions (verified: nothing in `defs/jobs.py` or `ml_core/_mlflow_runs.py`).
A run ID today cannot answer "exactly which code and which data produced this model?". Delta
Lake's time travel makes data versioning nearly free — logging the table version at read time
buys full data reproducibility for one integer per table. (MLflow *can* auto-set
`mlflow.source.git.commit`, but only when gitpython is installed and CWD is in the repo —
neither holds in a production container, so stamp explicitly.)

## Implementation

### 1. Git identity helper

New module `packages/ml_core/src/ml_core/_repro.py`:

- `get_git_info() -> dict[str, str]` returning `{"git_sha": ..., "git_dirty": "true"/"false"}`
  via `subprocess` (`git rev-parse HEAD`, `git status --porcelain`). Return
  `{"git_sha": "unknown", ...}` instead of raising when not in a git repo (containers) —
  callers must never fail because of provenance.
- `get_delta_versions(paths: dict[str, Path]) -> dict[str, str]` returning
  `{f"delta_version__{name}": str(DeltaTable(path).version())}` per table (`deltalake` is
  already a dependency). Missing table → `"absent"`, again never raising.

### 2. Stamp at registration

In `register_experiment` (`src/nged_substation_forecast/defs/jobs.py:108`), alongside the
existing experiment tags (`config`, `forecaster_target`, …): set `git_sha` / `git_dirty` tags
on the **parent run** in the existing `with mlflow.start_run(run_id=parent_run_id)` block
(`jobs.py:134-141`).

### 3. Stamp at training and prediction time

Registration SHA ≠ training SHA (folds can be trained days later on different code), so the
fold-level stamp is the load-bearing one:

- In the `trained_cv_model` asset (`defs/cv_assets.py`), where training params are already
  logged to the fold run: add `git_sha`/`git_dirty` tags plus
  `get_delta_versions({"power_time_series": ..., "nwp_data": ..., "eligible_time_series": ...})`
  using the paths from `Settings`. Log versions as tags (they're provenance, not metrics).
- In `cv_power_forecasts`, same stamps (prediction may run on yet another SHA/data state) —
  the forecast-metadata logging block is the natural place.

### 4. Surface in docs

One short paragraph in `docs/ml_experimentation/dagster-workflow.md`: what is stamped, where,
and how to reconstruct a run (`git checkout {sha}` + `pl.scan_delta(..., version=N)`).

## Verification

1. Unit tests for `_repro.py`: git info in a temp repo; `unknown` outside one; Delta version of
   a freshly written temp table is `0` and increments on append.
2. Extend `tests/test_register_experiment_job.py` to assert the parent run carries `git_sha`.
3. Run the smoke-test fold end-to-end and confirm the fold run shows git + Delta-version tags
   in the MLflow UI.
4. Round-trip check: `pl.scan_delta(power_time_series_path, version=<logged>)` returns the
   training-time state after a subsequent append.
