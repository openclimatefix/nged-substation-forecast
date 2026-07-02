# Engineering health

> **Status: 🚧 Planned.** Tooling, reproducibility, and rigour improvements from the 2026-07
> codebase review that don't change forecast behaviour. Each section is an independent,
> roughly one-PR piece of work; sections are deleted as they ship. Task ordering lives in the
> GitHub Project board (see the [map](index.md#map-of-substantial-work)).

## CI: run lint, types, and tests on every PR

The only GitHub workflows are `docs.yml` (MkDocs deploy) and `contribution-bot.yml`. Code
quality is enforced solely by local pre-commit hooks, so a contributor (or an agent) who skips
hooks can merge broken code. Additionally, `uv run pytest` from the repo root currently
**fails collection**: `tests/test_metrics.py` and `packages/ml_core/tests/test_metrics.py`
share a module basename and there are no `__init__.py` files in the test dirs, so pytest's
default import mode refuses to collect both. Nobody noticed because nothing runs the full
suite from the root.

This is the highest-value fix in the whole review: a few hours of work, permanent payoff, and
a prerequisite for trusting every other piece of planned work.

### Implementation details — CI (deleted when it ships)

**1. Fix root pytest collection.** In `pyproject.toml` `[tool.pytest.ini_options]` add:

```toml
addopts = "--import-mode=importlib"
```

`importlib` mode allows duplicate test-file basenames without `__init__.py` files. Verify
`uv run pytest --collect-only -q` succeeds from the root (~165 tests) before touching CI.
If any test breaks under importlib mode, fix that test rather than renaming files.

**2. Add `.github/workflows/ci.yml`.** Trigger: `pull_request` + `push` to `main`. Single job
on `ubuntu-latest`:

```yaml
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true
```

Steps:

1. `actions/checkout@v4`
2. `astral-sh/setup-uv@v5` with `enable-cache: true`
3. `uv sync --all-packages --dev` (workspace root alone doesn't install package test deps)
4. `uv run ruff check .`
5. `uv run ruff format --check .`
6. `uv run ty check`
7. `uv run pymarkdown scan -r docs README.md CLAUDE.md metadata/README.md packages/*/README.md`
   (same command as CLAUDE.md / pre-commit)
8. `uv run pytest` — with dummy env for the required `Settings` fields, since a few tests
   construct `Settings()` directly (`tests/test_trained_cv_model.py:186`) and rely on the
   local `.env` which CI doesn't have:

   ```yaml
   env:
     NGED_S3_BUCKET_URL: "https://example.com"
     NGED_S3_BUCKET_ACCESS_KEY: "dummy"
     NGED_S3_BUCKET_SECRET: "dummy"
   ```

   (Most tests already monkeypatch these — see `tests/test_cv_assets.py:80` — the job-level
   env just covers the stragglers.)

Notes:

- Python version comes from `requires-python` / `.python-version` via uv; no setup-python
  needed.
- The full-stack MLflow-server test (`tests/test_metrics.py`) spawns a real
  `mlflow server` subprocess — `mlflow` (full) is already in the dev group, so it should run
  in CI. If it proves flaky on runners, gate it behind `-m integration` and split into a
  separate non-required job, but try it as-is first.

**3. Branch protection.** After the workflow is green on `main`, mark the CI job as a required
status check in the GitHub repo settings (manual step, note it in the PR description).

**Verification.** (1) `uv run pytest` from the repo root passes locally after step 1. (2) Open
the PR; the CI workflow runs and passes. (3) Push a deliberately broken commit to the PR
branch (e.g. an unused import) and confirm the workflow fails, then revert.

## Reproducibility: stamp git SHA and Delta table versions on MLflow runs

Experiments log config, class targets, hyperparams, and window bounds to MLflow — but no git
commit and no data versions (verified: nothing in `defs/jobs.py` or `ml_core/_mlflow_runs.py`).
A run ID today cannot answer "exactly which code and which data produced this model?". Delta
Lake's time travel makes data versioning nearly free — logging the table version at read time
buys full data reproducibility for one integer per table. (MLflow *can* auto-set
`mlflow.source.git.commit`, but only when gitpython is installed and CWD is in the repo —
neither holds in a production container, so stamp explicitly.)

### Implementation details — stamping (deleted when it ships)

**1. Git identity helper.** New module `packages/ml_core/src/ml_core/_repro.py`:

- `get_git_info() -> dict[str, str]` returning `{"git_sha": ..., "git_dirty": "true"/"false"}`
  via `subprocess` (`git rev-parse HEAD`, `git status --porcelain`). Return
  `{"git_sha": "unknown", ...}` instead of raising when not in a git repo (containers) —
  callers must never fail because of provenance.
- `get_delta_versions(paths: dict[str, Path]) -> dict[str, str]` returning
  `{f"delta_version__{name}": str(DeltaTable(path).version())}` per table (`deltalake` is
  already a dependency). Missing table → `"absent"`, again never raising.

**2. Stamp at registration.** In `register_experiment`
(`src/nged_substation_forecast/defs/jobs.py:108`), alongside the existing experiment tags
(`config`, `forecaster_target`, …): set `git_sha` / `git_dirty` tags on the **parent run** in
the existing `with mlflow.start_run(run_id=parent_run_id)` block (`jobs.py:134-141`).

**3. Stamp at training and prediction time.** Registration SHA ≠ training SHA (folds can be
trained days later on different code), so the fold-level stamp is the load-bearing one:

- In the `trained_cv_model` asset (`defs/cv_assets.py`), where training params are already
  logged to the fold run: add `git_sha`/`git_dirty` tags plus
  `get_delta_versions({"power_time_series": ..., "nwp_data": ..., "eligible_time_series": ...})`
  using the paths from `Settings`. Log versions as tags (they're provenance, not metrics).
- In `cv_power_forecasts`, same stamps (prediction may run on yet another SHA/data state) —
  the forecast-metadata logging block is the natural place.

**4. Surface in docs.** One short paragraph in
`docs/ml_experimentation/dagster-workflow.md`: what is stamped, where, and how to reconstruct
a run (`git checkout {sha}` + `pl.scan_delta(..., version=N)`).

**Verification.** (1) Unit tests for `_repro.py`: git info in a temp repo; `unknown` outside
one; Delta version of a freshly written temp table is `0` and increments on append. (2) Extend
`tests/test_register_experiment_job.py` to assert the parent run carries `git_sha`. (3) Run
the smoke-test fold end-to-end and confirm the fold run shows git + Delta-version tags in the
MLflow UI. (4) Round-trip check: `pl.scan_delta(power_time_series_path, version=<logged>)`
returns the training-time state after a subsequent append.

## NWP quantisation: count and surface clipped values

`NwpOnDisk.from_nwp_in_memory` (`packages/contracts/src/contracts/weather_schemas.py:390`)
clips each variable to its buffered range before Int16 encoding:

```python
clipped_col = pl.col(col_name).clip(lower_bound=buffered_min, upper_bound=buffered_max)
```

The buffered ranges carry only a 5% margin over the min/max observed when
`compute_scaling_params` ran. If ECMWF ENS later produces values outside that range (extreme
storm winds, record pressure lows), they are silently flattened to the boundary — corrupting
exactly the extreme-weather inputs that matter most for peak forecasting, with no signal that
it happened.

Related: [#161](https://github.com/openclimatefix/nged-substation-forecast/issues/161) asks
for more Dagster-UI metrics and validation checks on NWP ingestion — the clip counts here are
the highest-value of those checks; close or trim #161 accordingly when this lands.

### Implementation details — clip logging (deleted when it ships)

**1. Pure counting helper in contracts.** New function in `weather_schemas.py` (near
`from_nwp_in_memory`):

```python
def count_out_of_range(
    nwp_in_memory: pt.DataFrame[NwpInMemory],
    scaling_params: pt.DataFrame[NwpScalingParams],
) -> dict[str, int]:
    """Per-variable count of values outside the buffered scaling range (i.e. values that
    from_nwp_in_memory would clip)."""
```

One `select` building `((col < buffered_min) | (col > buffered_max)).sum()` per scaling-params
row (mirror the loop structure at `weather_schemas.py:380-398`), returning only non-zero
entries. Keep `from_nwp_in_memory` itself pure/unchanged — counting is the caller's concern,
and the asset wants the numbers for metadata anyway.

**2. Call it in the `ecmwf_ens` asset.** In `src/nged_substation_forecast/defs/assets.py`,
where the day's `NwpInMemory` frame is converted via `from_nwp_in_memory`:

- Call `count_out_of_range` on the same frame.
- If any count > 0: `context.log.warning(...)` naming the variables and counts.
- Always attach the dict to the asset materialisation metadata (e.g.
  `clipped_values_total` plus per-variable entries), so drift is visible in the Dagster UI
  timeline even at zero.

Cost note: one extra aggregation pass over the day's frame (~7M rows) — negligible next to the
download.

**3. Docs.** One paragraph in the NWP section of `docs/roadmap/data-sources.md` (or wherever
the quantisation scheme is described): clipping is monitored, and persistent non-zero counts
mean `compute_scaling_params` should be re-run with a wider range (which requires re-encoding
the Delta table — a deliberate, documented operation, not something to automate).

**Verification.** (1) Unit test in `packages/contracts/tests/`: a small `NwpInMemory` fixture
with values pushed beyond `buffered_max` for one variable → `count_out_of_range` reports
exactly those counts, and zero for in-range variables. (2) Round-trip test: values at exactly
`buffered_min`/`buffered_max` count as in-range (clip is inclusive). (3) Materialise one
`ecmwf_ens` partition locally and confirm the metadata entry appears (expect zeros for a
normal day).

## Drop Hydra (and OmegaConf): plain YAML + importlib + pydantic

The project runs four config systems: Hydra, OmegaConf, pydantic-settings, and pydantic model
configs. Hydra's actual usage is tiny — `hydra.utils.get_class` / `hydra.utils.instantiate` in
`defs/jobs.py:80-81` and the experiment-reload path in `cv_assets.py`
(`load_experiment_forecaster`), plus `OmegaConf.load`/`merge` (`jobs.py:76-79`). None of
Hydra's value proposition (CLI composition, config groups, sweeps, launchers) is used, the
override merge is documented as whole-value replacement anyway, and Hydra has historically
lagged new Python releases (a real risk on 3.14+). Replacing it removes two dependencies for
~20 lines of code.

### Implementation details — Hydra removal (deleted when it ships)

**1. Replacement helpers.** In `packages/contracts/src/contracts/hydra_schemas.py` — renamed
to `config_schemas.py` (fix imports; check the module's existing contents first,
`load_cv_config` lives here):

- `import_class(target: str) -> type` — `importlib.import_module` on the module path +
  `getattr` for the class (replaces `hydra.utils.get_class`; also used by
  `load_experiment_forecaster` in `cv_assets.py` and `_class_target`'s round-trip).
- `load_model_config(path: Path, overrides: dict[str, Any]) -> tuple[type, dict[str, Any]]` —
  `yaml.safe_load`, shallow-merge `overrides` onto `model_params` (dict `update`; whole-value
  replacement, matching the documented semantics at `jobs.py:69-70`), pop the two `_target_`
  keys, return `(forecaster_cls_path, params)`.

**2. Rewire `_resolve_forecaster_config` (`defs/jobs.py:56-83`).**

```python
raw = yaml.safe_load((PROJECT_ROOT / base_model_config).read_text())
raw["model_params"].update(config_overrides)
forecaster_cls = import_class(raw["_target_"])
config_cls = import_class(raw["model_params"].pop("_target_"))
forecaster_config = config_cls(**raw["model_params"])  # pydantic validates
```

Pydantic replaces `hydra.utils.instantiate`'s validation role entirely (the configs are
already `BaseModel` subclasses). Same swap in `load_experiment_forecaster` (`cv_assets.py`),
which reconstructs classes from the `forecaster_target`/`config_target` MLflow tags.

**3. Dependency and reference cleanup.**

- Remove `hydra-core` from root `pyproject.toml` (and `omegaconf` if it's a direct dep
  anywhere; add `pyyaml` explicitly since it's currently only transitive).
- Grep for `hydra` / `OmegaConf` across `src/`, `packages/`, `tests/`, `conf/`, `docs/`,
  `CLAUDE.md`: update the `conf/model/xgboost.yaml` header comment
  (`conf/model/xgboost.yaml:5-9`), the `BaseForecasterConfig` docstring mentioning "Hydra
  config wiring" (`base_forecaster.py:23`), and the Configuration row in
  `docs/architecture/overview.md`.
- Keep the `_target_` YAML convention unchanged — it's a good convention; only the library
  interpreting it changes. The MLflow `forecaster_target`/`config_target` tag format is
  likewise unchanged, so previously registered experiments still reload.

**Verification.** (1) `tests/test_register_experiment_job.py` and
`tests/test_trained_cv_model.py` pass unchanged — they cover the register → reload-from-tags
path end-to-end and are the real safety net. (2) New unit test: `config_overrides` replaces
list values wholesale (the documented semantics), and an invalid override (e.g.
`max_depth: "high"`) raises a pydantic `ValidationError`. (3) `uv sync` then
`grep -r hydra src/ packages/ conf/ docs/ CLAUDE.md` returns nothing. (4) Register an
experiment against an existing local MLflow store and confirm `load_experiment_forecaster`
reloads a pre-migration experiment.

## Scientific-rigor tests and cleanup

*Runs after the [live service and monitoring](live-service.md) land.* The feature-level
no-lookahead tests, cross-mode equivalence test, idempotency tests, and the full-stack
cross-process MLflow test all exist. Three "not cheating" guardrail tests from the original
testing strategy remain unwritten, plus general cleanup.

### Implementation details — rigor tests (deleted when they ship)

**Part 1 — scientific-rigor tests:**

- **CV-windowing no-lookahead** (complements the feature-level tests, which cover lag leakage
  but not window construction): assert no *training* row has `valid_time >= val_start` for its
  fold — i.e. the training window built by `training_window(fold)` and applied in
  `trained_cv_model` never bleeds into validation.
- **Leaderboard fairness**: two different experiments over the same fold are scored on the
  **identical** `(time_series_id, fold)` population — a regression guard on the
  experiment-independence of `eligible_time_series`.
- **Determinism**: training a fold twice with a fixed `random_seed` yields identical
  predictions. This underpins idempotent retries and a stable leaderboard.

**Part 2 — cleanup:**

- Remove any remaining dead code/imports from the phased build-out.
- **Split `defs/cv_assets.py`** (898 lines — the complexity hotspot flagged in the 2026-07
  codebase review) into `cv_assets.py` / `production_assets.py` / `metric_assets.py`. The
  [`live_forecasts` work](live-service.md#the-live_forecasts-asset) already starts
  `production_assets.py`; move the `metrics` asset and its helpers into `metric_assets.py`
  here. Pure logic stays in `ml_core._cv_helpers`.

**Part 3 — docs freshness pass.** The permanent-docs migration from the old `dagster_plan.md`
is already done (July 2026): `docs/architecture/ml-orchestration.md` and
`docs/ml_experimentation/evaluating-new-data-sources.md` capture its important ideas. What
remains:

- Check `docs/` against the code after the live service and monitoring land — in particular
  extend `docs/ml_experimentation/dagster-workflow.md` with the live-forecast and monitoring
  flows, and update the "Known limitation" and MLflow-logging notes in
  `docs/architecture/ml-orchestration.md` if the implementations diverged from the plans.
- Run the ship-time triage (per CLAUDE.md) on any roadmap content the live-service work
  implemented — e.g. flip the relevant 🚧 statuses in
  `docs/roadmap/metrics-and-leaderboard.md` once monitoring lands.

**Verification.** Full `uv run pytest` green from the repo root, **including the full-stack
cross-process integration test**; `uv run pymarkdown scan` (per CLAUDE.md) green on the
touched docs; `grep -ri 'dagster_plan' docs/ src/ packages/` returns nothing.
