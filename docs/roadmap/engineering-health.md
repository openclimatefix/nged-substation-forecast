# Engineering health

> **Status: 🚧 Planned.** Tooling, reproducibility, and rigour improvements from the 2026-07
> codebase review that don't change forecast behaviour. Each section is an independent,
> roughly one-PR piece of work; sections are deleted as they ship. Mostly the v0.2 epic
> [#138](https://github.com/openclimatefix/nged-substation-forecast/issues/138):
> Hydra removal [#228](https://github.com/openclimatefix/nged-substation-forecast/issues/228) ·
> rigor tests [#229](https://github.com/openclimatefix/nged-substation-forecast/issues/229).
> NWP ingestion checks
> [#161](https://github.com/openclimatefix/nged-substation-forecast/issues/161) have shipped —
> as-built in
> [Known ECMWF ENS data-quality issues](../architecture/ecmwf-ens-known-issues.md#an-incomplete-run-tolerated-and-reported).
> Task ordering lives in the GitHub Project board.

## Drop Hydra (and OmegaConf): plain YAML + importlib + pydantic

Issue: [#228](https://github.com/openclimatefix/nged-substation-forecast/issues/228)

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

Issue: [#229](https://github.com/openclimatefix/nged-substation-forecast/issues/229)

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
- **Degradation smoke-tests**: ablate whole input groups — NWP absent, telemetry absent, a single
  weather variable nulled — and assert that a forecast is still produced for every time series, that
  every value stays inside physical bounds, and that nothing explodes. These consume the scenario
  vocabulary defined by the failure-scenario suite in
  [Metrics & Leaderboard](metrics-and-leaderboard.md#scoring-under-failure-scenarios), so they must
  land alongside it rather than against an ad-hoc vocabulary of their own. They are cheap and
  CI-fast — pure functions over an `AllFeatures` frame, no MLflow — and they check *survival*, not
  skill; skill under degradation is the leaderboard's job. The principle they enforce is
  [Inherent Stability](../design-philosophy/inherent-stability.md).

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
