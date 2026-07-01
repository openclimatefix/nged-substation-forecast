# Drop Hydra (and OmegaConf): plain YAML + importlib + pydantic

## Finding

The project runs four config systems: Hydra, OmegaConf, pydantic-settings, and pydantic model
configs. Hydra's actual usage is tiny — `hydra.utils.get_class` / `hydra.utils.instantiate` in
`defs/jobs.py:80-81` and the experiment-reload path in `cv_assets.py`
(`load_experiment_forecaster`), plus `OmegaConf.load`/`merge` (`jobs.py:76-79`). None of
Hydra's value proposition (CLI composition, config groups, sweeps, launchers) is used, the
override merge is documented as whole-value replacement anyway, and Hydra has historically
lagged new Python releases (a real risk on 3.14+). Replacing it removes two dependencies for
~20 lines of code.

## Implementation

### 1. Replacement helpers

In `packages/contracts/src/contracts/hydra_schemas.py` — renamed to `config_schemas.py` (fix
imports; check the module's existing contents first, `load_cv_config` lives here):

- `import_class(target: str) -> type` — `importlib.import_module` on the module path +
  `getattr` for the class (replaces `hydra.utils.get_class`; also used by
  `load_experiment_forecaster` in `cv_assets.py` and `_class_target`'s round-trip).
- `load_model_config(path: Path, overrides: dict[str, Any]) -> tuple[type, dict[str, Any]]` —
  `yaml.safe_load`, shallow-merge `overrides` onto `model_params` (dict `update`; whole-value
  replacement, matching the documented semantics at `jobs.py:69-70`), pop the two `_target_`
  keys, return `(forecaster_cls_path, params)`.

### 2. Rewire `_resolve_forecaster_config` (`defs/jobs.py:56-83`)

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

### 3. Dependency and reference cleanup

- Remove `hydra-core` from root `pyproject.toml` (and `omegaconf` if it's a direct dep
  anywhere; add `pyyaml` explicitly since it's currently only transitive).
- Grep for `hydra` / `OmegaConf` across `src/`, `packages/`, `tests/`, `conf/`, `docs/`,
  `CLAUDE.md`: update the `conf/model/xgboost.yaml` header comment (`conf/model/xgboost.yaml:5-9`),
  the `BaseForecasterConfig` docstring mentioning "Hydra config wiring"
  (`base_forecaster.py:23`), and the Configuration row in `docs/architecture/overview.md`.
- Keep the `_target_` YAML convention unchanged — it's a good convention; only the library
  interpreting it changes. The MLflow `forecaster_target`/`config_target` tag format is
  likewise unchanged, so previously registered experiments still reload.

### 4. Out of scope (deliberately)

The Patito friction budget noted in the review (four documented gotchas in CLAUDE.md) needs no
action now. Revisit only if a fifth workaround appears; the options then are boundary-only
validation or `dataframely`.

## Verification

1. `tests/test_register_experiment_job.py` and `tests/test_trained_cv_model.py` pass unchanged
   — they cover the register → reload-from-tags path end-to-end and are the real safety net.
2. New unit test: `config_overrides` replaces list values wholesale (the documented
   semantics), and an invalid override (e.g. `max_depth: "high"`) raises a pydantic
   `ValidationError`.
3. `uv sync` then `grep -r hydra src/ packages/ conf/ docs/ CLAUDE.md` returns nothing.
4. Register an experiment against an existing local MLflow store and confirm
   `load_experiment_forecaster` reloads a pre-migration experiment.
