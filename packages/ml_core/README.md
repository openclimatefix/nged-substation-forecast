# `ml_core`

The model-agnostic half of the forecasting stack: the interface every model implements, the feature
pipeline that feeds it, the metrics that score it, and the helpers that train, promote, and serve
it.

## Why this package exists

**`ml_core` owns everything about forecasting that is not specific to one model family.** A concrete
model supplies five members — `train`, `predict`, `save`, `load`, and the `trained_time_series_ids`
property — and inherits the rest from this package: the feature pipeline that builds its input, the
MLflow wiring that gives its run an identity, the archive format its weights are shipped in, the
checks that decide whether a saved model can still be served, and the scoring that puts it on the
leaderboard. `XGBoostForecaster` in `xgboost_forecaster` is the only subclass outside the tests
today, so the second model family is the real test of that split. Writing that second family should
mean writing five members, not a second pipeline.

**The Dagster assets delegate here rather than implementing the forecasting logic themselves.** The
cross-validation, metrics, and live-inference assets in `src/nged_substation_forecast/defs/` are
thin shells over the functions below. That thinness keeps every one of those functions unit-testable
without a Dagster instance, an MLflow server, or an object store. The assets hold the orchestration
— partitions, schedules, retries, and asset checks — and nothing else.

**Every neighbouring package owns a piece of the data; `ml_core` owns what is done with that data.**
`contracts` owns what every frame means, and `ml_core` consumes those Patito schemas without
declaring any of its own. `delta_store` owns how a table is physically written, and `ml_core` writes
no Delta table at all. The files `ml_core` does write are a saved model's frozen roster copy, the
gzipped tar archive that model ships in, and the `promotion.json` recording which run was promoted.
`nged_data` and `dynamical_data` own the ingest of observed power and of gridded weather, and
`ml_core` starts from whatever those two packages landed. `geo` owns H3 indexing, and
`weather_utils` owns the analysis-proxy query the weather-lag join is built on. What is left —
turning power, weather, and metadata into a model-ready frame, and turning a model's output into a
leaderboard row — is this package.

## Two invariants span the whole package

**Confusing the moment a forecast is issued with the moment the weather model ran is the failure
this package is built to prevent.** `power_fcst_init_time` is the first of those two moments and
`nwp_init_time` is the second. The pipeline carries both moments as separate columns from end to
end. The two columns differ by the publication delay — the hours between a numerical weather
prediction (NWP) run's `init_time` and the moment that run reaches our own disk. A feature is
therefore legitimate only if it was knowable at `power_fcst_init_time`, never if it was merely
knowable at `nwp_init_time`. Every power lag shorter than or equal to the forecast lead time is
nullified against `power_fcst_init_time`, in `_nullify_leaky_lags`. Weather lags reaching back
before `power_fcst_init_time` are answered from an earlier NWP run rather than from the current run.
Answering a past target time from the earlier run is the dual-strategy join in `_apply_weather_lag`.

**The train==predict population invariant keeps the leaderboard comparable.** A model scores exactly
the `time_series_id` population it trained on, whatever the eligibility rules would admit today, and
`BaseForecaster.trained_time_series_ids` is that frozen record. Power coverage changes over time, so
a series can newly qualify or newly drop out between a training run and a scoring run. Letting the
scored population drift with the coverage would silently compare two experiments over two different
populations. The same invariant stops the live service forecasting a series the promoted model never
saw.

## Research and production share one execution path

**What differs between a cross-validation run and a live forecast is failure policy, not code.** The
cross-validation, training, and metrics paths fail fast, because a quietly degraded training run
poisons every comparison built on it. The live path degrades instead: an absent input routes into
the always-output branch, the degradation is logged and reported to Sentry, and the forecast still
comes out with wider uncertainty. `_engineer_features` carries both policies in one function,
choosing between them on whether the caller supplied a `power_fcst_init_time`. The reasoning for
that asymmetry is on [Inherent
stability](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/)
and [design principle
3](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/design-principles/#3-one-execution-path-from-research-to-production).

## Contents

- `features` — the `FeatureEngineer` strategy interface, its default `TabularFeatureEngineer`
  implementation, and the declarative tabular pipeline behind that implementation. A forecaster
  wanting a different view of the data (a spatial crop for a convolutional neural network, say)
  supplies a different `FeatureEngineer` rather than editing the shared pipeline. The package
  docstring below lists the sub-modules and says what each sub-module owns.
- `base_forecaster` — `BaseForecaster`, the abstract interface every model implements, and
  `BaseForecasterConfig`, the serialisable config carrying a trained model's experiment identity.
  Also holds the MLflow round-trip that both research and production read a saved model through,
  which ships a model as one replaceable archive rather than as a directory of files. The constant
  `_MLFLOW_MODEL_ARTIFACT` documents why that distinction matters.
- `metrics` — the scoring pipeline: effective capacity, the ensemble-aware metrics (fair continuous
  ranked probability score, Fortin-corrected spread, pinball loss, prediction-interval coverage
  probability, and interval width), the horizon-slice bands, and the MLflow aggregate keys. The
  equations and the argument for each choice are on [Evaluation
  metrics](https://openclimatefix.github.io/nged-substation-forecast/techniques/evaluation-metrics/);
  the docstrings below say what this implementation does with those equations.
- `cv_helpers` — pure, input-only helpers for the cross-validation assets: fold date arithmetic, the
  eligibility rule, partition-key parsing, and config flattening. Fold design itself is on
  [Cross-validation
  folds](https://openclimatefix.github.io/nged-substation-forecast/ml_experimentation/cross-validation-folds/).
- `mlflow_runs` — idempotent resolution of an MLflow experiment, its parent run, and its per-fold
  child runs, always by tag rather than through a live handle, so separate processes and Dagster
  retries converge on the same run.
- `production_helpers` — the live-inference helpers: picking the NWP run a slot may legitimately
  use, building the dense forecast spine the power-centric join needs, and refusing to promote a
  saved model this code can no longer serve.
- `repro` — the git commit and the Delta-table versions stamped onto an MLflow run, so a result can
  be traced back to the exact code and data that produced it. Every function here is non-raising,
  because provenance must never fail the run it describes.
