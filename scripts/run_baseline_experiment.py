"""Reproduce the issue #179 weather/calendar-only baseline experiment end-to-end, headless.

This is the "shared baseline" for switching-event detection: the production XGBoost forecaster
configured with **no power-lag features** (weather + calendar covariates only), so its residual
(observed - expected) isolates switching events rather than absorbing them through a lagged-power
side door. See the roadmap:
<https://openclimatefix.github.io/nged-substation-forecast/roadmap/switching-events/#the-baseline-shared-foundation>

The whole pipeline runs in one process sharing a single ``DagsterInstance``. That is required, not
incidental: ``register_experiment`` adds the dynamic partition key
``xgboost_no_power_lags__mid_2025_to_mid_2026`` to the instance, and the subsequent
``materialize(..., partition_key=...)`` calls validate that key against the *same* instance's
dynamic-partition store. Splitting the steps across processes with separate ephemeral instances
would fail with "partition key not found". This mirrors the canonical pattern in
``tests/test_metrics.py``.

Only ``selected_features`` (and an honest ``training_strategy`` tag) are overridden; every other
hyperparameter is inherited from ``conf/model/xgboost.yaml``. The 20 features below are exactly the
base config's list minus the four ``power_lag_*h`` entries.

Run from the repo root, or a worktree where ``.env`` (and ``data``) are symlinked:

    uv run python scripts/run_baseline_experiment.py
"""

from __future__ import annotations

import mlflow
from contracts.settings import Settings
from dagster import DagsterInstance, RunConfig, materialize
from mlflow.tracking import MlflowClient
from nged_substation_forecast.defs.cv_assets import (
    MetricsConfig,
    PopulationFilter,
    cv_power_forecasts,
    effective_capacity,
    metrics,
    trained_cv_model,
)
from nged_substation_forecast.defs.jobs import (
    RegisterExperimentConfig,
    register_experiment_job,
)

EXPERIMENT_NAME = "xgboost_no_power_lags"
FOLD_ID = "mid_2025_to_mid_2026"
PARTITION_KEY = f"{EXPERIMENT_NAME}__{FOLD_ID}"

# The base conf/model/xgboost.yaml selected_features, minus the four power_lag_*h entries.
SELECTED_FEATURES: list[str] = [
    "local_time_of_day_sin",
    "local_time_of_day_cos",
    "local_time_of_year_sin",
    "local_time_of_year_cos",
    "local_day_of_week_sin",
    "local_day_of_week_cos",
    "local_day_of_week",
    "local_utc_offset",
    "temperature_2m",
    "dew_point_temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "pressure_surface",
    "pressure_reduced_to_mean_sea_level",
    "geopotential_height_500hpa",
    "downward_short_wave_radiation_flux_surface",
    "categorical_precipitation_type_surface",
    "windchill",
]


def _register(instance: DagsterInstance) -> None:
    """Create the MLflow experiment + parent run and register the fold's dynamic partition."""
    result = register_experiment_job.execute_in_process(
        run_config=RunConfig(
            ops={
                "register_experiment": RegisterExperimentConfig(
                    experiment_name=EXPERIMENT_NAME,
                    base_model_config="conf/model/xgboost.yaml",
                    config_overrides={
                        "selected_features": SELECTED_FEATURES,
                        # The base tag "horizon_as_feature" is inaccurate here (no lead-time
                        # feature, no lags); label this experiment for what it is.
                        "training_strategy": "weather_calendar_only",
                    },
                    run_mode="full_cv",
                    description=(
                        "Weather/calendar-only baseline (no power lags) — the switching-event"
                        " detection shared baseline (#179)."
                    ),
                )
            }
        ),
        instance=instance,
    )
    assert result.success, "register_experiment_job failed"


def _report_metrics() -> None:
    """Print the leaderboard aggregate metrics logged to the experiment's parent (cv_summary) run."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print("No MLflow experiment found; skipping metric report.")
        return
    parent_runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="tags.cv_role = 'parent'",
    )
    if not parent_runs:
        print("No parent run found; skipping metric report.")
        return
    metrics_dict = parent_runs[0].data.metrics
    print(f"\n=== Leaderboard metrics for {EXPERIMENT_NAME} (mean across folds) ===")
    for key in sorted(metrics_dict):
        print(f"  {key:<40} {metrics_dict[key]:.4f}")


def main() -> None:
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    instance = DagsterInstance.ephemeral()

    print(f"[1/5] Registering experiment {EXPERIMENT_NAME} ({len(SELECTED_FEATURES)} features)...")
    _register(instance)

    print(f"[2/5] Training fold {FOLD_ID}...")
    assert materialize(
        [trained_cv_model], partition_key=PARTITION_KEY, instance=instance
    ).success, "trained_cv_model failed"

    print("[3/5] Generating ensemble forecasts (this is the long step)...")
    assert materialize(
        [cv_power_forecasts], partition_key=PARTITION_KEY, instance=instance
    ).success, "cv_power_forecasts failed"

    print("[4/5] Materialising effective_capacity (NMAE denominator)...")
    assert materialize([effective_capacity], instance=instance).success, "effective_capacity failed"

    print("[5/5] Computing leaderboard metrics...")
    assert materialize(
        [metrics],
        run_config=RunConfig(
            ops={
                "metrics": MetricsConfig(
                    population_filter=PopulationFilter(
                        experiment_name=EXPERIMENT_NAME, fold_id=FOLD_ID
                    ),
                    evaluation_scope="leaderboard",
                )
            }
        ),
        instance=instance,
    ).success, "metrics failed"

    _report_metrics()
    print("\nDone. Now run: uv run python scripts/export_forecasts_for_alex.py")


if __name__ == "__main__":
    main()
