"""Integration test for the ``trained_cv_model`` asset.

Exercises the real wiring end-to-end against a file-based MLflow + temp Delta tables: register an
experiment (Phase 3), then materialise ``trained_cv_model`` for its fold and assert the model
artifact round-trips from MLflow and that training honoured the fold's eligible population and
inclusive training window.
"""

from datetime import datetime, timezone
from pathlib import Path

import mlflow
import polars as pl
import pytest
from contracts.ml_schemas import EligibleTimeSeries
from dagster import DagsterInstance, RunConfig, materialize
from deltalake import write_deltalake
from mlflow.tracking import MlflowClient
from xgboost_forecaster.forecaster import XGBoostForecaster

from contracts.settings import Settings

from nged_substation_forecast.defs.cv_assets import _load_engineering_inputs, trained_cv_model
from nged_substation_forecast.defs.jobs import RegisterExperimentConfig, register_experiment_job

pytestmark = pytest.mark.integration

FOLD_ID = "mid_2025_to_mid_2026"
EXPERIMENT_NAME = "exp_smoke"
PARTITION_KEY = f"{EXPERIMENT_NAME}__{FOLD_ID}"

# ts1 sits in H3 cell 10 with in-window data (trained); ts2 sits in cell 20 with data only after
# train_end 2025-06-30 (eligible, but excluded by the training-window filter).
_TS1_CELL = 10
_TS2_CELL = 20
_IN_WINDOW = datetime(2024, 6, 1, tzinfo=timezone.utc)
_AFTER_TRAIN_END = datetime(2025, 8, 1, tzinfo=timezone.utc)

_NWP_CONTINUOUS_COLS = (
    "temperature_2m",
    "dew_point_temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "pressure_surface",
    "pressure_reduced_to_mean_sea_level",
    "geopotential_height_500hpa",
    "downward_long_wave_radiation_flux_surface",
    "downward_short_wave_radiation_flux_surface",
    "precipitation_surface",
)


def _write_power(path: str) -> None:
    rows = [
        {"time_series_id": ts, "time": t, "power": 100.0 + i}
        for ts, day in ((1, _IN_WINDOW), (2, _AFTER_TRAIN_END))
        for i, t in enumerate(
            pl.datetime_range(
                day.replace(hour=6),
                day.replace(hour=8),
                interval="30m",
                time_zone="UTC",
                eager=True,
            )
        )
    ]
    pl.DataFrame(rows).cast(
        {"time_series_id": pl.Int32, "time": pl.Datetime("us", "UTC"), "power": pl.Float32}
    ).write_delta(path)


_NWP_ENSEMBLE_MEMBERS = (0, 1, 2)
"""Members written to the synthetic NWP. Member 0 is the control; 1 and 2 exercise the
``ensemble_members`` filter in ``_load_engineering_inputs`` (training keeps only the control)."""


def _write_nwp(path: str) -> None:
    """Write a minimal NwpOnDisk-shaped Delta (integer weather cols) for two cells/days.

    Each (cell, valid_time) carries all of ``_NWP_ENSEMBLE_MEMBERS`` so tests can assert that
    training narrows NWP to the control member while prediction would keep every member.
    """
    records = []
    for cell, day in ((_TS1_CELL, _IN_WINDOW), (_TS2_CELL, _AFTER_TRAIN_END)):
        init_time = day.replace(hour=0)
        for valid_time in pl.datetime_range(
            day.replace(hour=6), day.replace(hour=8), interval="30m", time_zone="UTC", eager=True
        ):
            for member in _NWP_ENSEMBLE_MEMBERS:
                record = {
                    "nwp_model_id": "ECMWF_ENS_0_25_degree",
                    "init_time": init_time,
                    "valid_time": valid_time,
                    "ensemble_member": member,
                    "h3_index": cell,
                    "categorical_precipitation_type_surface": None,
                }
                record.update({col: 2000 for col in _NWP_CONTINUOUS_COLS})
                records.append(record)
    df = pl.DataFrame(records).cast(
        {
            "init_time": pl.Datetime("us", "UTC"),
            "valid_time": pl.Datetime("us", "UTC"),
            "ensemble_member": pl.UInt8,
            "h3_index": pl.UInt64,
            "categorical_precipitation_type_surface": pl.UInt8,
            **{col: pl.Int16 for col in _NWP_CONTINUOUS_COLS},
        }
    )
    write_deltalake(table_or_uri=path, data=df.to_arrow())


def _write_metadata(path: Path) -> None:
    pl.DataFrame(
        {
            "time_series_id": pl.Series([1, 2], dtype=pl.Int32),
            "h3_res_5": pl.Series([_TS1_CELL, _TS2_CELL], dtype=pl.UInt64),
            "time_series_type": ["Primary", "Primary"],
        }
    ).write_parquet(path)


def _write_eligible(path: str) -> None:
    """Both ts1 and ts2 are eligible; the asset must still train only the in-window ts1."""
    eligible = EligibleTimeSeries.validate(
        pl.DataFrame(
            {
                "fold_id": pl.Series([FOLD_ID, FOLD_ID], dtype=pl.String),
                "time_series_id": pl.Series([1, 2], dtype=pl.Int32),
            }
        )
    )
    write_deltalake(table_or_uri=path, data=eligible.to_arrow(), partition_by=["fold_id"])


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    nged_path = tmp_path / "NGED"
    nged_path.mkdir()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("NGED_S3_BUCKET_URL", "https://example.com")
    monkeypatch.setenv("NGED_S3_BUCKET_ACCESS_KEY", "dummy")
    monkeypatch.setenv("NGED_S3_BUCKET_SECRET", "dummy")
    monkeypatch.setenv("NGED_DATA_PATH", str(nged_path))
    monkeypatch.setenv("NWP_DATA_PATH", str(tmp_path / "NWP"))
    monkeypatch.setenv("ELIGIBLE_TIME_SERIES_DATA_PATH", str(tmp_path / "eligible"))
    monkeypatch.setenv("MODEL_CACHE_BASE_PATH", str(tmp_path / "cache"))
    mlflow.set_tracking_uri(tracking_uri)

    _write_power(str(nged_path / "power_time_series.delta"))
    _write_nwp(str(tmp_path / "NWP"))
    _write_metadata(nged_path / "metadata.parquet")
    _write_eligible(str(tmp_path / "eligible"))
    return {"cache": str(tmp_path / "cache")}


def _register(instance: DagsterInstance) -> None:
    result = register_experiment_job.execute_in_process(
        run_config=RunConfig(
            ops={
                "register_experiment": RegisterExperimentConfig(
                    experiment_name=EXPERIMENT_NAME,
                    base_model_config="conf/model/xgboost.yaml",
                    config_overrides={"selected_features": ["temperature_2m"], "n_estimators": 5},
                    # full_cv so the leaderboard fold's partition (FOLD_ID) is registered; this test
                    # drives that fold's window and synthetic data, not the smoke_test fold.
                    run_mode="full_cv",
                )
            }
        ),
        instance=instance,
    )
    assert result.success


def test_load_engineering_inputs_filters_ensemble_members(env: dict[str, str]) -> None:
    """``ensemble_members`` narrows NWP at the scan; ``None`` keeps every member.

    This is the lever that keeps training (control member only) from fanning every forecast row out
    across all ~51 members against the same power target — the source of the training OOM.
    """
    settings = Settings()
    train_start = datetime(2024, 4, 1, tzinfo=timezone.utc)
    train_end = datetime(2025, 6, 30, 23, 59, 59, tzinfo=timezone.utc)

    _, _, nwp_control = _load_engineering_inputs(
        settings, [1, 2], train_start, train_end, ensemble_members=[0]
    )
    assert nwp_control.collect()["ensemble_member"].unique().sort().to_list() == [0]

    _, _, nwp_all = _load_engineering_inputs(settings, [1, 2], train_start, train_end)
    assert nwp_all.collect()["ensemble_member"].unique().sort().to_list() == list(
        _NWP_ENSEMBLE_MEMBERS
    )


def test_trained_cv_model_trains_and_saves_to_mlflow(env: dict[str, str]) -> None:
    instance = DagsterInstance.ephemeral()
    _register(instance)

    assert materialize([trained_cv_model], partition_key=PARTITION_KEY, instance=instance).success

    # The fold's child run exists with the logged training params.
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    assert experiment is not None
    fold_runs = MlflowClient().search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.cv_role = 'fold' and tags.fold_id = '{FOLD_ID}'",
    )
    assert len(fold_runs) == 1
    fold_run = fold_runs[0]
    assert fold_run.data.params["n_eligible_time_series"] == "2"

    # The model round-trips from MLflow, and only the in-window ts1 was trained (ts2's data is all
    # past train_end, so the inclusive-window filter excludes it).
    loaded = XGBoostForecaster.load_from_mlflow(fold_run.info.run_id, Path(env["cache"]))
    assert loaded.trained_time_series_ids == [1]
