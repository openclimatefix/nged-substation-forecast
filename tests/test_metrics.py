"""Integration tests for the ``metrics`` asset.

Tests at two tiers:

1. **In-process integration** (file-based MLflow + temp Delta): exercises the real asset wiring
   without a network server. Covers leaderboard loop, idempotency, and ad_hoc scope.

2. **Full-stack cross-process** (real ``mlflow server`` subprocess + temp Delta): the one test
   from §7.10 that proves the by-tag cross-process run resolution (§4.1.1) and the artifact
   upload/download round-trip (§4.5). Runs against a real HTTP tracking server so each
   ``materialize()`` call — like a separate Dagster process — uses a fresh MlflowClient
   connection to resolve runs by tag.
"""

import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import patito as pt
import polars as pl
import pytest
from contracts.ml_schemas import EligibleTimeSeries
from contracts.power_schemas import (
    LIST_OF_TIME_SERIES_TYPES,
    PowerForecast,
    TimeSeriesMetadata,
)
from dagster import DagsterInstance, RunConfig, materialize
from deltalake import write_deltalake
from mlflow.tracking import MlflowClient

from nged_substation_forecast.defs.cv_assets import (
    MetricsConfig,
    PopulationFilter,
    cv_power_forecasts,
    effective_capacity,
    metrics,
    trained_cv_model,
)
from nged_substation_forecast.defs.jobs import RegisterExperimentConfig, register_experiment_job

pytestmark = pytest.mark.integration

FOLD_ID = "mid_2025_to_mid_2026"
EXPERIMENT_NAME = "exp_metrics_smoke"
PARTITION_KEY = f"{EXPERIMENT_NAME}__{FOLD_ID}"

_TS1_CELL = 10
# Train day inside window [2024-04-01, 2025-06-30]; val day inside [2025-07-01, 2026-06-30].
_TRAIN_DAY = datetime(2024, 6, 1, tzinfo=timezone.utc)
_VAL_DAY = datetime(2025, 8, 1, tzinfo=timezone.utc)
_VAL_MEMBERS = (0, 1, 2)

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


def _half_hours(day: datetime) -> pl.Series:
    return pl.datetime_range(
        day.replace(hour=6), day.replace(hour=8), interval="30m", time_zone="UTC", eager=True
    )


def _write_power_with_actuals(path: str) -> None:
    """Power for ts1 in both the training window and the validation window.

    The metrics asset joins forecasts to observed power, so we need actuals for the
    validation window (the forecast valid times) as well as training-window data.
    """
    rows = []
    for i, t in enumerate(_half_hours(_TRAIN_DAY)):
        rows.append({"time_series_id": 1, "time": t, "power": 100.0 + i})
    for i, t in enumerate(_half_hours(_VAL_DAY)):
        rows.append({"time_series_id": 1, "time": t, "power": 80.0 + i})
    pl.DataFrame(rows).cast(
        {"time_series_id": pl.Int32, "time": pl.Datetime("us", "UTC"), "power": pl.Float32}
    ).write_delta(path)


def _nwp_records(cell: int, day: datetime, members: tuple[int, ...]) -> list[dict]:
    records = []
    init_time = day.replace(hour=0)
    for member in members:
        for valid_time in _half_hours(day):
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
    return records


def _write_nwp(path: str) -> None:
    records = _nwp_records(_TS1_CELL, _TRAIN_DAY, (0,)) + _nwp_records(
        _TS1_CELL, _VAL_DAY, _VAL_MEMBERS
    )
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
    # Full TimeSeriesMetadata frame plus h3_res_5 (used by the NWP spatial join).
    TimeSeriesMetadata.validate(
        pl.DataFrame(
            {
                "time_series_id": pl.Series([1], dtype=pl.Int32),
                "time_series_name": ["Test Substation"],
                "time_series_type": pl.Series(["Disaggregated Demand"]).cast(
                    pl.Enum(LIST_OF_TIME_SERIES_TYPES)
                ),
                "units": pl.Series(["MW"]).cast(pl.Enum(["MW", "MVA"])),
                "licence_area": pl.Series(["EMids"]).cast(pl.Enum(["EMids"])),
                "substation_number": pl.Series([1], dtype=pl.Int32),
                "substation_type": pl.Series(["Primary"]).cast(
                    pl.Enum(["BSP", "EHV Customer", "GSP", "HV Customer", "Primary"])
                ),
                "latitude": pl.Series([53.0], dtype=pl.Float32),
                "longitude": pl.Series([-0.5], dtype=pl.Float32),
                "h3_res_5": pl.Series([_TS1_CELL], dtype=pl.UInt64),
            }
        ),
        allow_superfluous_columns=True,
    ).write_parquet(path)


def _write_eligible(path: str) -> None:
    eligible = EligibleTimeSeries.validate(
        pl.DataFrame(
            {
                "fold_id": pl.Series([FOLD_ID], dtype=pl.String),
                "time_series_id": pl.Series([1], dtype=pl.Int32),
            }
        )
    )
    write_deltalake(table_or_uri=path, data=eligible.to_arrow(), partition_by=["fold_id"])


def _base_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tracking_uri: str
) -> dict[str, Path]:
    """Set up environment variables and write synthetic fixtures.

    Returns a dict of paths that tests use for assertions.
    """
    nged_path = tmp_path / "NGED"
    nged_path.mkdir()
    forecasts_path = tmp_path / "power_forecasts"
    metrics_path = tmp_path / "forecast_metrics"
    cache_path = tmp_path / "cache"
    effective_capacity_path = tmp_path / "effective_capacity"

    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("NGED_S3_BUCKET_URL", "https://example.com")
    monkeypatch.setenv("NGED_S3_BUCKET_ACCESS_KEY", "dummy")
    monkeypatch.setenv("NGED_S3_BUCKET_SECRET", "dummy")
    monkeypatch.setenv("NGED_DATA_PATH", str(nged_path))
    monkeypatch.setenv("NWP_DATA_PATH", str(tmp_path / "NWP"))
    monkeypatch.setenv("ELIGIBLE_TIME_SERIES_DATA_PATH", str(tmp_path / "eligible"))
    monkeypatch.setenv("MODEL_CACHE_BASE_PATH", str(cache_path))
    monkeypatch.setenv("POWER_FORECASTS_DATA_PATH", str(forecasts_path))
    monkeypatch.setenv("FORECAST_METRICS_DATA_PATH", str(metrics_path))
    # Point at a temp path so metrics never reads the repo's real effective_capacity table; the
    # table is absent until a test materialises it, and the metrics asset fails cleanly without it
    # (see test_metrics_raises_without_effective_capacity).
    monkeypatch.setenv("EFFECTIVE_CAPACITY_DATA_PATH", str(effective_capacity_path))

    _write_power_with_actuals(str(nged_path / "power_time_series.delta"))
    _write_nwp(str(tmp_path / "NWP"))
    _write_metadata(nged_path / "metadata.parquet")
    _write_eligible(str(tmp_path / "eligible"))

    return {
        "forecasts": forecasts_path,
        "metrics": metrics_path,
        "cache": cache_path,
        "effective_capacity": effective_capacity_path,
    }


def _register(instance: DagsterInstance) -> None:
    result = register_experiment_job.execute_in_process(
        run_config=RunConfig(
            ops={
                "register_experiment": RegisterExperimentConfig(
                    experiment_name=EXPERIMENT_NAME,
                    base_model_config="conf/model/xgboost.yaml",
                    config_overrides={"selected_features": ["temperature_2m"], "n_estimators": 5},
                    run_mode="full_cv",
                )
            }
        ),
        instance=instance,
    )
    assert result.success


def _metrics_run_config(scope: str = "leaderboard") -> RunConfig:
    return RunConfig(
        ops={
            "metrics": MetricsConfig(
                population_filter=PopulationFilter(
                    experiment_name=EXPERIMENT_NAME,
                    fold_id=FOLD_ID,
                ),
                evaluation_scope=scope,
            )
        }
    )


# ---------------------------------------------------------------------------
# In-process integration tests (file-based MLflow)
# ---------------------------------------------------------------------------


@pytest.fixture
def file_mlflow_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    return _base_env(tmp_path, monkeypatch, tracking_uri)


def _run_cv_pipeline(instance: DagsterInstance, materialise_capacity: bool = True) -> None:
    """Register, train, and predict for the leaderboard fold.

    Also materialises ``effective_capacity`` (the NMAE denominator ``metrics`` now requires) unless
    ``materialise_capacity`` is False — used by the test that asserts ``metrics`` fails without it.
    """
    _register(instance)
    assert materialize([trained_cv_model], partition_key=PARTITION_KEY, instance=instance).success
    assert materialize([cv_power_forecasts], partition_key=PARTITION_KEY, instance=instance).success
    if materialise_capacity:
        assert materialize([effective_capacity], instance=instance).success


def test_metrics_leaderboard_writes_forecast_metrics_delta(
    file_mlflow_env: dict[str, Path],
) -> None:
    instance = DagsterInstance.ephemeral()
    _run_cv_pipeline(instance)
    assert materialize(
        [metrics], run_config=_metrics_run_config("leaderboard"), instance=instance
    ).success

    fm = pl.read_delta(str(file_mlflow_env["metrics"]))
    assert fm.height > 0

    # Every row carries the correct scope, fold, and experiment.
    assert (fm["evaluation_scope"] == "leaderboard").all()
    assert (fm["fold_id"] == FOLD_ID).all()
    assert (fm["experiment_name"] == EXPERIMENT_NAME).all()

    # Window columns are populated.
    assert fm["window_start"].is_not_null().all()
    assert fm["window_end"].is_not_null().all()
    assert (fm["window_label"] == FOLD_ID).all()
    assert fm["computed_at"].is_not_null().all()

    # time_series_type is populated from metadata.
    assert (fm["time_series_type"] == "Disaggregated Demand").all()

    # mlflow_run_id links back to the fold run.
    assert fm["mlflow_run_id"].is_not_null().all()


def test_metrics_leaderboard_logs_to_mlflow(
    file_mlflow_env: dict[str, Path],
) -> None:
    instance = DagsterInstance.ephemeral()
    _run_cv_pipeline(instance)
    assert materialize(
        [metrics], run_config=_metrics_run_config("leaderboard"), instance=instance
    ).success

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    assert experiment is not None
    client = MlflowClient()

    # Fold run has per-type + overall metrics.
    fold_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.cv_role = 'fold' and tags.fold_id = '{FOLD_ID}'",
    )
    assert len(fold_runs) == 1
    fold_metrics = fold_runs[0].data.metrics
    assert "rmse__all" in fold_metrics
    assert "rmse__disaggregated_demand" in fold_metrics

    # Parent run has aggregate metrics (mean across folds — one fold here, so same values).
    parent_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.cv_role = 'parent'",
    )
    assert len(parent_runs) == 1
    parent_metrics = parent_runs[0].data.metrics
    assert "rmse__all" in parent_metrics


def test_metrics_is_idempotent(file_mlflow_env: dict[str, Path]) -> None:
    instance = DagsterInstance.ephemeral()
    _run_cv_pipeline(instance)

    assert materialize(
        [metrics], run_config=_metrics_run_config("leaderboard"), instance=instance
    ).success
    first_height = pl.read_delta(str(file_mlflow_env["metrics"])).height

    # Re-materialising overwrites the partition rather than appending.
    assert materialize(
        [metrics], run_config=_metrics_run_config("leaderboard"), instance=instance
    ).success
    assert pl.read_delta(str(file_mlflow_env["metrics"])).height == first_height


def _read_metric(metrics_path: Path, metric_name: str) -> float:
    """Read the overall (``horizon_slice="all"``) value of one metric for the single test series."""
    fm = pl.read_delta(str(metrics_path)).filter(
        (pl.col("metric_name") == metric_name) & (pl.col("horizon_slice") == "all")
    )
    assert fm.height == 1
    return float(fm["metric_value"][0])


def test_metrics_raises_without_effective_capacity(file_mlflow_env: dict[str, Path]) -> None:
    """The metrics asset requires the effective_capacity table and fails cleanly when it is absent."""
    instance = DagsterInstance.ephemeral()
    _run_cv_pipeline(instance, materialise_capacity=False)

    result = materialize(
        [metrics],
        run_config=_metrics_run_config("leaderboard"),
        instance=instance,
        raise_on_error=False,
    )
    assert not result.success


def test_metrics_nmae_denominator_is_effective_capacity(
    file_mlflow_env: dict[str, Path],
) -> None:
    """NMAE is MAE divided by the series' full-history effective_capacity_mw."""
    instance = DagsterInstance.ephemeral()
    _run_cv_pipeline(instance)
    assert materialize(
        [metrics], run_config=_metrics_run_config("leaderboard"), instance=instance
    ).success

    mae = _read_metric(file_mlflow_env["metrics"], "mae")
    nmae = _read_metric(file_mlflow_env["metrics"], "nmae")
    capacity = pl.read_delta(str(file_mlflow_env["effective_capacity"])).filter(
        pl.col("time_series_id") == 1
    )["effective_capacity_mw"][0]

    assert nmae == pytest.approx(mae / capacity, rel=1e-5)


def _batch_forecast_frame(
    time_series_id: int, times: list[datetime], power_fcst: list[float]
) -> pl.DataFrame:
    """One-member PowerForecast rows for the per-series-batching test."""
    n = len(times)
    return pl.DataFrame(
        {
            "time_series_id": pl.Series([time_series_id] * n, dtype=pl.Int32),
            "valid_time": pl.Series(times, dtype=pl.Datetime("us", "UTC")),
            "power_fcst_init_time": pl.Series([min(times)] * n, dtype=pl.Datetime("us", "UTC")),
            "ensemble_member": pl.Series([0] * n, dtype=pl.Int8),
            "nwp_init_time": pl.Series([None] * n, dtype=pl.Datetime("us", "UTC")),
            "power_fcst": pl.Series(power_fcst, dtype=pl.Float32),
            "power_fcst_model_name": pl.Series(["stub"] * n, dtype=pl.String),
            "power_fcst_model_version": pl.Series([1] * n, dtype=pl.Int16),
            "ml_flow_experiment_id": pl.Series([None] * n, dtype=pl.Int32),
            "experiment_name": pl.Series(["exp_batch"] * n, dtype=pl.String),
            "fold_id": pl.Series(["2025"] * n, dtype=pl.String),
        }
    )


def test_score_forecast_group_per_series_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-series batching writes exactly the rows a whole-group call would.

    With the batch size forced to 1, three series exercise the multi-batch path; the series
    with no overlapping actuals is skipped (as it silently vanishes from the inner join in a
    whole-group call) and the written Delta rows match ``compute_metrics`` on the overlapping
    series.
    """
    from contracts.power_schemas import EffectiveCapacity, PowerTimeSeries
    from ml_core.metrics import compute_metrics

    from nged_substation_forecast.defs import cv_assets

    times = [
        datetime(2025, 8, 1, 6, 0, tzinfo=timezone.utc),
        datetime(2025, 8, 1, 6, 30, tzinfo=timezone.utc),
    ]
    # Series 1 and 2 overlap the actuals below; series 3's valid times are a year later.
    far_times = [t.replace(year=2026) for t in times]
    group = PowerForecast.validate(
        pl.concat(
            [
                _batch_forecast_frame(1, times, [12.0, 8.0]),
                _batch_forecast_frame(2, times, [25.0, 19.0]),
                _batch_forecast_frame(3, far_times, [1.0, 1.0]),
            ]
        )
    )
    actuals = pt.LazyFrame.from_existing(
        pl.LazyFrame(
            {
                "time_series_id": pl.Series([1, 1, 2, 2], dtype=pl.Int32),
                "time": pl.Series(times * 2, dtype=pl.Datetime("us", "UTC")),
                "power": pl.Series([10.0, 10.0, 20.0, 20.0], dtype=pl.Float32),
            }
        )
    ).set_model(PowerTimeSeries)
    metadata_df = TimeSeriesMetadata.validate(
        pl.DataFrame(
            {
                "time_series_id": pl.Series([1, 2, 3], dtype=pl.Int32),
                "time_series_name": [f"Substation {i}" for i in (1, 2, 3)],
                "time_series_type": pl.Series(["Disaggregated Demand"] * 3).cast(
                    pl.Enum(LIST_OF_TIME_SERIES_TYPES)
                ),
                "units": pl.Series(["MW"] * 3).cast(pl.Enum(["MW", "MVA"])),
                "licence_area": pl.Series(["EMids"] * 3).cast(pl.Enum(["EMids"])),
                "substation_number": pl.Series([1, 2, 3], dtype=pl.Int32),
                "substation_type": pl.Series(["Primary"] * 3).cast(
                    pl.Enum(["BSP", "EHV Customer", "GSP", "HV Customer", "Primary"])
                ),
                "latitude": pl.Series([53.0] * 3, dtype=pl.Float32),
                "longitude": pl.Series([-0.5] * 3, dtype=pl.Float32),
                "h3_res_5": pl.Series([_TS1_CELL] * 3, dtype=pl.UInt64),
            }
        ),
        allow_superfluous_columns=True,
    )
    capacity_df = EffectiveCapacity.validate(
        pl.DataFrame(
            {
                "time_series_id": pl.Series([1, 2, 3], dtype=pl.Int32),
                "time": pl.Series([times[0]] * 3, dtype=pl.Datetime("us", "UTC")),
                "effective_capacity_mw": pl.Series([10.0, 20.0, 5.0], dtype=pl.Float32),
            }
        )
    )

    monkeypatch.setattr(cv_assets, "_METRICS_SERIES_BATCH_SIZE", 1)
    metrics_path = tmp_path / "forecast_metrics"
    n_rows, fold_metric_dict = cv_assets._score_forecast_group(
        "exp_batch",
        "2025",
        group.lazy(),
        actuals,
        metadata_df,
        capacity_df,
        "ad_hoc",
        str(metrics_path),
        datetime.now(timezone.utc),
        None,
    )

    assert fold_metric_dict is None  # ad_hoc scope never logs to MLflow
    written = pl.read_delta(str(metrics_path))
    assert set(written["time_series_id"].to_list()) == {1, 2}

    expected = compute_metrics(
        PowerForecast.validate(
            pl.concat(
                [
                    _batch_forecast_frame(1, times, [12.0, 8.0]),
                    _batch_forecast_frame(2, times, [25.0, 19.0]),
                ]
            )
        ),
        actuals,
        metadata_df,
        capacity_df,
    )
    assert n_rows == expected.height

    compare_cols = ["time_series_id", "horizon_slice", "metric_name", "metric_value"]
    sort_keys = ["time_series_id", "horizon_slice", "metric_name"]
    # Delta stores the Enum columns as String, so compare in String space. Expression casts
    # (not a {column: dtype} dict-cast) because `expected` is a model-bearing Patito frame.
    expected_str = expected.select(compare_cols).with_columns(
        pl.col("horizon_slice").cast(pl.String), pl.col("metric_name").cast(pl.String)
    )
    assert written.select(compare_cols).sort(sort_keys).equals(expected_str.sort(sort_keys))


def test_metrics_ad_hoc_no_mlflow_logging(file_mlflow_env: dict[str, Path]) -> None:
    instance = DagsterInstance.ephemeral()
    _run_cv_pipeline(instance)
    assert materialize(
        [metrics], run_config=_metrics_run_config("ad_hoc"), instance=instance
    ).success

    # Delta rows written.
    fm = pl.read_delta(str(file_mlflow_env["metrics"]))
    assert fm.height > 0
    assert (fm["evaluation_scope"] == "ad_hoc").all()
    assert fm["mlflow_run_id"].is_null().all()
    assert (fm["window_label"] == "ad_hoc").all()

    # No MLflow metrics logged (fold run was never created; parent run has no metrics).
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    assert experiment is not None
    client = MlflowClient()
    fold_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.cv_role = 'fold' and tags.fold_id = '{FOLD_ID}'",
    )
    # The fold run exists (created by trained_cv_model), but has no rmse metrics from metrics asset.
    for run in fold_runs:
        assert "rmse__all" not in run.data.metrics


def test_population_filter_prunes_partitions(tmp_path: Path) -> None:
    """``PopulationFilter.apply`` pushes its predicate into the Delta scan (partition pruning).

    Regression guard for partition pruning: ``experiment_name`` / ``fold_id`` are ``String`` in
    ``PowerForecast`` (matching how delta-rs stores them on disk), so the filter pushes straight
    into the Delta scan with no dtype cast in the way. The explain plan lists only the named
    partition's Parquet path and never the other experiment's.
    """
    path = str(tmp_path / "power_forecasts")
    valid_time = pl.Series([datetime(2025, 8, 1, 6, tzinfo=timezone.utc)] * 4).dt.cast_time_unit(
        "us"
    )
    df = pl.DataFrame(
        {
            "experiment_name": ["expA", "expA", "expB", "expB"],
            "fold_id": ["f1", "f1", "f1", "f1"],
            "valid_time": valid_time,
            "power_hat": [1.0, 2.0, 3.0, 4.0],
        }
    )
    write_deltalake(
        table_or_uri=path, data=df.to_arrow(), partition_by=["experiment_name", "fold_id"]
    )

    scan = pt.LazyFrame.from_existing(pl.scan_delta(path)).set_model(PowerForecast)
    plan = PopulationFilter(experiment_name="expA").apply(scan).explain()
    assert "experiment_name=expA" in plan
    assert "experiment_name=expB" not in plan


def test_metrics_no_filter_scores_every_group(file_mlflow_env: dict[str, Path]) -> None:
    """A default (no-filter) ``metrics`` run scores every ``(experiment_name, fold_id)`` group.

    The pipeline produces forecasts for one experiment; we duplicate them under a second
    experiment so two groups exist on disk, then run ``metrics`` with the default
    ``PopulationFilter()`` (no filter) and assert both groups land in ``forecast_metrics``.
    """
    instance = DagsterInstance.ephemeral()
    _run_cv_pipeline(instance)

    # Duplicate the produced forecasts under a second experiment so two groups exist on disk.
    forecasts_path = str(file_mlflow_env["forecasts"])
    second_experiment = f"{EXPERIMENT_NAME}_2"
    duplicate = pl.read_delta(forecasts_path).with_columns(
        experiment_name=pl.lit(second_experiment)
    )
    write_deltalake(
        table_or_uri=forecasts_path,
        data=duplicate.to_arrow(),
        mode="append",
        partition_by=["experiment_name", "fold_id"],
    )

    # Default PopulationFilter() = no filter → score everything.
    assert materialize(
        [metrics],
        run_config=RunConfig(ops={"metrics": MetricsConfig(evaluation_scope="ad_hoc")}),
        instance=instance,
    ).success

    fm = pl.read_delta(str(file_mlflow_env["metrics"]))
    scored_experiments = set(fm["experiment_name"].unique().to_list())
    assert scored_experiments == {EXPERIMENT_NAME, second_experiment}


# ---------------------------------------------------------------------------
# Full-stack cross-process test (real mlflow server)
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_mlflow_server(
    url: str,
    proc: subprocess.Popen[bytes],
    log_path: Path,
    timeout_s: float = 30.0,
) -> None:
    def _server_output() -> str:
        return log_path.read_text() if log_path.exists() else "(no output captured)"

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(
                f"MLflow server exited early with code {proc.returncode}:\n{_server_output()}"
            )
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        # Unparenthesised except-tuple (PEP 758, Python 3.14+); ruff format emits this form.
        except urllib.error.URLError, OSError:
            time.sleep(0.5)
    pytest.fail(
        f"MLflow server at {url} did not become ready within {timeout_s}s:\n{_server_output()}"
    )


def test_full_stack_real_mlflow_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full-stack test: real HTTP MLflow server + artifact round-trip + tag resolution.

    Proves §4.1.1 (cross-call tag resolution) and §4.5 (artifact upload/download).
    Each ``materialize()`` call starts with a fresh ``mlflow.set_tracking_uri`` inside
    the asset body — simulating what happens when assets run in separate Dagster processes.
    The artifact round-trip is proved by deleting the local model cache between
    ``trained_cv_model`` and ``cv_power_forecasts``.
    """
    # The real MLflow HTTP server needs the server runtime stack (full ``mlflow``, in the dev
    # group); a ``mlflow-skinny``-only environment cannot start it, so skip rather than fail.
    # This test runs the Flask app under gunicorn (see ``--gunicorn-opts`` below).
    pytest.importorskip("flask")
    pytest.importorskip("gunicorn")

    port = _find_free_port()
    # Capture server output to a file (not DEVNULL) so an early-dying server is diagnosable, and
    # not a PIPE, which the long-running server could otherwise fill and block on.
    server_log = tmp_path / "mlflow_server.log"
    log_file = server_log.open("wb")
    mlflow_proc = subprocess.Popen(
        [
            "mlflow",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--backend-store-uri",
            str(tmp_path / "mlruns"),
            "--default-artifact-root",
            str(tmp_path / "artifacts"),
            # Force the Flask/gunicorn server: mlflow 3.14's default uvicorn+FastAPI app imports
            # importlib.abc.Traversable, removed in Python 3.14, so it fails to start.
            "--gunicorn-opts",
            "--workers 1",
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        # mlflow 3.14 puts the file-store backend in maintenance mode unless this is set; the
        # subprocess does not inherit the fixture's env, so pass it through explicitly.
        env={**os.environ, "MLFLOW_ALLOW_FILE_STORE": "true"},
    )
    try:
        tracking_uri = f"http://127.0.0.1:{port}"
        _wait_for_mlflow_server(f"{tracking_uri}/health", mlflow_proc, server_log)
        mlflow.set_tracking_uri(tracking_uri)

        paths = _base_env(tmp_path, monkeypatch, tracking_uri)
        instance = DagsterInstance.ephemeral()
        _register(instance)

        # Train — uploads model artifacts to the real server.
        assert materialize(
            [trained_cv_model], partition_key=PARTITION_KEY, instance=instance
        ).success

        # Delete local cache to force artifact re-download on the next step (§4.5 cache miss).
        shutil.rmtree(paths["cache"], ignore_errors=True)

        # Predict — downloads artifacts from the real server on cache miss.
        assert materialize(
            [cv_power_forecasts], partition_key=PARTITION_KEY, instance=instance
        ).success

        # The NMAE denominator table that metrics requires.
        assert materialize([effective_capacity], instance=instance).success

        # Score — proves tag-based run resolution: the asset calls get_or_create_fold_run()
        # with a fresh MlflowClient connection and finds the same run that trained_cv_model used.
        assert materialize(
            [metrics],
            run_config=_metrics_run_config("leaderboard"),
            instance=instance,
        ).success

        # Assert leaderboard metrics landed on the real server.
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        assert experiment is not None
        parent_runs = MlflowClient().search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.cv_role = 'parent'",
        )
        assert len(parent_runs) == 1
        assert "rmse__all" in parent_runs[0].data.metrics

        # Assert forecast_metrics Delta has rows.
        fm = pl.read_delta(str(paths["metrics"]))
        assert (
            fm.filter(
                (pl.col("experiment_name") == EXPERIMENT_NAME)
                & (pl.col("fold_id") == FOLD_ID)
                & (pl.col("evaluation_scope") == "leaderboard")
            ).height
            > 0
        )

    finally:
        mlflow_proc.terminate()
        mlflow_proc.wait(timeout=10)
        log_file.close()
