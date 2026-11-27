"""Integration test for the ``cv_power_forecasts`` asset.

Exercises the real wiring end-to-end against a file-based MLflow + temp Delta tables: register an
experiment, train a fold model with ``trained_cv_model``, then materialise ``cv_power_forecasts``
and assert the forecasts land in the ``power_forecasts`` Delta table — stamped with the fold
label (not ``"live"``), restricted to the model's trained population, and written idempotently so a
re-materialisation does not duplicate rows.
"""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest
from _cleaned_power_test_data import write_cleaned_copy
from _nwp_test_data import half_hours, nwp_records, write_test_nwp
from baseline_forecasters.climatology import ClimatologyForecaster
from contracts.ml_schemas import EligibleTimeSeries
from contracts.power_schemas import POWER_FCST_QUANTILE_COLUMNS, PowerForecast
from dagster import DagsterInstance, materialize
from delta_store.power_forecasts import POWER_FCST_SIGNIFICAND_BITS
from delta_store.precision import FLOAT32_SIGNIFICAND_BITS
from deltalake import write_deltalake
from ml_core.cv_helpers import date_to_utc_datetime
from mlflow.tracking import MlflowClient

from nged_substation_forecast.defs import cv_assets
from nged_substation_forecast.defs._engineering_inputs import MAX_NWP_LEAD, load_engineering_inputs
from nged_substation_forecast.defs.cv_assets import cv_power_forecasts, trained_cv_model

pytestmark = pytest.mark.integration

RegisterExperiment = Callable[[DagsterInstance, str], None]
"""Type of the ``register_experiment`` fixture (``tests/conftest.py``)."""

RegisterExperimentWithConfig = Callable[..., None]
"""Type of the ``register_experiment`` fixture when called with its keyword-only
``base_model_config`` and ``config_overrides`` arguments."""

FOLD_ID = "mid_2025_to_mid_2026"
EXPERIMENT_NAME = "exp_predict_smoke"
PARTITION_KEY = f"{EXPERIMENT_NAME}__{FOLD_ID}"

# ts1 sits in H3 cell 10. It has training-window data (so the fold model trains on it)
# and validation-window NWP rows for three ensemble members, which the feature engineer drops.
_TS1_CELL = 10
# A weekday in the month before the validation day: climatology forecasts a validation row only
# from training samples on the same day type, at most one local month and half-hour away.
_TRAIN_DAY = datetime(2024, 7, 31, tzinfo=UTC)  # inside train window [2024-04-01, 2025-06-30]
_VAL_DAY = datetime(2025, 8, 1, tzinfo=UTC)  # inside val window [2025-07-01, 2026-06-30]
_VAL_MEMBERS = (0, 1, 2)  # stands in for the full 51-member ensemble in this synthetic fixture


def _write_power(path: str) -> None:
    """Observed power for ts1 in the training window (the validation forecast needs no actuals)."""
    rows = [
        {"time_series_id": 1, "time": t, "power": 100.0 + i}
        for i, t in enumerate(half_hours(_TRAIN_DAY))
    ]
    pl.DataFrame(rows).cast(
        {"time_series_id": pl.Int32, "time": pl.Datetime("us", "UTC"), "power": pl.Float32}
    ).write_delta(path)


def _write_nwp(path: str) -> None:
    """Training NWP (control member) plus validation NWP across the ensemble, both in ts1's cell."""
    records = nwp_records(_TS1_CELL, _TRAIN_DAY, (0,)) + nwp_records(
        _TS1_CELL, _VAL_DAY, _VAL_MEMBERS
    )
    write_test_nwp(path, records)


def _write_metadata(path: Path) -> None:
    pl.DataFrame(
        {
            "time_series_id": pl.Series([1], dtype=pl.Int32),
            "h3_res_5": pl.Series([_TS1_CELL], dtype=pl.UInt64),
            "time_series_type": ["Primary"],
            "latitude": [53.0] * 1,
            "longitude": [-0.5] * 1,
        }
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


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    nged_path = tmp_path / "NGED"
    nged_path.mkdir()
    forecasts_path = tmp_path / "power_forecasts"
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("NGED_DATA_PATH", str(nged_path))
    monkeypatch.setenv("NWP_DATA_PATH", str(tmp_path / "NWP"))
    monkeypatch.setenv("ELIGIBLE_TIME_SERIES_DATA_PATH", str(tmp_path / "eligible"))
    monkeypatch.setenv("POWER_FORECASTS_DATA_PATH", str(forecasts_path))
    mlflow.set_tracking_uri(tracking_uri)

    _write_power(str(nged_path / "power_time_series.delta"))
    write_cleaned_copy(nged_path / "power_time_series.delta")
    _write_nwp(str(tmp_path / "NWP"))
    _write_metadata(nged_path / "metadata.parquet")
    _write_eligible(str(tmp_path / "eligible"))
    return {"forecasts": str(forecasts_path)}


def _read_forecasts(env: dict[str, str]) -> pl.DataFrame:
    return pl.read_delta(env["forecasts"])


def test_cv_power_forecasts_predicts_validation_fold(
    env: dict[str, str],
    dagster_instance: DagsterInstance,
    register_experiment: RegisterExperiment,
) -> None:
    register_experiment(dagster_instance, EXPERIMENT_NAME)
    assert materialize(
        [trained_cv_model], partition_key=PARTITION_KEY, instance=dagster_instance
    ).success
    assert materialize(
        [cv_power_forecasts], partition_key=PARTITION_KEY, instance=dagster_instance
    ).success

    forecasts = _read_forecasts(env)
    assert forecasts.height > 0

    # Stamped with the fold label and the experiment, not the "live" production sentinel.
    assert (forecasts["fold_id"] == FOLD_ID).all()
    assert (forecasts["experiment_name"] == EXPERIMENT_NAME).all()

    # Only the model's trained population (ts1) is scored.
    assert set(forecasts["time_series_id"].unique().to_list()) == {1}

    # The output has one row per key, no ensemble_member column, and the 13 quantile columns.
    assert "ensemble_member" not in forecasts.columns
    assert not forecasts.select(PowerForecast.PRIMARY_KEY).is_duplicated().any()
    assert set(POWER_FCST_QUANTILE_COLUMNS) <= set(forecasts.columns)

    # The fold's child run gained prediction stats.
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    assert experiment is not None
    fold_runs = MlflowClient().search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.cv_role = 'fold' and tags.fold_id = '{FOLD_ID}'",
    )
    assert len(fold_runs) == 1
    # A tag, not a metric: the count shrinks when the trained population or the validation window
    # does, and MLflow reports a metric's latest value as the max over all writes.
    assert fold_runs[0].data.tags["n_forecast_rows"] == str(forecasts.height)


def test_cv_power_forecasts_derives_power_lookback_from_selected_features(
    env: dict[str, str],
    dagster_instance: DagsterInstance,
    register_experiment: RegisterExperimentWithConfig,
    spy_power_lookback: Callable[[], list[timedelta]],
) -> None:
    """``cv_power_forecasts`` derives ``power_lookback`` from the experiment's own
    ``selected_features`` and passes it through to ``load_engineering_inputs`` on every
    ``init_time`` chunk.

    Mirrors ``test_trained_cv_model_derives_power_lookback_from_selected_features`` in
    ``tests/test_trained_cv_model.py``, for this asset's own call site. The shared
    ``register_experiment`` fixture defaults to climatology, which selects no feature, so
    registering the manual heuristic with ``power_lag_24h`` is what exercises this call site's own
    derivation.
    """
    # Climatology selects no feature, so the manual heuristic carries the power lag here.
    register_experiment(
        dagster_instance,
        EXPERIMENT_NAME,
        base_model_config="conf/model/manual_heuristic.yaml",
        config_overrides={"selected_features": ["power_lag_24h"]},
    )
    assert materialize(
        [trained_cv_model], partition_key=PARTITION_KEY, instance=dagster_instance
    ).success

    # Install the spy only after training, so its calls list holds only cv_power_forecasts' own
    # load_engineering_inputs calls, not trained_cv_model's.
    calls = spy_power_lookback()

    assert materialize(
        [cv_power_forecasts], partition_key=PARTITION_KEY, instance=dagster_instance
    ).success

    assert calls
    assert all(call == timedelta(hours=24) for call in calls)


def test_cv_power_forecasts_storage_format(
    env: dict[str, str],
    dagster_instance: DagsterInstance,
    register_experiment: RegisterExperiment,
) -> None:
    """The written parquet files carry the compression-oriented storage format end-to-end.

    Guards that ``cv_power_forecasts`` writes through ``delta_store.power_forecasts``: ZSTD
    compression with the per-column encodings, rows sorted within each file, and ``power_fcst``
    rounded to ``POWER_FCST_SIGNIFICAND_BITS``. The format itself is unit-tested in
    ``packages/delta_store/tests/``; this asserts the real asset actually uses it.
    """
    register_experiment(dagster_instance, EXPERIMENT_NAME)
    assert materialize(
        [trained_cv_model], partition_key=PARTITION_KEY, instance=dagster_instance
    ).success
    assert materialize(
        [cv_power_forecasts], partition_key=PARTITION_KEY, instance=dagster_instance
    ).success

    parquet_files = sorted(Path(env["forecasts"]).rglob("*.parquet"))
    assert parquet_files

    for parquet_file in parquet_files:
        metadata = pq.ParquetFile(parquet_file).metadata
        row_group = metadata.row_group(0)
        encodings_by_column = {
            row_group.column(i).path_in_schema: (
                row_group.column(i).compression,
                set(row_group.column(i).encodings),
            )
            for i in range(row_group.num_columns)
        }
        for column, (compression, _encodings) in encodings_by_column.items():
            assert compression == "ZSTD", f"{column} written as {compression}, expected ZSTD"
        assert "DELTA_BINARY_PACKED" in encodings_by_column["valid_time"][1]
        for quantile_column in POWER_FCST_QUANTILE_COLUMNS:
            assert "BYTE_STREAM_SPLIT" in encodings_by_column[quantile_column][1]

        rows = pl.read_parquet(parquet_file)
        sort_key = rows.select(key=pl.struct(*PowerForecast.PRIMARY_KEY))["key"]
        assert sort_key.is_sorted()

    # Every quantile column is rounded to POWER_FCST_SIGNIFICAND_BITS: the low fraction bits of
    # every finite value are zero, yet the values were not destroyed outright. (The synthetic
    # fixture's constant NWP features make every prediction near-identical, so don't assert on
    # value diversity — the fixture trains on power ≈ 100, so surviving values must be in that
    # ballpark.)
    zeroed_low_bits = FLOAT32_SIGNIFICAND_BITS - POWER_FCST_SIGNIFICAND_BITS
    stored = _read_forecasts(env).select(POWER_FCST_QUANTILE_COLUMNS).to_numpy()
    assert np.isfinite(stored).all()
    assert (stored.view(np.uint32) & np.uint32((1 << zeroed_low_bits) - 1) == 0).all()
    assert (np.abs(stored) > 50).all()


def test_cv_power_forecasts_is_idempotent(
    env: dict[str, str],
    dagster_instance: DagsterInstance,
    register_experiment: RegisterExperiment,
) -> None:
    register_experiment(dagster_instance, EXPERIMENT_NAME)
    assert materialize(
        [trained_cv_model], partition_key=PARTITION_KEY, instance=dagster_instance
    ).success

    assert materialize(
        [cv_power_forecasts], partition_key=PARTITION_KEY, instance=dagster_instance
    ).success
    first_height = _read_forecasts(env).height

    # Re-materialising the same fold overwrites its partition rather than appending.
    assert materialize(
        [cv_power_forecasts], partition_key=PARTITION_KEY, instance=dagster_instance
    ).success
    assert _read_forecasts(env).height == first_height


def test_cv_power_forecasts_fails_loudly_when_a_trained_series_loses_its_metadata(
    env: dict[str, str],
    dagster_instance: DagsterInstance,
    tmp_path: Path,
    register_experiment: RegisterExperiment,
) -> None:
    """The metadata parquet can change between training and scoring, so check again at predict.

    The check in `trained_cv_model` cannot cover this: the parquet is re-read here.
    """
    register_experiment(dagster_instance, EXPERIMENT_NAME)
    assert materialize(
        [trained_cv_model], partition_key=PARTITION_KEY, instance=dagster_instance
    ).success

    # ts1 was trained, then vanishes from the metadata parquet before scoring.
    pl.DataFrame(
        {
            "time_series_id": pl.Series([2], dtype=pl.Int32),
            "h3_res_5": pl.Series([_TS1_CELL], dtype=pl.UInt64),
            "time_series_type": ["Primary"],
            "latitude": [53.0] * 1,
            "longitude": [-0.5] * 1,
        }
    ).write_parquet(tmp_path / "NGED" / "metadata.parquet")

    result = materialize(
        [cv_power_forecasts],
        partition_key=PARTITION_KEY,
        instance=dagster_instance,
        raise_on_error=False,
    )

    assert not result.success
    failure = result.failure_data_for_node("cv_power_forecasts")
    assert failure is not None
    assert "no row in the metadata parquet" in str(failure.error)
    assert "[1]" in str(failure.error)


_MAX_PLAUSIBLE_PREDICT_CHUNKS = 50
"""A fold's validation window needs about 40 chunks of 14 days at most; more means a loop bug."""


def test_cv_power_forecasts_walks_contiguous_init_time_chunks_from_the_nwp_look_back(
    env: dict[str, str],
    dagster_instance: DagsterInstance,
    register_experiment: RegisterExperiment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The chunks start one maximum NWP lead before the window and follow on 1 microsecond apart."""
    register_experiment(dagster_instance, EXPERIMENT_NAME)
    assert materialize(
        [trained_cv_model], partition_key=PARTITION_KEY, instance=dagster_instance
    ).success

    chunks: list[tuple[datetime, datetime]] = []

    def _spy(*args: Any, **kwargs: Any) -> Any:
        # Raise rather than loop for ever when the chunk step stops advancing.
        if len(chunks) >= _MAX_PLAUSIBLE_PREDICT_CHUNKS:
            raise AssertionError("cv_power_forecasts made an implausible number of chunks")
        chunks.append((kwargs["init_time_start"], kwargs["init_time_end"]))
        return load_engineering_inputs(*args, **kwargs)

    monkeypatch.setattr(cv_assets, "load_engineering_inputs", _spy)

    assert materialize(
        [cv_power_forecasts], partition_key=PARTITION_KEY, instance=dagster_instance
    ).success

    val_start = date_to_utc_datetime(cv_assets._cv_config.get_fold(FOLD_ID).val_start)
    assert len(chunks) > 1
    assert chunks[0][0] == val_start - MAX_NWP_LEAD
    for (_, previous_end), (next_start, _) in pairwise(chunks):
        assert next_start == previous_end + timedelta(microseconds=1)


def test_cv_power_forecasts_fails_loudly_when_the_model_trained_no_series(
    env: dict[str, str],
    dagster_instance: DagsterInstance,
    register_experiment: RegisterExperiment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loaded forecaster with an empty trained population has nothing to forecast."""
    register_experiment(dagster_instance, EXPERIMENT_NAME)
    assert materialize(
        [trained_cv_model], partition_key=PARTITION_KEY, instance=dagster_instance
    ).success
    monkeypatch.setattr(
        ClimatologyForecaster, "trained_time_series_ids", property(lambda _self: [])
    )

    result = materialize(
        [cv_power_forecasts],
        partition_key=PARTITION_KEY,
        instance=dagster_instance,
        raise_on_error=False,
    )

    assert not result.success
    failure = result.failure_data_for_node("cv_power_forecasts")
    assert failure is not None
    assert "has no trained time series" in str(failure.error)
