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
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
from xgboost_forecaster.forecaster import XGBoostForecaster

from contracts.settings import Settings

from _nwp_test_data import NWP_CONTINUOUS_COL_VALUES
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
    """Write a minimal Nwp-shaped Delta (Float32 physical-unit weather cols) for two cells/days.

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
                record.update(NWP_CONTINUOUS_COL_VALUES)
                records.append(record)
    df = pl.DataFrame(records).cast(
        {
            "init_time": pl.Datetime("us", "UTC"),
            "valid_time": pl.Datetime("us", "UTC"),
            "ensemble_member": pl.UInt8,
            "h3_index": pl.UInt64,
            "categorical_precipitation_type_surface": pl.UInt8,
            **{col: pl.Float32 for col in NWP_CONTINUOUS_COL_VALUES},
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


def _write_eligible(path: str, time_series_ids: tuple[int, ...] = (1, 2)) -> None:
    """Write the fold's eligible population, replacing any existing table (default: ts1 and ts2).

    The asset must still train only the in-window ts1. ``time_series_ids`` is a parameter, and the
    write replaces the table, so a test can shrink or grow the eligible set between
    materialisations of the same fold.
    """
    eligible = EligibleTimeSeries.validate(
        pl.DataFrame(
            {
                "fold_id": pl.Series([FOLD_ID] * len(time_series_ids), dtype=pl.String),
                "time_series_id": pl.Series(list(time_series_ids), dtype=pl.Int32),
            }
        )
    )
    write_deltalake(
        table_or_uri=path,
        data=eligible.to_arrow(),
        mode="overwrite",
        partition_by=["fold_id"],
    )


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    nged_path = tmp_path / "NGED"
    nged_path.mkdir()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("NGED_DATA_PATH", str(nged_path))
    monkeypatch.setenv("NWP_DATA_PATH", str(tmp_path / "NWP"))
    monkeypatch.setenv("ELIGIBLE_TIME_SERIES_DATA_PATH", str(tmp_path / "eligible"))
    mlflow.set_tracking_uri(tracking_uri)

    _write_power(str(nged_path / "power_time_series.delta"))
    _write_nwp(str(tmp_path / "NWP"))
    _write_metadata(nged_path / "metadata.parquet")
    _write_eligible(str(tmp_path / "eligible"))


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


def test_load_engineering_inputs_filters_ensemble_members(env: None) -> None:
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


def test_load_engineering_inputs_prunes_nwp_to_requested_cells_and_init_window(
    env: None,
) -> None:
    """NWP is pruned to the requested series' H3 cells and the window's ``init_time`` partitions."""
    settings = Settings()
    train_start = datetime(2024, 4, 1, tzinfo=timezone.utc)
    train_end = datetime(2025, 6, 30, 23, 59, 59, tzinfo=timezone.utc)

    # Requesting only ts1 must scan only ts1's cell, never ts2's.
    _, _, nwp_ts1 = _load_engineering_inputs(settings, [1], train_start, train_end)
    assert nwp_ts1.collect()["h3_index"].unique().to_list() == [_TS1_CELL]

    # Requesting both: ts2's cell is initialised at 2025-08-01 (after train_end), so the init_time
    # partition prune drops it entirely — only ts1's in-window cell survives.
    _, _, nwp_both = _load_engineering_inputs(settings, [1, 2], train_start, train_end)
    assert nwp_both.collect()["h3_index"].unique().to_list() == [_TS1_CELL]


def _fold_run(client: MlflowClient) -> Run:
    """Return the single MLflow fold run for this test's ``(experiment, fold)``."""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    assert experiment is not None
    fold_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.cv_role = 'fold' and tags.fold_id = '{FOLD_ID}'",
    )
    assert len(fold_runs) == 1
    return fold_runs[0]


def test_trained_cv_model_trains_and_saves_to_mlflow(
    env: None, dagster_instance: DagsterInstance
) -> None:
    _register(dagster_instance)

    assert materialize(
        [trained_cv_model], partition_key=PARTITION_KEY, instance=dagster_instance
    ).success

    # The fold's identity (fold_id) is already a tag from run creation; the training window and
    # counters are logged as tags too, since a re-materialisation is allowed to change them and
    # tags (unlike MLflow params) overwrite cleanly — issue #197.
    fold_run = _fold_run(MlflowClient())
    assert fold_run.data.tags["fold_id"] == FOLD_ID
    assert fold_run.data.tags["train_start"] == "2024-04-01T00:00:00+00:00"
    assert fold_run.data.tags["train_end"] == "2025-06-30T23:59:59+00:00"
    assert fold_run.data.tags["n_eligible_time_series"] == "2"
    assert fold_run.data.tags["n_trained_time_series"] == "1"

    # The model round-trips from MLflow, and only the in-window ts1 was trained (ts2's data is all
    # past train_end, so the inclusive-window filter excludes it).
    loaded = XGBoostForecaster.load_from_mlflow(fold_run.info.run_id)
    assert loaded.trained_time_series_ids == [1]


def test_re_materialising_a_fold_with_a_changed_eligible_count_succeeds(
    env: None, dagster_instance: DagsterInstance
) -> None:
    """The same ``(experiment, fold)`` partition materialises twice, updating the counters.

    Regression test for issue #197: ``get_or_create_fold_run`` reuses the fold run by tag, and
    MLflow params are write-once, so logging a changed ``n_eligible_time_series`` as a *param*
    (as this code used to) made the second materialisation fail with "Changing param values is
    not allowed" — after the model had already been uploaded, leaving the fold half-written and
    the Dagster run failed.
    """
    eligible_path = str(Settings().eligible_time_series_data_path)
    # First pass: only ts1 is eligible.
    _write_eligible(eligible_path, (1,))

    _register(dagster_instance)
    assert materialize(
        [trained_cv_model], partition_key=PARTITION_KEY, instance=dagster_instance
    ).success

    client = MlflowClient()
    first_run_id = _fold_run(client).info.run_id
    assert XGBoostForecaster.load_from_mlflow(first_run_id).trained_time_series_ids == [1]

    # Second pass: coverage has extended, so ts2 is now eligible too — the exact input change
    # that used to be rejected. ts2's data is still all past train_end (see the module
    # docstring), so it stays excluded by the training-window filter and the trained population
    # is unchanged; only the *eligible* count grows, which is what this test needs to change.
    _write_eligible(eligible_path, (1, 2))
    assert materialize(
        [trained_cv_model], partition_key=PARTITION_KEY, instance=dagster_instance
    ).success

    # Still one run per (experiment, fold) — the leaderboard's one-run-per-fold model is intact —
    # and it now carries the updated counter.
    fold_run = _fold_run(client)
    assert fold_run.info.run_id == first_run_id
    assert fold_run.data.tags["n_eligible_time_series"] == "2"

    # The model still loads cleanly from the reused run after the second training pass.
    assert XGBoostForecaster.load_from_mlflow(first_run_id).trained_time_series_ids == [1]


def test_trained_cv_model_fails_loudly_when_no_eligible_series(
    env: None, dagster_instance: DagsterInstance
) -> None:
    """With no eligible series for the fold, the asset must fail loudly, not silently succeed."""
    # Replace the eligible table so this fold has no rows (only an unrelated fold), mirroring an
    # un-materialised / coverage-excluded fold in production.
    empty_for_fold = EligibleTimeSeries.validate(
        pl.DataFrame(
            {
                "fold_id": pl.Series(["unrelated_fold"], dtype=pl.String),
                "time_series_id": pl.Series([1], dtype=pl.Int32),
            }
        )
    )
    write_deltalake(
        table_or_uri=str(Settings().eligible_time_series_data_path),
        data=empty_for_fold.to_arrow(),
        mode="overwrite",
        partition_by=["fold_id"],
    )

    _register(dagster_instance)
    result = materialize(
        [trained_cv_model],
        partition_key=PARTITION_KEY,
        instance=dagster_instance,
        raise_on_error=False,
    )

    assert not result.success
    failure = result.failure_data_for_node("trained_cv_model")
    assert failure is not None
    assert "No eligible time series" in str(failure.error)
