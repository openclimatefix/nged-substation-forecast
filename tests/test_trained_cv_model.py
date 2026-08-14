"""Integration test for the ``trained_cv_model`` asset.

Exercises the real wiring end-to-end against a file-based MLflow + temp Delta tables: register an
experiment (Phase 3), then materialise ``trained_cv_model`` for its fold and assert the model
artifact round-trips from MLflow and that training honoured the fold's eligible population and
inclusive training window.
"""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import mlflow
import polars as pl
import pytest
from _nwp_test_data import half_hours, nwp_records, write_test_nwp
from contracts.ml_schemas import EligibleTimeSeries
from contracts.settings import Settings
from dagster import DagsterInstance, materialize
from deltalake import write_deltalake
from ml_core._production_helpers import fetch_model_artifacts
from ml_core.base_forecaster import load_trained_metadata
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
from xgboost_forecaster.forecaster import XGBoostForecaster

from nged_substation_forecast.defs._engineering_inputs import load_engineering_inputs
from nged_substation_forecast.defs.cv_assets import _load_roster, trained_cv_model

pytestmark = pytest.mark.integration

RegisterExperiment = Callable[[DagsterInstance, str], None]
"""Type of the ``register_experiment`` fixture (``tests/conftest.py``)."""

FOLD_ID = "mid_2025_to_mid_2026"
EXPERIMENT_NAME = "exp_smoke"
PARTITION_KEY = f"{EXPERIMENT_NAME}__{FOLD_ID}"

# ts1 sits in H3 cell 10 with in-window data (trained); ts2 sits in cell 20 with data only after
# train_end 2025-06-30 (eligible, but excluded by the training-window filter).
_TS1_CELL = 10
_TS2_CELL = 20
_IN_WINDOW = datetime(2024, 6, 1, tzinfo=UTC)
_AFTER_TRAIN_END = datetime(2025, 8, 1, tzinfo=UTC)

_TRAIN_START = datetime(2024, 4, 1, tzinfo=UTC)
"""The fold's inclusive training-window start, from ``conf/cv/default.yaml``."""

_EARLY_INIT_TIME = _TRAIN_START - timedelta(days=10)
"""``init_time`` of one NWP run forecasting into the window from before it.

Inside ``MAX_NWP_LEAD`` (16 days) of ``_TRAIN_START``, so the default ``init_time`` lower bound
must reach it; before ``_TRAIN_START``, so nothing else does.
"""


def _write_power(path: str) -> None:
    rows = [
        {"time_series_id": ts, "time": t, "power": 100.0 + i}
        for ts, day in ((1, _IN_WINDOW), (2, _AFTER_TRAIN_END))
        for i, t in enumerate(half_hours(day))
    ]
    pl.DataFrame(rows).cast(
        {"time_series_id": pl.Int32, "time": pl.Datetime("us", "UTC"), "power": pl.Float32}
    ).write_delta(path)


_NWP_ENSEMBLE_MEMBERS = (0, 1, 2)
"""Members written to the synthetic NWP. Member 0 is the control; 1 and 2 exercise the
``ensemble_members`` filter in ``load_engineering_inputs`` (training keeps only the control)."""


def _write_nwp(path: str) -> None:
    """Write a minimal Nwp-shaped Delta (Float32 physical-unit weather cols).

    Each (cell, valid_time) carries all of ``_NWP_ENSEMBLE_MEMBERS`` so tests can assert that
    training narrows NWP to the control member while prediction would keep every member.

    Three (cell, day) combinations, so each of ``load_engineering_inputs``'s scan predicates is the
    *only* thing that can remove one of them:

    - ts1's cell, in-window — the rows every test expects to survive.
    - ts2's cell, in-window on the same ``init_time`` and ``valid_time``s as ts1's. Only the
      ``h3_index`` predicate can drop these, so a test that requests ts1 alone and still sees ts2's
      cell has caught the cell prune going missing.
    - ts1's cell again, initialised at ``_EARLY_INIT_TIME`` — before the training window — and
      forecasting into its first day. Only the ``MAX_NWP_LEAD`` lookback keeps these, so their
      absence means the lookback has gone.
    """
    records = (
        nwp_records(_TS1_CELL, _IN_WINDOW, _NWP_ENSEMBLE_MEMBERS)
        + nwp_records(_TS2_CELL, _IN_WINDOW, _NWP_ENSEMBLE_MEMBERS)
        + nwp_records(_TS1_CELL, _TRAIN_START, _NWP_ENSEMBLE_MEMBERS, init_time=_EARLY_INIT_TIME)
    )
    write_test_nwp(path, records)


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


def test_load_engineering_inputs_filters_ensemble_members(env: None) -> None:
    """``ensemble_members`` narrows NWP at the scan; ``None`` keeps every member.

    This is the lever that keeps training (control member only) from fanning every forecast row out
    across all ~51 members against the same power target — the source of the training OOM.
    """
    settings = Settings()
    train_start = datetime(2024, 4, 1, tzinfo=UTC)
    train_end = datetime(2025, 6, 30, 23, 59, 59, tzinfo=UTC)

    metadata = _load_roster(settings, [1, 2])
    _, nwp_control = load_engineering_inputs(
        settings, [1, 2], metadata, train_start, train_end, ensemble_members=[0]
    )
    assert nwp_control.collect()["ensemble_member"].unique().sort().to_list() == [0]

    _, nwp_all = load_engineering_inputs(settings, [1, 2], metadata, train_start, train_end)
    assert nwp_all.collect()["ensemble_member"].unique().sort().to_list() == list(
        _NWP_ENSEMBLE_MEMBERS
    )


def test_load_engineering_inputs_prunes_nwp_to_requested_cells_and_init_window(
    env: None,
) -> None:
    """NWP is pruned to the requested series' H3 cells, over an ``init_time`` range that reaches
    back ``MAX_NWP_LEAD`` before the window."""
    settings = Settings()
    train_start = _TRAIN_START
    train_end = datetime(2025, 6, 30, 23, 59, 59, tzinfo=UTC)

    # Both cells carry the same in-window init_time and valid_times, so requesting ts1 alone can
    # only drop ts2's cell through the h3_index predicate.
    _, nwp_ts1 = load_engineering_inputs(
        settings,
        time_series_ids=[1],
        metadata=_load_roster(settings, [1]),
        window_start=train_start,
        window_end=train_end,
    )
    assert nwp_ts1.collect()["h3_index"].unique().to_list() == [_TS1_CELL]
    # ts2's only power is after train_end, so widen the window before asserting the power scan is
    # pruned by population — otherwise the window filter would be doing the work.
    power_wide, _ = load_engineering_inputs(
        settings,
        time_series_ids=[1],
        metadata=_load_roster(settings, [1]),
        window_start=train_start,
        window_end=_AFTER_TRAIN_END + timedelta(days=1),
    )
    assert power_wide.collect()["time_series_id"].unique().to_list() == [1]

    # Requesting both keeps both, so the line above is a prune rather than an empty table.
    _, nwp_both = load_engineering_inputs(
        settings,
        time_series_ids=[1, 2],
        metadata=_load_roster(settings, [1, 2]),
        window_start=train_start,
        window_end=train_end,
    )
    assert sorted(nwp_both.collect()["h3_index"].unique().to_list()) == [_TS1_CELL, _TS2_CELL]

    # The run initialised before the window still reaches it: init_time_start defaults to
    # window_start - MAX_NWP_LEAD, not to window_start.
    assert _EARLY_INIT_TIME in nwp_ts1.collect()["init_time"].unique().to_list()


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
    env: None,
    dagster_instance: DagsterInstance,
    tmp_path: Path,
    register_experiment: RegisterExperiment,
) -> None:
    register_experiment(dagster_instance, EXPERIMENT_NAME)

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

    # The archive also carries the roster rows the model was engineered against, which is what
    # `live_forecasts` locates its time series by instead of reading the roster.
    model_dir = tmp_path / "promoted"
    fetch_model_artifacts(fold_run.info.run_id, model_dir)
    frozen = load_trained_metadata(model_dir)
    # Narrowed to the trained population, not the wider eligible one it was engineered over.
    assert frozen["time_series_id"].to_list() == [1]
    assert frozen["h3_res_5"].item() == _TS1_CELL


def test_re_materialising_a_fold_with_a_changed_eligible_count_succeeds(
    env: None,
    dagster_instance: DagsterInstance,
    register_experiment: RegisterExperiment,
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

    register_experiment(dagster_instance, EXPERIMENT_NAME)
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
    env: None,
    dagster_instance: DagsterInstance,
    register_experiment: RegisterExperiment,
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

    register_experiment(dagster_instance, EXPERIMENT_NAME)
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


def test_trained_cv_model_fails_loudly_when_an_eligible_series_has_no_metadata(
    env: None,
    dagster_instance: DagsterInstance,
    register_experiment: RegisterExperiment,
) -> None:
    """R&D fails fast: a CV run must not quietly train fewer series than its population names.

    A series with no metadata row has no H3 cell, so it can never be joined to NWP. Training
    around it silently would make this fold's leaderboard numbers incomparable with every other
    experiment's, which is exactly what the shared eligible population exists to prevent.
    """
    _write_eligible(str(Settings().eligible_time_series_data_path), time_series_ids=(1, 2, 9))

    register_experiment(dagster_instance, EXPERIMENT_NAME)
    result = materialize(
        [trained_cv_model],
        partition_key=PARTITION_KEY,
        instance=dagster_instance,
        raise_on_error=False,
    )

    assert not result.success
    failure = result.failure_data_for_node("trained_cv_model")
    assert failure is not None
    assert "no row in the metadata parquet" in str(failure.error)
    assert "[9]" in str(failure.error)
