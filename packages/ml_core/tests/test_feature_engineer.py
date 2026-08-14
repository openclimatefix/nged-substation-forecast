"""Unit tests for the FeatureEngineer strategy and its nearest-cell NWP spatial join."""

import inspect
from datetime import datetime

import patito as pt
import polars as pl
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import Nwp
from ml_core.features.feature_engineer import DEFAULT_LOCAL_TIMEZONE, FeatureEngineer
from ml_core.features.tabular_feature_engineer import (
    TabularFeatureEngineer,
    _attach_nearest_nwp_cell,
    _engineer_features,
)
from polars.testing import assert_frame_equal


def _nwp_two_cells() -> pt.LazyFrame[Nwp]:
    """NWP for two H3 cells (10, 20) and one unmatched cell (99)."""
    valid_time = datetime(2024, 6, 1, 12, 0)
    init_time = datetime(2024, 6, 1, 0, 0)
    df = pl.DataFrame(
        {
            "h3_index": pl.Series([10, 20, 99], dtype=pl.UInt64),
            "valid_time": [valid_time, valid_time, valid_time],
            "init_time": [init_time, init_time, init_time],
            "ensemble_member": pl.Series([0, 0, 0], dtype=pl.UInt8),
            "temperature_2m": [10.0, 12.0, 14.0],
        }
    )
    return pt.LazyFrame.from_existing(df.lazy()).set_model(Nwp)


def _metadata_two_series() -> pt.DataFrame[TimeSeriesMetadata]:
    """ts1 -> cell 10, ts2 -> cell 20 (both real cells); no series in cell 99."""
    df = pl.DataFrame(
        {
            "time_series_id": [1, 2],
            "h3_res_5": pl.Series([10, 20], dtype=pl.UInt64),
            "time_series_type": ["Primary", "Primary"],
        }
    )
    return pt.DataFrame(df).set_model(TimeSeriesMetadata)


def test_attach_nearest_nwp_cell_maps_cells_to_series() -> None:
    """Each NWP cell becomes the time series in it; the unmatched cell is dropped."""
    result = _attach_nearest_nwp_cell(_nwp_two_cells(), _metadata_two_series()).collect()

    assert "h3_index" not in result.columns
    assert "time_series_id" in result.columns
    # Cells 10 and 20 matched to ts1 and ts2; cell 99 dropped (inner join).
    assert sorted(result["time_series_id"].to_list()) == [1, 2]
    temp_by_ts = dict(zip(result["time_series_id"], result["temperature_2m"], strict=True))
    assert temp_by_ts == {1: 10.0, 2: 12.0}


def test_attach_nearest_nwp_cell_replicates_shared_cell() -> None:
    """Two time series in the same cell both receive that cell's weather."""
    metadata = pt.DataFrame(
        pl.DataFrame(
            {
                "time_series_id": [1, 2],
                "h3_res_5": pl.Series([10, 10], dtype=pl.UInt64),
                "time_series_type": ["Primary", "Primary"],
            }
        )
    ).set_model(TimeSeriesMetadata)

    result = _attach_nearest_nwp_cell(_nwp_two_cells(), metadata).collect()

    cell_10_rows = result.filter(pl.col("temperature_2m") == 10.0)
    assert sorted(cell_10_rows["time_series_id"].to_list()) == [1, 2]


def test_tabular_feature_engineer_returns_all_features_shape() -> None:
    """The default engineer maps cells then runs the tabular pipeline, yielding AllFeatures."""
    valid_time = datetime(2024, 6, 1, 12, 0)
    power = pt.LazyFrame.from_existing(
        pl.DataFrame({"time_series_id": [1], "time": [valid_time], "power": [100.0]}).lazy()
    ).set_model(PowerTimeSeries)

    result = TabularFeatureEngineer().engineer(
        selected_features={"temperature_2m"},
        power_time_series=power,
        time_series_metadata=_metadata_two_series(),
        nwp=_nwp_two_cells(),
    )

    collected = result.collect()
    # Output is the base AllFeatures columns plus the requested feature.
    assert {"time_series_id", "valid_time", "power", "temperature_2m"} <= set(collected.columns)
    # ts1's row picks up cell-10 weather (10.0); the join keyed on time_series_id.
    ts1 = collected.filter(pl.col("time_series_id") == 1)
    assert ts1["temperature_2m"].to_list() == [10.0]


def test_tabular_feature_engineer_drops_series_with_no_metadata_row() -> None:
    """A series with power observations but no metadata row produces no output rows at all.

    Single-run mode is power-centric, so without the semi-join in ``_engineer_features`` the
    series would survive with every weather feature null and be predicted on regardless — a
    garbage forecast that reads as healthy to ``live_forecasts_are_healthy`` because the series
    is present. ``AllFeatures.time_series_type`` is declared non-nullable on the strength of this.
    """
    power_fcst_init_time = datetime(2024, 6, 1, 12, 0)
    # ts1 has a metadata row; ts3 does not.
    power = pt.LazyFrame.from_existing(
        pl.DataFrame(
            {
                "time_series_id": [1, 3],
                "time": [power_fcst_init_time, power_fcst_init_time],
                "power": [100.0, 200.0],
            }
        ).lazy()
    ).set_model(PowerTimeSeries)

    result = (
        TabularFeatureEngineer()
        .engineer(
            selected_features={"temperature_2m", "time_series_type"},
            power_time_series=power,
            time_series_metadata=_metadata_two_series(),
            nwp=_nwp_two_cells(),
            power_fcst_init_time=power_fcst_init_time,
            nwp_init_time=datetime(2024, 6, 1, 0, 0),
        )
        .collect()
    )

    assert result["time_series_id"].to_list() == [1]
    assert result["time_series_type"].null_count() == 0


def test_tabular_feature_engineer_single_run_params_reach_engineer_features() -> None:
    """``TabularFeatureEngineer.engineer``'s single-run kwargs pass through unchanged.

    The public ``engineer()`` interface doesn't implement single-run mode itself — it just
    forwards to ``_engineer_features``, which already does. This locks that passthrough: calling
    through ``engineer()`` must equal calling ``_engineer_features`` directly with the same
    single-run args on the same (already cell-mapped) NWP.
    """
    power_fcst_init_time = datetime(2024, 6, 1, 12, 0)
    nwp_init_time = datetime(2024, 6, 1, 0, 0)
    power = pt.LazyFrame.from_existing(
        pl.DataFrame(
            {"time_series_id": [1], "time": [power_fcst_init_time], "power": [100.0]}
        ).lazy()
    ).set_model(PowerTimeSeries)
    metadata = _metadata_two_series()
    nwp = _nwp_two_cells()

    via_engineer = (
        TabularFeatureEngineer()
        .engineer(
            selected_features={"temperature_2m"},
            power_time_series=power,
            time_series_metadata=metadata,
            nwp=nwp,
            power_fcst_init_time=power_fcst_init_time,
            nwp_init_time=nwp_init_time,
        )
        .collect()
    )

    nwp_per_time_series = _attach_nearest_nwp_cell(nwp, metadata)
    direct = _engineer_features(
        {"temperature_2m"},
        power,
        metadata,
        nwp=nwp_per_time_series,
        power_fcst_init_time=power_fcst_init_time,
        nwp_init_time=nwp_init_time,
    ).collect()

    assert_frame_equal(
        via_engineer.sort("time_series_id"), direct.sort("time_series_id"), check_dtypes=False
    )


def test_tabular_feature_engineer_threads_local_timezone() -> None:
    """``local_timezone`` reaches the local-time features through the public ``engineer()`` call.

    Unlike ``test_apply_local_time_features_non_london_timezone`` in ``test_features.py``, which
    calls the bottom-most helper directly, this goes through ``TabularFeatureEngineer.engineer()``
    — the composition point every production call site uses
    (``forecaster.feature_engineer.engineer(...)``). ``local_timezone`` passes through two
    intermediate layers (``_engineer_features``, then ``_apply_post_join_features``) before
    reaching ``_apply_local_time_features``; a dropped passthrough at either layer would silently
    revert every caller to ``Europe/London`` while the direct-call unit test kept passing.
    """
    valid_time = datetime(2024, 6, 1, 12, 0)  # 17:30 in Asia/Kolkata (fixed UTC+5:30, no DST).
    power = pt.LazyFrame.from_existing(
        pl.DataFrame({"time_series_id": [1], "time": [valid_time], "power": [100.0]}).lazy()
    ).set_model(PowerTimeSeries)

    result = (
        TabularFeatureEngineer()
        .engineer(
            selected_features={"local_utc_offset_minutes"},
            power_time_series=power,
            time_series_metadata=_metadata_two_series(),
            nwp=_nwp_two_cells(),
            local_timezone="Asia/Kolkata",
        )
        .collect()
    )

    # Bulk mode is NWP-centric, so both cells' time series appear even though only ts1 has a
    # power observation; every row shares the same valid_time, so every offset is 330.
    assert set(result["local_utc_offset_minutes"].to_list()) == {330}
    assert result.height == 2


def test_tabular_feature_engineer_default_local_timezone_is_london() -> None:
    """Calling ``engineer()`` with no ``local_timezone`` must default production to Europe/London.

    Every production and CV call site (``production_assets.py``, ``cv_assets.py``) calls
    ``engineer()`` without passing ``local_timezone`` as of writing, so a wrong default would
    reach them silently. This test pins only the parameter's default value, not those call sites
    themselves, and it is invisible to every other test in this module, which passes
    ``local_timezone`` explicitly. British Summer Time gives a UTC+1 offset in June, so the
    correct default (``Europe/London``) yields ``local_utc_offset_minutes == 60``; a wrong
    default (e.g. ``Asia/Kolkata``'s fixed +5:30) would silently produce ``330`` instead, with
    every other test in the suite still green.
    """
    valid_time = datetime(2024, 6, 1, 12, 0)
    power = pt.LazyFrame.from_existing(
        pl.DataFrame({"time_series_id": [1], "time": [valid_time], "power": [100.0]}).lazy()
    ).set_model(PowerTimeSeries)

    result = (
        TabularFeatureEngineer()
        .engineer(
            selected_features={"local_utc_offset_minutes"},
            power_time_series=power,
            time_series_metadata=_metadata_two_series(),
            nwp=_nwp_two_cells(),
        )
        .collect()
    )

    assert set(result["local_utc_offset_minutes"].to_list()) == {60}


def test_feature_engineer_abc_declares_local_timezone_parameter() -> None:
    """``local_timezone`` is part of the ``FeatureEngineer`` interface, not just the one subclass.

    Inspects the *abstract* method's own signature, deliberately not a concrete instance assigned
    to a ``FeatureEngineer``-typed variable: ``ty`` narrows a declared-``FeatureEngineer`` receiver
    back to the concrete ``TabularFeatureEngineer`` on assignment, so a test written that way would
    still type-check even if ``local_timezone`` were deleted from the ABC — silently passing both
    pytest and ``ty``. Reading ``FeatureEngineer.engineer``'s signature directly has no such
    loophole: a future ``FeatureEngineer`` implementation for another region can only be relied on
    to accept ``local_timezone`` if the interface itself declares it.
    """
    parameters = inspect.signature(FeatureEngineer.engineer).parameters
    assert "local_timezone" in parameters
    assert parameters["local_timezone"].default == DEFAULT_LOCAL_TIMEZONE
