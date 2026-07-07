"""Unit tests for the FeatureEngineer strategy and its nearest-cell NWP spatial join."""

from datetime import datetime

import patito as pt
import polars as pl
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import Nwp
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
    temp_by_ts = dict(zip(result["time_series_id"], result["temperature_2m"]))
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
