from datetime import UTC, datetime, timedelta

import patito as pt
import polars as pl
import pytest
from contracts.common import DELIVERY_QUANTILES, quantile_label
from contracts.power_schemas import POWER_FCST_QUANTILE_COLUMNS, PowerForecast

_INCREASING_QUANTILES: list[float] = [float(i) for i in range(len(DELIVERY_QUANTILES))]


def _forecast_rows(
    valid_time_offsets_minutes: list[int], quantiles: list[list[float]] | None = None
) -> pt.DataFrame:
    """One forecast row per offset, with the 13 quantile columns taken from ``quantiles``."""
    n_rows = len(valid_time_offsets_minutes)
    quantiles = quantiles or [_INCREASING_QUANTILES] * n_rows
    return (
        pt.DataFrame(
            {
                "valid_time": [
                    datetime(2026, 1, 1, 0, 0, tzinfo=UTC) + timedelta(minutes=offset)
                    for offset in valid_time_offsets_minutes
                ],
                "time_series_id": [123] * n_rows,
                "ml_flow_experiment_id": [1] * n_rows,
                "nwp_init_time": [datetime(2026, 1, 1, 0, 0, tzinfo=UTC)] * n_rows,
                "power_fcst_model_name": ["model_a"] * n_rows,
                "power_fcst_model_version": [1] * n_rows,
                "power_fcst_init_time": [datetime(2026, 1, 1, 0, 0, tzinfo=UTC)] * n_rows,
                "experiment_name": ["baseline"] * n_rows,
                "fold_id": ["live"] * n_rows,
            }
            | {
                column: [row[position] for row in quantiles]
                for position, column in enumerate(POWER_FCST_QUANTILE_COLUMNS)
            }
        )
        .set_model(PowerForecast)
        .cast()
    )


def test_power_forecast_quantile_columns_follow_the_delivery_levels():
    """The 13 columns are the delivery levels, in order, and each is ``Float32``.

    ``delta_store`` writes and rounds ``Float32``; nothing inside `contracts` itself would
    otherwise pin the dtype this schema declares.
    """
    assert (
        tuple(f"power_fcst_{quantile_label(quantile)}" for quantile in DELIVERY_QUANTILES)
        == POWER_FCST_QUANTILE_COLUMNS
    )
    assert all(PowerForecast.dtypes[column] == pl.Float32 for column in POWER_FCST_QUANTILE_COLUMNS)
    assert "ensemble_member" not in PowerForecast.columns


def test_power_forecast_validation():
    _forecast_rows(valid_time_offsets_minutes=[30]).validate()


def test_power_forecast_accepts_equal_quantiles():
    """A deterministic forecast carries one value in all 13 columns."""
    _forecast_rows(valid_time_offsets_minutes=[30], quantiles=[[5.0] * 13]).validate()


def test_power_forecast_rejects_a_quantile_below_the_level_beneath_it():
    crossed = list(_INCREASING_QUANTILES)
    crossed[3], crossed[4] = crossed[4], crossed[3]  # power_fcst_p10 > power_fcst_p20
    with pytest.raises(Exception, match="power_fcst_p20"):
        _forecast_rows(valid_time_offsets_minutes=[30], quantiles=[crossed]).validate()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_power_forecast_rejects_a_non_finite_quantile(bad_value: float):
    """A NaN passes the ordering constraint, so each column carries its own finite check."""
    quantiles = [*_INCREASING_QUANTILES[:-1], bad_value]
    with pytest.raises(Exception, match="power_fcst_p99"):
        _forecast_rows(valid_time_offsets_minutes=[30], quantiles=[quantiles]).validate()


def test_power_forecast_rejects_duplicate_primary_key():
    """Two rows sharing the full primary key mean the same forecast reached us twice."""
    with pytest.raises(ValueError, match="Duplicate entries found for primary key"):
        _forecast_rows(valid_time_offsets_minutes=[30, 30]).validate()


def test_power_forecast_accepts_rows_differing_only_in_valid_time():
    """`valid_time` is part of the key, so two lead times of one run are not duplicates."""
    _forecast_rows(valid_time_offsets_minutes=[30, 60]).validate()
