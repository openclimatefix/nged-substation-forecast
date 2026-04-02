from datetime import datetime, timedelta, timezone

import polars as pl
import pytest
from hypothesis import given, strategies as st

from contracts.data_schemas import Nwp, PowerForecast, SubstationFlows


def test_substation_flows_validation_mw_or_mva():
    # Valid with MW
    df_mw = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            "substation_number": [123],
            "MW": [10.0],
            "MVA": [None],
            "MVAr": [None],
            "ingested_at": [None],
        }
    ).cast(
        {
            "substation_number": pl.Int32,
            "MW": pl.Float32,
            "MVA": pl.Float32,
            "MVAr": pl.Float32,
            "ingested_at": pl.Datetime(time_unit="us", time_zone="UTC"),
        }
    )

    # Should pass
    SubstationFlows.validate(df_mw)

    # Valid with MVA
    df_mva = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            "substation_number": [123],
            "MW": [None],
            "MVA": [10.0],
            "MVAr": [None],
            "ingested_at": [None],
        }
    ).cast(
        {
            "substation_number": pl.Int32,
            "MW": pl.Float32,
            "MVA": pl.Float32,
            "MVAr": pl.Float32,
            "ingested_at": pl.Datetime(time_unit="us", time_zone="UTC"),
        }
    )

    # Should pass
    SubstationFlows.validate(df_mva)

    # Invalid: neither MW nor MVA has data
    df_none = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            "substation_number": [123],
            "MW": [None],
            "MVA": [None],
            "MVAr": [5.0],
            "ingested_at": [None],
        }
    ).cast(
        {
            "substation_number": pl.Int32,
            "MW": pl.Float32,
            "MVA": pl.Float32,
            "MVAr": pl.Float32,
            "ingested_at": pl.Datetime(time_unit="us", time_zone="UTC"),
        }
    )

    with pytest.raises(ValueError, match="must have non-null data in either 'MW' or 'MVA'"):
        SubstationFlows.validate(df_none)


def test_substation_flows_validation_both():
    # Valid with both
    df_both = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            "substation_number": [123],
            "MW": [10.0],
            "MVA": [12.0],
            "MVAr": [5.0],
            "ingested_at": [datetime(2026, 3, 20, tzinfo=timezone.utc)],
        }
    ).cast(
        {
            "substation_number": pl.Int32,
            "MW": pl.Float32,
            "MVA": pl.Float32,
            "MVAr": pl.Float32,
            "ingested_at": pl.Datetime(time_unit="us", time_zone="UTC"),
        }
    )

    # Should pass
    SubstationFlows.validate(df_both)


def test_power_forecast_validation():
    df = pl.DataFrame(
        {
            "valid_time": [datetime(2026, 1, 2, tzinfo=timezone.utc)],
            "substation_number": [123],
            "ensemble_member": [0],
            "power_fcst_model_name": ["xgboost"],
            "power_fcst_init_time": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            "nwp_init_time": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            "nwp_init_hour": [0],
            "lead_time_hours": [24.0],
            "power_fcst_init_year_month": ["2026-01"],
            "MW_or_MVA": [50.5],
        }
    ).cast(
        {
            "substation_number": pl.Int32,
            "ensemble_member": pl.UInt8,
            "power_fcst_model_name": pl.Categorical,
            "nwp_init_hour": pl.Int32,
            "lead_time_hours": pl.Float32,
            "MW_or_MVA": pl.Float32,
        }
    )

    # Should pass
    PowerForecast.validate(df)


def test_nwp_validation():
    init_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    base_data = {
        "init_time": [init_time, init_time],
        "ensemble_member": [0, 0],
        "h3_index": [123, 123],
        "temperature_2m": [10, 11],
        "dew_point_temperature_2m": [5, 6],
        "wind_u_10m": [2.0, 3.0],
        "wind_v_10m": [0.0, 0.0],
        "wind_u_100m": [4.0, 5.0],
        "wind_v_100m": [0.0, 0.0],
        "pressure_surface": [100, 101],
        "pressure_reduced_to_mean_sea_level": [102, 103],
        "geopotential_height_500hpa": [50, 51],
        "categorical_precipitation_type_surface": [0, 0],
    }

    nwp_vars_to_uint8 = {
        col: pl.UInt8
        for col in [
            "ensemble_member",
            "categorical_precipitation_type_surface",
        ]
    }
    nwp_vars_to_float32 = {
        col: pl.Float32
        for col in [
            "temperature_2m",
            "dew_point_temperature_2m",
            "pressure_surface",
            "pressure_reduced_to_mean_sea_level",
            "geopotential_height_500hpa",
            "precipitation_surface",
            "downward_short_wave_radiation_flux_surface",
            "downward_long_wave_radiation_flux_surface",
            "wind_u_10m",
            "wind_v_10m",
            "wind_u_100m",
            "wind_v_100m",
        ]
    }

    # Case 1: Valid (Null at step 0, value at step 1)
    df_valid = pl.DataFrame(
        {
            **base_data,
            "valid_time": [init_time, init_time + timedelta(hours=1)],
            "precipitation_surface": [None, 1],
            "downward_short_wave_radiation_flux_surface": [None, 100],
            "downward_long_wave_radiation_flux_surface": [None, 200],
        }
    ).cast({**nwp_vars_to_uint8, **nwp_vars_to_float32, "h3_index": pl.UInt64})

    # Should pass
    Nwp.validate(df_valid)

    # Case 2: Invalid (Null at step 1)
    df_invalid = pl.DataFrame(
        {
            **base_data,
            "valid_time": [init_time, init_time + timedelta(hours=1)],
            "precipitation_surface": [None, None],
            "downward_short_wave_radiation_flux_surface": [None, 100],
            "downward_long_wave_radiation_flux_surface": [None, 200],
        }
    ).cast({**nwp_vars_to_uint8, **nwp_vars_to_float32, "h3_index": pl.UInt64})

    with pytest.raises(ValueError, match="Column 'precipitation_surface' contains 1 null values"):
        Nwp.validate(df_invalid)


@given(
    mw=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)
    | st.none(),
    mva=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)
    | st.none(),
)
def test_substation_flows_property_based(mw, mva):
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            "substation_number": [123],
            "MW": [mw],
            "MVA": [mva],
            "MVAr": [None],
            "ingested_at": [None],
        }
    ).cast(
        {
            "substation_number": pl.Int32,
            "MW": pl.Float32,
            "MVA": pl.Float32,
            "MVAr": pl.Float32,
            "ingested_at": pl.Datetime(time_unit="us", time_zone="UTC"),
        }
    )

    if mw is None and mva is None:
        with pytest.raises(ValueError, match="must have non-null data in either 'MW' or 'MVA'"):
            SubstationFlows.validate(df)
    else:
        SubstationFlows.validate(df)
