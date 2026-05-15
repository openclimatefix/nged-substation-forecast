from datetime import datetime, timezone

import pytest
import polars as pl
import patito as pt
from contracts.weather_schemas import NwpInMemory

def test_categorical_precipitation_type_surface_validation():
    # Test case 1: Valid data (all null before 2024-11-13, not null after)
    df_valid = pl.DataFrame({
        "nwp_model_id": ["ECMWF_ENS_0_25_degree", "ECMWF_ENS_0_25_degree"],
        "init_time": [
            datetime(2024, 11, 13, tzinfo=timezone.utc),
            datetime(2024, 11, 14, tzinfo=timezone.utc)
        ],
        "valid_time": [
            datetime(2024, 11, 13, 0, tzinfo=timezone.utc),
            datetime(2024, 11, 14, 1, tzinfo=timezone.utc)
        ],
        "ensemble_member": [1, 1],
        "h3_index": [1, 1],
        "categorical_precipitation_type_surface": [None, 1],
        "temperature_2m": [10.0, 10.0],
        "dew_point_temperature_2m": [5.0, 5.0],
        "wind_speed_10m": [5.0, 5.0],
        "wind_direction_10m": [180.0, 180.0],
        "wind_speed_100m": [5.0, 5.0],
        "wind_direction_100m": [180.0, 180.0],
        "pressure_surface": [1000.0, 1000.0],
        "pressure_reduced_to_mean_sea_level": [1000.0, 1000.0],
        "geopotential_height_500hpa": [5000.0, 5000.0],
        "downward_long_wave_radiation_flux_surface": [None, 100.0],
        "downward_short_wave_radiation_flux_surface": [None, 100.0],
        "precipitation_surface": [None, 0.01],
    })
    
    # This should pass
    NwpInMemory.validate(pt.DataFrame(df_valid).set_model(NwpInMemory).cast())

    # Test case 2: Invalid data (not null before 2024-11-13)
    df_invalid_before = df_valid.with_columns(
        pl.Series("categorical_precipitation_type_surface", [1, 1])
    )
    with pytest.raises(ValueError, match="must be all null for init_time <= 2024-11-13"):
        NwpInMemory.validate(pt.DataFrame(df_invalid_before).set_model(NwpInMemory).cast())

    # Test case 3: Invalid data (null after 2024-11-13)
    df_invalid_after = df_valid.with_columns(
        pl.Series("categorical_precipitation_type_surface", [None, None])
    )
    with pytest.raises(ValueError, match="must not be null for init_time > 2024-11-13"):
        NwpInMemory.validate(pt.DataFrame(df_invalid_after).set_model(NwpInMemory).cast())
