from datetime import datetime, timezone

import patito as pt
import polars as pl
from contracts.weather_schemas import NwpInMemory, NwpOnDisk, NwpScalingParams


def test_nwp_scaling_roundtrip():
    # Create in-memory data
    in_memory_df = (
        pt.DataFrame(
            {
                "init_time": [datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)],
                "valid_time": [datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)],
                "ensemble_member": [1],
                "h3_index": [123456789],
                "categorical_precipitation_type_surface": [0],
                "temperature_2m": [20.0],
                "dew_point_temperature_2m": [10.0],
                "wind_speed_10m": [5.0],
                "wind_direction_10m": [180.0],
                "wind_speed_100m": [10.0],
                "wind_direction_100m": [180.0],
                "pressure_surface": [1000.0],
                "pressure_reduced_to_mean_sea_level": [1013.0],
                "geopotential_height_500hpa": [5000.0],
                "downward_long_wave_radiation_flux_surface": [0.0],
                "downward_short_wave_radiation_flux_surface": [0.0],
                "precipitation_surface": [0.0],
            }
        )
        .set_model(NwpInMemory)
        .cast()
    )

    # Create scaling params for all columns
    all_scaling_params = pt.DataFrame(
        {
            "col_name": [
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
            ],
            "buffered_min": [0.0] * 12,
            "buffered_max": [10000.0] * 12,
            "buffered_range": [10000.0] * 12,
        }
    ).set_model(NwpScalingParams)

    # Convert to on-disk
    on_disk_df = NwpOnDisk.from_nwp_in_memory(in_memory_df, all_scaling_params)
    assert on_disk_df.schema["temperature_2m"] == pl.Int16

    # Convert back to in-memory
    back_in_memory_df = NwpOnDisk.to_nwp_in_memory(on_disk_df, all_scaling_params)

    # Check that temperature_2m is close to original (due to integer rounding)
    assert abs(back_in_memory_df["temperature_2m"][0] - in_memory_df["temperature_2m"][0]) < 1.0
