from datetime import datetime, timedelta

import patito as pt
import polars as pl
import pytest
from pydantic import ValidationError
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import NwpInMemory
from ml_core.features import (
    STATIC_FEATURE_REGISTRY,
    LagFeature,
    ParsedFeatures,
    _apply_local_time_features,
    _apply_power_lag,
    _apply_rolling_mean_feature,
    _nullify_leaky_lags,
    _process_nwp,
    _upsample_nwp_to_half_hourly,
    engineer_features,
)


def test_apply_power_lag_with_source():
    # Test that _apply_power_lag correctly pulls from source_lf
    target_df = pl.DataFrame(
        {
            "time_series_id": ["ts1"],
            "ensemble_member": [0],
            "valid_time": [datetime(2023, 1, 1, 12, 0)],
            "power": [100.0],
        }
    )

    source_df = pl.DataFrame(
        {
            "time_series_id": ["ts1"],
            "ensemble_member": [0],
            "valid_time": [datetime(2023, 1, 1, 11, 0)],
            "power": [90.0],
        }
    )

    lag_feat = LagFeature.from_str("power_lag_1h")
    result = _apply_power_lag(target_df.lazy(), source_df.lazy(), lag_feat).collect()

    assert result["power_lag_1h"][0] == 90.0


def test_process_nwp():
    df = pl.DataFrame(
        {
            "valid_time": [datetime(2020, 1, 1, 12), datetime(2020, 1, 1, 15)],
            "init_time": [datetime(2020, 1, 1, 0), datetime(2020, 1, 1, 0)],
        }
    )

    result = (
        _process_nwp(pt.LazyFrame.from_existing(df.lazy()).set_model(NwpInMemory))
        .collect()
        .sort("valid_time")
    )

    assert "nwp_lead_time_hours" in result.columns
    assert "nwp_init_time" in result.columns
    assert "init_time" not in result.columns
    # 3-hourly input (12:00→15:00) upsampled to 30-min gives 7 rows
    assert len(result) == 7
    assert result["nwp_lead_time_hours"].to_list() == [12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0]


def test_upsample_nwp_to_half_hourly_interpolates_continuous_vars():
    """Continuous weather vars are linearly interpolated between steps."""
    df = pl.DataFrame(
        {
            "nwp_init_time": [datetime(2020, 1, 1, 0)] * 2,
            "ensemble_member": [0, 0],
            "valid_time": [datetime(2020, 1, 1, 0), datetime(2020, 1, 1, 3)],
            "temperature_2m": [10.0, 16.0],
        }
    )
    result = _upsample_nwp_to_half_hourly(df.lazy()).collect().sort("valid_time")

    assert len(result) == 7  # 0:00, 0:30, 1:00, 1:30, 2:00, 2:30, 3:00
    # Midpoint at 1:30: 10.0 + (3/6) * (16.0 - 10.0) = 13.0
    mid = result.filter(pl.col("valid_time") == datetime(2020, 1, 1, 1, 30))
    assert mid["temperature_2m"][0] == pytest.approx(13.0)


def test_upsample_nwp_to_half_hourly_forward_fills_categorical_vars():
    """Categorical vars are forward-filled, not interpolated."""
    df = pl.DataFrame(
        {
            "nwp_init_time": [datetime(2020, 1, 1, 0)] * 2,
            "ensemble_member": [0, 0],
            "valid_time": [datetime(2020, 1, 1, 0), datetime(2020, 1, 1, 3)],
            "categorical_precipitation_type_surface": pl.Series([0, 5], dtype=pl.UInt8),
        }
    )
    result = _upsample_nwp_to_half_hourly(df.lazy()).collect().sort("valid_time")

    # At 1:30 (before the 3:00 step), categorical should still be 0
    before = result.filter(pl.col("valid_time") == datetime(2020, 1, 1, 1, 30))
    assert before["categorical_precipitation_type_surface"][0] == 0
    # At 3:00, it transitions to 5
    at = result.filter(pl.col("valid_time") == datetime(2020, 1, 1, 3, 0))
    assert at["categorical_precipitation_type_surface"][0] == 5


def test_upsample_nwp_no_cross_group_interpolation():
    """interpolate().over() must not fill nulls across group boundaries.

    precipitation_surface is legitimately null at lead time 0 in each NWP group.
    After upsampling, those nulls must stay null — they must not be back-filled by
    the preceding group's last known value via global interpolation.
    """
    df = pl.DataFrame(
        {
            "nwp_init_time": [
                datetime(2020, 1, 1, 0),
                datetime(2020, 1, 1, 0),  # group A: 0h, 3h
                datetime(2020, 1, 1, 6),
                datetime(2020, 1, 1, 6),  # group B: 0h, 3h
            ],
            "ensemble_member": [0, 0, 0, 0],
            "valid_time": [
                datetime(2020, 1, 1, 0),
                datetime(2020, 1, 1, 3),
                datetime(2020, 1, 1, 6),
                datetime(2020, 1, 1, 9),
            ],
            "precipitation_surface": pl.Series([None, 0.002, None, 0.003], dtype=pl.Float32),
        }
    )
    result = _upsample_nwp_to_half_hourly(df.lazy()).collect().sort(["nwp_init_time", "valid_time"])

    # Group A lead-time-0 null stays null
    a_lt0 = result.filter(
        (pl.col("nwp_init_time") == datetime(2020, 1, 1, 0))
        & (pl.col("valid_time") == datetime(2020, 1, 1, 0))
    )
    assert a_lt0["precipitation_surface"][0] is None

    # Group B lead-time-0 null must NOT be contaminated by Group A's last value (0.002)
    b_lt0 = result.filter(
        (pl.col("nwp_init_time") == datetime(2020, 1, 1, 6))
        & (pl.col("valid_time") == datetime(2020, 1, 1, 6))
    )
    assert b_lt0["precipitation_surface"][0] is None


def test_upsample_nwp_no_cross_group_forward_fill():
    """forward_fill().over() must not fill categorical nulls across group boundaries.

    Group B starts with a null at lead time 0. After forward-filling, Group B's leading
    null must NOT be filled with Group A's last known categorical value.
    """
    df = pl.DataFrame(
        {
            "nwp_init_time": [
                datetime(2020, 1, 1, 0),
                datetime(2020, 1, 1, 0),  # group A
                datetime(2020, 1, 1, 6),
                datetime(2020, 1, 1, 6),  # group B
            ],
            "ensemble_member": [0, 0, 0, 0],
            "valid_time": [
                datetime(2020, 1, 1, 0),
                datetime(2020, 1, 1, 3),
                datetime(2020, 1, 1, 6),
                datetime(2020, 1, 1, 9),
            ],
            # Group A: both steps have value 5. Group B: lead-time-0 is null, then 2.
            "categorical_precipitation_type_surface": pl.Series([5, 5, None, 2], dtype=pl.UInt8),
        }
    )
    result = _upsample_nwp_to_half_hourly(df.lazy()).collect().sort(["nwp_init_time", "valid_time"])

    # Group B lead-time-0 (06:00) must stay null — not forward-filled from Group A's 5
    b_lt0 = result.filter(
        (pl.col("nwp_init_time") == datetime(2020, 1, 1, 6))
        & (pl.col("valid_time") == datetime(2020, 1, 1, 6))
    )
    assert b_lt0["categorical_precipitation_type_surface"][0] is None

    # The upsampled rows between 6:00 and 9:00 should also be null (forward-filled from None)
    b_mid = result.filter(
        (pl.col("nwp_init_time") == datetime(2020, 1, 1, 6))
        & (pl.col("valid_time") == datetime(2020, 1, 1, 7, 30))
    )
    assert b_mid["categorical_precipitation_type_surface"][0] is None


def test_upsample_nwp_to_half_hourly_groups_independently():
    """Each (nwp_init_time, ensemble_member) group is upsampled independently."""
    df = pl.DataFrame(
        {
            "nwp_init_time": [
                datetime(2020, 1, 1, 0),
                datetime(2020, 1, 1, 0),
                datetime(2020, 1, 1, 6),
                datetime(2020, 1, 1, 6),
            ],
            "ensemble_member": [0, 0, 0, 0],
            "valid_time": [
                datetime(2020, 1, 1, 6),
                datetime(2020, 1, 1, 9),
                datetime(2020, 1, 1, 6),
                datetime(2020, 1, 1, 9),
            ],
            "temperature_2m": [10.0, 16.0, 20.0, 26.0],
        }
    )
    result = _upsample_nwp_to_half_hourly(df.lazy()).collect()

    # Each group spans 3h at 30-min resolution → 7 rows; 2 groups → 14 total
    assert len(result) == 14

    mid_a = result.filter(
        (pl.col("nwp_init_time") == datetime(2020, 1, 1, 0))
        & (pl.col("valid_time") == datetime(2020, 1, 1, 7, 30))
    )
    assert mid_a["temperature_2m"][0] == pytest.approx(13.0)

    mid_b = result.filter(
        (pl.col("nwp_init_time") == datetime(2020, 1, 1, 6))
        & (pl.col("valid_time") == datetime(2020, 1, 1, 7, 30))
    )
    assert mid_b["temperature_2m"][0] == pytest.approx(23.0)


def test_windchill_feature():
    df = pl.DataFrame(
        {
            "temperature_2m": [0.0, -10.0],
            "wind_speed_10m": [5.0, 10.0],  # m/s
        }
    )

    result = df.with_columns(STATIC_FEATURE_REGISTRY["windchill"])

    assert "windchill" in result.columns
    # Formula: 13.12 + 0.6215 * T - 11.37 * (V ** 0.16) + 0.3965 * T * (V ** 0.16)
    # V is in km/h, so V = wind_speed_10m * 3.6
    # For T=0, V=18: 13.12 + 0 - 11.37 * (18 ** 0.16) + 0 = 13.12 - 11.37 * 1.583 = -4.88
    # For T=-10, V=36: 13.12 - 6.215 - 11.37 * (36 ** 0.16) - 3.965 * (36 ** 0.16) = 6.905 - 15.335 * 1.768 = -20.2
    assert result["windchill"][0] == pytest.approx(-4.88, abs=0.1)
    assert result["windchill"][1] == pytest.approx(-20.3, abs=0.1)


def test_nullify_leaky_lags():
    power_fcst_init_time = datetime(2023, 1, 1, 0, 0)
    df = pl.DataFrame(
        {
            "power_fcst_init_time": [power_fcst_init_time] * 4,
            "valid_time": [
                power_fcst_init_time + timedelta(hours=10),  # 10h lead < 24h -> keep
                power_fcst_init_time + timedelta(hours=24),  # 24h lead == 24h -> nullify (>=)
                power_fcst_init_time + timedelta(hours=30),  # 30h lead > 24h -> nullify
                power_fcst_init_time + timedelta(hours=50),  # 50h lead > 24h -> nullify
            ],
            "power_lag_24h": [100.0, 200.0, 300.0, 400.0],
        }
    )
    lf = df.lazy()
    leaky_features = [LagFeature.from_str("power_lag_24h")]
    result = _nullify_leaky_lags(lf, leaky_features).collect()

    assert result["power_lag_24h"].to_list() == [100.0, None, None, None]


def test_apply_local_time_features():
    # Test with a winter date (GMT) and a summer date (BST)
    df = pl.DataFrame(
        {
            "valid_time": [
                datetime(2023, 1, 10, 12, 0),  # Winter, UTC=GMT
                datetime(2023, 7, 10, 12, 0),  # Summer, UTC=BST-1
            ]
        }
    )

    result = _apply_local_time_features(df.lazy()).collect()

    assert "local_utc_offset" in result.columns
    assert result["local_utc_offset"].to_list() == [0.0, 1.0]

    # In winter, 12:00 UTC is 12:00 local.
    # In summer, 12:00 UTC is 13:00 local.
    # local_time_of_day_sin for 12:00 is sin(pi) = 0
    # local_time_of_day_sin for 13:00 is sin(13/24 * 2pi) = -0.2588
    assert result["local_time_of_day_sin"][0] == pytest.approx(0.0, abs=1e-5)
    assert result["local_time_of_day_sin"][1] == pytest.approx(-0.258819, abs=1e-5)

    assert "local_day_of_week" in result.columns
    # 2023-01-10 is a Tuesday (2)
    # 2023-07-10 is a Monday (1)
    assert result["local_day_of_week"].to_list() == ["Tuesday", "Monday"]


def test_parsed_features_from_selected_features():
    selected = {
        "power_lag_24h",
        "temperature_2m_lag_12h",
        "temperature_2m_rolling_mean_6h",
        "windchill",
        "local_time_of_day_sin",
    }
    parsed = ParsedFeatures.from_strings(selected)

    assert len(parsed.lags) == 2
    power_lag = next(f for f in parsed.lags if f.base_col == "power" and f.hours == 24)
    assert power_lag is not None

    temp_lag = next(f for f in parsed.lags if f.base_col == "temperature_2m" and f.hours == 12)
    assert temp_lag is not None

    assert len(parsed.rolling_means) == 1
    temp_rolling = parsed.rolling_means[0]
    assert temp_rolling.base_col == "temperature_2m"
    assert temp_rolling.hours == 6

    assert parsed.static_features == ["windchill"]
    assert parsed.time_features == ["local_time_of_day_sin"]

    # Leaky lags should track power lags
    leaky_features = parsed.get_leaky_features()
    assert len(leaky_features) == 1
    assert leaky_features[0].string_repr == "power_lag_24h"
    assert leaky_features[0].hours == 24


def test_parsed_features_from_selected_features_invalid_stacking():
    with pytest.raises(ValueError, match="Feature stacking is not supported"):
        ParsedFeatures.from_strings({"power_lag_24h_rolling_mean_6h"})


def test_parsed_features_from_selected_features_invalid_hours():
    # Lag and rolling window hours must be gt=0 and le=17520 (2 years)
    with pytest.raises(ValidationError):
        ParsedFeatures.from_strings({"power_lag_0h"})
    with pytest.raises(ValidationError):
        ParsedFeatures.from_strings({"power_lag_17521h"})
    with pytest.raises(ValidationError):
        ParsedFeatures.from_strings({"temperature_2m_rolling_mean_0h"})
    with pytest.raises(ValidationError):
        ParsedFeatures.from_strings({"temperature_2m_rolling_mean_17521h"})


def test_parsed_features_from_selected_features_invalid_base_column():
    # Base column must be a valid BaseColumn literal
    with pytest.raises(ValidationError):
        ParsedFeatures.from_strings({"invalid_col_lag_24h"})

    with pytest.raises(ValidationError):
        ParsedFeatures.from_strings({"invalid_col_rolling_mean_6h"})


def test_parsed_features_from_selected_features_malformed_patterns():
    # Negative hours, non-integer hours, or other malformed patterns should raise ValueError
    # because they do not match the regex and fall through to unrecognized feature check.
    with pytest.raises(ValueError, match="Invalid lag feature name format"):
        ParsedFeatures.from_strings({"power_lag_-5h"})

    with pytest.raises(ValueError, match="Invalid rolling_mean feature name format"):
        ParsedFeatures.from_strings({"power_rolling_mean_-2h"})

    with pytest.raises(ValueError, match="Invalid lag feature name format"):
        ParsedFeatures.from_strings({"power_lag_12.5h"})

    with pytest.raises(ValueError, match="Invalid rolling_mean feature name format"):
        ParsedFeatures.from_strings({"power_rolling_mean_abch"})


def test_apply_rolling_mean_feature():
    nwp_init_time = datetime(2023, 1, 1, 0, 0)
    df = pl.DataFrame(
        {
            "time_series_id": ["ts1"] * 4,
            "nwp_init_time": [nwp_init_time] * 4,
            "ensemble_member": [0] * 4,
            "valid_time": [
                datetime(2023, 1, 1, 0, 0),
                datetime(2023, 1, 1, 1, 0),
                datetime(2023, 1, 1, 2, 0),
                datetime(2023, 1, 1, 3, 0),
            ],
            "temperature_2m": [10.0, 20.0, 30.0, 40.0],
        }
    )
    # Rolling mean with window of 2 hours
    # For 0:00: mean([10.0]) = 10.0
    # For 1:00: mean([10.0, 20.0]) = 15.0
    # For 2:00: mean([20.0, 30.0]) = 25.0
    # For 3:00: mean([30.0, 40.0]) = 35.0
    result = _apply_rolling_mean_feature(df.lazy(), "temperature_2m", 2).collect()
    assert "temperature_2m_rolling_mean_2h" in result.columns
    assert result["temperature_2m_rolling_mean_2h"].to_list() == [10.0, 15.0, 25.0, 35.0]


def test_apply_rolling_mean_feature_with_ensemble():
    nwp_init_time = datetime(2023, 1, 1, 0, 0)
    df = pl.DataFrame(
        {
            "time_series_id": ["ts1"] * 4,
            "nwp_init_time": [nwp_init_time] * 4,
            "ensemble_member": [1, 1, 2, 2],
            "valid_time": [
                datetime(2023, 1, 1, 0, 0),
                datetime(2023, 1, 1, 1, 0),
                datetime(2023, 1, 1, 0, 0),
                datetime(2023, 1, 1, 1, 0),
            ],
            "temperature_2m": [10.0, 20.0, 100.0, 200.0],
        }
    )
    result = _apply_rolling_mean_feature(df.lazy(), "temperature_2m", 2).collect()
    assert "temperature_2m_rolling_mean_2h" in result.columns
    # Sort to ensure deterministic order
    sorted_result = result.sort(["ensemble_member", "valid_time"])
    assert sorted_result["temperature_2m_rolling_mean_2h"].to_list() == [10.0, 15.0, 100.0, 150.0]


def test_apply_rolling_mean_feature_does_not_mix_nwp_runs():
    # Two NWP runs covering the same valid_times; the rolling mean must stay within each run.
    init_time_a = datetime(2023, 1, 1, 0, 0)
    init_time_b = datetime(2023, 1, 1, 6, 0)
    df = pl.DataFrame(
        {
            "time_series_id": ["ts1"] * 4,
            "nwp_init_time": [init_time_a, init_time_a, init_time_b, init_time_b],
            "ensemble_member": [0] * 4,
            "valid_time": [
                datetime(2023, 1, 1, 6, 0),
                datetime(2023, 1, 1, 7, 0),
                datetime(2023, 1, 1, 6, 0),
                datetime(2023, 1, 1, 7, 0),
            ],
            "temperature_2m": [10.0, 20.0, 100.0, 200.0],
        }
    )
    result = _apply_rolling_mean_feature(df.lazy(), "temperature_2m", 2).collect()
    sorted_result = result.sort(["nwp_init_time", "valid_time"])
    # Run A: mean([10]) = 10, mean([10, 20]) = 15
    # Run B: mean([100]) = 100, mean([100, 200]) = 150
    assert sorted_result["temperature_2m_rolling_mean_2h"].to_list() == [10.0, 15.0, 100.0, 150.0]


def test_parsed_features_from_selected_features_forbids_power_rolling_mean():
    with pytest.raises(ValidationError):
        ParsedFeatures.from_strings({"power_rolling_mean_6h"})


def test_engineer_features_no_nwp():
    valid_time = datetime(2023, 1, 1, 12, 0)
    power_df = pl.DataFrame(
        {
            "time_series_id": [123],
            "time": [valid_time],
            "power": [100.0],
        }
    )
    metadata_df = pl.DataFrame(
        {
            "time_series_id": [123],
            "time_series_name": ["ALFORD 33 11kV S STN"],
            "time_series_type": ["BESS"],
            "units": ["MW"],
            "licence_area": ["EMids"],
            "substation_number": [1],
            "substation_type": ["BSP"],
            "latitude": [52.0],
            "longitude": [-1.0],
            "h3_res_5": [123456789],
        }
    )

    # Run engineer_features with nwp=None
    engineered = engineer_features(
        power_time_series=pt.LazyFrame.from_existing(power_df.lazy()).set_model(PowerTimeSeries),
        time_series_metadata=pt.DataFrame(metadata_df).set_model(TimeSeriesMetadata),
        nwp=None,
        selected_features={"power_lag_24h", "local_time_of_day_sin"},
    ).collect()

    assert len(engineered) == 1
    assert "power_lag_24h" in engineered.columns
    assert "local_time_of_day_sin" in engineered.columns


def test_engineer_features_empty_data():
    # Test with empty DataFrames to ensure it doesn't crash and handles empty inputs gracefully
    power_df = pl.DataFrame(
        {
            "time_series_id": pl.Series([], dtype=pl.Int64),
            "time": pl.Series([], dtype=pl.Datetime),
            "power": pl.Series([], dtype=pl.Float32),
        }
    )
    metadata_df = pl.DataFrame(
        {
            "time_series_id": pl.Series([], dtype=pl.Int64),
            "time_series_name": pl.Series([], dtype=pl.String),
            "time_series_type": pl.Series([], dtype=pl.String),
            "units": pl.Series([], dtype=pl.String),
            "licence_area": pl.Series([], dtype=pl.String),
            "substation_number": pl.Series([], dtype=pl.Int32),
            "substation_type": pl.Series([], dtype=pl.String),
            "latitude": pl.Series([], dtype=pl.Float32),
            "longitude": pl.Series([], dtype=pl.Float32),
            "h3_res_5": pl.Series([], dtype=pl.UInt64),
        }
    )

    engineered = engineer_features(
        power_time_series=pt.LazyFrame.from_existing(power_df.lazy()).set_model(PowerTimeSeries),
        time_series_metadata=pt.DataFrame(metadata_df).set_model(TimeSeriesMetadata),
        nwp=None,
        selected_features={"power_lag_24h", "local_time_of_day_sin"},
    ).collect()

    assert len(engineered) == 0
    assert "power_lag_24h" in engineered.columns
    assert "local_time_of_day_sin" in engineered.columns


def test_engineer_features_raises_value_error_when_weather_requested_but_nwp_none():
    valid_time = datetime(2023, 1, 1, 12, 0)
    power_df = pl.DataFrame(
        {
            "time_series_id": [123],
            "time": [valid_time],
            "power": [100.0],
        }
    )
    metadata_df = pl.DataFrame(
        {
            "time_series_id": [123],
            "time_series_name": ["ALFORD 33 11kV S STN"],
            "time_series_type": ["BESS"],
            "units": ["MW"],
            "licence_area": ["EMids"],
            "substation_number": [1],
            "substation_type": ["BSP"],
            "latitude": [52.0],
            "longitude": [-1.0],
            "h3_res_5": [123456789],
        }
    )

    def _call(features: set[str]) -> None:
        engineer_features(
            power_time_series=pt.LazyFrame.from_existing(power_df.lazy()).set_model(
                PowerTimeSeries
            ),
            time_series_metadata=pt.DataFrame(metadata_df).set_model(TimeSeriesMetadata),
            nwp=None,
            selected_features=features,
        )

    for features in [
        {"temperature_2m_lag_12h"},  # weather lag
        {"temperature_2m_rolling_mean_6h"},  # weather rolling mean
        {"windchill"},  # static feature that needs weather
        {"temperature_2m"},  # raw weather feature
    ]:
        with pytest.raises(
            ValueError, match="Weather features were requested but no NWP data was provided"
        ):
            _call(features)


def test_parsed_features_raw_weather_features():
    """Verify that raw weather features are parsed correctly.

    This test ensures that raw weather variables (e.g., 'temperature_2m') are correctly
    identified and parsed into the `weather_features` list, and that requesting them
    correctly triggers the requirement for weather (NWP) data.
    """
    selected = {"temperature_2m", "wind_speed_10m"}
    parsed = ParsedFeatures.from_strings(selected)

    assert set(parsed.weather_features) == {"temperature_2m", "wind_speed_10m"}
    assert parsed.requires_weather_data() is True


def test_parsed_features_safe_input_base_columns():
    """Verify that safe input base columns are parsed correctly.

    This test ensures that base columns that are safe to use as direct input features
    (e.g., 'nwp_lead_time_hours', 'time_series_type') are correctly identified and parsed
    into the `base_features` list, allowing downstream models to use them without
    triggering target leakage or index column errors.
    """
    selected = {"nwp_lead_time_hours", "time_series_type"}
    parsed = ParsedFeatures.from_strings(selected)

    assert set(parsed.base_features) == {"nwp_lead_time_hours", "time_series_type"}


def test_parsed_features_forbids_power_target_leakage():
    """Verify that requesting 'power' as an input feature raises a ValueError.

    This test ensures that the target leakage prevention guardrail is active. Requesting
    the raw target variable 'power' as an input feature must raise a clear, helpful
    ValueError explaining target leakage and guiding the user to use lagged power features.
    """
    with pytest.raises(
        ValueError,
        match="The target variable 'power' cannot be requested as an input feature",
    ) as exc_info:
        ParsedFeatures.from_strings({"power"})

    assert "prevent target leakage" in str(exc_info.value)
    assert "power_lag_24h" in str(exc_info.value)


def test_parsed_features_forbids_valid_time_index():
    """Verify that requesting 'valid_time' as an input feature raises a ValueError.

    This test ensures that the index column guardrail is active. Requesting 'valid_time'
    as an input feature must raise a clear, helpful ValueError explaining that it is an
    index column and guiding the user to use local time features instead.
    """
    with pytest.raises(
        ValueError,
        match="The index column 'valid_time' cannot be requested as an input feature",
    ) as exc_info:
        ParsedFeatures.from_strings({"valid_time"})

    assert "local_time_of_day_sin" in str(exc_info.value)


def test_engineer_features_multi_run_backtest_uses_bulk_mode():
    """Multi-run backtesting must use power_fcst_init_time=None (bulk training mode).

    Passing power_fcst_init_time=<scalar> with multi-run data stamps the same constant
    nwp_init_time on every row, so the NWP join matches only one run and leaves everything
    else null. This test verifies that bulk mode (power_fcst_init_time=None) correctly
    generates one row per (nwp_init_time, valid_time) combination, which is what a
    backtest needs.
    """
    valid_time = datetime(2023, 1, 1, 12, 0)
    nwp_init_time_1 = datetime(2022, 12, 31, 0, 0)
    nwp_init_time_2 = datetime(2023, 1, 1, 0, 0)

    nwp_df = pl.DataFrame(
        {
            "time_series_id": ["ts1", "ts1"],
            "valid_time": [valid_time, valid_time],
            "ensemble_member": [0, 0],
            "init_time": [nwp_init_time_1, nwp_init_time_2],
            "temperature_2m": [10.0, 12.0],
        }
    )

    power_df = pl.DataFrame(
        {
            "time_series_id": ["ts1"],
            "time": [valid_time],
            "power": [100.0],
        }
    )

    metadata_df = pl.DataFrame(
        {
            "time_series_id": ["ts1"],
            "time_series_type": ["substation"],
        }
    )

    # Bulk mode: power_fcst_init_time=None — both NWP runs produce a row.
    result = engineer_features(
        power_time_series=pt.LazyFrame.from_existing(power_df.lazy()).set_model(PowerTimeSeries),
        time_series_metadata=pt.DataFrame(metadata_df).set_model(TimeSeriesMetadata),
        nwp=pt.LazyFrame.from_existing(nwp_df.lazy()).set_model(NwpInMemory),
        selected_features={"temperature_2m"},
        power_fcst_init_time=None,  # backtest / training mode
    ).collect()

    # Both NWP runs should produce a row, each with the correct temperature.
    assert len(result) == 2
    nwp_init_times = set(result["nwp_init_time"].to_list())
    assert nwp_init_times == {nwp_init_time_1, nwp_init_time_2}


def test_engineer_features_raises_when_nwp_init_time_given_without_power_fcst_init_time():
    power_df = pl.DataFrame(
        {"time_series_id": ["ts1"], "time": [datetime(2023, 1, 1, 12)], "power": [100.0]}
    )
    metadata_df = pl.DataFrame({"time_series_id": ["ts1"], "time_series_type": ["substation"]})

    with pytest.raises(ValueError, match="nwp_init_time can only be provided in single-run mode"):
        engineer_features(
            power_time_series=pt.LazyFrame.from_existing(power_df.lazy()).set_model(
                PowerTimeSeries
            ),
            time_series_metadata=pt.DataFrame(metadata_df).set_model(TimeSeriesMetadata),
            selected_features={"power_lag_24h"},
            power_fcst_init_time=None,
            nwp_init_time=datetime(2023, 1, 1, 0),
        )


def test_engineer_features_weather_lag_leakage_prevention():
    # Create dummy data to verify weather lag leakage prevention
    valid_time = datetime(2026, 6, 11, 12, 0)
    power_fcst_init_time = datetime(2026, 6, 10, 6, 0)
    nwp_init_time = datetime(2026, 6, 10, 0, 0)

    # NWP data has two runs:
    # 1. Run initialized at 2026-06-10 00:00:00 (the one we should use)
    # 2. Run initialized at 2026-06-10 12:00:00 (future run relative to power_fcst_init_time!)
    nwp_df = pl.DataFrame(
        {
            "time_series_id": ["ts1", "ts1", "ts1", "ts1"],
            "valid_time": [
                valid_time,  # target time
                valid_time - timedelta(hours=2),  # lag target time (2026-06-11 10:00:00)
                valid_time - timedelta(hours=2),  # lag target time (2026-06-11 10:00:00)
                valid_time - timedelta(hours=36),  # lag target time (2026-06-10 00:00:00)
            ],
            "ensemble_member": [0, 0, 0, 0],
            "init_time": [
                nwp_init_time,  # run 1
                nwp_init_time,  # run 1 (same run)
                datetime(2026, 6, 10, 12, 0),  # run 2 (future run!)
                nwp_init_time,  # run 1
            ],
            "temperature_2m": [
                15.0,  # temp at valid_time
                10.0,  # temp at lag target time in run 1
                12.0,  # temp at lag target time in run 2 (future run!)
                8.0,  # temp at lag target time (36h lag)
            ],
        }
    )

    power_df = pl.DataFrame(
        {
            "time_series_id": ["ts1"],
            "time": [valid_time],
            "power": [100.0],
        }
    )

    metadata_df = pl.DataFrame(
        {
            "time_series_id": ["ts1"],
            "time_series_type": ["substation"],
        }
    )

    # Run engineer_features in production/backtest mode
    engineered = engineer_features(
        power_time_series=pt.LazyFrame.from_existing(power_df.lazy()).set_model(PowerTimeSeries),
        time_series_metadata=pt.DataFrame(metadata_df).set_model(TimeSeriesMetadata),
        nwp=pt.LazyFrame.from_existing(nwp_df.lazy()).set_model(NwpInMemory),
        selected_features={"temperature_2m", "temperature_2m_lag_2h", "temperature_2m_lag_36h"},
        power_fcst_init_time=power_fcst_init_time,
        nwp_init_time=nwp_init_time,
    ).collect()

    # Verify that temperature_2m_lag_2h is 10.0 (from run 1), NOT 12.0 (from run 2)
    # This proves that the same-run join was used for target_time > power_fcst_init_time
    assert engineered["temperature_2m_lag_2h"][0] == 10.0

    # Verify that temperature_2m_lag_36h is 8.0 (from freshest run)
    assert engineered["temperature_2m_lag_36h"][0] == 8.0


def test_engineer_features_power_lag_nullification_end_to_end():
    """Power lags are null when lead_time >= lag_hours, including the exact boundary case."""
    power_fcst_init_time = datetime(2023, 1, 3, 0, 0)
    # Power data spans both the lag source times and the forecast valid times so we confirm
    # that the lag VALUE exists before nullification (it's not null due to missing data).
    power_df = pl.DataFrame(
        {
            "time_series_id": ["ts1"] * 6,
            "time": [
                datetime(2023, 1, 2, 6, 0),  # lag source for 6h-lead row
                datetime(2023, 1, 3, 0, 0),  # lag source for 24h-lead row (boundary)
                datetime(2023, 1, 3, 6, 0),  # valid_time_1: 6h lead  — lag kept
                datetime(2023, 1, 3, 12, 0),  # lag source for 36h-lead row
                datetime(2023, 1, 4, 0, 0),  # valid_time_2: 24h lead — nullified (lead == lag)
                datetime(2023, 1, 4, 12, 0),  # valid_time_3: 36h lead — nullified
            ],
            "power": [50.0, 60.0, 80.0, 70.0, 90.0, 100.0],
        }
    )
    metadata_df = pl.DataFrame({"time_series_id": ["ts1"], "time_series_type": ["substation"]})

    result = engineer_features(
        power_time_series=pt.LazyFrame.from_existing(power_df.lazy()).set_model(PowerTimeSeries),
        time_series_metadata=pt.DataFrame(metadata_df).set_model(TimeSeriesMetadata),
        selected_features={"power_lag_24h"},
        power_fcst_init_time=power_fcst_init_time,
    ).collect()

    row_6h = result.filter(pl.col("valid_time") == datetime(2023, 1, 3, 6, 0))
    row_24h = result.filter(pl.col("valid_time") == datetime(2023, 1, 4, 0, 0))
    row_36h = result.filter(pl.col("valid_time") == datetime(2023, 1, 4, 12, 0))

    assert row_6h["power_lag_24h"][0] == 50.0  # 6h lead < 24h lag: kept
    assert row_24h["power_lag_24h"][0] is None  # 24h lead == 24h lag: nullified (>= boundary)
    assert row_36h["power_lag_24h"][0] is None  # 36h lead > 24h lag: nullified


def test_engineer_features_bulk_mode_weather_lag_uses_correct_nwp_run():
    """In bulk mode, weather lag boundary differs per row because power_fcst_init_time varies.

    NWP run A: power_fcst_init_time = 2023-01-01 06:00  (init 00:00 + 6h delay)
    NWP run B: power_fcst_init_time = 2023-01-02 06:00  (init 00:00 + 6h delay)
    valid_time = 2023-01-02 12:00, lag = 12h → target_time = 2023-01-02 00:00

    Run A: target_time (Jan-02 00:00) > power_fcst_init_time (Jan-01 06:00) → same-run join → 10.0
    Run B: target_time (Jan-02 00:00) < power_fcst_init_time (Jan-02 06:00) → freshest-run join → 20.0
    """
    nwp_df = pl.DataFrame(
        {
            "time_series_id": ["ts1"] * 4,
            "valid_time": [
                datetime(2023, 1, 2, 12),  # run A: main valid_time
                datetime(2023, 1, 2, 0),  # run A: lag target (temperature = 10.0)
                datetime(2023, 1, 2, 12),  # run B: main valid_time
                datetime(
                    2023, 1, 2, 0
                ),  # run B: lag target and freshest at that time (temperature = 20.0)
            ],
            "ensemble_member": [0, 0, 0, 0],
            "init_time": [
                datetime(2023, 1, 1, 0),
                datetime(2023, 1, 1, 0),
                datetime(2023, 1, 2, 0),
                datetime(2023, 1, 2, 0),
            ],
            "temperature_2m": [15.0, 10.0, 18.0, 20.0],
        }
    )
    power_df = pl.DataFrame(
        {
            "time_series_id": ["ts1"],
            "time": [datetime(2023, 1, 2, 12)],
            "power": [100.0],
        }
    )
    metadata_df = pl.DataFrame({"time_series_id": ["ts1"], "time_series_type": ["substation"]})

    all_rows = engineer_features(
        power_time_series=pt.LazyFrame.from_existing(power_df.lazy()).set_model(PowerTimeSeries),
        time_series_metadata=pt.DataFrame(metadata_df).set_model(TimeSeriesMetadata),
        nwp=pt.LazyFrame.from_existing(nwp_df.lazy()).set_model(NwpInMemory),
        selected_features={"temperature_2m_lag_12h"},
        power_fcst_init_time=None,
        nwp_publication_delay_hours=6,
    ).collect()

    # Bulk mode is NWP-centric: it fans out over all NWP (valid_time, init_time) pairs.
    # Filter to only the forecast valid_time rows (the lag-target rows also appear with power=null).
    result = all_rows.filter(pl.col("valid_time") == datetime(2023, 1, 2, 12, 0)).sort(
        "nwp_init_time"
    )

    assert len(result) == 2
    # Run A row: same-run join because target_time is in run A's forecast window
    assert result["temperature_2m_lag_12h"][0] == 10.0
    # Run B row: freshest-run join because target_time is before run B's power_fcst_init_time
    assert result["temperature_2m_lag_12h"][1] == 20.0


def test_engineer_features_bulk_mode_derives_power_fcst_init_time():
    """In bulk mode, power_fcst_init_time must equal nwp_init_time + nwp_publication_delay_hours."""
    nwp_init_time = datetime(2023, 1, 1, 0, 0)
    valid_time = datetime(2023, 1, 1, 12, 0)
    nwp_publication_delay_hours = 9  # non-default to confirm the parameter is used

    nwp_df = pl.DataFrame(
        {
            "time_series_id": ["ts1"],
            "valid_time": [valid_time],
            "ensemble_member": [0],
            "init_time": [nwp_init_time],
            "temperature_2m": [10.0],
        }
    )
    power_df = pl.DataFrame({"time_series_id": ["ts1"], "time": [valid_time], "power": [100.0]})
    metadata_df = pl.DataFrame({"time_series_id": ["ts1"], "time_series_type": ["substation"]})

    result = engineer_features(
        power_time_series=pt.LazyFrame.from_existing(power_df.lazy()).set_model(PowerTimeSeries),
        time_series_metadata=pt.DataFrame(metadata_df).set_model(TimeSeriesMetadata),
        nwp=pt.LazyFrame.from_existing(nwp_df.lazy()).set_model(NwpInMemory),
        selected_features={"temperature_2m"},
        nwp_publication_delay_hours=nwp_publication_delay_hours,
    ).collect()

    expected = nwp_init_time + timedelta(hours=nwp_publication_delay_hours)
    assert result["power_fcst_init_time"][0] == expected


def test_engineer_features_raises_when_no_control_member_for_weather_lag():
    """Weather lag features require ensemble_member==0; missing control member must raise loudly."""
    valid_time = datetime(2023, 1, 1, 12, 0)
    nwp_init_time = datetime(2023, 1, 1, 0, 0)

    nwp_df = pl.DataFrame(
        {
            "time_series_id": ["ts1"],
            "valid_time": [valid_time],
            "ensemble_member": [1],  # no control member
            "init_time": [nwp_init_time],
            "temperature_2m": [10.0],
        }
    )
    power_df = pl.DataFrame({"time_series_id": ["ts1"], "time": [valid_time], "power": [100.0]})
    metadata_df = pl.DataFrame({"time_series_id": ["ts1"], "time_series_type": ["substation"]})

    with pytest.raises(ValueError, match="control member"):
        engineer_features(
            power_time_series=pt.LazyFrame.from_existing(power_df.lazy()).set_model(
                PowerTimeSeries
            ),
            time_series_metadata=pt.DataFrame(metadata_df).set_model(TimeSeriesMetadata),
            nwp=pt.LazyFrame.from_existing(nwp_df.lazy()).set_model(NwpInMemory),
            selected_features={"temperature_2m_lag_6h"},
            power_fcst_init_time=datetime(2023, 1, 1, 6, 0),
            nwp_init_time=nwp_init_time,
        )
