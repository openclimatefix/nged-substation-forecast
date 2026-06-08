from datetime import datetime, timedelta
from typing import cast

import patito as pt
import polars as pl
import pytest
from pydantic import ValidationError
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import NwpOnDisk
from ml_core.features import (
    STATIC_FEATURE_REGISTRY,
    ParsedFeatures,
    apply_lag_feature,
    apply_local_time_features,
    calculate_lead_time,
    engineer_features,
    nullify_leaky_lags,
)


def test_engineer_features_nwp_historical_best():
    # Create dummy data
    # 2 ensemble members, 2 lead times for the same valid_time
    # We want to ensure only the one with minimum lead_time is kept
    valid_time = datetime(2023, 1, 1, 12, 0)

    nwp_df = pl.DataFrame(
        {
            "time_series_id": ["ts1", "ts1", "ts1", "ts1"],
            "valid_time": [valid_time, valid_time, valid_time, valid_time],
            "ensemble_member": [1, 1, 2, 2],
            "init_time": [
                valid_time - timedelta(hours=10),  # lead 10
                valid_time - timedelta(hours=20),  # lead 20
                valid_time - timedelta(hours=5),  # lead 5
                valid_time - timedelta(hours=15),  # lead 15
            ],
            "temperature_2m": [10.0, 11.0, 12.0, 13.0],
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

    # Run engineer_features
    engineered = engineer_features(
        power_time_series=pt.LazyFrame.from_existing(power_df.lazy()).set_model(PowerTimeSeries),
        time_series_metadata=pt.DataFrame(metadata_df).set_model(TimeSeriesMetadata),
        nwp=pt.LazyFrame.from_existing(nwp_df.lazy()).set_model(NwpOnDisk),
        selected_features={"temperature_2m", "temperature_2m_lag_24h"},
    ).collect()

    # Check that we have 2 ensemble members (1 and 2)
    # And for each, we have the temperature corresponding to the minimum lead time
    # Member 1: min lead is 10h (temp 10.0)
    # Member 2: min lead is 5h (temp 12.0)

    assert len(engineered) == 2
    assert set(engineered["ensemble_member"].to_list()) == {1, 2}

    # Check temperatures
    temp_1 = engineered.filter(pl.col("ensemble_member") == 1)["temperature_2m"][0]
    temp_2 = engineered.filter(pl.col("ensemble_member") == 2)["temperature_2m"][0]

    assert temp_1 == 10.0
    assert temp_2 == 12.0


def test_apply_lag_feature_with_source():
    # Test that apply_lag_feature correctly pulls from source_lf
    target_df = pl.DataFrame(
        {
            "time_series_id": ["ts1"],
            "valid_time": [datetime(2023, 1, 1, 12, 0)],
            "power": [100.0],
        }
    )

    source_df = pl.DataFrame(
        {
            "time_series_id": ["ts1"],
            "valid_time": [datetime(2023, 1, 1, 11, 0)],
            "power": [90.0],
        }
    )

    result = cast(
        pl.DataFrame,
        apply_lag_feature(target_df.lazy(), source_df.lazy(), "power", 1).collect(),
    )

    assert result["power_lag_1h"][0] == 90.0


def test_calculate_lead_time():
    df = pl.DataFrame(
        {
            "valid_time": [
                datetime(2020, 1, 1, 12),
                datetime(2020, 1, 1, 13),
            ],
            "init_time": [
                datetime(2020, 1, 1, 0),
                datetime(2020, 1, 1, 0),
            ],
        }
    )

    lf = df.lazy()
    result_lf = calculate_lead_time(lf)
    result = cast(pl.DataFrame, result_lf.collect())

    assert "lead_time_hours" in result.columns
    assert result["lead_time_hours"].to_list() == [12.0, 13.0]


def test_calculate_lead_time_no_init_time():
    df = pl.DataFrame(
        {
            "valid_time": [
                datetime(2020, 1, 1, 12),
                datetime(2020, 1, 1, 13),
            ]
        }
    )

    lf = df.lazy()
    result_lf = calculate_lead_time(lf)
    result = cast(pl.DataFrame, result_lf.collect())

    assert "lead_time_hours" not in result.columns


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
    df = pl.DataFrame(
        {
            "lead_time_hours": [10.0, 30.0, 50.0],
            "power_lag_24h": [100.0, 200.0, 300.0],
        }
    )
    lf = df.lazy()
    # 24h lag:
    # lead_time 10h < 24h -> keep
    # lead_time 30h >= 24h -> nullify
    # lead_time 50h >= 24h -> nullify
    result = cast(pl.DataFrame, nullify_leaky_lags(lf, {"power_lag_24h": 24}).collect())

    assert result["power_lag_24h"].to_list() == [100.0, None, None]


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

    result = cast(pl.DataFrame, apply_local_time_features(df.lazy()).collect())

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
    parsed = ParsedFeatures.from_selected_features(selected)

    assert "power_lag_24h" in parsed.lags
    assert parsed.lags["power_lag_24h"].base_col == "power"
    assert parsed.lags["power_lag_24h"].lag_hours == 24

    assert "temperature_2m_lag_12h" in parsed.lags
    assert parsed.lags["temperature_2m_lag_12h"].base_col == "temperature_2m"
    assert parsed.lags["temperature_2m_lag_12h"].lag_hours == 12

    assert "temperature_2m_rolling_mean_6h" in parsed.rolling_means
    assert parsed.rolling_means["temperature_2m_rolling_mean_6h"].base_col == "temperature_2m"
    assert parsed.rolling_means["temperature_2m_rolling_mean_6h"].window_hours == 6

    assert parsed.static == ["windchill"]
    assert parsed.local_time == ["local_time_of_day_sin"]

    # Leaky lags should track power lags
    assert parsed.leaky_lags == {"power_lag_24h": 24}


def test_parsed_features_from_selected_features_invalid_stacking():
    with pytest.raises(ValueError, match="Feature stacking is not supported"):
        ParsedFeatures.from_selected_features({"power_lag_24h_rolling_mean_6h"})


def test_parsed_features_from_selected_features_invalid_lag_hours():
    # Lag hours must be gt=0 and le=8760
    with pytest.raises(ValidationError):
        ParsedFeatures.from_selected_features({"power_lag_0h"})

    with pytest.raises(ValidationError):
        ParsedFeatures.from_selected_features({"power_lag_8761h"})


def test_parsed_features_from_selected_features_invalid_rolling_hours():
    # Rolling window hours must be gt=0 and le=8760
    with pytest.raises(ValidationError):
        ParsedFeatures.from_selected_features({"power_rolling_mean_0h"})

    with pytest.raises(ValidationError):
        ParsedFeatures.from_selected_features({"power_rolling_mean_8761h"})


def test_parsed_features_from_selected_features_invalid_base_column():
    # Base column must be a valid BaseColumn literal
    with pytest.raises(ValidationError):
        ParsedFeatures.from_selected_features({"invalid_col_lag_24h"})

    with pytest.raises(ValidationError):
        ParsedFeatures.from_selected_features({"invalid_col_rolling_mean_6h"})


def test_parsed_features_from_selected_features_malformed_patterns():
    # Negative hours, non-integer hours, or other malformed patterns should raise ValueError
    # because they do not match the regex and fall through to unrecognized feature check.
    with pytest.raises(ValueError, match="Unrecognised feature name"):
        ParsedFeatures.from_selected_features({"power_lag_-5h"})

    with pytest.raises(ValueError, match="Unrecognised feature name"):
        ParsedFeatures.from_selected_features({"power_rolling_mean_-2h"})

    with pytest.raises(ValueError, match="Unrecognised feature name"):
        ParsedFeatures.from_selected_features({"power_lag_12.5h"})

    with pytest.raises(ValueError, match="Unrecognised feature name"):
        ParsedFeatures.from_selected_features({"power_rolling_mean_abch"})
