from datetime import datetime, timedelta

import patito as pt
import polars as pl
import pytest
from pydantic import ValidationError
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import NwpOnDisk
from ml_core.features import (
    STATIC_FEATURE_REGISTRY,
    LagFeature,
    ParsedFeatures,
    apply_lag_feature,
    apply_local_time_features,
    calculate_nwp_lead_time,
    engineer_features,
    nullify_leaky_lags,
    apply_rolling_mean_feature,
)


def test_engineer_features_bulk_mode_keeps_all_nwp_runs():
    # Create dummy data
    # 2 ensemble members, 2 lead times for the same valid_time
    # In bulk training mode, we keep all available forecast runs (init_times)
    # because they represent different initialization times under the expanded primary key design.
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

    # In bulk training mode, we keep all 4 rows because they represent different forecast runs
    assert len(engineered) == 4
    assert set(engineered["ensemble_member"].to_list()) == {1, 2}


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

    lag_feat = LagFeature.from_str("power_lag_1h")
    result = apply_lag_feature(target_df.lazy(), source_df.lazy(), lag_feat).collect()

    assert result["power_lag_1h"][0] == 90.0


def test_calculate_nwp_lead_time():
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
    result_lf = calculate_nwp_lead_time(lf)
    result = result_lf.collect()

    assert "nwp_lead_time_hours" in result.columns
    assert "nwp_init_time" in result.columns
    assert "init_time" not in result.columns
    assert result["nwp_lead_time_hours"].to_list() == [12.0, 13.0]


def test_calculate_nwp_lead_time_no_init_time():
    df = pl.DataFrame(
        {
            "valid_time": [
                datetime(2020, 1, 1, 12),
                datetime(2020, 1, 1, 13),
            ]
        }
    )

    lf = df.lazy()
    result_lf = calculate_nwp_lead_time(lf)
    result = result_lf.collect()

    assert "nwp_lead_time_hours" not in result.columns


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
            "power_fcst_init_time": [power_fcst_init_time] * 3,
            "valid_time": [
                power_fcst_init_time + timedelta(hours=10),
                power_fcst_init_time + timedelta(hours=30),
                power_fcst_init_time + timedelta(hours=50),
            ],
            "power_lag_24h": [100.0, 200.0, 300.0],
        }
    )
    lf = df.lazy()
    # 24h lag:
    # lead_time 10h < 24h -> keep
    # lead_time 30h >= 24h -> nullify
    # lead_time 50h >= 24h -> nullify
    leaky_features = [LagFeature.from_str("power_lag_24h")]
    result = nullify_leaky_lags(lf, leaky_features).collect()

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

    result = apply_local_time_features(df.lazy()).collect()

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
    # Create a simple dataframe with hourly data
    df = pl.DataFrame(
        {
            "time_series_id": ["ts1"] * 4,
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
    result = apply_rolling_mean_feature(df.lazy(), "temperature_2m", 2).collect()
    assert "temperature_2m_rolling_mean_2h" in result.columns
    assert result["temperature_2m_rolling_mean_2h"].to_list() == [10.0, 15.0, 25.0, 35.0]


def test_apply_rolling_mean_feature_with_ensemble():
    df = pl.DataFrame(
        {
            "time_series_id": ["ts1"] * 4,
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
    result = apply_rolling_mean_feature(df.lazy(), "temperature_2m", 2).collect()
    assert "temperature_2m_rolling_mean_2h" in result.columns
    # Sort to ensure deterministic order
    sorted_result = result.sort(["ensemble_member", "valid_time"])
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
        nwp=pt.LazyFrame.from_existing(nwp_df.lazy()).set_model(NwpOnDisk),
        selected_features={"temperature_2m"},
        power_fcst_init_time=None,  # backtest / training mode
    ).collect()

    # Both NWP runs should produce a row, each with the correct temperature.
    assert len(result) == 2
    nwp_init_times = set(result["nwp_init_time"].to_list())
    assert nwp_init_times == {nwp_init_time_1, nwp_init_time_2}


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
        nwp=pt.LazyFrame.from_existing(nwp_df.lazy()).set_model(NwpOnDisk),
        selected_features={"temperature_2m", "temperature_2m_lag_2h", "temperature_2m_lag_36h"},
        power_fcst_init_time=power_fcst_init_time,
        nwp_publication_delay_hours=6,
    ).collect()

    # Verify that temperature_2m_lag_2h is 10.0 (from run 1), NOT 12.0 (from run 2)
    # This proves that the same-run join was used for target_time > power_fcst_init_time
    assert engineered["temperature_2m_lag_2h"][0] == 10.0

    # Verify that temperature_2m_lag_36h is 8.0 (from freshest run)
    assert engineered["temperature_2m_lag_36h"][0] == 8.0
