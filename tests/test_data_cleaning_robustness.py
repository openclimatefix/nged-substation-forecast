import polars as pl
import dagster as dg
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

from src.nged_substation_forecast.defs.data_cleaning_assets import cleaned_actuals
from contracts.settings import Settings, DataQualitySettings
from contracts.data_schemas import UTC_DATETIME_DTYPE


def test_cleaned_actuals_lookback_logic(tmp_path: Path):
    """Test that cleaned_actuals correctly uses a 1-day lookback from Delta table."""

    # 1. Setup paths
    delta_dir = tmp_path / "delta"
    raw_flows_path = delta_dir / "raw_power_time_series"
    raw_flows_path.mkdir(parents=True)

    # 2. Create dummy data for two days
    # Day 1: 2026-03-09
    # Day 2: 2026-03-10 (The partition we will run)

    t1 = datetime(2026, 3, 9, 12, tzinfo=timezone.utc)
    t2 = datetime(2026, 3, 10, 12, tzinfo=timezone.utc)

    df = pl.DataFrame(
        {
            "time_series_id": [1, 1],
            "start_time": [t1, t2],
            "period_end_time": [t1 + timedelta(minutes=30), t2 + timedelta(minutes=30)],
            "power": [10.0, 10.0],
        }
    ).with_columns(
        [
            pl.col("time_series_id").cast(pl.Int32),
            pl.col("start_time").cast(UTC_DATETIME_DTYPE),
            pl.col("period_end_time").cast(UTC_DATETIME_DTYPE),
            pl.col("power").cast(pl.Float32),
        ]
    )

    df.write_delta(str(raw_flows_path))

    # 3. Setup settings
    settings = Settings(
        nged_data_path=tmp_path,
        data_quality=DataQualitySettings(
            stuck_std_threshold=0.01, max_mw_threshold=100.0, min_mw_threshold=-20.0
        ),
    )

    # 4. Run the asset for partition 2026-03-10
    with dg.build_asset_context(
        partition_key="2026-03-10",
    ) as context:
        # We need to mock clean_substation_flows to verify what data it receives
        with patch(
            "src.nged_substation_forecast.defs.data_cleaning_assets.clean_power_time_series"
        ) as mock_clean:
            # Mock return value to be a valid DataFrame
            mock_clean.return_value = df.filter(pl.col("start_time") == t2)

            cleaned_actuals(context, settings)

            # Verify that clean_substation_flows was called with data from BOTH days
            # because of the 1-day lookback.
            called_df = mock_clean.call_args[0][0]
            assert len(called_df) == 2
            assert t1 in called_df["start_time"].to_list()
            assert t2 in called_df["start_time"].to_list()


def test_cleaned_actuals_idempotency(tmp_path: Path):
    """Test that cleaned_actuals correctly overwrites only its partition."""

    delta_dir = tmp_path / "delta"
    raw_flows_path = delta_dir / "raw_power_time_series"
    cleaned_actuals_path = delta_dir / "cleaned_actuals"
    raw_flows_path.mkdir(parents=True)

    t1 = datetime(2026, 3, 9, 0, tzinfo=timezone.utc)
    # 48 hours = 96 periods.
    # Need at least 96 rows.
    times = [t1 + timedelta(minutes=30 * i) for i in range(200)]
    powers = [10.0 + (i % 2) * 50 for i in range(200)]

    df = pl.DataFrame(
        {
            "time_series_id": [1 for _ in range(200)],
            "period_end_time": times,
            "power": powers,
        }
    ).with_columns(
        [
            pl.col("time_series_id").cast(pl.Int32),
            pl.col("period_end_time").cast(UTC_DATETIME_DTYPE),
            pl.col("power").cast(pl.Float32),
        ]
    )

    df.write_delta(str(raw_flows_path))

    settings = Settings(nged_data_path=tmp_path)

    with dg.build_asset_context(
        partition_key="2026-03-10",
    ) as context:
        # Run once
        cleaned_actuals(context, settings)

        # Verify data exists
        df_result = pl.read_delta(str(cleaned_actuals_path)).sort(
            ["time_series_id", "period_end_time"]
        )
        print(df_result)
        assert len(df_result) > 0
        assert df_result["power"][0] == 10.0

        # Run again with different data in source for same partition
        df_new = pl.DataFrame(
            {
                "time_series_id": [1 for _ in range(200)],
                "period_end_time": times,
                "power": [20.0 + (i % 2) * 50 for i in range(200)],
            }
        ).with_columns(
            [
                pl.col("time_series_id").cast(pl.Int32),
                pl.col("period_end_time").cast(UTC_DATETIME_DTYPE),
                pl.col("power").cast(pl.Float32),
            ]
        )

        df_new.write_delta(
            str(raw_flows_path),
            mode="overwrite",
            delta_write_options={"predicate": f"period_end_time >= '{times[0].isoformat()}'"},
        )

        cleaned_actuals(context, settings)

        # Verify data was overwritten
        df_result = pl.read_delta(str(cleaned_actuals_path)).sort(
            ["time_series_id", "period_end_time"]
        )
        assert len(df_result) > 0
        assert df_result["power"][0] == 20.0
