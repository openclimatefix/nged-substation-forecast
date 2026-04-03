import polars as pl
import dagster as dg
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src.nged_substation_forecast.defs.data_cleaning_assets import cleaned_actuals
from contracts.settings import Settings, DataQualitySettings


def test_cleaned_actuals_lookback_logic(tmp_path: Path):
    """Test that cleaned_actuals correctly uses a 1-day lookback from Delta table."""

    # 1. Setup paths
    delta_dir = tmp_path / "delta"
    live_flows_path = delta_dir / "live_primary_flows"
    live_flows_path.mkdir(parents=True)

    # 2. Create dummy data for two days
    # Day 1: 2026-03-09
    # Day 2: 2026-03-10 (The partition we will run)

    t1 = datetime(2026, 3, 9, 12, tzinfo=timezone.utc)
    t2 = datetime(2026, 3, 10, 12, tzinfo=timezone.utc)

    df = pl.DataFrame(
        {
            "timestamp": [t1, t2],
            "substation_number": [1, 1],
            "MW": [10.0, 10.0],
            "MVA": [10.0, 10.0],
            "MVAr": [0.0, 0.0],
            "ingested_at": [t1, t2],
        }
    ).with_columns(
        [
            pl.col("substation_number").cast(pl.Int32),
            pl.col("MW").cast(pl.Float32),
            pl.col("MVA").cast(pl.Float32),
            pl.col("MVAr").cast(pl.Float32),
        ]
    )

    df.write_delta(str(live_flows_path))

    # 3. Setup settings
    settings = Settings(
        nged_data_path=tmp_path,
        data_quality=DataQualitySettings(
            stuck_std_threshold=0.01, max_mw_threshold=100.0, min_mw_threshold=-20.0
        ),
    )

    # 4. Run the asset for partition 2026-03-10
    context = dg.build_asset_context(
        partition_key="2026-03-10",
    )

    # We need to mock clean_substation_flows to verify what data it receives
    with patch(
        "src.nged_substation_forecast.defs.data_cleaning_assets.clean_substation_flows"
    ) as mock_clean:
        # Mock return value to be a valid DataFrame
        mock_clean.return_value = df.filter(pl.col("timestamp") == t2)

        cleaned_actuals(context, settings)

        # Verify that clean_substation_flows was called with data from BOTH days
        # because of the 1-day lookback.
        called_df = mock_clean.call_args[0][0]
        assert len(called_df) == 2
        assert t1 in called_df["timestamp"].to_list()
        assert t2 in called_df["timestamp"].to_list()


def test_cleaned_actuals_idempotency(tmp_path: Path):
    """Test that cleaned_actuals correctly overwrites only its partition."""

    delta_dir = tmp_path / "delta"
    live_flows_path = delta_dir / "live_primary_flows"
    cleaned_actuals_path = delta_dir / "cleaned_actuals"
    live_flows_path.mkdir(parents=True)

    t1 = datetime(2026, 3, 10, 12, tzinfo=timezone.utc)

    df = pl.DataFrame(
        {
            "timestamp": [t1],
            "substation_number": [1],
            "MW": [10.0],
            "MVA": [10.0],
            "MVAr": [0.0],
            "ingested_at": [t1],
        }
    ).with_columns(
        [
            pl.col("substation_number").cast(pl.Int32),
            pl.col("MW").cast(pl.Float32),
            pl.col("MVA").cast(pl.Float32),
            pl.col("MVAr").cast(pl.Float32),
        ]
    )

    df.write_delta(str(live_flows_path))

    settings = Settings(nged_data_path=tmp_path)

    context = dg.build_asset_context(
        partition_key="2026-03-10",
    )

    # Run once
    cleaned_actuals(context, settings)

    # Verify data exists
    df_result = pl.read_delta(str(cleaned_actuals_path))
    assert len(df_result) == 1
    assert df_result["MW"][0] == 10.0

    # Run again with different data in source for same partition
    df_new = pl.DataFrame(
        {
            "timestamp": [t1],
            "substation_number": [1],
            "MW": [20.0],
            "MVA": [20.0],
            "MVAr": [0.0],
            "ingested_at": [t1],
        }
    ).with_columns(
        [
            pl.col("substation_number").cast(pl.Int32),
            pl.col("MW").cast(pl.Float32),
            pl.col("MVA").cast(pl.Float32),
            pl.col("MVAr").cast(pl.Float32),
        ]
    )

    df_new.write_delta(
        str(live_flows_path),
        mode="overwrite",
        delta_write_options={"predicate": "timestamp >= '2026-03-10'"},
    )

    cleaned_actuals(context, settings)

    # Verify data was overwritten
    df_result = pl.read_delta(str(cleaned_actuals_path))
    assert len(df_result) == 1
    assert df_result["MW"][0] == 20.0
