from datetime import datetime, timezone

import polars as pl
import pytest
from contracts.data_schemas import SubstationFlows


def test_substation_flows_validation_mw_or_mva():
    # Valid with MW
    df_mw = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            "MW": [10.0],
        }
    ).with_columns(
        [
            pl.col("MW").cast(pl.Float32),
        ]
    )

    # Should pass
    SubstationFlows.validate(df_mw)

    # Valid with MVA
    df_mva = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            "MVA": [10.0],
        }
    ).with_columns(
        [
            pl.col("MVA").cast(pl.Float32),
        ]
    )

    # Should pass
    SubstationFlows.validate(df_mva)

    # Invalid: neither MW nor MVA
    df_none = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            "MVAr": [5.0],
        }
    ).with_columns(
        [
            pl.col("MVAr").cast(pl.Float32),
        ]
    )

    with pytest.raises(ValueError, match="at least one of 'MW' or 'MVA' columns"):
        SubstationFlows.validate(df_none)


def test_substation_flows_validation_both():
    # Valid with both
    df_both = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            "MW": [10.0],
            "MVA": [12.0],
        }
    ).with_columns(
        [
            pl.col("MW").cast(pl.Float32),
            pl.col("MVA").cast(pl.Float32),
        ]
    )

    # Should pass
    SubstationFlows.validate(df_both)
