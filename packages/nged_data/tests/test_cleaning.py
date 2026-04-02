import polars as pl
from contracts.settings import DataQualitySettings, Settings
from nged_data.cleaning import clean_substation_flows


def test_clean_substation_flows_isolation():
    """Test that rolling standard deviation is computed per-substation.

    This ensures that data from one substation does not affect the calculation
    for another, preventing data leakage and incorrect stale detection.
    """
    # Create data for two substations.
    # Substation 1: Constant values (stale)
    # Substation 2: Varying values (not stale)
    # If they are not isolated, the rolling std might be affected by the other substation.

    # We need at least 48 periods for the rolling window.
    n_periods = 100

    df1 = pl.DataFrame(
        {
            "substation_number": [1] * n_periods,
            "MW": [10.0] * n_periods,
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 1) + pl.duration(minutes=30 * (n_periods - 1)),
                interval="30m",
                eager=True,
            ),
        }
    )

    df2 = pl.DataFrame(
        {
            "substation_number": [2] * n_periods,
            "MW": [10.0 + (i % 10) for i in range(n_periods)],
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 1) + pl.duration(minutes=30 * (n_periods - 1)),
                interval="30m",
                eager=True,
            ),
        }
    )

    # Interleave them to make it harder if not using .over()
    df = pl.concat([df1, df2]).sort("timestamp", "substation_number")

    settings = Settings.model_construct(
        data_quality=DataQualitySettings(
            stuck_std_threshold=0.01,
            max_mw_threshold=100.0,
            min_mw_threshold=-20.0,
        )
    )

    cleaned_df = clean_substation_flows(df, settings)

    # Substation 1 should have nulls (after the first 47 periods)
    s1_mw = cleaned_df.filter(pl.col("substation_number") == 1).sort("timestamp")["MW"]
    # The first 47 values are NOT null because rolling_std is null -> inf -> not < threshold
    assert not s1_mw[:47].is_null().any()
    # The rest should be null because they are stuck (std is 0.0 < 0.01)
    assert s1_mw[47:].is_null().all()

    # Substation 2 should NOT have nulls (except maybe the first 47 if they are filled with null)
    # Actually, in the code: .fill_null(float("inf")) is used before comparing with threshold.
    # So the first 47 values should NOT be null unless they are actually stuck.
    s2_mw = cleaned_df.filter(pl.col("substation_number") == 2).sort("timestamp")["MW"]
    assert not s2_mw.is_null().any()
