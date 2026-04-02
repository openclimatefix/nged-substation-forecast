"""Data cleaning logic for substation flows.

This module provides functions to identify and handle problematic telemetry data:
- "Stuck" values: Values where the rolling standard deviation is below a threshold,
  indicating a sensor that hasn't changed reading.
- "Insane" values: Values that fall outside physically plausible min/max bounds.

All operations use backward-looking rolling windows and group by substation to prevent
data leakage. Bad values are replaced with null to preserve the temporal grid rather
than being removed or imputed.
"""

from collections.abc import Sequence

import polars as pl

from contracts.settings import Settings


def clean_substation_flows(
    df: pl.DataFrame, settings: Settings, group_by_cols: Sequence[str] | None = None
) -> pl.DataFrame:
    """Clean substation flows by replacing stuck and insane values with null.

    This function identifies problematic telemetry data and replaces these values
    with null to preserve the temporal grid. It performs the following checks:

    1. **Stuck sensor detection**: Uses a backward-looking rolling standard deviation
       (24-hour window) to detect sensors that haven't changed reading.
       Values are marked as stuck if the rolling standard deviation falls below
       `settings.data_quality.stuck_std_threshold`. We use time-based rolling
       to correctly handle different temporal resolutions and missing data.

    2. **Insane value detection**: Flags values outside the range
       `[min_mw_threshold, max_mw_threshold]` as insane. This captures physically
       implausible readings (e.g., MW > 100 for primary substations, or extreme
       negative values).

    Both checks are performed per-substation to prevent cross-substation data leakage.

    Args:
        df: Raw substation flows DataFrame with 'MW' and 'MVA' columns.
        settings: Configuration containing data quality thresholds.
        group_by_cols: Columns to group by for per-substation operations.
                     Defaults to ["substation_number"].

    Returns:
        DataFrame with stuck and insane values replaced with null. The temporal grid
        is preserved (no rows removed, no imputation).

    Example:
        >>> from contracts.settings import Settings
        >>> df = clean_substation_flows(raw_df, Settings(...))

    Notes:
        - All rolling window operations are backward-looking to prevent data leakage.
        - Null values from the original data are preserved as-is.
        - Both 'MW' and 'MVA' columns are checked independently.
        - After cleaning, downstream logic should use `pl.coalesce(['MW', 'MVA'])`
          to create the `MW_or_MVA` column.
    """

    def _compute_insane_mask(power_col: str, min_thresh: float, max_thresh: float) -> pl.Expr:
        """Compute a boolean expression for insane value detection."""
        return (pl.col(power_col) < min_thresh) | (pl.col(power_col) > max_thresh)

    # Determine the power columns to check
    power_columns = [col for col in ["MW", "MVA"] if col in df.columns]

    if not power_columns:
        # No power columns to check, return early
        return df

    # Determine grouping columns
    group_by = list(group_by_cols) if group_by_cols is not None else ["substation_number"]

    # Sort by timestamp as required by .rolling()
    df_sorted = df.sort("timestamp")

    # Build the mask for stuck sensors using time-based rolling windows.
    # This is more robust than row-based rolling as it handles missing data
    # and different temporal resolutions correctly.
    stuck_mask_expr = pl.lit(False)
    rolling_std_cols = []

    for col in power_columns:
        std_col_name = f"{col}_rolling_std"
        count_col_name = f"{col}_rolling_count"
        rolling_std_cols.extend([std_col_name, count_col_name])

        rolling_df = df_sorted.rolling(
            index_column="timestamp",
            period="24h",
            group_by=group_by,
            closed="right",
        ).agg(
            [
                pl.col(col).std().alias(std_col_name),
                pl.col(col).count().alias(count_col_name),
            ]
        )

        df_sorted = df_sorted.join(rolling_df, on=["timestamp", *group_by], how="left")

        # A sensor is stuck if its rolling std is below the threshold.
        # We require at least 48 periods (24 hours at 30m resolution) to avoid
        # false positives from short-term constant values.
        stuck_mask_expr = stuck_mask_expr | (
            (pl.col(count_col_name) >= 48)
            & (
                pl.col(std_col_name).fill_null(float("inf"))
                < settings.data_quality.stuck_std_threshold
            )
        )

    # Build the mask for insane values
    insane_mask_expr = pl.any_horizontal(
        [
            _compute_insane_mask(
                col,
                settings.data_quality.min_mw_threshold,
                settings.data_quality.max_mw_threshold,
            )
            for col in power_columns
        ]
    )

    # Combine masks: a value is bad if it's stuck OR insane
    bad_value_mask = stuck_mask_expr | insane_mask_expr

    # Replace bad values with null and drop temporary rolling columns
    df_cleaned = df_sorted.with_columns(
        [
            pl.when(bad_value_mask).then(pl.lit(None)).otherwise(pl.col(col)).alias(col)
            for col in power_columns
        ]
    ).drop(rolling_std_cols)

    return df_cleaned
