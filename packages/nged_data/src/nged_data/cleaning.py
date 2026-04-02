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
       (48-period/24-hour window) to detect sensors that haven't changed reading.
       Values are marked as stuck if the rolling standard deviation falls below
       `settings.data_quality.stuck_std_threshold`.

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

    def _compute_rolling_std(
        power_col: str, std_threshold: float, window_size: int = 48
    ) -> pl.Expr:
        """Compute a boolean expression for stuck sensor detection.

        A sensor is considered "stuck" if its rolling standard deviation falls below
        the threshold over a 48-period (24-hour) window. The window is strictly
        backward-looking to prevent data leakage from future values.

        This must be computed per-substation using `.over("substation_number")`.
        Without this, the rolling std would be computed for the entire DataFrame,
        mixing data from all substations and causing incorrect stale detection.

        Args:
            power_col: Column name to check (e.g., "MW" or "MVA").
            std_threshold: Threshold below which the sensor is considered stuck.
            window_size: Number of periods for the rolling window. Defaults to 48
                        (24 hours at 30-minute resolution).

        Returns:
            A Polars expression that evaluates to True for stuck values,
            grouped by substation_number for per-substation rolling calculations.
        """
        return (
            pl.col(power_col)
            .rolling_std(window_size)
            .fill_null(float("inf"))
            .over("substation_number")
            < std_threshold
        )

    def _compute_insane_mask(power_col: str, min_thresh: float, max_thresh: float) -> pl.Expr:
        """Compute a boolean expression for insane value detection.

        A value is considered "insane" if it falls outside the physically plausible
        range [min_thresh, max_thresh].

        Args:
            power_col: Column name to check (e.g., "MW" or "MVA").
            min_thresh: Minimum threshold for valid values.
            max_thresh: Maximum threshold for valid values.

        Returns:
            A Polars expression that evaluates to True for insane values.
        """
        return (pl.col(power_col) < min_thresh) | (pl.col(power_col) > max_thresh)

    # Determine the power columns to check
    power_columns = [col for col in ["MW", "MVA"] if col in df.columns]

    if not power_columns:
        # No power columns to check, return early
        return df

    # Build the mask for stuck sensors
    stuck_masks = [
        _compute_rolling_std(col, settings.data_quality.stuck_std_threshold)
        for col in power_columns
    ]
    stuck_mask = pl.any_horizontal(stuck_masks)

    # Build the mask for insane values
    insane_masks = [
        _compute_insane_mask(
            col, settings.data_quality.min_mw_threshold, settings.data_quality.max_mw_threshold
        )
        for col in power_columns
    ]
    insane_mask = pl.any_horizontal(insane_masks)

    # Combine masks: a value is bad if it's stuck OR insane
    bad_value_mask = stuck_mask | insane_mask

    # Replace bad values with null
    df_cleaned = df.with_columns(
        [
            pl.when(bad_value_mask).then(pl.lit(None)).otherwise(pl.col(col)).alias(col)
            for col in power_columns
        ]
    )

    return df_cleaned
