import polars as pl
from contracts.settings import Settings


# TODO: Rename function to clean_power_time_series.
# TODO: Rename substation_flows_df to power_time_series, and its type should be
# `pt.DataFrame[PowerTimeSeries]`
# TODO: The return type should be pt.DataFrame[PowerTimeSeries]
def clean_substation_flows(
    substation_flows_df: pl.DataFrame,
    settings: Settings,
) -> pl.DataFrame:
    """Clean substation flows by replacing stuck and insane values with null."""

    def _compute_rolling_std(
        power_col: str, std_threshold: float, window_size: int = 48
    ) -> pl.Expr:
        return (
            pl.col(power_col).rolling_std(window_size).fill_null(0).over("substation_number")
            < std_threshold
        )

    def _compute_insane_mask(power_col: str, min_thresh: float, max_thresh: float) -> pl.Expr:
        return (pl.col(power_col) < min_thresh) | (pl.col(power_col) > max_thresh)

    # TODO: This logic is a hang-over from when the dataframe contained MW and MVA. But now the code
    # uses the PowerTimeSeries data contract, which just has a single `power` column.
    power_columns = [col for col in ["MW", "MVA"] if col in substation_flows_df.columns]

    if not power_columns:
        return substation_flows_df

    stuck_masks = [
        _compute_rolling_std(col, settings.data_quality.stuck_std_threshold)
        for col in power_columns
    ]
    stuck_mask = pl.any_horizontal(stuck_masks)

    insane_masks = [
        _compute_insane_mask(
            col, settings.data_quality.min_mw_threshold, settings.data_quality.max_mw_threshold
        )
        for col in power_columns
    ]
    insane_mask = pl.any_horizontal(insane_masks)

    bad_value_mask = stuck_mask | insane_mask

    df_cleaned = substation_flows_df.with_columns(
        [
            pl.when(bad_value_mask).then(pl.lit(None)).otherwise(pl.col(col)).alias(col)
            for col in power_columns
        ]
    )

    return df_cleaned
