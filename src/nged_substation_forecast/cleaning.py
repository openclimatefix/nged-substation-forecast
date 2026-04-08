from collections.abc import Sequence
import polars as pl
from contracts.settings import Settings


def clean_substation_flows(
    df: pl.DataFrame, settings: Settings, group_by_cols: Sequence[str] | None = None
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

    power_columns = [col for col in ["MW", "MVA"] if col in df.columns]

    if not power_columns:
        return df

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

    df_cleaned = df.with_columns(
        [
            pl.when(bad_value_mask).then(pl.lit(None)).otherwise(pl.col(col)).alias(col)
            for col in power_columns
        ]
    )

    return df_cleaned
