"""Batteries' half-hourly output, the day-ahead and system prices, and the calendar constants.

The loaders that more than one study folder needs. The batteries' half-hourly settled output
(B1610) is the solar-BMU census's download. The prices are the public downloads under
`data/studies/downloads/market/`. B1610 stamps each half-hour at its end and the price files stamp
each period at its start, so `battery_output` moves the output to period starts before anything
joins it. Every time here is UTC.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

from studies.sources import MARKET_DOWNLOADS_DIR, SOLAR_BMU_CENSUS_INPUTS_DIR

B1610_DIR: Final[Path] = SOLAR_BMU_CENSUS_INPUTS_DIR / "b1610"
"""The census's half-hourly output, one parquet file per BMU."""
B1610_SUFFIX: Final[str] = "_20250901_20260901.parquet"
WINDOW_START: Final[datetime] = datetime(2025, 9, 1, tzinfo=UTC)
"""The first half-hour's start; the window is one year long."""
HALF_HOURS_PER_DAY: Final[int] = 48
MINUTES_PER_HALF_HOUR: Final[int] = 30
FOLD_MONTHS: Final[dict[int, int]] = {
    9: 0,
    10: 0,
    11: 0,
    12: 1,
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 2,
    6: 3,
    7: 3,
    8: 3,
}
"""The held-out fold of each calendar month: four contiguous blocks of three whole months."""
WEEK_START: Final[datetime] = datetime(2025, 12, 15, tzinfo=UTC)
"""The Monday of the week containing the winter solstice, fixed by the calendar rather than by how
the output looks. The solar studies use the same week."""


def market_frame() -> pl.DataFrame:
    """Return the three price series on one half-hourly grid.

    The hourly N2EX day-ahead price is repeated for both half-hours of its hour.

    Returns:
        Columns `time` (the period start), `day_ahead_gbp_per_mwh`, `system_price_gbp_per_mwh`
        (the single imbalance price, the sell price), `system_buy_price_gbp_per_mwh`, and
        `apx_index_gbp_per_mwh`, for the half-hours in the window.
    """
    base = MARKET_DOWNLOADS_DIR
    n2ex = pl.read_parquet(base / "neso_n2ex_day_ahead" / "neso_n2ex_day_ahead.parquet").select(
        hour=pl.col("time"), day_ahead_gbp_per_mwh=pl.col("price_gbp_per_mwh")
    )
    system = pl.read_parquet(base / "elexon_system_prices" / "elexon_system_prices.parquet").select(
        "time",
        system_price_gbp_per_mwh=pl.col("system_sell_price_gbp_per_mwh"),
        system_buy_price_gbp_per_mwh=pl.col("system_buy_price_gbp_per_mwh"),
    )
    apx = pl.read_parquet(base / "elexon_mid_apx" / "elexon_mid_apx.parquet").select(
        "time", apx_index_gbp_per_mwh=pl.col("price_gbp_per_mwh")
    )
    return (
        system.join(apx, on="time", how="left")
        .with_columns(hour=pl.col("time").dt.truncate("1h"))
        .join(n2ex, on="hour", how="left")
        .drop("hour")
        .filter(pl.col("time") >= WINDOW_START)
        .filter(pl.col("time") < WINDOW_START + timedelta(days=365))
        .sort("time")
    )


def battery_output(*, bmu_id: str) -> pl.DataFrame:
    """Return a battery's output on period starts.

    Args:
        bmu_id: The Elexon BMU identifier.

    Returns:
        Columns `time` (the period start, UTC), `output_mwh` (positive is export, negative is
        import), and `output_mw`, which is twice `output_mwh`. A half-hour that B1610 did not
        publish has no row.
    """
    frame = pl.read_parquet(B1610_DIR / f"{bmu_id}{B1610_SUFFIX}")
    return frame.select(
        time=pl.col("half_hour_end_time").dt.cast_time_unit("us")
        - timedelta(minutes=MINUTES_PER_HALF_HOUR),
        output_mwh=pl.col("output_mwh"),
        output_mw=pl.col("output_mwh") * 2.0,
    )


def p99_output_mw(*, frame: pl.DataFrame) -> float:
    """Return the 99th percentile of a battery's absolute output, the study's scale for errors.

    Args:
        frame: A frame with `output_mw`.

    Returns:
        The percentile, in megawatts.
    """
    return float(np.quantile(frame["output_mw"].abs().to_numpy(), 0.99))
