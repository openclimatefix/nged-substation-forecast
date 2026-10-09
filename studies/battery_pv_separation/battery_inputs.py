"""Load the inputs that every script of the battery study shares.

The batteries' half-hourly settled output (B1610) is the census study's download. The prices are
the public downloads under `data/studies/downloads/market/`. B1610 stamps each half-hour at its
end and the price files stamp each period at its start, so `battery_frame` moves the output to
period starts before it joins anything. Every time here is UTC.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import (
    BATTERY_PV_SEPARATION_DIR,
    MARKET_DOWNLOADS_DIR,
    SOLAR_BMU_CENSUS_INPUTS_DIR,
)

OUTPUT_DIR: Final[Path] = BATTERY_PV_SEPARATION_DIR
"""Where this study's results live."""
B1610_DIR: Final[Path] = SOLAR_BMU_CENSUS_INPUTS_DIR / "b1610"
"""The census's half-hourly output, one parquet file per BMU."""
B1610_SUFFIX: Final[str] = "_20250901_20260901.parquet"
BATTERIES: Final[tuple[str, ...]] = ("T_LKSDB-1", "E_DOLLB-1", "T_THURB-1", "T_OCHLB-1")
"""The four batteries. The first two were named in advance. The other two have the largest energy
throughput among the list's batteries (pumped storage excluded) that are not a second unit at a
site already listed."""
NAMES: Final[dict[str, str]] = {
    "T_LKSDB-1": "Lakeside",
    "E_DOLLB-1": "Dollymans",
    "T_THURB-1": "Thurrock",
    "T_OCHLB-1": "Ocker Hill",
}
"""A short name for each battery, used in chart labels."""
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
TERCILE_LABELS: Final[tuple[str, str, str]] = (
    "Cheapest third of the day",
    "Middle third",
    "Dearest third of the day",
)
"""The within-day price terciles, cheapest first."""
EXPECTED_ROWS: Final[int] = 365 * HALF_HOURS_PER_DAY
"""Half-hours in the window."""


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


def battery_frame(*, bmu_id: str) -> pl.DataFrame:
    """Return a battery's output joined to prices, with the within-day price rank.

    Args:
        bmu_id: The Elexon BMU identifier.

    Returns:
        One row per half-hour in which both the output and the day-ahead price exist, with the
        output, the prices, `date` (the UTC day), `tod` (the half-hour of the UTC day, 0 to 47),
        `rank_pct` (the half-hour's day-ahead price rank among its day's 48, ties averaged, scaled
        to the open interval 0 to 1), `tercile` (0 for the cheapest third of the day), `month`
        (`YYYY-MM`), and `fold`.
    """
    joined = battery_output(bmu_id=bmu_id).join(market_frame(), on="time", how="inner")
    return (
        joined.drop_nulls("day_ahead_gbp_per_mwh")
        .with_columns(
            date=pl.col("time").dt.date(),
            tod=pl.col("time").dt.hour().cast(pl.Int32) * 2 + pl.col("time").dt.minute() // 30,
            month=pl.col("time").dt.strftime("%Y-%m"),
            calendar_month=pl.col("time").dt.month(),
        )
        .with_columns(
            rank_pct=(
                (pl.col("day_ahead_gbp_per_mwh").rank("average").over("date") - 0.5)
                / HALF_HOURS_PER_DAY
            ),
            fold=pl.col("calendar_month").replace_strict(FOLD_MONTHS, return_dtype=pl.Int64),
        )
        .with_columns(tercile=(pl.col("rank_pct") * 3).floor().cast(pl.Int64).clip(0, 2))
        .drop("calendar_month")
        .sort("time")
    )


def p99_output_mw(*, frame: pl.DataFrame) -> float:
    """Return the 99th percentile of a battery's absolute output, the study's scale for errors.

    Args:
        frame: A frame with `output_mw`.

    Returns:
        The percentile, in megawatts.
    """
    return float(np.quantile(frame["output_mw"].abs().to_numpy(), 0.99))
