"""Build experiment 15's input table: experiment 00006's features plus the same-time price columns.

`valid_time` in experiment 00006's table is the end of the half-hour, and Elexon's `time` is the
start, so the target half-hour starts 30 minutes before `valid_time`. Three column sets are added,
none lagged:

- `price_now`: the `APXMIDP` price of the target half-hour, missing if its volume is zero.
- `shuffled_now`: the price of the same weekday and half-hour 1 to 4 weeks away in either
  direction, drawn per week with a fixed seed.
- `demand_now` and `wind_now`: GB national demand (`INDO`) and wind (`FUELHH` `WIND`) of the
  target half-hour.

The price and national series files are those experiment 10 pulled.

    uv run python ml-experiments/00015-same-time-price-stage1-demand/e00015_inputs.py
"""

from datetime import timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00015-same-time-price-stage1-demand"
EXPERIMENT_00006_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00006-stage1a-demand-input-feature-sweep" / "inputs"
)
"""Experiment 00006's private inputs, read as data only."""
EXPERIMENT_00010_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00010-elexon-price-lags-stage1a-demand"
)
"""Experiment 10's price and national-series pulls, read as data only."""
INPUTS_PATH: Final[Path] = EXPERIMENT_DIR / "inputs" / "features.parquet"
HALF_HOUR: Final[timedelta] = timedelta(minutes=30)
SHUFFLE_WEEKS: Final[tuple[int, ...]] = (-4, -3, -2, -1, 1, 2, 3, 4)
SHUFFLE_SEED: Final[int] = 10015
PRICE_COLUMNS: Final[tuple[str, ...]] = ("price_now",)
SHUFFLED_COLUMNS: Final[tuple[str, ...]] = ("shuffled_now",)
CONTROL_COLUMNS: Final[tuple[str, ...]] = ("demand_now", "wind_now")


def half_hourly_series() -> pl.DataFrame:
    """Return, per half-hour start `time`, the price, national demand, and wind."""
    prices = (
        pl.read_parquet(EXPERIMENT_00010_DIR / "prices" / "market_index.parquet")
        .filter(pl.col("dataProvider") == "APXMIDP")
        .unique("time")
        .select("time", price=pl.when(pl.col("volume") > 0).then(pl.col("price")))
    )
    control = EXPERIMENT_00010_DIR / "control"
    demand = (
        pl.read_parquet(control / "elexon_demand_outturn" / "elexon_demand_outturn.parquet")
        .unique("time")
        .select("time", demand=pl.col("indo_mw"))
    )
    wind = (
        pl.read_parquet(control / "elexon_fuelhh" / "elexon_fuelhh.parquet")
        .filter(pl.col("fuel_type") == "WIND")
        .unique("time")
        .select("time", wind=pl.col("generation_mw"))
    )
    return prices.join(demand, on="time", how="full", coalesce=True).join(
        wind, on="time", how="full", coalesce=True
    )


def build() -> pl.DataFrame:
    """Return experiment 00006's feature table with the same-time price and control columns."""
    features = pl.read_parquet(EXPERIMENT_00006_DIR / "features.parquet")
    series = half_hourly_series()
    origin = features["valid_time"].min()
    n_days = features.select(
        (pl.col("valid_time").max() - pl.col("valid_time").min()).dt.total_days()
    ).item()
    n_weeks = int(n_days) // 7 + 2
    offsets = np.random.default_rng(SHUFFLE_SEED).choice(np.array(SHUFFLE_WEEKS), size=n_weeks)
    frame = features.with_columns(
        start=pl.col("valid_time") - HALF_HOUR,
        week=((pl.col("valid_time") - origin).dt.total_days() // 7).cast(pl.Int64),
    ).with_columns(
        shuffled_start=pl.col("start")
        + pl.duration(days=7 * pl.col("week").replace_strict(dict(enumerate(offsets.tolist()))))
    )
    return (
        frame.join(
            series.select(
                pl.col("time").alias("start"),
                price_now=pl.col("price"),
                demand_now=pl.col("demand"),
                wind_now=pl.col("wind"),
            ),
            on="start",
            how="left",
            validate="m:1",
        )
        .join(
            series.select(pl.col("time").alias("shuffled_start"), shuffled_now=pl.col("price")),
            on="shuffled_start",
            how="left",
            validate="m:1",
        )
        .drop("start", "week", "shuffled_start")
    )


def main() -> None:
    """Write the input table and print the share of rows with each added column missing."""
    frame = build()
    INPUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(INPUTS_PATH, compression="zstd")
    for column in (*PRICE_COLUMNS, *SHUFFLED_COLUMNS, *CONTROL_COLUMNS):
        print(f"{column}: {frame[column].null_count() / frame.height:.4f} missing")
    print(f"wrote {frame.height} rows to {INPUTS_PATH}")


if __name__ == "__main__":
    main()
