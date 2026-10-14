"""Build experiment 10's input table: experiment 00006's features plus the lagged price columns.

The price columns come from Elexon's Market Index Data (`APXMIDP` provider). Each is read at the
half-hour 7 days before the target half-hour, so none post-dates the valid time minus 7 days.
`valid_time` in experiment 00006's table is the end of the half-hour, and Elexon's `time` is the
start, so the target half-hour starts 30 minutes before `valid_time`.

Three column sets are added, each of three columns:

- `price_lag7d`, `price_min_lag7d`, `price_max_lag7d`: the price at the half-hour, and the minimum
  and maximum of the 48 half-hours ending there.
- `shuffled_*`: the same three, taken from one earlier week (1 to 4 weeks further back than the
  true lag), drawn with a fixed seed.
- `demand_lag7d`, `wind_lag7d`, `residual_range_lag7d`: national demand (INDO) and wind (FUELHH
  `WIND`) at the half-hour, and the range of demand minus wind over the 48 half-hours ending there.

    uv run python ml-experiments/00010-elexon-price-lags-stage1a-demand/e00010_inputs.py
"""

from datetime import timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00010-elexon-price-lags-stage1a-demand"
EXPERIMENT_00006_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00006-stage1a-demand-input-feature-sweep" / "inputs"
)
"""Experiment 00006's private inputs, read as data only."""
INPUTS_PATH: Final[Path] = EXPERIMENT_DIR / "inputs" / "features.parquet"
LAG: Final[timedelta] = timedelta(days=7)
HALF_HOUR: Final[timedelta] = timedelta(minutes=30)
WINDOW: Final[int] = 48
MIN_WINDOW_SAMPLES: Final[int] = 40
SHUFFLE_WEEKS: Final[tuple[int, ...]] = (1, 2, 3, 4)
SHUFFLE_SEED: Final[int] = 10010
PRICE_COLUMNS: Final[tuple[str, ...]] = ("price_lag7d", "price_min_lag7d", "price_max_lag7d")
SHUFFLED_COLUMNS: Final[tuple[str, ...]] = (
    "shuffled_lag7d",
    "shuffled_min_lag7d",
    "shuffled_max_lag7d",
)
CONTROL_COLUMNS: Final[tuple[str, ...]] = ("demand_lag7d", "wind_lag7d", "residual_range_lag7d")


def _grid() -> pl.DataFrame:
    """Return the half-hourly grid of start times that covers the price and control series."""
    return pl.DataFrame(
        {
            "time": pl.datetime_range(
                pl.datetime(2018, 12, 1, time_zone="UTC"),
                pl.datetime(2026, 10, 14, time_zone="UTC"),
                interval="30m",
                eager=True,
                closed="left",
            )
        }
    )


def price_table() -> pl.DataFrame:
    """Return, per half-hour start `time`, the price, and the min and max of the 48 ending there."""
    raw = (
        pl.read_parquet(EXPERIMENT_DIR / "prices" / "market_index.parquet")
        .filter(pl.col("dataProvider") == "APXMIDP")
        .unique("time")
        .with_columns(price=pl.when(pl.col("volume") > 0).then(pl.col("price")))
    )
    grid = _grid().join(raw.select("time", "price"), on="time", how="left").sort("time")
    return grid.select(
        "time",
        price=pl.col("price"),
        price_min=pl.col("price").rolling_min(WINDOW, min_samples=MIN_WINDOW_SAMPLES),
        price_max=pl.col("price").rolling_max(WINDOW, min_samples=MIN_WINDOW_SAMPLES),
    )


def control_table() -> pl.DataFrame:
    """Return, per half-hour start `time`, demand, wind, and the range of demand minus wind."""
    base = EXPERIMENT_DIR / "control"
    demand = (
        pl.read_parquet(base / "elexon_demand_outturn" / "elexon_demand_outturn.parquet")
        .unique("time")
        .select("time", demand=pl.col("indo_mw"))
    )
    wind = (
        pl.read_parquet(base / "elexon_fuelhh" / "elexon_fuelhh.parquet")
        .filter(pl.col("fuel_type") == "WIND")
        .unique("time")
        .select("time", wind=pl.col("generation_mw"))
    )
    grid = (
        _grid().join(demand, on="time", how="left").join(wind, on="time", how="left").sort("time")
    )
    residual = pl.col("demand") - pl.col("wind")
    return grid.select(
        "time",
        "demand",
        "wind",
        residual_range=residual.rolling_max(WINDOW, min_samples=MIN_WINDOW_SAMPLES)
        - residual.rolling_min(WINDOW, min_samples=MIN_WINDOW_SAMPLES),
    )


def shuffle_weeks(*, n_weeks: int) -> np.ndarray:
    """Return, for each week index, an extra lag of 1 to 4 weeks, from a fixed seed."""
    rng = np.random.default_rng(SHUFFLE_SEED)
    return rng.choice(np.array(SHUFFLE_WEEKS), size=n_weeks)


def build() -> pl.DataFrame:
    """Return experiment 00006's feature table with the price, shuffled, and control columns."""
    features = pl.read_parquet(EXPERIMENT_00006_DIR / "features.parquet")
    prices = price_table()
    controls = control_table()
    origin = features["valid_time"].min()
    n_days = features.select(
        (pl.col("valid_time").max() - pl.col("valid_time").min()).dt.total_days()
    ).item()
    offsets = shuffle_weeks(n_weeks=int(n_days) // 7 + 2)
    frame = features.with_columns(
        start=pl.col("valid_time") - HALF_HOUR,
        week=((pl.col("valid_time") - origin).dt.total_days() // 7).cast(pl.Int64),
    ).with_columns(
        lag_start=pl.col("start") - LAG,
        shuffled_start=pl.col("start")
        - LAG
        - pl.duration(days=7 * pl.col("week").replace_strict(dict(enumerate(offsets.tolist())))),
    )
    return (
        frame.join(
            prices.rename(
                {"time": "lag_start", "price": "price_lag7d"}
                | {"price_min": "price_min_lag7d", "price_max": "price_max_lag7d"}
            ),
            on="lag_start",
            how="left",
            validate="m:1",
        )
        .join(
            prices.rename(
                {"time": "shuffled_start", "price": "shuffled_lag7d"}
                | {"price_min": "shuffled_min_lag7d", "price_max": "shuffled_max_lag7d"}
            ),
            on="shuffled_start",
            how="left",
            validate="m:1",
        )
        .join(
            controls.rename(
                {"time": "lag_start", "demand": "demand_lag7d", "wind": "wind_lag7d"}
                | {"residual_range": "residual_range_lag7d"}
            ),
            on="lag_start",
            how="left",
            validate="m:1",
        )
        .drop("start", "week", "lag_start", "shuffled_start")
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
