"""Build experiment 25's input tables: price, system-price, and national columns, past values only.

Two tables are written under the experiment's private data folder:

- `series.parquet`: one row per half-hour `start` (UTC start of the half-hour) on a dense grid, with
  the raw series (`price`, `ssp`, `niv`, `demand`, `wind`) and every derived column.
- `features.parquet`: experiment 00006's feature table, with the derived columns joined at the
  target half-hour's `start` (`valid_time` minus 30 minutes).

Every derived column at `start` uses values at `start` or earlier: differences against earlier
half-hours, and trailing windows that end at `start`. `check_no_lookahead` proves it by removing
the future rows of the series and checking that no column changes.

    uv run python ml-experiments/00025-price-change-features-stage1-demand/e00025_inputs.py
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00025-price-change-features-stage1-demand"
EXPERIMENT_00006_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00006-stage1a-demand-input-feature-sweep" / "inputs"
)
EXPERIMENT_00010_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00010-elexon-price-lags-stage1a-demand"
)
"""Experiment 10's pull of Market Index Data and national series, read as data only."""
SYSTEM_PRICE_DIRS: Final[tuple[Path, ...]] = (
    EXPERIMENT_DIR / "downloads" / "elexon_system_prices" / "_day_cache",
    ML_EXPERIMENTS_DATA_DIR.parent / "studies" / "downloads" / "market" / "elexon_system_prices",
)
"""2019-09 to 2025-08 fetched for this experiment; 2025-09 to 2026-09 was already on disk."""
SERIES_PATH: Final[Path] = EXPERIMENT_DIR / "inputs" / "series.parquet"
FEATURES_PATH: Final[Path] = EXPERIMENT_DIR / "inputs" / "features.parquet"
HALF_HOUR: Final[timedelta] = timedelta(minutes=30)
GRID_START: Final[datetime] = datetime(2019, 1, 1, tzinfo=UTC)
GRID_END: Final[datetime] = datetime(2026, 10, 13, tzinfo=UTC)
DAY: Final[int] = 48
"""Half-hours in 24 hours: the trailing window of the shape columns."""
SIX_HOURS: Final[int] = 12
MIN_WINDOW_SAMPLES: Final[int] = 24
SPIKE_STANDARD_DEVIATIONS: Final[float] = 2.0
CHANGE_LAGS: Final[tuple[int, ...]] = (1, 2, 4, 12)

CHANGE_COLUMNS: Final[tuple[str, ...]] = (
    "price_now",
    *(f"price_d{lag}" for lag in CHANGE_LAGS),
)
SHAPE_COLUMNS: Final[tuple[str, ...]] = (
    "price_now",
    "price_minus_day_mean",
    "price_day_rank",
    "price_half_hours_since_spike",
    "price_six_hour_range",
)
NATIONAL_COLUMNS: Final[tuple[str, ...]] = (
    "demand_now",
    "wind_now",
    "demand_d1",
    "wind_d1",
    "demand_d4",
)
SYSTEM_COLUMNS: Final[tuple[str, ...]] = (
    "price_now",
    "ssp_now",
    "niv_now",
    "ssp_d1",
    "niv_d1",
)
DERIVED_COLUMNS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys((*CHANGE_COLUMNS, *SHAPE_COLUMNS, *NATIONAL_COLUMNS, *SYSTEM_COLUMNS))
)


def raw_series() -> pl.DataFrame:
    """Return the dense half-hourly grid with `price`, `ssp`, `niv`, `demand`, and `wind`."""
    grid = pl.DataFrame(
        {
            "start": pl.datetime_range(
                GRID_START, GRID_END, interval="30m", time_zone="UTC", eager=True
            )
        }
    )
    prices = (
        pl.read_parquet(EXPERIMENT_00010_DIR / "prices" / "market_index.parquet")
        .filter(pl.col("dataProvider") == "APXMIDP")
        .unique("time")
        .select(pl.col("time").alias("start"), price=pl.when(pl.col("volume") > 0).then("price"))
    )
    system_frames = [
        pl.read_parquet(path)
        for directory in SYSTEM_PRICE_DIRS
        for path in sorted(directory.glob("*.parquet"))
    ]
    system = (
        pl.concat(system_frames)
        .unique("time")
        .select(
            pl.col("time").alias("start"),
            ssp=pl.col("system_sell_price_gbp_per_mwh"),
            niv=pl.col("net_imbalance_volume_mwh"),
        )
    )
    control = EXPERIMENT_00010_DIR / "control"
    demand = (
        pl.read_parquet(control / "elexon_demand_outturn" / "elexon_demand_outturn.parquet")
        .unique("time")
        .select(pl.col("time").alias("start"), demand=pl.col("indo_mw"))
    )
    wind = (
        pl.read_parquet(control / "elexon_fuelhh" / "elexon_fuelhh.parquet")
        .filter(pl.col("fuel_type") == "WIND")
        .unique("time")
        .select(pl.col("time").alias("start"), wind=pl.col("generation_mw"))
    )
    out = grid
    for table in (prices, system, demand, wind):
        out = out.join(table, on="start", how="left", validate="1:1")
    return out.sort("start")


def day_rank(column: str) -> pl.Expr:
    """Return the share of the trailing 24 hours (including now) priced at or below now.

    A half-hour whose own value is null gives null, and null half-hours in the window are skipped.
    """
    now = pl.col(column)
    below = pl.sum_horizontal(
        (pl.col(column).shift(lag) <= now).cast(pl.Int32) for lag in range(DAY)
    )
    seen = pl.sum_horizontal(
        pl.col(column).shift(lag).is_not_null().cast(pl.Int32) for lag in range(DAY)
    )
    return pl.when(now.is_not_null() & (seen >= MIN_WINDOW_SAMPLES)).then(below / seen)


def derive(series: pl.DataFrame) -> pl.DataFrame:
    """Add every derived column to the dense grid, using values at or before each half-hour."""
    window = {"window_size": DAY, "min_samples": MIN_WINDOW_SAMPLES}
    with_changes = series.with_columns(
        price_now=pl.col("price"),
        ssp_now=pl.col("ssp"),
        niv_now=pl.col("niv"),
        demand_now=pl.col("demand"),
        wind_now=pl.col("wind"),
        **{f"price_d{lag}": pl.col("price") - pl.col("price").shift(lag) for lag in CHANGE_LAGS},
        ssp_d1=pl.col("ssp") - pl.col("ssp").shift(1),
        niv_d1=pl.col("niv") - pl.col("niv").shift(1),
        demand_d1=pl.col("demand") - pl.col("demand").shift(1),
        wind_d1=pl.col("wind") - pl.col("wind").shift(1),
        demand_d4=pl.col("demand") - pl.col("demand").shift(4),
        price_day_mean=pl.col("price").rolling_mean(**window),
        price_day_std=pl.col("price").rolling_std(**window),
        price_six_hour_range=(
            pl.col("price").rolling_max(window_size=SIX_HOURS, min_samples=SIX_HOURS // 2)
            - pl.col("price").rolling_min(window_size=SIX_HOURS, min_samples=SIX_HOURS // 2)
        ),
        price_day_rank=day_rank("price"),
    )
    return (
        with_changes.with_columns(
            price_minus_day_mean=pl.col("price") - pl.col("price_day_mean"),
            spike_index=pl.when(
                pl.col("price")
                > pl.col("price_day_mean") + SPIKE_STANDARD_DEVIATIONS * pl.col("price_day_std")
            )
            .then(pl.int_range(pl.len()))
            .otherwise(None),
            row=pl.int_range(pl.len()),
        )
        .with_columns(last_spike=pl.col("spike_index").forward_fill())
        .with_columns(
            price_half_hours_since_spike=pl.when(pl.col("price").is_not_null()).then(
                pl.min_horizontal(pl.col("row") - pl.col("last_spike"), pl.lit(DAY)).fill_null(DAY)
            )
        )
        .drop("price_day_mean", "price_day_std", "spike_index", "row", "last_spike")
    )


def check_no_lookahead(series: pl.DataFrame) -> None:
    """Fail if any derived column changes when the rows after a cut-off are removed."""
    cut = series.height * 3 // 5
    full = derive(series).head(cut)
    truncated = derive(series.head(cut))
    for column in DERIVED_COLUMNS:
        left, right = full[column], truncated[column]
        same = ((left == right) | (left.is_null() & right.is_null())).all()
        if not same:
            raise AssertionError(f"{column} changes when future rows are removed")


def build_features(series: pl.DataFrame) -> pl.DataFrame:
    """Return experiment 00006's feature table with the derived columns joined at `start`."""
    features = pl.read_parquet(EXPERIMENT_00006_DIR / "features.parquet")
    return features.with_columns(start=pl.col("valid_time") - HALF_HOUR).join(
        series.select("start", *DERIVED_COLUMNS), on="start", how="left", validate="m:1"
    )


def main() -> None:
    """Write both tables and print the share of each column that is missing."""
    series = derive(raw_series())
    check_no_lookahead(raw_series())
    SERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    series.write_parquet(SERIES_PATH, compression="zstd")
    features = build_features(series)
    features.write_parquet(FEATURES_PATH, compression="zstd")
    for column in ("price", "ssp", "niv", "demand", "wind", *DERIVED_COLUMNS):
        print(f"{column}: {series[column].null_count() / series.height:.4f} missing in the grid")
    print(f"wrote {series.height} grid rows and {features.height} feature rows")


if __name__ == "__main__":
    main()
