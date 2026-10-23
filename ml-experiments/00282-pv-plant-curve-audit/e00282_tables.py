"""Residual tables of experiment 282: measured minus modelled output, in points of capacity.

Every table is on the scored year (fold 1 of experiment 218), grouped by a slice, with a 95%
interval from resampling the 12 scored calendar months as blocks. Run it under
`systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue` with `uv run --no-sync python`.
"""

import e00282_common as common
import numpy as np
import polars as pl

SCORED_MONTH_COUNT = 12


def scored_long() -> pl.DataFrame:
    """Return the scored rows with one row per model: residual in points of capacity."""
    frame = pl.read_parquet(common.FRAME_PATH).filter(pl.col("scored"))
    parts = [
        frame.with_columns(model=pl.lit(model), residual=100.0 * (pl.col("output") - pl.col(model)))
        for model in common.MODELS
    ]
    return pl.concat(parts).with_columns(
        season=pl.col("calendar_month").replace_strict(
            {m: s for s, months in common.SEASONS.items() for m in months}, return_dtype=pl.String
        ),
        half=pl.when(pl.col("hour_of_day") < 12.0)
        .then(pl.lit("morning"))
        .otherwise(pl.lit("afternoon")),
    )


def scored_months() -> list[int]:
    """Return the sorted month indices of the scored year, whatever slice is being tabulated."""
    frame = pl.read_parquet(common.FRAME_PATH).filter(pl.col("scored"))
    return sorted(frame["month_index"].unique().to_list())


def month_block_draws() -> np.ndarray:
    """Return the bootstrap draws: an array of draws by 12 positions into the sorted month list."""
    rng = np.random.default_rng(common.BOOTSTRAP_SEED)
    return rng.integers(0, SCORED_MONTH_COUNT, size=(common.BOOTSTRAP_DRAWS, SCORED_MONTH_COUNT))


def residual_table(*, long: pl.DataFrame, by: list[str], value: str = "residual") -> pl.DataFrame:
    """Return the mean `value` per group with a 95% month-block interval and the row count.

    Args:
        long: Scored rows with a `month_index` column.
        by: The grouping columns.
        value: The column to average.

    Returns:
        One row per group with `mean`, `low`, `high`, and `n`.
    """
    months = scored_months()
    draws = month_block_draws()
    per_month = long.group_by([*by, "month_index"]).agg(total=pl.col(value).sum(), count=pl.len())
    out = []
    for key, group in per_month.group_by(by, maintain_order=True):
        total = np.zeros(len(months))
        count = np.zeros(len(months))
        for row in group.iter_rows(named=True):
            position = months.index(row["month_index"])
            total[position] = row["total"]
            count[position] = row["count"]
        numerator = total[draws].sum(axis=1)
        denominator = count[draws].sum(axis=1)
        valid = denominator > 0
        boot = numerator[valid] / denominator[valid]
        n = int(count.sum())
        out.append(
            dict(zip(by, key, strict=True))
            | {
                "mean": float(total.sum() / n),
                "low": float(np.quantile(boot, 0.025)),
                "high": float(np.quantile(boot, 0.975)),
                "n": n,
            }
        )
    return pl.DataFrame(out).sort(by)


def flag_departures(*, table: pl.DataFrame) -> pl.DataFrame:
    """Mark bins with enough rows whose interval excludes zero and whose size exceeds 5 points."""
    return table.with_columns(
        departs=(pl.col("n") >= common.MIN_BIN_ROWS)
        & ((pl.col("low") > 0) | (pl.col("high") < 0))
        & (pl.col("mean").abs() > common.DEPARTURE_POINTS),
        exceeds_stop=(pl.col("n") >= common.MIN_BIN_ROWS)
        & (pl.col("mean").abs() > common.STOP_POINTS),
    )


def main() -> None:
    """Write the check 1 to 4 residual tables and print the stop-rule summary."""
    long = scored_long()
    clear = long.filter(pl.col("clear_index") >= common.CLEAR_INDEX)
    tables = {
        "check1_bins": flag_departures(
            table=residual_table(long=long, by=["model", "site", "poa_bin"])
        ),
        "check1_season_half": residual_table(
            long=long, by=["model", "site", "season", "half", "poa_bin"]
        ),
        "check3_low_sun": residual_table(
            long=long.filter(pl.col("elevation_deg") < common.LOW_SUN_DEG),
            by=["model", "site"],
        ),
        "check3_clear_midday": residual_table(
            long=clear.filter(pl.col("elevation_deg") > common.MIDDAY_DEG),
            by=["model", "site"],
        ),
    }
    hot = long.filter(
        (pl.col("poa_fleet_w_m2") >= 700) & (pl.col("poa_fleet_w_m2") < 900)
    ).with_columns(
        temp_tercile=pl.col("temperature_c")
        .bin_quantiles([1 / 3, 2 / 3], labels=["cool", "mid", "warm"])
        .over(["model", "site"])
        .cast(pl.String)
    )
    tables["check4_temperature"] = residual_table(long=hot, by=["model", "site", "temp_tercile"])
    for name, table in tables.items():
        table.write_csv(common.DATA_DIR / f"{name}.csv")
    summary = (
        tables["check1_bins"]
        .group_by("model", "site")
        .agg(
            max_abs=pl.col("mean").abs().filter(pl.col("n") >= common.MIN_BIN_ROWS).max(),
            departs=pl.col("departs").any(),
            exceeds_stop=pl.col("exceeds_stop").any(),
        )
        .sort("model", "site")
    )
    print(summary)
    print(
        summary.group_by("model").agg(
            departs=pl.col("departs").sum(), exceeds_stop=pl.col("exceeds_stop").sum()
        )
    )


if __name__ == "__main__":
    main()
