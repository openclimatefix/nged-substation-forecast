"""Score experiment 17: the planned contrasts, absolute errors, yearly bias, and monthly errors.

Reads the fits `e00017_fit.py` wrote, adds the 14-day correction arm K and a climatology refitted
each month, and writes tables under the experiment's private `results/` folder.

    uv run python ml-experiments/00017-training-window-and-retraining-rule-drift-demand/e00017_report.py
"""

from datetime import UTC, datetime
from typing import Final

import e00017_fit as fit
import polars as pl
from baseline_forecasters.climatology import (
    CLIMATOLOGY_QUANTILE_COLUMNS,
    _local_calendar_cell_keys,
    _pooled_quantile_lookup,
)
from studies.bootstrap import (
    bootstrap_absolute,
    bootstrap_difference,
    bootstrap_difference_at_level,
)
from studies.experiment_store import losses_for_bootstrap

RESULTS_DIR: Final = fit.EXPERIMENT_DIR / "results"
REFERENCE: Final[str] = "A"
CONTRASTS: Final[tuple[str, ...]] = ("W6", "W12", "W18", "H0.5", "H1", "H2", "F", "Y", "K")
"""C1 to C9, in the order the plan lists them."""
LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / len(CONTRASTS))
N_RESAMPLES: Final[int] = 20_000
CORRECTION_FIRST_DAY: Final[int] = 15
CORRECTION_LAST_DAY: Final[int] = 2
"""The correction for local day d averages the errors of days d - 15 to d - 2 (14 days, 1 day lag)."""
MIN_CORRECTION_DAYS: Final[int] = 7
CLIMATOLOGY_MEDIAN: Final[str] = CLIMATOLOGY_QUANTILE_COLUMNS[25]


def read_fits() -> pl.DataFrame:
    """Return every series' predictions: `(time_series_id, valid_time, target, <arms>)`."""
    return pl.concat([pl.read_parquet(f) for f in sorted(fit.FITS_DIR.glob("*.parquet"))])


def add_correction(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Add arm K: A plus the mean signed error of A over local days d - 15 to d - 2.

    Only errors of A's own earlier forecasts are used, so nothing from the day being forecast or
    the day before enters. A day with fewer than 7 days of errors gets no correction.
    """
    local_date = pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.date()
    keyed = rows.with_columns(date=local_date, error=pl.col("target") - pl.col(REFERENCE))
    daily = keyed.group_by("time_series_id", "date").agg(
        error_sum=pl.col("error").sum(), n=pl.len()
    )
    out = []
    for series_id, group in daily.group_by("time_series_id", maintain_order=True):
        dense = (
            pl.date_range(group["date"].min(), group["date"].max(), "1d", eager=True)
            .alias("date")
            .to_frame()
            .join(group.drop("time_series_id"), on="date", how="left")
            .with_columns(pl.col("error_sum", "n").fill_null(0))
        )
        window = CORRECTION_FIRST_DAY - CORRECTION_LAST_DAY + 1
        dense = dense.with_columns(
            roll_sum=pl.col("error_sum").rolling_sum(window).shift(CORRECTION_LAST_DAY),
            roll_n=pl.col("n").rolling_sum(window).shift(CORRECTION_LAST_DAY),
            days=(pl.col("n") > 0).cast(pl.Int32).rolling_sum(window).shift(CORRECTION_LAST_DAY),
        ).with_columns(
            correction=pl.when(pl.col("days") >= MIN_CORRECTION_DAYS)
            .then(pl.col("roll_sum") / pl.col("roll_n"))
            .otherwise(0.0),
            time_series_id=pl.lit(series_id[0], dtype=pl.Int32),
        )
        out.append(dense.select("time_series_id", "date", "correction"))
    return (
        keyed.join(pl.concat(out), on=["time_series_id", "date"], how="left")
        .with_columns(K=pl.col(REFERENCE) + pl.col("correction").fill_null(0.0))
        .drop("date", "error", "correction")
    )


def climatology_rows(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Add `climatology`: the month-by-half-hour median of all history before each month's purge."""
    features = pl.read_parquet(
        fit.SIXTH_DIR / "inputs" / "features.parquet",
        columns=["time_series_id", "valid_time", "power_mw", "effective_capacity_mw"],
    )
    capacity = features.group_by("time_series_id").agg(c=pl.col("effective_capacity_mw").first())
    observed = features.filter(pl.col("power_mw").is_not_null())
    pieces = []
    for month in fit.months():
        month_rows = rows.filter(pl.col("valid_time").dt.truncate("1mo") == month)
        if month_rows.is_empty():
            continue
        training = observed.filter(
            pl.col("valid_time") >= fit.TRAIN_START, pl.col("valid_time") < month - fit.PURGE
        ).select("time_series_id", "valid_time", power=pl.col("power_mw"))
        lookup = _pooled_quantile_lookup(training).select(
            "time_series_id",
            "local_month",
            "local_half_hour_of_day",
            "local_is_weekend",
            median_mw=pl.col(CLIMATOLOGY_MEDIAN),
        )
        keys = ["time_series_id", "local_month", "local_half_hour_of_day", "local_is_weekend"]
        pieces.append(
            month_rows.with_columns(
                **_local_calendar_cell_keys(pl.col("valid_time"), time_zone="Europe/London")
            )
            .join(lookup, on=keys, how="left")
            .join(capacity, on="time_series_id")
            .with_columns(climatology=pl.col("median_mw") / pl.col("c"))
            .drop("median_mw", "c", "local_month", "local_half_hour_of_day", "local_is_weekend")
        )
    return pl.concat(pieces)


def loss_frame(*, rows: pl.DataFrame, arms: list[str]) -> pl.DataFrame:
    """Return the bootstrap's loss frame with `abs_error` in points of capacity."""
    keyed = rows.with_columns(
        power_fcst_init_time=pl.col("valid_time").dt.truncate("1mo"), seed=pl.lit(0)
    )
    return losses_for_bootstrap(
        arm_losses={
            arm: keyed.select(
                "time_series_id",
                "power_fcst_init_time",
                "valid_time",
                "seed",
                abs_error=(pl.col(arm) - pl.col("target")).abs() * 100.0,
            ).drop_nulls("abs_error")
            for arm in arms
        }
    )


def main() -> None:
    """Write and print every table."""
    rows = add_correction(rows=read_fits())
    rows = climatology_rows(rows=rows)
    arms = [REFERENCE, *CONTRASTS]
    shared = rows.filter(pl.col("climatology").is_not_null())
    print(f"rows {rows.height:,}; with a climatology cell {shared.height:,}")
    rows = shared
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(RESULTS_DIR / "scored_rows.parquet")

    frame = loss_frame(rows=rows, arms=arms)
    contrasts, absolute = [], []
    for arm in arms:
        result = bootstrap_absolute(losses=frame, arm=arm, metric="abs_error")
        absolute.append({"arm": arm, "mae": result["value"], "lower": result["lower_95"],
                         "upper": result["upper_95"], "n_months": result["n_months"]})
    clim = loss_frame(rows=shared, arms=["climatology"])
    result = bootstrap_absolute(losses=clim, arm="climatology", metric="abs_error")
    absolute.append({"arm": "climatology", "mae": result["value"], "lower": result["lower_95"],
                     "upper": result["upper_95"], "n_months": result["n_months"]})
    for number, arm in enumerate(CONTRASTS, start=1):
        low, high = bootstrap_difference_at_level(
            losses=frame, treatment=arm, reference=REFERENCE, metric="abs_error",
            level=LEVEL, n_resamples=N_RESAMPLES,
        )
        point = bootstrap_difference(
            losses=frame, treatment=arm, reference=REFERENCE, metric="abs_error"
        )
        contrasts.append({"contrast": f"C{number}", "arm": arm, "difference": point["difference"],
                          "lower": low, "upper": high, "level": LEVEL})
    year = pl.col("valid_time").dt.year()
    by_year = rows.group_by(year.alias("year")).agg(
        [(pl.col(a) - pl.col("target")).abs().mean().mul(100).alias(f"mae_{a}") for a in arms]
        + [(pl.col("target") - pl.col(a)).mean().mul(100).alias(f"bias_{a}") for a in arms]
    ).sort("year")
    by_month = rows.group_by(pl.col("valid_time").dt.truncate("1mo").alias("month")).agg(
        [(pl.col(a) - pl.col("target")).abs().mean().mul(100).alias(a) for a in arms]
        + [(pl.col("target") - pl.col(a)).mean().mul(100).alias(f"bias_{a}") for a in arms]
    ).sort("month")
    tables = {
        "absolute": pl.DataFrame(absolute),
        "contrasts": pl.DataFrame(contrasts),
        "by_year": by_year,
        "by_month": by_month,
    }
    for name, table in tables.items():
        table.write_parquet(RESULTS_DIR / f"{name}.parquet")
        with pl.Config(tbl_rows=100, tbl_cols=30, tbl_width_chars=250):
            print(f"\n## {name}\n{table}")


if __name__ == "__main__":
    main()
