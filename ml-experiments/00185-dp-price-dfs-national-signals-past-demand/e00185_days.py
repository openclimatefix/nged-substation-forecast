"""Build the day tables of experiment 185: arm D's daily loss, signed residual, and calendar flags.

The scored-year table reproduces experiment 172's `days.parquet` (control). The tuning-month table
applies the same rules to arm D's tuning-period forecasts, for choosing the signal in Part 3.
A day is a local (Europe/London) calendar day, dated by the half-hour's start; the rows are arm D's
D+1 band (lead 15 to 39 hours) outside hand-filtered switching events.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00185_common as common
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
LEVELS: Final[dict[str, float]] = {
    "q01": 0.01, "q05": 0.05, "q10": 0.10, "q25": 0.25, "q40": 0.40, "q50": 0.50,
    "q60": 0.60, "q75": 0.75, "q90": 0.90, "q95": 0.95, "q99": 0.99,
}  # fmt: skip
D1_LEAD_HOURS: Final[tuple[float, float]] = (15.0, 39.0)
MIN_ROWS_PER_DAY: Final[int] = 24
SCORED_FIRST: Final[datetime] = datetime(2025, 7, 2, tzinfo=UTC)
SCORED_LAST: Final[datetime] = datetime(2026, 6, 30, 23, 59, tzinfo=UTC)
HALF_HOUR_NS: Final[int] = 30 * 60 * 10**9


def pinball_score_expr() -> pl.Expr:
    """Return twice the mean pinball loss over the 11 quantiles, as a Polars expression."""
    losses = []
    for column, level in LEVELS.items():
        error = pl.col("actual") - pl.col(column)
        losses.append(pl.max_horizontal(level * error, (level - 1.0) * error))
    return 2.0 * pl.mean_horizontal(*losses)


def read_rows(*, period: str, arm: str = "d") -> pl.DataFrame:
    """Return an arm's rows for `period` (`scored` or `tuning`) in the D+1 band, scored by row."""
    frames = [
        pl.read_parquet(common.PREDS_DIR / f"{arm}_{period}_{site}.parquet").with_columns(
            site=pl.lit(site)
        )
        for site in common.SITES
    ]
    rows = pl.concat(frames).filter(
        pl.col("actual").is_finite(),
        pl.col("lead_hours") >= D1_LEAD_HOURS[0],
        pl.col("lead_hours") < D1_LEAD_HOURS[1],
        ~pl.col("switching_mask"),
    )
    if period == "scored":
        rows = rows.filter(
            pl.col("valid_time") >= SCORED_FIRST, pl.col("valid_time") <= SCORED_LAST
        )
    start = pl.col("valid_time") - pl.duration(minutes=30)
    local = start.dt.convert_time_zone(common.LOCAL_TZ)
    return rows.with_columns(score=pinball_score_expr(), local_date=local.dt.date())


def load_cache() -> FeatureCache:
    """Return the feature cache that experiment 155 used."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def day_temperature_and_holiday(*, cache: FeatureCache) -> pl.DataFrame:
    """Return, per site and local day, the ERA5 day-mean temperature and a bank-holiday flag."""
    end_times = cache.end_times
    start = pl.Series(end_times.astype("datetime64[us]") - np.timedelta64(30, "m"))
    local_date = start.dt.replace_time_zone("UTC").dt.convert_time_zone(common.LOCAL_TZ).dt.date()
    pieces = []
    for site in common.SITES:
        frame = pl.DataFrame(
            {
                "site": site,
                "local_date": local_date,
                "temperature_c": cache.sites[site].temperature_c,
                "holiday": cache.calendar.holiday_period >= 0,
            }
        )
        pieces.append(
            frame.group_by(["site", "local_date"]).agg(
                day_temperature_c=pl.col("temperature_c")
                .filter(pl.col("temperature_c").is_finite())
                .mean(),
                holiday=pl.col("holiday").any(),
            )
        )
    return pl.concat(pieces)


def build_days(*, period: str, cache: FeatureCache) -> pl.DataFrame:
    """Return one row per (site, local day): loss, excess loss, signed residual, and flags."""
    rows = read_rows(period=period)
    days = rows.group_by(["site", "local_date"]).agg(
        n_rows=pl.len(),
        score=pl.col("score").mean(),
        signed_error=(pl.col("actual") - pl.col("mean")).mean(),
        mean_abs_error=(pl.col("actual") - pl.col("mean")).abs().mean(),
        actual_mean=pl.col("actual").mean(),
    )
    days = days.filter(pl.col("n_rows") >= MIN_ROWS_PER_DAY)
    median = days.group_by("site").agg(site_median=pl.col("score").median())
    days = days.join(median, on="site").with_columns(excess=pl.col("score") - pl.col("site_median"))
    days = days.join(
        day_temperature_and_holiday(cache=cache), on=["site", "local_date"], how="left"
    )
    weekday = pl.col("local_date").dt.weekday()
    return days.with_columns(
        day_type=pl.when(pl.col("holiday"))
        .then(pl.lit("holiday"))
        .when(weekday == 6)
        .then(pl.lit("saturday"))
        .when(weekday == 7)
        .then(pl.lit("sunday"))
        .otherwise(pl.lit("weekday")),
        month=pl.col("local_date").dt.month(),
        year_month=pl.col("local_date").dt.year() * 100 + pl.col("local_date").dt.month(),
        period=pl.lit(period),
    ).sort(["site", "local_date"])


def main() -> None:
    """Write the scored-year and tuning-month day tables, and check the scored one against 172."""
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache()
    for period in ("scored", "tuning"):
        days = build_days(period=period, cache=cache)
        days.write_parquet(common.WORK_DIR / f"days_{period}.parquet")
        print(period, days.shape, days["local_date"].min(), days["local_date"].max())
    scored = pl.read_parquet(common.WORK_DIR / "days_scored.parquet")
    e172 = pl.read_parquet(common.E172_WORK / "days.parquet")
    joined = scored.join(e172, on=["site", "local_date"], how="full", suffix="_172")
    both = joined.filter(pl.col("excess").is_not_null() & pl.col("excess_172").is_not_null())
    print(
        "control vs 172: rows",
        scored.height,
        e172.height,
        "max |excess diff|",
        (both["excess"] - both["excess_172"]).abs().max(),
        "max |temp diff|",
        (both["day_temperature_c"] - both["day_temperature_c_172"]).abs().max(),
    )


if __name__ == "__main__":
    main()
