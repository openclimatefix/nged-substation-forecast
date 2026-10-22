"""Build the day table: each site's local days with their score, excess loss, and labels.

The excess of a day is its mean row score minus the median of the site's day scores. Labels are
the five of the plan: `switching`, `flagged`, `cold`, `economy7`, and `holiday`.
"""

from typing import Final

import e00172_common as common
import polars as pl

WORST_RANK_COLUMN: Final[str] = "rank"


def build_days(
    *, rows: pl.DataFrame, switching: pl.DataFrame, flagged: pl.DataFrame
) -> pl.DataFrame:
    """Return one row per (site, local day) with at least `MIN_ROWS_PER_DAY` rows."""
    day_score = rows.group_by(["site", "local_date"]).agg(
        n_rows=pl.len(),
        score=pl.col("score").mean(),
        month=pl.col("month").first(),
        year_month=(pl.col("local_date").dt.year() * 100 + pl.col("month")).first(),
        holiday=pl.col("holiday").any(),
        day_temperature_c=pl.col("day_temperature_c").first(),
        mean_abs_error=(pl.col("actual") - pl.col("mean")).abs().mean(),
    )
    days = day_score.filter(pl.col("n_rows") >= common.MIN_ROWS_PER_DAY)
    median = days.group_by("site").agg(site_median=pl.col("score").median())
    days = days.join(median, on="site").with_columns(excess=pl.col("score") - pl.col("site_median"))
    days = _add_economy_7(days=days, rows=rows)
    covered = switching.select("site", "local_date").unique().with_columns(switching=pl.lit(True))
    days = days.join(covered, on=["site", "local_date"], how="left").join(
        flagged, on=["site", "local_date"], how="left"
    )
    days = days.with_columns(
        switching=pl.col("switching").fill_null(False),
        flagged=(pl.col("n_flagged").fill_null(0) > 0)
        | (pl.col("n_readings").fill_null(0) < common.MIN_READINGS),
        cold=pl.col("day_temperature_c") < common.COLD_DAY_C,
    )
    labelled = pl.any_horizontal(*[pl.col(name) for name in common.LABELS])
    return days.with_columns(labelled=labelled).sort(["site", "local_date"])


def _add_economy_7(*, days: pl.DataFrame, rows: pl.DataFrame) -> pl.DataFrame:
    """Add `economy7`: a winter day on which the four step half-hours carry the loss."""
    median = days.select("site", "site_median").unique()
    step = pl.struct("utc_hour", "utc_minute").map_elements(
        lambda s: (s["utc_hour"], s["utc_minute"]) in common.E7_UTC_HALF_HOURS,
        return_dtype=pl.Boolean,
    )
    above = (
        rows.join(median, on="site")
        .with_columns(
            above=(pl.col("score") - pl.col("site_median")).clip(lower_bound=0.0),
            in_step=step,
        )
        .group_by(["site", "local_date"])
        .agg(
            step_share=(pl.col("above") * pl.col("in_step")).sum()
            / pl.col("above").sum().clip(lower_bound=1e-9),
        )
    )
    winter = pl.col("month").is_in(sorted(common.WINTER_MONTHS))
    return days.join(above, on=["site", "local_date"], how="left").with_columns(
        economy7=winter & (pl.col("step_share") >= common.E7_SHARE)
    )


def worst_days(*, days: pl.DataFrame, fraction: float = common.WORST_FRACTION) -> pl.DataFrame:
    """Return each site's worst `fraction` of days (rounded up) by excess loss, with their rank."""
    ranked = days.with_columns(
        rank=pl.col("excess").rank(method="ordinal", descending=True).over("site"),
        n_days=pl.len().over("site"),
    )
    return ranked.filter(pl.col("rank") <= (pl.col("n_days") * fraction).ceil())
