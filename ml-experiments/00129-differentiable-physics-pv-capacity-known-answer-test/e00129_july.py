"""Rows of experiment 103 extended through July 2026, for experiment 129's hindsight table.

Experiment 103's `rows.parquet` ends at 2026-07-01 00:00 UTC. The maintainer-approved exception
lets experiment 129 fit to 2026-08-31, but August's final ERA5 is not out, so the data stops at
2026-07-31. This script appends July's half-hours to a copy of `rows.parquet`, derived from
experiment 103's `panel.parquet` by the same rules as `e00103_decide.py`: `in_pool` is the pooled
roles with a clean (present and unflagged) reading, `early_cut` and `lockdown` are false after
2021, `day_type` has no bank holiday in July, and `scarce_kind` is empty because no July day has a
mean temperature below zero. The copy is `work/rows_to_july.parquet`; experiment 103's file is not
touched.

    uv run --no-sync python ml-experiments/00129-*/e00129_july.py
"""

from datetime import UTC, datetime
from typing import Final

import e00129_common as common
import polars as pl
from studies.dp_features import PART_1_DIR, ROWS_NAME

JULY_START: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
JULY_END: Final[datetime] = datetime(2026, 8, 1, tzinfo=UTC)
POOLED_ROLES: Final[tuple[str, ...]] = ("mw_long", "mw_short", "mva_dark")


def july_rows(*, existing: pl.DataFrame) -> pl.DataFrame:
    """Return July's rows, with the columns and dtypes of experiment 103's `rows.parquet`."""
    role_of = dict(existing.group_by("site").agg(pl.col("role").first()).iter_rows())
    panel = pl.read_parquet(PART_1_DIR / "panel.parquet").filter(
        pl.col("time") > JULY_START,
        pl.col("time") <= JULY_END,
        pl.col("site").is_in(list(role_of)),
    )
    weekday = pl.col("local_date").dt.weekday()
    return panel.select(
        "time_series_id",
        "site",
        "time",
        pl.col("power").cast(pl.Float32),
        "present",
        in_pool=pl.col("site").replace_strict(role_of).is_in(POOLED_ROLES)
        & pl.col("present")
        & ~pl.col("flagged"),
        default_rows=pl.col("site").replace_strict(role_of).is_in(POOLED_ROLES)
        & pl.col("present")
        & ~pl.col("flagged")
        & ~pl.col("switching_mask"),
        early_cut=pl.lit(value=False),
        lockdown=pl.lit(value=False),
        switching_mask="switching_mask",
        dark="dark",
        scarce_kind=pl.lit(None, dtype=pl.String),
        season_year=pl.lit(2025, dtype=pl.Int32),
        day_type=pl.when(weekday == 6)
        .then(pl.lit("saturday"))
        .when(weekday == 7)
        .then(pl.lit("sunday"))
        .otherwise(pl.lit("weekday")),
        role=pl.col("site").replace_strict(role_of),
    )


def main() -> None:
    """Write `work/rows_to_july.parquet`."""
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    existing = pl.read_parquet(PART_1_DIR / ROWS_NAME)
    extended = pl.concat([existing, july_rows(existing=existing).select(existing.columns)])
    assert (
        extended.select(pl.struct("time_series_id", "time").is_duplicated().any()).item() is False
    )
    extended.write_parquet(common.WORK_DIR / "rows_to_july.parquet")
    print(extended.group_by("site").agg(last=pl.col("time").max()).sort("site"))
    print(extended.filter(pl.col("time") > JULY_START).select(pl.col("in_pool").mean()))


if __name__ == "__main__":
    main()
