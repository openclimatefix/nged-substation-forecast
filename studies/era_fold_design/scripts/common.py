"""Shared frame builders and fold designs for the era-fold measurement (scratch)."""

import sys
from pathlib import Path

WORKTREE = Path("/home/jack/dev/nged-substation-forecast/.claude/worktrees/era-fold-design")
sys.path.insert(0, str(WORKTREE / "studies" / "beam_diffuse_split"))
# pin the ENS script to the commit the D0 numbers came from (the worktree branch has since changed it)
sys.path.insert(0, "/home/jack/dev/nged-substation-forecast/.claude/worktrees/scratch/era-fold/code")
OUT = Path("/home/jack/dev/nged-substation-forecast/.claude/worktrees/scratch/era-fold/out")

import polars as pl  # noqa: E402
from build_dataset import _wind_sites  # noqa: E402
from export_cap import with_export_cap  # noqa: E402
from run_experiment import _add_time_features  # noqa: E402
from studies.cross_validation import (  # noqa: E402
    UKV_UPGRADE_MONTH,
    assign_folds,
    cut_eras,
)
from weather_products import PANELS, joined as solar_joined, with_eras  # noqa: E402
from weather_products import common_rows as solar_common_rows  # noqa: E402
from wind_products import common_rows as wind_common_rows  # noqa: E402
from wind_products import joined as wind_joined  # noqa: E402

DATA = Path("/home/jack/dev/nged-substation-forecast/data/studies/beam_diffuse_split")


def wind_pre_fold() -> pl.DataFrame:
    """The wind main frame before the era and fold step."""
    sites = _wind_sites()
    return _add_time_features(dataset=wind_common_rows(frame=wind_joined(sites=sites)))


def solar_pre_fold() -> pl.DataFrame:
    """The solar `long` panel rows before the era and fold step."""
    panel = PANELS["long"]
    rows = solar_common_rows(frame=solar_joined(products=panel.products))
    if panel.first_time is not None:
        rows = rows.filter(pl.col("time") >= panel.first_time)
    return _add_time_features(dataset=rows)


def d0(frame: pl.DataFrame) -> pl.DataFrame:
    return with_eras(frame=frame)


def d2(frame: pl.DataFrame) -> pl.DataFrame:
    """No era cut: folds by site only, era feature kept."""
    post = pl.col("month") >= UKV_UPGRADE_MONTH
    labelled = frame.with_columns(
        era=pl.when(post).then(pl.lit("post")).otherwise(pl.lit("pre")),
        era_code=post.cast(pl.Int8),
    )
    return assign_folds(dataset=labelled, by=("site",))


def d1(frame: pl.DataFrame, offsets: dict[int, int]) -> pl.DataFrame:
    return cut_eras(frame=frame, first_months=[UKV_UPGRADE_MONTH], fold_offsets=offsets)


def finish_solar(frame: pl.DataFrame) -> pl.DataFrame:
    return with_export_cap(dataset=frame)


SEEDS_N = 3


def expected_rows(*, kind: str, study: str, design: str) -> int:
    """Rows one arm must hold under one setting: sites x hours x seeds, from the saved fold table."""
    return pl.read_parquet(OUT / f"folds_{kind}{study}_{design}.parquet").height * SEEDS_N


def take(*, losses: pl.DataFrame, setting: str, arms: list[str], expected: int) -> pl.DataFrame:
    """Filter to one setting and the named arms, and assert each arm holds exactly `expected` rows."""
    sub = losses.filter(pl.col("setting") == setting, pl.col("arm").is_in(arms))
    counts = dict(sub.group_by("arm").len().iter_rows())
    assert set(counts) == set(arms), (setting, arms, counts)
    assert all(n == expected for n in counts.values()), (setting, expected, counts)
    return sub
