"""Summarise the stored bias rows of experiment 274 and print every number the page quotes.

uv run --no-sync python e00274_report.py > printed_report.txt
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final

import e00274_fit as fit
import numpy as np
import polars as pl

N_DRAWS: Final[int] = 20_000
SEED: Final[int] = 274
TOLERANCE_POINTS: Final[float] = 0.1
TOLERANCE_SHARE: Final[float] = 0.05
PHASES: Final[tuple[str, ...]] = ("BEFORE", "DURING", "AFTER_48H", "LATER_3M", "LATER_4_12M")
UNIT_KEYS: Final[list[str]] = ["site", "onset"]
SCENARIO_KEYS: Final[list[str]] = ["size", "sign", "shape", "duration_days", "window"]


def load_rows() -> pl.DataFrame:
    """Return every job's rows with the end date, the months since the end, and the phase group.

    The `LATER` phase is split at three months after the end of the transfer.
    """
    rows = pl.read_parquet(fit.ROWS_DIR / "*.parquet").with_columns(
        end=pl.when(pl.col("duration_days") > 0).then(
            pl.col("onset") + pl.duration(days=pl.col("duration_days"))
        )
    )
    months_after_end = (pl.col("cut") - pl.col("end")).dt.total_days() / 30.4375
    return rows.with_columns(
        phase_group=pl.when(pl.col("phase") != "LATER")
        .then(pl.col("phase"))
        .when(months_after_end < 3)
        .then(pl.lit("LATER_3M"))
        .otherwise(pl.lit("LATER_4_12M")),
        season=pl.when(pl.col("onset").dt.month() == 12)
        .then(pl.lit("winter onset"))
        .otherwise(pl.lit("summer onset")),
    )


def unit_means(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return the row-weighted mean bias per unit, scenario, and phase group, and the share learned.

    The share is the bias divided by the mean transfer of the same unit and scenario over its
    DURING rows, because the transfer is zero after it ends.
    """
    keys = [*UNIT_KEYS, *SCENARIO_KEYS]
    weight = pl.col("n_rows")

    def mean_of(column: str) -> pl.Expr:
        return (pl.col(column) * weight).sum() / weight.sum()

    transfer = (
        rows.filter(pl.col("phase") == "DURING")
        .group_by(keys)
        .agg(transfer=mean_of("mean_transfer"))
    )
    return (
        rows.group_by([*keys, "phase_group"])
        .agg(
            n_rows=weight.sum(),
            bias=mean_of("mean_bias"),
            error_injected=mean_of("mean_error_injected_vs_nra"),
            error_clean=mean_of("mean_error_clean_vs_nra"),
        )
        .join(transfer, on=keys, how="left")
        .with_columns(share=pl.col("bias") / pl.col("transfer"))
    )


def two_way_interval(*, units: pl.DataFrame, column: str) -> tuple[float, float, float]:
    """Return the mean and the 95% interval, resampling substations and onsets independently.

    Args:
        units: One row per (site, onset) with the column to average.
        column: The column.

    Returns:
        (mean, lower, upper); NaN when no unit has a value.
    """
    table = units.pivot(on="onset", index="site", values=column).sort("site")
    grid = table.select(sorted(c for c in table.columns if c != "site")).to_numpy()
    if grid.size == 0 or not np.isfinite(grid).any():
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(SEED)
    n_sites, n_onsets = grid.shape
    site_draws = rng.integers(0, n_sites, (N_DRAWS, n_sites))
    onset_draws = rng.integers(0, n_onsets, (N_DRAWS, n_onsets))
    picked = grid[site_draws[:, :, None], onset_draws[:, None, :]]
    draws = np.nanmean(picked, axis=(1, 2))
    return float(np.nanmean(grid)), *np.quantile(draws, [0.025, 0.975])


def cell(
    *, units: pl.DataFrame, where: Mapping[str, object], column: str = "bias"
) -> tuple[float, ...]:
    """Return (mean, lower, upper) of the scenario cell whose columns equal `where`."""
    selected = units.filter(*[pl.col(k) == v for k, v in where.items()])
    return two_way_interval(units=selected, column=column)


def fmt(values: tuple[float, ...]) -> str:
    """Format a mean and interval."""
    return f"{values[0]:+.3f} ({values[1]:+.3f} to {values[2]:+.3f})"


def controls(*, rows: pl.DataFrame) -> None:
    """Print the two positive controls."""
    before = rows.filter(pl.col("phase") == "BEFORE")
    zero = rows.filter(pl.col("size") == 0)
    print("## Controls")
    print(f"BEFORE rows: {before.height} groups, max |bias| = {before['mean_bias'].abs().max()}")
    print(f"size-0 rows: {zero.height} groups, max |bias| = {zero['mean_bias'].abs().max()}")


def headline(*, units: pl.DataFrame) -> None:
    """Print the 30-day, 25% headline cell for the 24-month window and its neighbours."""
    print("\n## Headline: share transfer, receiver, 24-month window")
    for days in (7, 14, 30, 60, 90):
        for size in fit.SIZES:
            for phase in ("DURING", "AFTER_48H", "LATER_3M", "LATER_4_12M"):
                where = {
                    "window": "24m",
                    "shape": "share",
                    "sign": 1,
                    "size": size,
                    "duration_days": days,
                    "phase_group": phase,
                }
                mean = cell(units=units, where=where)
                share = cell(units=units, column="share", where=where)
                transfer = cell(units=units, column="transfer", where=where)
                print(
                    f"{days:>3} d  size {size:.2f}  {phase:<12} bias {fmt(mean)}  "
                    f"share {fmt(share)}  mean transfer {transfer[0]:.2f}"
                )


def limits(*, units: pl.DataFrame) -> None:
    """Print, per size and window, the longest tested duration whose AFTER_48H bias is tolerable."""
    print(
        "\n## Limit: longest duration with the AFTER_48H interval upper bound under the tolerance"
    )
    for window in fit.WINDOWS:
        for size in fit.SIZES:
            rows = []
            for days in (7, 14, 30, 60, 90):
                where = {
                    "window": window,
                    "shape": "share",
                    "sign": 1,
                    "size": size,
                    "duration_days": days,
                    "phase_group": "AFTER_48H",
                }
                _, _, upper_points = cell(units=units, where=where)
                _, _, upper_share = cell(units=units, column="share", where=where)
                rows.append(
                    f"{days} d: up {upper_points:+.3f} pts "
                    f"({'ok' if upper_points < TOLERANCE_POINTS else 'over'}), "
                    f"share up {upper_share:+.3f} "
                    f"({'ok' if upper_share < TOLERANCE_SHARE else 'over'})"
                )
            print(f"{window} size {size:.2f}: " + "; ".join(rows))


def splits(*, units: pl.DataFrame) -> None:
    """Print the 25% AFTER_48H and DURING bias split by window, season, sign, shape, and site."""
    print("\n## Splits at size 0.25, 30 d, AFTER_48H and DURING (share of transfer in brackets)")
    base = {"size": 0.25, "duration_days": 30}
    for phase in ("AFTER_48H", "DURING"):
        for window in fit.WINDOWS:
            for sign, shape in ((1, "share"), (-1, "share"), (1, "fixed")):
                where = {
                    **base,
                    "window": window,
                    "sign": sign,
                    "shape": shape,
                    "phase_group": phase,
                }
                print(
                    f"{phase} {window} sign {sign:+d} {shape}: "
                    f"{fmt(cell(units=units, where=where))} "
                    f"share {fmt(cell(units=units, column='share', where=where))}"
                )
        season_units = units.filter(
            pl.col("window") == "24m", pl.col("sign") == 1, pl.col("shape") == "share"
        )
        for season in ("winter", "summer"):
            sel = season_units.filter(
                pl.col("size") == 0.25,
                pl.col("duration_days") == 30,
                pl.col("phase_group") == phase,
                pl.col("onset").dt.month() == (12 if season == "winter" else 6),
            )
            print(
                f"{phase} 24m {season} onset: {fmt(two_way_interval(units=sel, column='bias'))} "
                f"share {fmt(two_way_interval(units=sel, column='share'))}"
            )
        for site in fit.SITES:
            sel = units.filter(
                pl.col("window") == "24m",
                pl.col("sign") == 1,
                pl.col("shape") == "share",
                pl.col("size") == 0.25,
                pl.col("duration_days") == 30,
                pl.col("phase_group") == phase,
                pl.col("site") == site,
            )
            if sel.is_empty():
                continue
            print(
                f"{phase} 24m {site}: mean bias {sel['bias'].mean():+.3f} "
                f"share {sel['share'].mean():+.3f}"
            )


def errors_against_nra(*, units: pl.DataFrame) -> None:
    """Print the signed error against the true NRA series, clean-trained and injected-trained."""
    print("\n## Signed error against the NRA series, 24m, size 0.25, receiver, share (AFTER_48H)")
    for days in (7, 14, 30, 60, 90):
        sel = units.filter(
            pl.col("window") == "24m",
            pl.col("sign") == 1,
            pl.col("shape") == "share",
            pl.col("size") == 0.25,
            pl.col("duration_days") == days,
            pl.col("phase_group") == "AFTER_48H",
        )
        print(
            f"{days:>3} d: clean {sel['error_clean'].mean():+.3f}  "
            f"injected {sel['error_injected'].mean():+.3f}"
        )


def permanent(*, rows: pl.DataFrame) -> None:
    """Print the learned share of a permanent step by months since the onset, per window."""
    print("\n## Permanent step, 25% share: learned share by months since onset")
    sel = rows.filter(
        pl.col("duration_days") == -1,
        pl.col("size") == 0.25,
        pl.col("sign") == 1,
        pl.col("shape") == "share",
        pl.col("phase") == "DURING",
    ).with_columns(
        months=((pl.col("cut") - pl.col("onset")).dt.total_days() / 30.4375).round().cast(pl.Int32)
    )
    for window in fit.WINDOWS:
        curve = (
            sel.filter(pl.col("window") == window)
            .group_by("months")
            .agg(
                share=(pl.col("mean_bias") / pl.col("mean_transfer")).mean(),
                n=pl.len(),
            )
            .sort("months")
        )
        text = ", ".join(f"m{m}: {s:.2f}" for m, s in curve.select("months", "share").iter_rows())
        print(f"{window}: {text}")


def main() -> None:
    """Print the full report."""
    rows = load_rows()
    print(f"rows: {rows.height}, jobs: {rows.select(UNIT_KEYS).unique().height} of 40")
    controls(rows=rows)
    units = unit_means(rows=rows)
    headline(units=units)
    limits(units=units)
    splits(units=units)
    errors_against_nra(units=units)
    permanent(rows=rows)


if __name__ == "__main__":
    main()
