"""Load experiment 00039's frame, and build the farm tables, Primary rows, and units.

The frame is read and never rebuilt. The farm convention is experiment 00039's: power divided by
the farm's 99th percentile over its daylight half-hours, meter dropouts removed, and remaining
negative readings set to zero, so a unit of injected PV has a true capacity of 1 MW.
"""

from collections.abc import Iterator

import e00040_common as common
import e00040_model as model
import numpy as np
import polars as pl


def load_prepared() -> pl.DataFrame:
    """Return the daylight rows of 00039's frame, after the row-count and site-set checks.

    Returns:
        The prepared rows (`model.prepare`).

    Raises:
        AssertionError: When the frame or the site set differs from experiment 00039's.
    """
    frame = pl.read_parquet(common.FRAME_PATH)
    assert frame.height == common.FRAME_ROWS, f"frame has {frame.height} rows"
    sites = set(frame["site"].unique().to_list())
    expected = {*common.DIRECTIONAL_PRIMARIES, *common.FARMS}
    assert expected <= sites, f"frame lacks sites {expected - sites}"
    return model.prepare(frame=frame)


def primary_rows(*, prepared: pl.DataFrame) -> pl.DataFrame:
    """Return the directional Primaries' rows with non-null power and temperature.

    A `day_cloud` column holds the day's mean cloud response over the day's daylight rows.
    """
    rows = prepared.filter(
        pl.col("site").is_in(common.DIRECTIONAL_PRIMARIES)
        & pl.col("power").is_not_null()
        & pl.col("hdd").is_not_null()
    )
    return rows.with_columns(day_cloud=pl.col("cloud").mean().over("site", "date"))


def farm_tables(*, prepared: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return each farm's normalised power (`farm_norm`) without dropouts, by farm label."""
    out = {}
    for farm in common.FARMS:
        rows = prepared.filter((pl.col("site") == farm) & pl.col("power").is_not_null())
        p99 = float(np.nanpercentile(rows["power"].to_numpy(), 99))
        rows = rows.filter(
            ~((pl.col("power") <= 0.0) & (pl.col("ghi_u") > common.DROPOUT_GHI_W_M2))
        )
        out[farm] = rows.select("time", farm_norm=pl.col("power").clip(lower_bound=0.0) / p99)
    return out


def season_year_rows(*, rows: pl.DataFrame) -> Iterator[tuple[int, pl.DataFrame]]:
    """Yield (season-year, rows) for each season-year with enough half-hours."""
    for year in sorted(rows["season_year"].unique().to_list()):
        block = rows.filter(pl.col("season_year") == year)
        if block.height >= common.MIN_WINDOW_ROWS:
            yield year, block


def units(
    *, primaries: pl.DataFrame, farms: dict[str, pl.DataFrame]
) -> Iterator[tuple[str, str, int, pl.DataFrame]]:
    """Yield (Primary, farm, season-year, shared rows) for every unit with enough shared rows."""
    for primary in common.DIRECTIONAL_PRIMARIES:
        base = primaries.filter(pl.col("site") == primary)
        for farm in common.FARMS:
            joined = base.join(farms[farm], on="time", how="inner")
            for year, rows in season_year_rows(rows=joined):
                yield primary, farm, year, rows
