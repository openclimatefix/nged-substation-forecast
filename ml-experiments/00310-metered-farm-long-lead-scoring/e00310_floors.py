# ruff: noqa: E501
"""Write the `climatology` and `manual_heuristic` floors for every scored row of every day band.

Both follow experiment 224's definitions. `floor_climatology` is the farm's powers before the scored
year in the same local month, half-hour, and weekend cell, pooled over neighbouring cells with
`baseline_forecasters.climatology.CLIMATOLOGY_POOLING_OFFSETS`. `floor_heuristic` takes the
empirical quantiles of 13 earlier observed powers (analogues 1 to 6 weeks and 49 to 55 weeks
before the half-hour). Experiment 224 scored leads of 12 to 36 hours only; here an analogue counts
only when it was observed before the run (its time is at or before `init_time`), so the 1-week
analogue is dropped at leads over 7 days. The floors are the same in both history settings.

    uv run --no-sync python ml-experiments/00310-*/e00310_floors.py
"""

from datetime import timedelta
from typing import Final

import e00310_common as common
import e00310_features as feats
import numpy as np
import polars as pl
from baseline_forecasters.climatology import CLIMATOLOGY_POOLING_OFFSETS
from studies.tail_calibration import level_column

HEURISTIC_LAG_WEEKS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6, 49, 50, 51, 52, 53, 54, 55)
QCOLS: Final[list[str]] = [level_column(level=lv) for lv in common.LEVELS]


def scored_rows(*, site: str, band: str) -> pl.DataFrame:
    """Return a farm's scored rows (runs in the scored year) for a band."""
    frame = feats.load_band_farm(site, band).frame
    return frame.filter(
        (pl.col("init_time") >= common.SCORED_START) & (pl.col("time") < common.SCORED_END)
    ).select("time", "init_time", "lead_hours", "output")


def climatology_lookup(*, site: str) -> pl.DataFrame:
    """Return the 11 pooled quantiles per (month, half-hour, weekend) from powers before the scored year."""
    history = feats.output_series(site).filter(pl.col("time") < common.SCORED_START)
    keyed = (
        history.with_columns(local=pl.col("time").dt.convert_time_zone("Europe/London"))
        .select(
            "output",
            month=pl.col("local").dt.month(),
            half_hour=pl.col("local").dt.hour() * 2 + pl.col("local").dt.minute() // 30,
            weekend=pl.col("local").dt.weekday() >= 6,
        )
        .join(CLIMATOLOGY_POOLING_OFFSETS, how="cross")
        .select(
            "output",
            "weekend",
            month=((pl.col("month") - 1 + pl.col("month_offset")) % 12 + 1),
            half_hour=((pl.col("half_hour") + pl.col("half_hour_offset")) % 48),
        )
    )
    return keyed.group_by("month", "half_hour", "weekend").agg(
        *[
            pl.col("output").quantile(float(lv), "linear").alias(c)
            for lv, c in zip(common.LEVELS, QCOLS, strict=True)
        ]
    )


def floors_for(*, site: str, band: str) -> pl.DataFrame:
    """Return both floors' quantiles for the farm's scored rows in a band."""
    rows = scored_rows(site=site, band=band)
    history = feats.output_series(site).select("time", "output")
    columns = []
    for weeks in HEURISTIC_LAG_WEEKS:
        lag_time = pl.col("time") - timedelta(weeks=weeks)
        lagged = rows.select("time", "init_time", lag_time=lag_time).join(
            history.rename({"time": "lag_time", "output": "lag"}), on="lag_time", how="left"
        )
        known = lagged["lag_time"] <= lagged["init_time"]
        columns.append(
            lagged.select(pl.when(known).then(pl.col("lag")).otherwise(None)).to_series().to_numpy()
        )
    stack = np.column_stack(columns).astype(np.float64)
    valid = np.sum(~np.isnan(stack), axis=1) > 0
    heuristic = np.full((rows.height, len(common.LEVELS)), np.nan)
    heuristic[valid] = np.nanquantile(stack[valid], common.LEVELS, axis=1).T
    local = rows.select(
        month=pl.col("time").dt.convert_time_zone("Europe/London").dt.month(),
        half_hour=pl.col("time").dt.convert_time_zone("Europe/London").dt.hour() * 2
        + pl.col("time").dt.convert_time_zone("Europe/London").dt.minute() // 30,
        weekend=pl.col("time").dt.convert_time_zone("Europe/London").dt.weekday() >= 6,
    )
    climatology = (
        local.join(climatology_lookup(site=site), on=["month", "half_hour", "weekend"], how="left")
        .select(QCOLS)
        .to_numpy()
        .astype(np.float64)
    )
    pieces = []
    for arm, quantiles in (("floor_heuristic", heuristic), ("floor_climatology", climatology)):
        pieces.append(
            rows.select("time", "init_time", "lead_hours").with_columns(
                time_series_id=pl.lit(site),
                arm=pl.lit(arm),
                seed=pl.lit(0),
                band=pl.lit(band),
                actual=rows["output"].cast(pl.Float64),
                **{c: pl.Series(quantiles[:, i]) for i, c in enumerate(QCOLS)},
            )
        )
    return pl.concat(pieces)


def main() -> None:
    """Write one floors file per band."""
    common.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for band in common.BAND_ORDER:
        frame = pl.concat([floors_for(site=s, band=band) for s in common.FARMS])
        frame.write_parquet(common.DATA_DIR / f"floors_{band}.parquet")
        finite = frame.filter(pl.all_horizontal([pl.col(c).is_finite() for c in QCOLS]))
        print(band, frame.height, finite.height, flush=True)


if __name__ == "__main__":
    main()
