r"""Blocked out-of-sample residuals for P12 and P13 before their experiment-17 panel begins.

Experiment 17's fits start a series only after it has a year of observed half-hours, so P13 enters
the residual panel in February 2026 and P12 in March 2026, after the six real transfers of
experiment 101 that lie at those two series. This script predicts every half-hour from the series'
first reading up to the start of its existing panel with the model of experiment 17's arm H0.5
(XGBoost on ERA5 weather and the clock, weights halving every 0.5 years), fitted with the predicted
block held out: calendar months for the main residual, and half-months from the 16th for the offset
residual. Each fit uses the series' own rows outside the block and a 7-day purge on each side,
before 2026-07-01, and weights each row by its distance from the block. The fits use rows from after
the block, which a live service could not do.

    systemd-run --user --scope -p MemoryMax=4G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00132-*/e00132_fit.py
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import xgboost as xgb
from e00132_common import GRID_START, RESTORE_PAD, VARIANTS, WORK_DIR, fits_folder
from studies.sources import ML_EXPERIMENTS_DATA_DIR

FEATURES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00006-stage1a-demand-input-feature-sweep"
    / "inputs"
    / "features.parquet"
)
SERIES: Final[tuple[int, ...]] = (12, 13)
SCORE_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""No experiment scores or trains on a valid time on or after 2026-07-01."""
PURGE: Final[timedelta] = timedelta(days=7)
N_ROUNDS: Final[int] = 500
HALF_LIFE_YEARS: Final[float] = 0.5
CHRISTMAS_WINDOW_DAYS: Final[int] = 14
N_THREADS: Final[int] = 2
PARAMETERS: Final[dict[str, float | int | str]] = {
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "colsample_bynode": 1.0,
    "tree_method": "hist",
    "objective": "reg:absoluteerror",
    "seed": 0,
}
COLUMNS: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
    "solar_zenith",
    "solar_azimuth",
    "hour_local",
    "day_of_week_local",
    "covid_indicator",
    "days_to_christmas_window",
)
"""Experiment 17's input columns."""
PANEL_START: Final[dict[int, datetime]] = {
    12: datetime(2026, 3, 1, tzinfo=UTC),
    13: datetime(2026, 2, 1, tzinfo=UTC),
}
"""First instant of each series' stored panel (experiment 37), which this script leaves alone."""
OFFSET_PANEL_START: Final[dict[int, datetime]] = {
    12: datetime(2026, 3, 16, tzinfo=UTC),
    13: datetime(2026, 2, 16, tzinfo=UTC),
}
"""First instant of each series' stored offset panel (experiment 71): its first 16th origin."""
FITS: Final[dict[str, tuple[int, dict[int, datetime]]]] = {
    "main": (1, PANEL_START),
    "offset": (16, OFFSET_PANEL_START),
}
"""Each residual's block day of the month, and where its stored panel begins."""


def _christmas_window() -> pl.Expr:
    """Return days to the nearest 25 December, signed, null outside the 14-day window."""
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.date()
    year = local.dt.year()
    offsets = [(pl.date(year + shift, 12, 25) - local).dt.total_days() for shift in (-1, 0, 1)]
    nearest = pl.min_horizontal([o.abs() for o in offsets])
    signed = (
        pl.when(offsets[0].abs() == nearest)
        .then(offsets[0])
        .when(offsets[1].abs() == nearest)
        .then(offsets[1])
        .otherwise(offsets[2])
    )
    return pl.when(nearest <= CHRISTMAS_WINDOW_DAYS).then(signed).otherwise(None)


def series_frame(*, time_series_id: int) -> pl.DataFrame:
    """The series' half-hours before `SCORE_END`, with experiment 17's input columns.

    Half-hours whose `target` is null (flagged by cleaning, or never read) stay in, so that they
    can be predicted; `training_rows` drops them from the fits.
    """
    return (
        pl.scan_parquet(FEATURES_PATH)
        .filter(pl.col("time_series_id") == time_series_id, pl.col("valid_time") < SCORE_END)
        .with_columns(days_to_christmas_window=_christmas_window())
        .select("time_series_id", "valid_time", "target", *COLUMNS)
        .sort("valid_time")
        .collect()
    )


def block_starts(*, first: datetime, last: datetime, day: int) -> list[datetime]:
    """The instants on `day` of each month from `first`'s month up to, not including, `last`."""
    out, month = [], datetime(first.year, first.month, day, tzinfo=UTC)
    while month < last:
        out.append(month)
        month = datetime(month.year + month.month // 12, month.month % 12 + 1, day, tzinfo=UTC)
    return out


def transfer_windows() -> list[tuple[datetime, datetime]]:
    """The six labelled transfers at P12 and P13, padded by `RESTORE_PAD` on each side."""
    events = pl.read_parquet(
        WORK_DIR.parent
        / "00101-ro-nc-detector-on-real-hand-filtered-switching-events"
        / "event_table.parquet"
    )
    both = events.filter(pl.col("series") == "12,13", pl.col("hours") > 24).sort("onset")
    return [
        (
            GRID_START + timedelta(minutes=30 * onset) - RESTORE_PAD,
            GRID_START + timedelta(minutes=30 * end) + RESTORE_PAD,
        )
        for onset, end in both.select("onset", "end").iter_rows()
    ]


def training_rows(
    *, frame: pl.DataFrame, start: datetime, end: datetime, mask_transfers: bool = False
) -> pl.DataFrame:
    """The observed rows outside `[start - PURGE, end + PURGE)`, which never include the block.

    With `mask_transfers`, the padded windows of the six labelled transfers are also left out.
    """
    rows = frame.filter(
        pl.col("target").is_not_null(),
        (pl.col("valid_time") < start - PURGE) | (pl.col("valid_time") >= end + PURGE),
    )
    if mask_transfers:
        for window_start, window_end in transfer_windows():
            rows = rows.filter(
                ~pl.col("valid_time").is_between(window_start - PURGE, window_end + PURGE)
            )
    return rows


def weights(
    *, rows: pl.DataFrame, start: datetime, end: datetime, half_life: bool = True
) -> np.ndarray:
    """0.5 ** (years from the block / half-life): the distance is to the nearer edge of the block.

    Without `half_life` every row has weight 1.
    """
    if not half_life:
        return np.ones(rows.height)
    before = (start - rows["valid_time"]).dt.total_seconds().to_numpy()
    after = (rows["valid_time"] - end).dt.total_seconds().to_numpy()
    distance_years = np.maximum(np.maximum(before, after), 0.0) / (365.25 * 86400)
    return 0.5 ** (distance_years / HALF_LIFE_YEARS)


def predict_block(
    *, frame: pl.DataFrame, start: datetime, end: datetime, variant: str
) -> pl.DataFrame:
    """Predictions for `[start, end)` from a model fitted without the block and its purge."""
    rows = training_rows(frame=frame, start=start, end=end, mask_transfers=variant == "masked")
    valid = frame.filter(pl.col("valid_time") >= start, pl.col("valid_time") < end)
    matrix = xgb.DMatrix(rows.select([pl.col(c).cast(pl.Float32) for c in COLUMNS]).to_numpy())
    matrix.set_label(rows["target"].to_numpy())
    matrix.set_weight(weights(rows=rows, start=start, end=end, half_life=variant != "equal"))
    booster = xgb.train({**PARAMETERS, "nthread": N_THREADS}, matrix, num_boost_round=N_ROUNDS)
    valid_matrix = xgb.DMatrix(
        valid.select([pl.col(c).cast(pl.Float32) for c in COLUMNS]).to_numpy()
    )
    prediction = booster.predict(valid_matrix).astype(np.float32)
    in_block_or_purge = rows.filter(
        pl.col("valid_time") >= start - PURGE, pl.col("valid_time") < end + PURGE
    ).height
    return valid.select("time_series_id", "valid_time", "target").with_columns(
        H0_5=pl.Series(prediction),
        block_start=pl.lit(start),
        block_end=pl.lit(end),
        n_train=pl.lit(rows.height, dtype=pl.UInt32),
        train_rows_in_block_or_purge=pl.lit(in_block_or_purge, dtype=pl.UInt32),
    )


def fit_series(*, time_series_id: int, name: str, variant: str) -> str:
    """Predict every half-hour before the series' stored panel, one held-out block at a time."""
    day, panel_start = FITS[name]
    out_dir = WORK_DIR / fits_folder(name=name, variant=variant)
    path = out_dir / f"{time_series_id}.parquet"
    if path.exists():
        return f"{variant} {name} {path.name}: exists"
    frame = series_frame(time_series_id=time_series_id)
    first = frame.filter(pl.col("target").is_not_null())["valid_time"].min()
    if not isinstance(first, datetime):
        raise TypeError(f"series {time_series_id} has no observed half-hours")
    starts = block_starts(first=first, last=panel_start[time_series_id], day=day)
    starts[0] = first
    ends = [*starts[1:], panel_start[time_series_id]]
    pieces = [
        predict_block(frame=frame, start=start, end=end, variant=variant)
        for start, end in zip(starts, ends, strict=True)
    ]
    out = pl.concat(pieces)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path.with_suffix(".tmp"))
    path.with_suffix(".tmp").rename(path)
    leaked = int(out["train_rows_in_block_or_purge"].sum())
    return (
        f"{variant} {name} {path.name}: {out.height:,} rows, {len(pieces)} blocks, "
        f"{leaked} leaked rows"
    )


def main() -> None:
    """Fit both series' main and offset blocks."""
    for variant in VARIANTS:
        for name in FITS:
            for series in SERIES:
                print(fit_series(time_series_id=series, name=name, variant=variant), flush=True)


if __name__ == "__main__":
    main()
