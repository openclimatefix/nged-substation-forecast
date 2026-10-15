"""Tier A: the proxy stack (experiment 00017's `H0.5` plus a correction) under every arm.

Daily origins at 09:00 UTC. Point forecast: absolute error and threshold-weighted absolute error at
each series' historical P99 (experiment 00033), in points of capacity. ERA5 arrives about 5 days
after the event, so the proxy's weather is not available at forecast time: the numbers are a ceiling
for the correction rule. Writes `tier_a_rows.parquet`.
"""

from datetime import UTC, datetime
from typing import Final

import numpy as np
import polars as pl
from e00038_base import (
    COVERAGE_END_STAMP,
    FITS_DIR,
    GRID_LENGTH,
    GRID_START,
    SERIES_IDS,
    SOURCE_DIR,
    THRESHOLDS_PATH,
    WORK_DIR,
)
from e00038_corrections import (
    NEWEST_AGE,
    correction_series,
    detected_mask,
    open_interval_start,
    static_mask,
)
from e00038_intervals import Interval

FIRST_ORIGIN: Final[int] = 14 * 48 + 18
"""2022-01-15 09:00 UTC, so a 14-day window is complete."""
BANDS: Final[dict[str, range]] = {"A": range(88), "B": range(144, 192)}
"""Valid-time offsets from the origin: 3.5 to 48 hours after the newest observation, and day 6-7."""
PAD: Final[int] = 4
"""Half-hours (2 hours) added either side of a logged interval when choosing out-of-event rows."""
QUIET_STAMPS: Final[int] = 16 * 48
ARMS: Final[dict[str, tuple[int, str]]] = {
    "plain14": (14, "none"),
    "plain7": (7, "none"),
    "d1_mask14": (14, "d1"),
    "d1_mask7": (7, "d1"),
    "d0w48_mask14": (14, "d0"),
    "oracle_conf14": (14, "oracle_conf"),
    "oracle_conf7": (7, "oracle_conf"),
    "oracle_logged14": (14, "oracle_logged"),
    "random14": (14, "random"),
    "random7": (7, "random"),
    "d1_persist14": (14, "d1_persist"),
    # Post hoc: the same arms with the revised interval builder (`d1fx`, `d0fx`).
    "d1fx_mask14": (14, "d1fx"),
    "d1fx_mask7": (7, "d1fx"),
    "d0fx_mask14": (14, "d0fx"),
    "d1fx_persist14": (14, "d1fx_persist"),
    "randomfx14": (14, "randomfx"),
    "randomfx7": (7, "randomfx"),
}


def frame_to_intervals(*, frame: pl.DataFrame) -> list[Interval]:
    """Rebuild `Interval` objects from the saved table."""
    return [Interval(**row) for row in frame.iter_rows(named=True)]


def dilate(*, mask: np.ndarray, by: int) -> np.ndarray:
    """Return `mask` widened by `by` stamps either side."""
    cumulative = np.concatenate([[0], np.cumsum(mask)])
    t = np.arange(len(mask))
    lo, hi = np.clip(t - by, 0, len(mask)), np.clip(t + by + 1, 0, len(mask))
    return cumulative[hi] - cumulative[lo] > 0


def logged_masks(*, oracle: pl.DataFrame, series: int, confirmed_only: bool) -> np.ndarray:
    """Return the stamps inside a logged closure of 2 hours to 60 days on `series`."""
    mask = np.zeros(GRID_LENGTH, dtype=bool)
    rows = oracle.filter((pl.col("series_a") == series) | (pl.col("series_b") == series))
    if confirmed_only:
        rows = rows.filter("confirmed")
    for first, last in rows.select("first_stamp", "last_stamp_exclusive").iter_rows():
        mask[max(first, 0) : min(last, GRID_LENGTH)] = True
    return mask


def shifted_interval_mask(*, intervals: list[Interval], series: int) -> np.ndarray:
    """Mask of the series' D1 intervals moved by a random whole number of days (+/-30 to 90)."""
    rng = np.random.default_rng(series)
    mask = np.zeros(GRID_LENGTH, dtype=bool)
    for iv in intervals:
        if iv.series == series:
            mask[iv.onset : iv.end] = True
    days = int(rng.integers(30, 91)) * int(rng.choice([-1, 1]))
    return np.roll(mask, days * 48)


def load_series() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return `(y, f)` per series on the grid, in fractions of capacity."""
    out = {}
    for sid in SERIES_IDS:
        part = pl.read_parquet(FITS_DIR / f"{sid}.parquet").select("valid_time", "target", "H0.5")
        stamps = (
            (part["valid_time"].dt.epoch("s") - int(GRID_START.timestamp())) // 1800
        ).to_numpy()
        keep = (stamps >= 0) & (stamps < GRID_LENGTH)
        y, f = np.full(GRID_LENGTH, np.nan), np.full(GRID_LENGTH, np.nan)
        y[stamps[keep]] = part["target"].to_numpy()[keep]
        f[stamps[keep]] = part["H0.5"].to_numpy()[keep]
        out[sid] = (y, f)
    return out


def main() -> None:
    """Compute every arm's corrections and score the rows."""
    oracle = pl.read_parquet(SOURCE_DIR / "oracle_real.parquet").filter("primary")
    thresholds = dict(
        pl.read_parquet(THRESHOLDS_PATH)
        .select(pl.col("time_series_id").cast(pl.Int64), "p99")
        .iter_rows()
    )
    d1 = frame_to_intervals(frame=pl.read_parquet(WORK_DIR / "intervals_d1.parquet"))
    d0 = frame_to_intervals(frame=pl.read_parquet(WORK_DIR / "intervals_d0.parquet"))
    d1fx = frame_to_intervals(frame=pl.read_parquet(WORK_DIR / "intervals_d1fx.parquet"))
    d0fx = frame_to_intervals(frame=pl.read_parquet(WORK_DIR / "intervals_d0fx.parquet"))
    d1_alarms = pl.read_parquet(WORK_DIR / "alarms_d1.parquet")
    origins = np.arange(FIRST_ORIGIN, COVERAGE_END_STAMP - 48, 48)
    newest = origins - NEWEST_AGE
    pieces = []
    for sid, (y, f) in load_series().items():
        residual = y - f
        out_of_event = ~dilate(
            mask=logged_masks(oracle=oracle, series=sid, confirmed_only=False), by=PAD
        )
        near_log = dilate(
            mask=logged_masks(oracle=oracle, series=sid, confirmed_only=False), by=QUIET_STAMPS
        )
        alarm_mask = np.zeros(GRID_LENGTH, dtype=bool)
        alarm_mask[d1_alarms.filter(pl.col("series") == sid)["stamp"].to_numpy()] = True
        quiet = out_of_event & ~near_log & ~dilate(mask=alarm_mask, by=QUIET_STAMPS)
        conf = logged_masks(oracle=oracle, series=sid, confirmed_only=True)
        masks = {
            "oracle_conf": static_mask(mask=conf),
            "oracle_logged": static_mask(
                mask=logged_masks(oracle=oracle, series=sid, confirmed_only=False)
            ),
            "random": static_mask(mask=shifted_interval_mask(intervals=d1, series=sid)),
            "d1": detected_mask(intervals=d1, series=sid, window=8),
            "d1_persist": detected_mask(intervals=d1, series=sid, window=8),
            "d0": detected_mask(intervals=d0, series=sid, window=48),
            "d1fx": detected_mask(intervals=d1fx, series=sid, window=8),
            "d1fx_persist": detected_mask(intervals=d1fx, series=sid, window=8),
            "d0fx": detected_mask(intervals=d0fx, series=sid, window=48),
            "randomfx": static_mask(mask=shifted_interval_mask(intervals=d1fx, series=sid)),
            "none": None,
        }
        persist = open_interval_start(intervals=d1, series=sid, window=8)
        persist_fx = open_interval_start(intervals=d1fx, series=sid, window=8)
        corrections = {
            arm: correction_series(
                residual=residual,
                newest=newest,
                window=days * 48,
                masked=masks[kind],
                open_start={"d1_persist": persist, "d1fx_persist": persist_fx}.get(kind),
            )
            for arm, (days, kind) in ARMS.items()
        }
        threshold = thresholds[sid]
        for band, offsets in BANDS.items():
            valid = origins[:, None] + np.array(offsets)[None, :]
            origin_index = np.repeat(np.arange(len(origins)), len(offsets))
            v = valid.ravel()
            keep = v < COVERAGE_END_STAMP
            v, origin_index = v[keep], origin_index[keep]
            actual = y[v]
            columns: dict[str, np.ndarray] = {}
            for arm, correction in corrections.items():
                forecast = f[v] + correction[origin_index]
                columns[f"ae_{arm}"] = (np.abs(actual - forecast) * 100).astype(np.float32)
                columns[f"tw_{arm}"] = (
                    np.abs(np.maximum(actual, threshold) - np.maximum(forecast, threshold)) * 100
                ).astype(np.float32)
            pieces.append(
                pl.DataFrame(
                    {
                        "series": np.full(len(v), sid),
                        "band": np.full(len(v), band),
                        "origin": origins[origin_index],
                        "v": v,
                        "out_of_event": out_of_event[v],
                        "quiet": quiet[v],
                        "in_confirmed": conf[v],
                        **columns,
                    }
                ).filter(np.isfinite(actual))
            )
        print(f"series {sid} done", flush=True)
    rows = pl.concat(pieces).with_columns(
        month=(
            pl.lit(datetime(2022, 1, 1, tzinfo=UTC)) + pl.duration(minutes=30 * pl.col("v"))
        ).dt.strftime("%Y-%m")
    )
    rows.write_parquet(WORK_DIR / "tier_a_rows.parquet")
    print(rows.shape)


if __name__ == "__main__":
    main()
