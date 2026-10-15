"""Tier B: experiment 00019's real stack (`n1h`, three seeds) with the correction under every arm.

Runs every 5 days from July 2025 to 2026-03-25. The correction uses 00019's rule and its spread of
1.15 around the median. Writes `tier_b_rows.parquet` (nine-level quantile score and the
threshold-weighted score at P99, in points of capacity, per case and arm).
"""

from datetime import UTC, datetime
from typing import Final

import numpy as np
import polars as pl
from e00038_base import (
    COVERAGE_END_STAMP,
    GRID_LENGTH,
    GRID_START,
    LEVEL_COLUMNS,
    SOURCE_DIR,
    THRESHOLDS_PATH,
    WORK_DIR,
    quantile_score,
    threshold_weighted_score,
)
from e00038_corrections import (
    NEWEST_AGE,
    correction_series,
    detected_mask,
    open_interval_start,
    static_mask,
)
from e00038_proxy import (
    frame_to_intervals,
    load_series,
    logged_masks,
    shifted_interval_mask,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ARMS_DIR: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00019-primary-demand-stack-vs-manual-heuristic" / "work" / "arms"
)
SPREAD: Final[float] = 1.15
ARMS: Final[dict[str, tuple[int, str]]] = {
    "plain14": (14, "none"),
    "plain7": (7, "none"),
    "d1_mask14": (14, "d1"),
    "d1_mask7": (7, "d1"),
    "oracle_conf14": (14, "oracle_conf"),
    "oracle_logged14": (14, "oracle_logged"),
    "random14": (14, "random"),
    "d1_persist14": (14, "d1_persist"),
    "d1fx_mask14": (14, "d1fx"),
    "d1fx_mask7": (7, "d1fx"),
    "randomfx14": (14, "randomfx"),
    "d1fx_persist14": (14, "d1fx_persist"),
}


def stamps(*, times: pl.Series) -> np.ndarray:
    """Half-hour stamps of a datetime series."""
    return ((times.dt.epoch("s") - int(GRID_START.timestamp())) // 1800).to_numpy()


def main() -> None:
    """Score every case under every arm."""
    oracle = pl.read_parquet(SOURCE_DIR / "oracle_real.parquet").filter("primary")
    thresholds = dict(
        pl.read_parquet(THRESHOLDS_PATH)
        .select(pl.col("time_series_id").cast(pl.Int64), "p99")
        .iter_rows()
    )
    d1 = frame_to_intervals(frame=pl.read_parquet(WORK_DIR / "intervals_d1.parquet"))
    d1fx = frame_to_intervals(frame=pl.read_parquet(WORK_DIR / "intervals_d1fx.parquet"))
    power = {sid: y for sid, (y, _) in load_series().items()}
    arms_data = pl.concat(
        [pl.read_parquet(ARMS_DIR / f"n1h_seed{s}.parquet") for s in (0, 1, 2)]
    ).filter(pl.col("valid_time") < datetime(2026, 3, 26, tzinfo=UTC))
    pieces = []
    for (sid, seed), part in arms_data.group_by("time_series_id", "seed"):
        sid, seed = int(sid), int(seed)  # noqa: PLW2901
        if sid not in power:
            continue
        pw = power[sid]
        runs = part["power_fcst_init_time"].unique().sort()
        origin = stamps(times=runs)
        median = np.full(GRID_LENGTH, np.nan)
        for p, run in enumerate(runs):
            q = part.filter(pl.col("power_fcst_init_time") == run)
            i = stamps(times=q["valid_time"])
            stop = origin[p + 1] if p + 1 < len(origin) else 10**9
            keep = (i >= origin[p]) & (i < stop) & (i >= 0) & (i < GRID_LENGTH)
            median[i[keep]] = q["q50"].to_numpy()[keep]
        residual = pw - median
        masks = {
            "oracle_conf": static_mask(
                mask=logged_masks(oracle=oracle, series=sid, confirmed_only=True)
            ),
            "oracle_logged": static_mask(
                mask=logged_masks(oracle=oracle, series=sid, confirmed_only=False)
            ),
            "random": static_mask(mask=shifted_interval_mask(intervals=d1, series=sid)),
            "d1": detected_mask(intervals=d1, series=sid, window=8),
            "d1_persist": detected_mask(intervals=d1, series=sid, window=8),
            "d1fx": detected_mask(intervals=d1fx, series=sid, window=8),
            "d1fx_persist": detected_mask(intervals=d1fx, series=sid, window=8),
            "randomfx": static_mask(mask=shifted_interval_mask(intervals=d1fx, series=sid)),
            "none": None,
        }
        persist = {
            "d1_persist": open_interval_start(intervals=d1, series=sid, window=8),
            "d1fx_persist": open_interval_start(intervals=d1fx, series=sid, window=8),
        }
        conf = logged_masks(oracle=oracle, series=sid, confirmed_only=True)
        logged = logged_masks(oracle=oracle, series=sid, confirmed_only=False)
        valid_stamp = stamps(times=part["valid_time"])
        init_stamp = stamps(times=part["power_fcst_init_time"])
        in_grid = (valid_stamp >= 0) & (valid_stamp < COVERAGE_END_STAMP)
        part = part.filter(pl.Series(in_grid))  # noqa: PLW2901
        valid_stamp, init_stamp = valid_stamp[in_grid], init_stamp[in_grid]
        actual = pw[valid_stamp]
        quantiles = part.select(LEVEL_COLUMNS).to_numpy().astype(float)
        columns: dict[str, np.ndarray] = {}
        for arm, (days, kind) in ARMS.items():
            per_run = correction_series(
                residual=residual,
                newest=origin - NEWEST_AGE,
                window=days * 48,
                masked=masks[kind],
                open_start=persist.get(kind),
            )
            shift = dict(zip(origin.tolist(), per_run.tolist(), strict=True))
            c = np.array([shift[int(o)] for o in init_stamp])
            q = quantiles + c[:, None]
            mid = q[:, [4]]
            q = np.sort(mid + SPREAD * (q - mid), axis=1)
            columns[f"qs_{arm}"] = (quantile_score(quantiles=q, actual=actual) * 100).astype(
                np.float32
            )
            columns[f"tw_{arm}"] = (
                threshold_weighted_score(
                    quantiles=q, actual=actual, threshold=np.full(len(actual), thresholds[sid])
                )
                * 100
            ).astype(np.float32)
        pieces.append(
            pl.DataFrame(
                {
                    "series": np.full(len(actual), sid),
                    "seed": np.full(len(actual), seed),
                    "v": valid_stamp,
                    "out_of_event": ~_dilate(logged, 4)[valid_stamp],
                    "in_confirmed": conf[valid_stamp],
                    **columns,
                }
            ).filter(np.isfinite(actual))
        )
    rows = pl.concat(pieces).with_columns(
        month=(
            pl.lit(datetime(2022, 1, 1, tzinfo=UTC)) + pl.duration(minutes=30 * pl.col("v"))
        ).dt.strftime("%Y-%m")
    )
    rows.write_parquet(WORK_DIR / "tier_b_rows.parquet")
    print(rows.shape)


def _dilate(mask: np.ndarray, by: int) -> np.ndarray:
    cumulative = np.concatenate([[0], np.cumsum(mask)])
    t = np.arange(len(mask))
    return (
        cumulative[np.clip(t + by + 1, 0, len(mask))] - cumulative[np.clip(t - by, 0, len(mask))]
        > 0
    )


if __name__ == "__main__":
    main()
