"""Paths and the residual panel shared by the scripts of experiment 00037."""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00037_grid import GRID_LENGTH, GRID_START
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00037-switching-first-look"
"""Private tables and figures; never committed."""
FITS_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00017-training-window-and-retraining-rule-drift-demand" / "fits"
)
FEATURES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00006-stage1a-demand-input-feature-sweep"
    / "inputs"
    / "features.parquet"
)
INTERCONNECTIONS_PATH: Final[Path] = (
    Path("/mnt/data/nged-substation-forecast/data/NGED/substation_hierarchy_2026-10-15")
    / "Primary Substation Interconnections.csv"
)
SERIES_IDS: Final[tuple[int, ...]] = (3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 25, 26)
MVA_SERIES: Final[frozenset[int]] = frozenset({3, 9, 11, 14, 15, 24, 25, 26})
"""Non-directional apparent-power series; conservation holds only approximately for them."""
TUNING_MONTHS: Final[frozenset[int]] = frozenset({1, 3, 5, 7, 9, 11})
"""Odd calendar months set thresholds; even months validate."""
H3_RESIDUAL_ARM: Final[str] = "H0.5"
# Static topology between the 16 Primaries, from the interconnection list mapped to series
# (substation 900019 is series 12 and 13, two transformers of one Primary, adjacent to each other).
ADJACENCY: Final[dict[int, tuple[int, ...]]] = {
    3: (15, 16, 26),
    7: (24,),
    8: (10, 14, 24),
    9: (11, 24),
    10: (8, 12, 13, 14, 24, 25),
    11: (9, 25),
    12: (10, 13, 14, 16, 25),
    13: (10, 12, 14, 16, 25),
    14: (8, 10, 12, 13, 15, 16),
    15: (3, 14, 17, 18),
    16: (3, 12, 13, 14, 26),
    17: (15, 18, 26),
    18: (15, 17),
    24: (7, 8, 9, 10, 25),
    25: (10, 11, 12, 13, 24),
    26: (3, 16, 17),
}


def build_panel() -> pl.DataFrame:
    """Return `series, stamp, valid_time, obs_mw, expected_mw, res_mw` for every series.

    `res_mw` is `(target - H0.5)` times the series' effective capacity; `expected_mw` is `H0.5`
    times capacity. Every `H0.5` prediction is out of sample (experiment 00017).
    """
    capacity = (
        pl.scan_parquet(FEATURES_PATH)
        .group_by("time_series_id")
        .agg(pl.col("effective_capacity_mw").median())
        .collect()
    )
    frames = [pl.read_parquet(FITS_DIR / f"{sid}.parquet") for sid in SERIES_IDS]
    return (
        pl.concat(frames)
        .join(capacity, on="time_series_id")
        .select(
            series=pl.col("time_series_id").cast(pl.Int64),
            valid_time=pl.col("valid_time"),
            obs_mw=(pl.col("target") * pl.col("effective_capacity_mw")).cast(pl.Float64),
            expected_mw=(pl.col(H3_RESIDUAL_ARM) * pl.col("effective_capacity_mw")).cast(
                pl.Float64
            ),
        )
        .with_columns(res_mw=pl.col("obs_mw") - pl.col("expected_mw"))
        .with_columns(
            stamp=((pl.col("valid_time").dt.epoch("s") - int(GRID_START.timestamp())) // 1800)
        )
        .filter(pl.col("stamp").is_between(0, GRID_LENGTH - 1))
        .sort("series", "stamp")
    )


def load_arrays(*, panel: pl.DataFrame, column: str) -> dict[int, np.ndarray]:
    """Return one float array of length `GRID_LENGTH` per series, NaN where missing."""
    out: dict[int, np.ndarray] = {}
    for (series,), group in panel.group_by(["series"]):
        array = np.full(GRID_LENGTH, np.nan)
        array[group["stamp"].to_numpy()] = group[column].to_numpy()
        out[int(series)] = array
    return out
