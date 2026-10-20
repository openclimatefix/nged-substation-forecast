"""Constants and helpers shared by the scripts of experiment 113 (DP N5, convex identification).

The experiment asks whether the convex per-site model of `studies.dp_gam` identifies each hour
band's heating slope and the latest season-year's PV capacity at realistic noise, and whether its
80% month-block bootstrap intervals cover the truth. Every fit is a column subset of one design
per substation and window, so the scripts build one Gram matrix per calendar month and combine the
months with the bootstrap's month counts.
"""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

import torch
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 113
SLUG: Final[str] = "differentiable-physics-n5-convex-identification-at-realistic-noise"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
"""Small result files (cells, controls, reports); private and never committed."""
CELLS_DIR: Final[Path] = WORK_DIR / "cells"
"""One parquet file per (site, window, cell), written atomically."""
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
"""The feature cache that experiment 106 built; the same inputs hash to the same file."""

DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
"""The five decision substations (D9 holds 9 months at the replicate cut)."""
REPLICATE_CUT: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
"""The synthetic cut: it falls on a season-year boundary, so the latest season-year is complete."""
SCORED_CUT: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)

RIDGE: Final[float] = 0.1
"""Experiment 104's tuned ridge penalty."""
DRIFT_STEP_PART_2: Final[float] = 3000.0
"""Experiment 104's tuned drift step."""
DRIFT_STEP_PART_4: Final[float] = 30.0
"""Experiment 106's tuned drift step."""

N_DRAWS_PER_SITE: Final[int] = 20
N_BOOTSTRAP: Final[int] = 100
"""Resamples per interval method; the weighting set holds the full sample, I1, and I1u."""
INTERVAL_Z: Final[float] = 1.2815515655446004
"""The 90th percentile of the standard normal, for an 80% two-sided interval."""
HEATING_SPLIT: Final[tuple[float, ...]] = (0.2, 0.1, 0.3, 0.4)
"""Each band's total heating slope split over the 6, 24, 72, and 168 hour smoothings."""
MIN_BAND_SLOPE: Final[float] = 2.0
"""The floor of a truth band slope, in points per 10 degrees C."""
CAPACITY_FIRST_RANGE: Final[tuple[float, float]] = (5.0, 25.0)
CAPACITY_INCREMENT_RANGE: Final[tuple[float, float]] = (1.0, 5.0)
IID_SD: Final[float] = 5.0
CAPACITY_MONTHS: Final[int] = 6
"""The summer months on which the latest season-year's capacity mostly rests."""


def configure_threads(*, threads: int = 1) -> None:
    """Limit the torch and BLAS threads of this process."""
    torch.set_num_threads(threads)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(threads)


def load_cache() -> FeatureCache:
    """Return the feature cache of experiments 104 and 106."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def as_float(value: object) -> float:
    """Return a polars scalar (typed as a wide union) as a float."""
    return float(cast("float", value))
