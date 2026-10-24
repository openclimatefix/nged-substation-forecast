# ruff: noqa: E501
r"""Paths and constants shared by the scripts of experiment 334 (other neural architectures).

Run every script with the folders of experiments 164, 278, 311, 317, 320, and 322 on `PYTHONPATH`:

    export PYTHONPATH=$(ls -d ml-experiments/00{164,278,311,317,319,320,322,334}-* | tr '\\n' ':')
    uv run --no-sync python <script>
"""

import json
from pathlib import Path
from typing import Final

import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 334
SLUG: Final[str] = "primary-demand-other-neural-architectures"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
PREP_DIR: Final[Path] = WORK_DIR / "prep"
FIT_DIR: Final[Path] = WORK_DIR / "fits"
REPORT_DIR: Final[Path] = WORK_DIR / "report"
E201_FEATURES: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00201-dp-gam-residual-mlp" / "work" / "features"
)
E311_PREP: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00311-primary-demand-mixture-density-band-head" / "work" / "prep"
)
E111_STORE: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00111-differentiable-physics-6-ens-forecast-engine-against-bg2"
    / "work"
    / "ens_store"
)

SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16", "D9")
STEPS: Final[int] = 480
"""Half-hour lead steps per run: 0.5 to 240 hours."""
SHORT_STEPS: Final[int] = 336
"""Steps to 168 hours, which the earlier experiments stored."""
HISTORY_STEPS: Final[int] = 336
"""Half-hours of power history the recurrent and transformer encoders read (7 days)."""
LATENCY_STEPS: Final[int] = 6
"""The newest power reading available at the origin is 3 hours old."""
STANDARDISE_READINGS: Final[int] = 4000
"""Per-site mean and spread of power come from the site's first 4000 readings (about 83 days), which precede every scored month."""

ARMS: Final[tuple[str, ...]] = ("m1", "m2", "r", "t")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 5)
REPLICATE_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
POOL_MONTHS: Final[int] = 24
VAL_MONTH_STRIDE: Final[int] = 4
SCORED_CUTS: Final[tuple[str, ...]] = tuple(
    f"{y}-{m:02d}" for y, m in [(2025, k) for k in range(7, 13)] + [(2026, k) for k in range(1, 7)]
)
REPLICATE_CUTS: Final[tuple[str, ...]] = (
    "2024-08",
    "2024-10",
    "2024-12",
    "2025-02",
    "2025-04",
    "2025-06",
)

QUANTILES: Final[tuple[str, ...]] = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
MEDIAN_INDEX: Final[int] = 5

FEATURES: Final[tuple[str, ...]] = (
    "lead_h",
    "temp",
    "cold_hinge",
    "warm_hinge",
    "temp_prev",
    "sw",
    "solar_term",
    "dark",
    "elevation",
    "hh_s1",
    "hh_c1",
    "hh_s2",
    "hh_c2",
    "dt0",
    "dt1",
    "dt2",
    "dt3",
    "dt4",
    "dow_s",
    "dow_c",
    "doy_s1",
    "doy_c1",
    "doy_s2",
    "doy_c2",
    "bank_holiday",
    "christmas",
    "e7",
    "lag_power",
    "lag_missing",
    "lag_hours",
    "last_reading",
    "mean24",
    "recent_missing",
    "unit_pv",
)
"""Columns of the per-step input tensor `x`; `unit_pv` is the derated unit PV shape per MW AC."""
COL: Final[dict[str, int]] = {name: i for i, name in enumerate(FEATURES)}
M1_COLUMNS: Final[tuple[int, ...]] = tuple(COL[n] for n in FEATURES if n != "unit_pv")
"""M1, R, and T read every served input except the explicit solar signal."""
M2_DROPPED: Final[tuple[str, ...]] = ("sw", "solar_term", "dark", "elevation", "unit_pv")
M2_COLUMNS: Final[tuple[int, ...]] = tuple(COL[n] for n in FEATURES if n not in M2_DROPPED)
"""M2 drops the weather inputs but for temperature; the explicit solar signal is added at fit time."""


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)
