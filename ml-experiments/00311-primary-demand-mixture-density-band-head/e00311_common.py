"""Constants and helpers shared by the scripts of experiment 311 (mixture-density band head).

The feature builder of experiment 278 (`e00278_h_data`) supplies the rows and the input matrix, so
run every script with that folder on `PYTHONPATH`:

    PYTHONPATH=ml-experiments/00278-stack-retraining-cadence uv run --no-sync python <script>
"""

import json
from pathlib import Path
from typing import Final

import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 311
SLUG: Final[str] = "primary-demand-mixture-density-band-head"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
PREP_DIR: Final[Path] = WORK_DIR / "prep"
FIT_DIR: Final[Path] = WORK_DIR / "fits"
REPORT_DIR: Final[Path] = WORK_DIR / "report"

E251_SCORED: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2"
    / "work" / "arms" / "scored"
)  # fmt: skip
E169_PREDS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00169-calibration-combined-fixes-and-ensemble-spread"
    / "work" / "dp_preds"
)  # fmt: skip

SCORED_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16", "D9")
QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
MEDIAN_INDEX: Final[int] = 5
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
INPUT_ARM: Final[int] = 2
"""Experiment 278's input arm 2: arm D's mean, level, anomaly, width, lead, weather, calendar."""
POOL_MONTHS: Final[int] = 24
MAX_LEAD_HOURS: Final[int] = 168
VAL_MONTH_STRIDE: Final[int] = 4

SCORED_CUTS: Final[tuple[str, ...]] = tuple(
    f"{y}-{m:02d}" for y, m in [(2025, k) for k in range(7, 13)] + [(2026, k) for k in range(1, 7)]
)
REPLICATE_CUTS: Final[tuple[str, ...]] = (
    "2024-08", "2024-10", "2024-12", "2025-02", "2025-04", "2025-06",
)  # fmt: skip

WIDENING_COLD_C: Final[float] = 2.0
WIDENING_FACTORS: Final[dict[str, float]] = {
    "q75": 1.653, "q90": 1.363, "q95": 1.261, "q99": 1.162,
}  # fmt: skip
"""The default's cold-day widening of the upper offsets, read from experiment 169's stored arms."""


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)
