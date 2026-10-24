# ruff: noqa: E501
"""Constants and helpers shared by the scripts of experiment 317 (default 1 at 7 to 10 days).

Every input is a stored file of an earlier experiment, read and never written. This experiment
writes only under its own `WORK_DIR`.
"""

import json
from pathlib import Path
from typing import Final

import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 317
SLUG: Final[str] = "primary-demand-default-1-at-7-to-10-days"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
CHAIN_DIR: Final[Path] = WORK_DIR / "chain"
REPORT_DIR: Final[Path] = WORK_DIR / "report"

E87_PREDICTIONS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00087-bg2-best-guess-assembled-with-drop-one-ablations" / "predictions"
)  # fmt: skip
E136_PREDS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00136-dp-bg-stage1-assembly-on-the-ensemble-mean" / "work" / "dp_preds"
)  # fmt: skip
E155_PREDS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00155-dp-bg-stage2-assembly-night-offset-and-error-correction" / "work" / "dp_preds"
)  # fmt: skip
E160_WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00160-calibration-band-width-fix-on-stage-2-winner" / "work"
)  # fmt: skip
E164_WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00164-calibration-cold-night-tail-under-coverage" / "work"
)  # fmt: skip
E169_PREDS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00169-calibration-combined-fixes-and-ensemble-spread" / "work" / "dp_preds"
)  # fmt: skip
E178_ROWS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00178-dp-sunny-daylight-gap-diagnosis" / "work" / "report" / "joined_rows.parquet"
)  # fmt: skip
E220_ARMS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00220-dp-bg-stage4-assembly-with-drop-one" / "work" / "arms"
)  # fmt: skip
E228_SCORED: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00228-dp-bg-rolling-origin-stack-against-bg2" / "work" / "arms" / "scored"
)  # fmt: skip
E251_ARMS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2" / "work" / "arms" / "scored"
)  # fmt: skip

DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
SCORED_SITES: Final[tuple[str, ...]] = (*DECISION_SITES, "D9")
PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]

HORIZON_HOURS: Final[float] = 240.0
HORIZON_HALF_HOURS: Final[int] = 480
STORED_HORIZON_HOURS: Final[float] = 168.0

LEAD_BAND_UPPER_HOURS: Final[tuple[float, ...]] = (24.0, 48.0, 144.0, 168.0, 192.0, 216.0, 240.0)
"""Upper edges of the spread table's lead bands: the four stored bands, then one per day."""
LEAD_BAND_CENTRES_HOURS: Final[tuple[float, ...]] = (
    12.0,
    36.0,
    96.0,
    156.0,
    168.0,
    180.0,
    204.0,
    228.0,
)
"""Interpolation knots. The knot at 168 hours holds the fourth band's table, so below 168 hours the
interpolation equals experiment 160's arm I."""
KNOT_TABLE_INDEX: Final[tuple[int, ...]] = (0, 1, 2, 3, 3, 4, 5, 6)
N_LEAD_BANDS: Final[int] = 7
COLD_GATE_C: Final[float] = 2.0
UPPER_FIX_LEVELS: Final[tuple[float, ...]] = (0.75, 0.9, 0.95, 0.99)
WINDOW_HALF_HOURS: Final[int] = 24


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)
