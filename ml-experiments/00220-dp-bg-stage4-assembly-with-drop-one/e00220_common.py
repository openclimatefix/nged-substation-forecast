"""Constants and helpers shared by the scripts of experiment 220 (stage 4 assembly).

Every input is a stored file of an earlier experiment, read and never written. This experiment
writes only under its own `WORK_DIR`.
"""

import json
from pathlib import Path
from typing import Final

import numpy as np
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 220
SLUG: Final[str] = "dp-bg-stage4-assembly-with-drop-one"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
ARM_DIR: Final[Path] = WORK_DIR / "arms"
REPORT_DIR: Final[Path] = WORK_DIR / "report"

E155_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00155-dp-bg-stage2-assembly-night-offset-and-error-correction"
    / "work" / "dp_preds"
)  # fmt: skip
E169_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00169-calibration-combined-fixes-and-ensemble-spread"
    / "work" / "dp_preds"
)  # fmt: skip
E201_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00201-dp-gam-residual-mlp" / "work" / "preds"
E202_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00202-dp-ens-two-number-rolling-replay" / "work" / "arms"
)
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)

DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
SCORED_SITES: Final[tuple[str, ...]] = (*DECISION_SITES, "D9")
N_SEEDS: Final[int] = 3
QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
HORIZON_HOURS: Final[float] = 168.0
KEYS: Final[list[str]] = ["site", "run", "valid_time"]
COLD_DAY_C: Final[float] = 2.0
WINTER_MONTHS: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
DAY_AHEAD_LEADS: Final[tuple[float, float]] = (15.5, 39.0)
"""The leads on which experiment 201 scored the network (hours, inclusive)."""

ARMS: Final[tuple[str, ...]] = (
    "a", "a_s", "a_h", "a_hs", "h_own",
    "drop_widen", "drop_r", "drop_offset",
    "a_s_drop_r", "a_s_drop_offset",
)  # fmt: skip
ARM_LABELS: Final[dict[str, str]] = {
    "a": "A: arm D with experiment 169's bands (reference)",
    "a_s": "A+S: A with the two-number solar summary",
    "a_h": "A+H: A with the network's median correction",
    "a_hs": "A+H+S: both",
    "h_own": "H-own: the network's own 11 quantiles",
    "drop_widen": "A+H+S without experiment 164's cold-day widening",
    "drop_r": "A+H+S without recipe R",
    "drop_offset": "A+H+S without the winter night offset",
    "a_s_drop_r": "A+S without recipe R (no network)",
    "a_s_drop_offset": "A+S without the winter night offset (no network)",
}


def load_cache() -> FeatureCache:
    """Return the feature cache that experiments 106 and 155 built."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)


def write_npy_atomic(*, path: Path, array: np.ndarray) -> None:
    """Write a NumPy array through a temporary name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.npy")
    np.save(temporary, array)
    temporary.rename(path)
