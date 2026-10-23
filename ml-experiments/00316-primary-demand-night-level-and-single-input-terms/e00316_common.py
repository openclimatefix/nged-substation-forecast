# ruff: noqa: E501
"""Paths, levels, and scoring helpers shared by the scripts of experiment 316."""

import json
from pathlib import Path
from typing import Final

import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 316
SLUG: Final[str] = "primary-demand-night-level-and-single-input-terms"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
REPORT_DIR: Final[Path] = WORK_DIR / "report"

E251_SCORED: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2"
    / "work" / "arms" / "scored"
)  # fmt: skip
E314_SCORED: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00314-primary-demand-gated-or-pinball-band-head"
    / "work" / "arms" / "scored"
)  # fmt: skip
E266_MASTER: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00266-summer-gap-diagnostics-solar-term" / "work" / "master.parquet"
)
E305_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00305-summer-gap-what-bg2-knows-that-arm-d-does-not" / "work"
)

QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
MEDIAN_INDEX: Final[int] = 5
N_RESAMPLES: Final[int] = 20_000


def qs_per_row(*, actual: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss over the 11 quantiles, per row."""
    error = actual.astype(np.float64)[:, None] - q.astype(np.float64)
    pin = np.maximum(LEVELS[None, :] * error, (LEVELS[None, :] - 1.0) * error)
    return 2.0 * pin.mean(axis=1)


def ratio_interval(
    *, num: np.ndarray, den: np.ndarray, blocks: np.ndarray, seed: int = 0, level: float = 0.95
) -> dict:
    """Return sum(num) / sum(den) with a block-bootstrap interval that resamples the block labels."""
    labels, codes = np.unique(blocks, return_inverse=True)
    n = len(labels)
    nums = np.bincount(codes, weights=num, minlength=n)
    dens = np.bincount(codes, weights=den, minlength=n)
    draws = np.random.default_rng(seed).integers(0, n, size=(N_RESAMPLES, n))
    res = nums[draws].sum(axis=1) / dens[draws].sum(axis=1)
    tail = 100.0 * (1.0 - level) / 2.0
    lo, hi = np.percentile(res, [tail, 100.0 - tail])
    return {"est": float(nums.sum() / dens.sum()), "lo": float(lo), "hi": float(hi), "blocks": int(n)}  # fmt: skip


def mean_interval(*, values: np.ndarray, blocks: np.ndarray, seed: int = 0) -> dict:
    """Return the mean of `values` with a 95% block-bootstrap interval."""
    return ratio_interval(num=values, den=np.ones_like(values), blocks=blocks, seed=seed)


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)
