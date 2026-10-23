"""Constants and helpers shared by the scripts of experiment 266 (April to June gap diagnostics)."""

import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 266
SLUG: Final[str] = "summer-gap-diagnostics-solar-term"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
E251_SCORED: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2"
    / "work" / "arms" / "scored"
)  # fmt: skip
E178_ROWS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00178-dp-sunny-daylight-gap-diagnosis" / "work" / "report" / "joined_rows.parquet"
)  # fmt: skip
E111_PREDS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00111-differentiable-physics-6-ens-forecast-engine-against-bg2" / "work" / "dp_preds"
)  # fmt: skip
E139_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00139-differentiable-physics-pv-as-of-capacity-in-gam-k" / "work"
)  # fmt: skip
SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
STACK_ARMS: Final[tuple[str, ...]] = ("arm1", "a_s", "dh", "d", "bg2")
N_RESAMPLES: Final[int] = 20_000
AMJ: Final[tuple[str, ...]] = ("2026-04", "2026-05", "2026-06")
JAS: Final[tuple[str, ...]] = ("2025-07", "2025-08", "2025-09")
NDJFM: Final[tuple[str, ...]] = ("2025-11", "2025-12", "2026-01", "2026-02", "2026-03")


def qs_per_row(*, actual: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss over the 11 quantiles, per row."""
    error = actual.astype(np.float64)[:, None] - q.astype(np.float64)
    pin = np.maximum(LEVELS[None, :] * error, (LEVELS[None, :] - 1.0) * error)
    return 2.0 * pin.mean(axis=1)


def mean_interval(*, values: np.ndarray, mask: np.ndarray, blocks: np.ndarray, seed: int = 0) -> dict:
    """Mean of `values` over `mask`, with a 95% interval that resamples the `blocks` labels."""
    rng = np.random.default_rng(seed)
    keep = mask & np.isfinite(values)
    labels, codes = np.unique(blocks[keep], return_inverse=True)
    n = len(labels)
    if n == 0:
        return {"est": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    sums = np.bincount(codes, weights=values[keep], minlength=n)
    counts = np.bincount(codes, minlength=n).astype(np.float64)
    draws = rng.integers(0, n, size=(N_RESAMPLES, n))
    res = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.percentile(res, [2.5, 97.5])
    return {"est": float(sums.sum() / counts.sum()), "lo": float(lo), "hi": float(hi), "n": int(keep.sum())}


def write_json_atomic(*, path: Path, content: object) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)


def read_rows() -> pl.DataFrame:
    """Return the built master frame of experiment 266."""
    return pl.read_parquet(WORK_DIR / "master.parquet")
