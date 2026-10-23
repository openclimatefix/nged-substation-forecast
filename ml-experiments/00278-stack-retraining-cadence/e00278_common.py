# ruff: noqa: E501
"""Paths, constants, and scoring helpers shared by the scripts of experiment 278.

The quantile score is twice the mean pinball loss over the 11 quantiles, in points of effective
capacity. The tail-weighted score raises the quantiles and the observation to the series' historical
95th (or 99th) percentile of power before 2025-07-02 first, as experiment 251 does. Intervals
resample the months of the forecast run's initialisation time.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00278-stack-retraining-cadence"
WORK_DIR: Final[Path] = DATA_DIR / "work"
PREDS_DIR: Final[Path] = WORK_DIR / "dp_preds"
REPORT_DIR: Final[Path] = WORK_DIR / "report"
E220_ARMS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00220-dp-bg-stage4-assembly-with-drop-one" / "work" / "arms"
)
E155_PREDS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00155-dp-bg-stage2-assembly-night-offset-and-error-correction"
    / "work"
    / "dp_preds"
)
E169_PREDS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00169-calibration-combined-fixes-and-ensemble-spread"
    / "work"
    / "dp_preds"
)
E87_PREDICTIONS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "predictions"
)
SCORED_START: Final[datetime] = datetime(2025, 7, 2, tzinfo=UTC)
SCORED_SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
KEYS: Final[list[str]] = ["site", "run", "valid_time"]
TAILS: Final[tuple[tuple[str, int], ...]] = (("q01", 0), ("q05", 1), ("q95", 9), ("q99", 10))
N_RESAMPLES: Final[int] = 20_000
WINTER_MONTHS: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
SUMMER_MONTHS: Final[frozenset[int]] = frozenset({4, 5, 6, 7, 8, 9})
BONFERRONI_LEVEL: Final[float] = 99.0
"""Five planned contrasts at the 5% level: 1 - 0.05 / 5 = 99% two-sided."""


def thresholds() -> dict[str, dict[str, float]]:
    """Return each site's historical 95th and 99th percentile of power, in points."""
    pieces = [
        pl.read_parquet(
            E169_PREDS / f"ref_{period}_{site}.parquet", columns=["actual", "valid_time"]
        ).with_columns(site=pl.lit(site))
        for site in SCORED_SITES
        for period in ("tuning", "replicate")
    ]
    frame = (
        pl.concat(pieces)
        .filter(pl.col("valid_time") < SCORED_START)
        .unique(subset=["site", "valid_time"])
    )
    out: dict[str, dict[str, float]] = {}
    for (site,), group in frame.group_by("site"):
        values = group["actual"].to_numpy().astype(np.float64)
        out[str(site)] = {
            "95": float(np.quantile(values, 0.95)),
            "99": float(np.quantile(values, 0.99)),
        }
    return out


def pinball(*, actual: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss over the 11 quantiles, per row."""
    error = actual[:, None] - q
    pin = np.maximum(LEVELS[None, :] * error, (LEVELS[None, :] - 1.0) * error)
    return 2.0 * pin.mean(axis=1)


def edges_of(*, rows: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return each row's site threshold at the 95th and 99th percentile."""
    table = thresholds()
    return {
        label: np.array([table[s][label] for s in rows["site"].to_list()]) for label in ("95", "99")
    }


def per_row(
    *, actual: np.ndarray, q: np.ndarray, edges: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Return per-row score columns: quantile score, median error, tail-weighted, tail pinball."""
    actual = actual.astype(np.float64)
    q = q.astype(np.float64)
    error = actual[:, None] - q
    pin = np.maximum(LEVELS[None, :] * error, (LEVELS[None, :] - 1.0) * error)
    out = {"qs": 2.0 * pin.mean(axis=1), "mae": np.abs(error[:, 5])}
    for label in ("95", "99"):
        edge = edges[label]
        out[f"tw{label}"] = pinball(actual=np.maximum(actual, edge), q=np.maximum(q, edge[:, None]))
    for name, pos in TAILS:
        out[f"pin_{name}"] = pin[:, pos]
        out[f"cov_{name}"] = (error[:, pos] <= 0).astype(np.float64)
    return out


def mean_interval(
    *,
    values: np.ndarray,
    mask: np.ndarray,
    blocks: np.ndarray,
    level: float,
    rng: np.random.Generator,
) -> dict:
    """Return the mean of `values` over `mask` with a block-resampled interval at `level` percent."""
    labels, codes = np.unique(blocks[mask], return_inverse=True)
    n = len(labels)
    if n == 0:
        return {"est": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    sums = np.bincount(codes, weights=values[mask], minlength=n)
    counts = np.bincount(codes, minlength=n).astype(np.float64)
    draws = rng.integers(0, n, size=(N_RESAMPLES, n))
    res = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (100.0 - level) / 2.0
    lo, hi = np.percentile(res, [tail, 100.0 - tail])
    return {
        "est": float(sums.sum() / counts.sum()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(mask.sum()),
    }


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)
