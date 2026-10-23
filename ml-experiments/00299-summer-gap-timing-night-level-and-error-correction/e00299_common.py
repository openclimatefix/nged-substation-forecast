"""Constants and helpers shared by the scripts of experiment 299 (spring-gap timing diagnostics)."""

import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 299
SLUG: Final[str] = "summer-gap-timing-night-level-and-error-correction"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
MASTER: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00266-summer-gap-diagnostics-solar-term" / "work" / "master.parquet"
)
E155_PREDS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00155-dp-bg-stage2-assembly-night-offset-and-error-correction"
    / "work" / "dp_preds"
)  # fmt: skip
E111_PREDS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00111-differentiable-physics-6-ens-forecast-engine-against-bg2"
    / "work" / "dp_preds"
)  # fmt: skip
E201_FEATURES: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00201-dp-gam-residual-mlp" / "work" / "features"
)  # fmt: skip
SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
N_RESAMPLES: Final[int] = 5_000
CLOCK_CHANGE_SPRING_2026: Final[str] = "2026-03-29"
CLOCK_CHANGE_AUTUMN_2025: Final[str] = "2025-10-26"
AMJ: Final[tuple[str, ...]] = ("2026-04", "2026-05", "2026-06")
JAS: Final[tuple[str, ...]] = ("2025-07", "2025-08", "2025-09")


def qs_per_row(*, actual: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss over the 11 quantiles, per row."""
    error = actual.astype(np.float64)[:, None] - q.astype(np.float64)
    pin = np.maximum(LEVELS[None, :] * error, (LEVELS[None, :] - 1.0) * error)
    return 2.0 * pin.mean(axis=1)


def block_mean_interval(
    *, values: np.ndarray, mask: np.ndarray, blocks: np.ndarray, seed: int = 0
) -> dict:
    """Mean of `values` over `mask`, with a 95% interval resampling the `blocks` labels."""
    keep = mask & np.isfinite(values)
    if not keep.any():
        return {"est": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "blocks": 0}
    labels, codes = np.unique(blocks[keep], return_inverse=True)
    n = len(labels)
    sums = np.bincount(codes, weights=values[keep], minlength=n)
    counts = np.bincount(codes, minlength=n).astype(np.float64)
    draws = np.random.default_rng(seed).integers(0, n, size=(N_RESAMPLES, n))
    res = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.percentile(res, [2.5, 97.5])
    return {
        "est": float(sums.sum() / counts.sum()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(keep.sum()),
        "blocks": int(n),
    }


def write_json_atomic(*, path: Path, content: object) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)


def read_master() -> pl.DataFrame:
    """Return experiment 266's master frame with the week blocks and local hour added."""
    local = (pl.col("valid_time") - pl.duration(minutes=30)).dt.convert_time_zone("Europe/London")
    return pl.read_parquet(common_master_path()).with_columns(
        gap=pl.col("qs_d") - pl.col("qs_bg2"),
        day=(pl.col("clear") > 20.0) & pl.col("csi").is_finite(),
        init_week=pl.col("power_fcst_init_time").dt.truncate("1w").dt.strftime("%Y-%m-%d"),
        valid_week=pl.col("valid_time").dt.truncate("1w").dt.strftime("%Y-%m-%d"),
        valid_date=pl.col("valid_time").dt.strftime("%Y-%m-%d"),
        local_hour=local.dt.hour(),
        utc_month=pl.col("valid_time").dt.month(),
        lead_band=pl.when(pl.col("lead_hours") < 6)
        .then(pl.lit("0-6"))
        .when(pl.col("lead_hours") < 24)
        .then(pl.lit("6-24"))
        .when(pl.col("lead_hours") < 48)
        .then(pl.lit("24-48"))
        .otherwise(pl.lit("48-168")),
    )


def common_master_path() -> Path:
    """Return the path of experiment 266's master frame."""
    return MASTER
