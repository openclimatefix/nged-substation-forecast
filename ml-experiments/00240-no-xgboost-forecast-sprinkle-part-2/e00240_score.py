# ruff: noqa: E501
"""Score experiment 240's solar arms against experiment 224's stored arm 3 and arm 5.

Joins the new arms (`predictions/<fold>.parquet`) to experiment 224's stored arm 3
(`a3_two_number`) and arm 5 (`a5_xgboost`) on farm and valid time, band 0 (12 to 36 hour leads),
so every arm is scored on the same rows. Reports, per fold, the quantile score (twice the mean
pinball loss over 11 levels, points of capacity), the MAE of the median, coverage and pinball loss
at the 0.01, 0.05, 0.95, and 0.99 levels, and the planned contrasts against arm 3 with intervals
from resampling whole months and the three seeds. Output: `scores.json`.

    uv run --no-sync python ml-experiments/00240-no-xgboost-forecast-sprinkle-part-2/e00240_score.py
"""

import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from studies.ens_members import MIXTURE_LEVELS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00240-no-xgboost-forecast-sprinkle-part-2"
SOURCE: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00224-metered-farm-dayahead-two-number" / "predictions"
)
LEVELS: Final[np.ndarray] = np.array(MIXTURE_LEVELS)
COLUMNS: Final[tuple[str, ...]] = tuple(f"q{round(100 * level):02d}" for level in MIXTURE_LEVELS)
TAILS: Final[tuple[int, ...]] = (0, 1, 9, 10)
NEW_ARMS: Final[tuple[str, ...]] = (
    "A3r",
    "R",
    "S",
    "RS",
    "R0",
    "PR",
    "Rg",
    "PRg",
    "RSg",
    "SQ",
    "SC",
    "X5r",
)
OLD_ARMS: Final[dict[str, str]] = {"a3_two_number": "A3", "a5_xgboost": "A5"}
GAP_HALF_THRESHOLD: Final[float] = 0.5
CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("C1", "R", "A3"),
    ("C2", "S", "A3"),
    ("C3", "RS", "A3"),
    ("D1", "R0", "A3"),
    ("D2", "PR", "A3"),
    ("D3", "R", "R0"),
    ("D4", "A5", "A3"),
    ("D5", "RS", "A5"),
    ("D6", "R", "A3r"),
    ("D7", "S", "A3r"),
    ("D8", "RS", "A3r"),
    ("D9", "A3r", "A3"),
    ("H1", "Rg", "A3"),
    ("H2", "PRg", "A3"),
    ("H3", "RSg", "A3"),
    ("H4", "SQ", "S"),
    ("H5", "SC", "S"),
    ("H6", "X5r", "A5"),
    ("H7", "S", "X5r"),
    ("H8", "SC", "A3"),
    ("H9", "X5r", "A3"),
    ("H10", "Rg", "A3r"),
    ("H11", "S", "A5"),
    ("H12", "SC", "A5"),
)
BONFERRONI_LEVEL: Final[float] = 1 - 0.05 / 3
RESAMPLES: Final[int] = 20_000
KEY: Final[tuple[str, ...]] = ("time_series_id", "valid_time")


def per_row(*, frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return per-row quantile score, median absolute error, and tail pinball and coverage."""
    actual = frame["actual"].to_numpy().astype(np.float64)
    quantiles = frame.select(COLUMNS).to_numpy().astype(np.float64)
    error = actual[:, None] - quantiles
    pinball = np.maximum(LEVELS * error, (LEVELS - 1) * error)
    out = {"qs": 2.0 * pinball.mean(axis=1) * 100.0, "mae_median": np.abs(error[:, 5]) * 100.0}
    for index in TAILS:
        out[f"pinball_{COLUMNS[index]}"] = pinball[:, index] * 100.0
        out[f"covered_{COLUMNS[index]}"] = (actual <= quantiles[:, index]).astype(np.float64)
    return out


def bootstrap(*, difference: np.ndarray, month: np.ndarray, seed: np.ndarray) -> list[float]:
    """Return the mean difference and the 95% and Bonferroni-level interval ends."""
    months, seeds = np.unique(month), np.unique(seed)
    sums = np.zeros((months.size, seeds.size))
    counts = np.zeros_like(sums)
    for i, m in enumerate(months):
        for j, s in enumerate(seeds):
            keep = (month == m) & (seed == s)
            sums[i, j], counts[i, j] = difference[keep].sum(), keep.sum()
    rng = np.random.default_rng(0)
    draw_m = rng.integers(0, months.size, size=(RESAMPLES, months.size))
    draw_s = rng.integers(0, seeds.size, size=(RESAMPLES, seeds.size))
    picked = (draw_m[:, :, None], draw_s[:, None, :])
    means = sums[picked].sum(axis=(1, 2)) / counts[picked].sum(axis=(1, 2))
    out = [float(sums.sum() / counts.sum())]
    for level in (0.95, BONFERRONI_LEVEL):
        tail = (1 - level) / 2
        out += [float(x) for x in np.quantile(means, [tail, 1 - tail])]
    return out


def arms_of(*, fold: str) -> dict[str, pl.DataFrame]:
    """Return every arm on the rows all arms share, sorted by seed and key; old arms repeat per seed."""
    new = pl.read_parquet(OUT_DIR / "predictions" / f"{fold}.parquet").filter(pl.col("band") == 0)
    old = pl.read_parquet(SOURCE / f"{fold}.parquet").filter(
        (pl.col("band") == 0) & pl.col("arm").is_in(list(OLD_ARMS))
    )
    shared = old.filter(pl.col("arm") == "a3_two_number").select(*KEY)
    for arm in NEW_ARMS:
        shared = shared.join(new.filter(pl.col("arm") == arm).select(*KEY).unique(), on=list(KEY))
    shared = shared.join(old.filter(pl.col("arm") == "a5_xgboost").select(*KEY), on=list(KEY))
    seeds = pl.DataFrame(
        {"seed": sorted(new["seed"].unique().to_list())}, schema={"seed": pl.Int32}
    )
    out = {}
    for arm in NEW_ARMS:
        frame = new.filter(pl.col("arm") == arm).join(shared, on=list(KEY), how="semi")
        out[arm] = frame.sort("seed", *KEY)
    for source, label in OLD_ARMS.items():
        frame = old.filter(pl.col("arm") == source).join(shared, on=list(KEY), how="semi")
        out[label] = (
            frame.drop("seed")
            .join(seeds, how="cross")
            .with_columns(pl.col("seed"))
            .sort("seed", *KEY)
        )
    heights = {len(f) for f in out.values()}
    assert len(heights) == 1, {a: len(f) for a, f in out.items()}
    return out


def main() -> None:
    """Score each fold and write the report."""
    report: dict[str, dict] = {}
    for fold in ("scored_year", "replicate"):
        arms = arms_of(fold=fold)
        reference = arms["A3"]
        for arm, frame in arms.items():
            assert frame["valid_time"].equals(reference["valid_time"]), arm
            assert frame["seed"].equals(reference["seed"]), arm
        month = reference["valid_time"].dt.strftime("%Y-%m").to_numpy()
        seed = reference["seed"].to_numpy()
        rows = {arm: per_row(frame=frame) for arm, frame in arms.items()}
        entry: dict[str, Any] = {"n_rows": reference.height // len(np.unique(seed)), "arms": {}}
        for arm, values in rows.items():
            entry["arms"][arm] = {name: float(v.mean()) for name, v in values.items()}
        entry["contrasts"] = {}
        for name, first, second in CONTRASTS:
            entry["contrasts"][name] = {
                "first": first,
                "second": second,
                **{
                    metric: bootstrap(
                        difference=rows[first][metric] - rows[second][metric],
                        month=month,
                        seed=seed,
                    )
                    for metric in ("qs", "mae_median")
                },
            }
        by_farm = {}
        farms = reference["time_series_id"].to_numpy()
        for farm in np.unique(farms):
            keep = farms == farm
            by_farm[str(farm)] = {arm: float(rows[arm]["qs"][keep].mean()) for arm in rows}
        entry["by_farm"] = by_farm
        report[fold] = entry
    (OUT_DIR / "scores.json").write_text(json.dumps(report, indent=1))
    scored = report["scored_year"]
    gap = scored["arms"]["A3"]["qs"] - scored["arms"]["A5"]["qs"]
    print(f"rows {scored['n_rows']}  gap A3-A5 {gap:.3f}  half {GAP_HALF_THRESHOLD * gap:.3f}")
    for arm, values in scored["arms"].items():
        print(f"{arm:4s} qs {values['qs']:.3f}  mae {values['mae_median']:.3f}")
    for name, c in scored["contrasts"].items():
        q = c["qs"]
        print(f"{name} {c['first']}-{c['second']}: {q[0]:+.4f} (98.3%: {q[3]:+.4f} to {q[4]:+.4f})")


if __name__ == "__main__":
    main()
