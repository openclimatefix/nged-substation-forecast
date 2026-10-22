"""Score the XGBoost-free forecast against the XGBoost forecast for demand, solar, and wind.

Reads stored predictions only (experiments 224, 228, and 230) and writes
`data/ml-experiments/00234-no-xgboost-forecast-audit-part-1/scores.json`. For each series type and
lead band it reports the quantile score (twice the mean pinball loss over the 11 levels), the MAE of
the median, coverage and pinball loss at the four tail levels, and the gap between the two arms as
a percentage with a 95% interval from resampling whole months.

uv run --no-sync python ml-experiments/00234-no-xgboost-forecast-audit-part-1/e00234_score.py
"""

import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_row_difference
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR
OUT: Final[Path] = DATA / "00234-no-xgboost-forecast-audit-part-1" / "scores.json"
LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99)
COLUMNS: Final[tuple[str, ...]] = tuple(f"q{round(100 * level):02d}" for level in LEVELS)
TAILS: Final[tuple[int, ...]] = (0, 1, 9, 10)
BANDS: Final[dict[str, tuple[float, float]]] = {
    "12-36 h": (12.0, 36.0),
    "all leads": (0.0, 1e9),
}
WIND_DIR: Final[Path] = DATA / "00230-wind-physics-core-part-1" / "work"
BEST_GUESS: Final[Path] = (
    DATA / "00033-best-guess-architecture-vs-production" / "work" / "generator_tables"
) / "bg_full.parquet"
KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time", "seed")


def loss_columns(*, frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return per-row quantile score, median absolute error, and tail pinball and coverage."""
    actual = frame["actual"].to_numpy().astype(np.float64)
    quantiles = frame.select(COLUMNS).to_numpy().astype(np.float64)
    levels = np.array(LEVELS)
    error = actual[:, None] - quantiles
    pinball = np.maximum(levels * error, (levels - 1) * error)
    out = {
        "qs": 2.0 * pinball.mean(axis=1),
        "mae_median": np.abs(error[:, 5]),
    }
    for index in TAILS:
        name = COLUMNS[index]
        out[f"pinball_{name}"] = pinball[:, index]
        out[f"covered_{name}"] = (actual <= quantiles[:, index]).astype(np.float64)
    return out


def score_pair(*, free: pl.DataFrame, xgb: pl.DataFrame, scale: float) -> dict[str, dict]:
    """Score two arms on identical rows, per lead band; `scale` converts values to the page unit."""
    assert free.height == xgb.height
    months = free["valid_time"].dt.strftime("%Y-%m").to_numpy()
    free_loss, xgb_loss = loss_columns(frame=free), loss_columns(frame=xgb)
    lead = free["lead_hours"].to_numpy()
    result = {}
    for band, (low, high) in BANDS.items():
        keep = (lead >= low) & (lead <= high)
        entry: dict[str, Any] = {"n_rows": int(keep.sum())}
        for name in free_loss:
            unit = 1.0 if name.startswith("covered") else scale
            a, b = free_loss[name][keep] * unit, xgb_loss[name][keep] * unit
            diff = bootstrap_row_difference(values=a - b, months=months[keep])
            entry[name] = {
                "free": float(a.mean()),
                "xgboost": float(b.mean()),
                "diff": diff["difference"],
                "lo": diff["lower_95"],
                "hi": diff["upper_95"],
                "gap_percent": 100.0 * diff["difference"] / float(b.mean()),
                "gap_lo_percent": 100.0 * diff["lower_95"] / float(b.mean()),
                "gap_hi_percent": 100.0 * diff["upper_95"] / float(b.mean()),
            }
        result[band] = entry
    return result


def demand() -> dict:
    """Score arm D plus the correction network and solar summary (dhs) against BG2, six sites."""
    folder = next(DATA.glob("00228-*")) / "work" / "dp_preds"

    def read(prefix: str) -> pl.DataFrame:
        parts = [pl.read_parquet(p) for p in sorted(folder.glob(f"{prefix}_scored_D*.parquet"))]
        return pl.concat(parts, how="diagonal").sort("site", "run", "valid_time", "seed")

    free, xgb = read("dhs"), read("bg2")
    return score_pair(free=free, xgb=xgb, scale=1.0)


def solar() -> dict:
    """Score the plant fed the mean and spread of 51 members against XGBoost, six farms."""
    frame = pl.read_parquet(
        DATA / "00224-metered-farm-dayahead-two-number" / "predictions" / "scored_year.parquet"
    ).filter(pl.col("band") == 0)
    sort = ["time_series_id", "power_fcst_init_time", "valid_time", "seed"]
    free = frame.filter(pl.col("arm") == "a3_two_number").sort(sort)
    xgb = frame.filter(pl.col("arm") == "a5_xgboost").sort(sort)
    return score_pair(free=free, xgb=xgb, scale=100.0)


def wind() -> dict:
    """Score the physics core B1 against the 11-level XGBoost refit A0r, three farms."""
    cases = (
        pl.read_parquet(BEST_GUESS)
        .filter((pl.col("technology") == "wind") & (pl.col("seed") == 0))
        .select(*KEYS[:3], "lead_hours", pl.col("actual_norm").alias("actual"))
    )

    def read(name: str) -> pl.DataFrame:
        frame = pl.read_parquet(WIND_DIR / "arms" / f"{name}_spread.parquet")
        return frame.join(cases, on=list(KEYS[:3]), how="inner").sort(*KEYS)

    return score_pair(free=read("B1"), xgb=read("A0r"), scale=100.0)


def main() -> None:
    """Write the scores of the three series types."""
    scores = {"demand": demand(), "solar": solar(), "wind": wind()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(scores, indent=1))
    for kind, bands in scores.items():
        for band, entry in bands.items():
            qs = entry["qs"]
            print(
                f"{kind:7s} {band:10s} n={entry['n_rows']:8d} free {qs['free']:.3f} "
                f"xgb {qs['xgboost']:.3f} gap {qs['gap_percent']:+.2f}% "
                f"({qs['gap_lo_percent']:+.2f} to {qs['gap_hi_percent']:+.2f}) "
                f"MAE {entry['mae_median']['free']:.3f} vs {entry['mae_median']['xgboost']:.3f}"
            )


if __name__ == "__main__":
    main()
