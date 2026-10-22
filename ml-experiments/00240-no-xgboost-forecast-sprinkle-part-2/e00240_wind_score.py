# ruff: noqa: E501
"""Score experiment 230's stored wind arms with experiment 240's rule (read only).

Experiment 230's B5 (XGBoost with the physics core's expected power as base margin), B9 (neural
calibrator), and B14 (XGBoost with the core's columns added) are sprinkled arms. This script
scores them against B1 (the XGBoost-free physics core) and A0r (XGBoost) at 12 to 36 hour leads and
over all leads, on the rows all arms share, with whole months and the three seeds resampled.
The arm files may change until experiment 230 merges, so re-run this script after it merges.
Output: `wind_scores.json`.

    uv run --no-sync python ml-experiments/00240-no-xgboost-forecast-sprinkle-part-2/e00240_wind_score.py
"""

import json
from pathlib import Path
from typing import Final

import e00240_score as solar_score
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WIND_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00230-wind-physics-core-part-1" / "work" / "arms"
BEST_GUESS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00033-best-guess-architecture-vs-production"
    / "work"
    / "generator_tables"
    / "bg_full.parquet"
)
ARMS: Final[tuple[str, ...]] = ("B1", "A0r", "B5", "B9", "B14")
KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time", "seed")
BANDS: Final[dict[str, tuple[float, float]]] = {"12-36 h": (12.0, 36.0), "all leads": (0.0, 1e9)}
CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("W1", "B5", "B1"),
    ("W2", "B9", "B1"),
    ("W3", "B14", "B1"),
    ("W4", "A0r", "B1"),
    ("W5", "B5", "A0r"),
    ("W6", "B14", "A0r"),
)


def main() -> None:
    """Score each wind arm and write the report."""
    cases = (
        pl.read_parquet(BEST_GUESS)
        .filter((pl.col("technology") == "wind") & (pl.col("seed") == 0))
        .select(*KEYS[:3], "lead_hours", pl.col("actual_norm").alias("actual"))
    )
    frames = {
        arm: pl.read_parquet(WIND_DIR / f"{arm}_spread.parquet")
        .join(cases, on=list(KEYS[:3]), how="inner")
        .sort(*KEYS)
        for arm in ARMS
    }
    reference = frames["B1"]
    for arm, frame in frames.items():
        assert frame.height == reference.height, (arm, frame.height, reference.height)
        for key in KEYS:
            assert frame[key].equals(reference[key]), (arm, key)
    month = reference["valid_time"].dt.strftime("%Y-%m").to_numpy()
    seed = reference["seed"].to_numpy()
    lead = reference["lead_hours"].to_numpy()
    rows = {arm: solar_score.per_row(frame=frame) for arm, frame in frames.items()}
    report: dict[str, dict] = {}
    for band, (low, high) in BANDS.items():
        keep = (lead >= low) & (lead <= high)
        entry = {
            "n_rows": int(keep.sum()),
            "arms": {
                arm: {k: float(v[keep].mean()) for k, v in r.items()} for arm, r in rows.items()
            },
            "contrasts": {
                name: {
                    "first": first,
                    "second": second,
                    **{
                        metric: solar_score.bootstrap(
                            difference=(rows[first][metric] - rows[second][metric])[keep],
                            month=month[keep],
                            seed=seed[keep],
                        )
                        for metric in ("qs", "mae_median")
                    },
                }
                for name, first, second in CONTRASTS
            },
        }
        report[band] = entry
    out = solar_score.OUT_DIR / "wind_scores.json"
    out.write_text(json.dumps(report, indent=1))
    for band, entry in report.items():
        print(band, entry["n_rows"])
        for arm, v in entry["arms"].items():
            print(f"  {arm:4s} qs {v['qs']:.3f} mae {v['mae_median']:.3f}")
        for name, c in entry["contrasts"].items():
            q = c["qs"]
            print(
                f"  {name} {c['first']}-{c['second']}: {q[0]:+.4f} (98.3%: {q[3]:+.4f} to {q[4]:+.4f})"
            )


if __name__ == "__main__":
    main()
