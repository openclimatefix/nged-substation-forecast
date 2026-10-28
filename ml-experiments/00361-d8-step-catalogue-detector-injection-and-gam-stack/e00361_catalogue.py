"""Step 1: catalogue D8's detected steps and multi-day drops, and print every number quoted.

PYTHONPATH=<the folders of experiments 252, 264, 354, and 361> \
        OMP_NUM_THREADS=2 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00361-*/e00361_catalogue.py
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Final

import e00361_detect as detect
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00361-d8-step-catalogue-detector-injection-and-gam-stack"
)
FIRST: Final[datetime] = datetime(2023, 8, 15)  # noqa: DTZ001  (the grid is naive UTC)


def catalogue(
    *, site: str, threshold_mw: float, negative: bool = True, drops: bool = False
) -> pl.DataFrame:
    """Return the steps (or, with `drops`, the multi-day drops) detected since 2023-08-15."""
    times, power = detect.load_power_mw(site=site)
    if drops:
        segments = detect.detect_drops(power_mw=power, threshold_mw=threshold_mw)
    else:
        segments = detect.detect(power_mw=power, threshold_mw=threshold_mw, negative=negative)
    frame = pl.DataFrame(
        {
            "start": times[segments.start],
            "end": times[segments.end - 1],
            "level_mw": segments.level_mw,
            "dwell_days": (segments.end - segments.start) / detect.HALF_HOURS_PER_DAY,
        }
    )
    return frame.filter(pl.col("start") >= FIRST)


def main() -> None:
    """Write the catalogue tables and print the summary."""
    WORK.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    for threshold in (1.0, 1.5, 2.0):
        neg = catalogue(site="D8", threshold_mw=threshold)
        pos = catalogue(site="D8", threshold_mw=threshold, negative=False)
        drops = catalogue(site="D8", threshold_mw=threshold, drops=True)
        report[f"t{threshold}"] = {
            "negative_steps": neg.height,
            "negative_drops_2d": drops.height,
            "positive_steps": pos.height,
            "median_level_mw": float(neg["level_mw"].median()) if neg.height else None,  # ty: ignore[invalid-argument-type]
            "median_drop_level_mw": float(drops["level_mw"].median()) if drops.height else None,  # ty: ignore[invalid-argument-type]
            "median_dwell_days": float(neg["dwell_days"].median()) if neg.height else None,  # ty: ignore[invalid-argument-type]
            "median_drop_dwell_days": float(drops["dwell_days"].median()) if drops.height else None,  # ty: ignore[invalid-argument-type]
        }
        if threshold == detect.THRESHOLD_MW:
            neg.write_parquet(WORK / "d8_negative_segments.parquet")
            drops.write_parquet(WORK / "d8_drops.parquet")
            print(drops)
            pos.write_parquet(WORK / "d8_positive_segments.parquet")
            per_month = (
                neg.with_columns(month=pl.col("start").dt.strftime("%Y-%m"))
                .group_by("month")
                .len()
                .sort("month")
            )
            report["per_month"] = per_month.to_dicts()
            print(neg)
            print(per_month)
    print(json.dumps({k: v for k, v in report.items() if k != "per_month"}, indent=1))
    (WORK / "catalogue_report.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
