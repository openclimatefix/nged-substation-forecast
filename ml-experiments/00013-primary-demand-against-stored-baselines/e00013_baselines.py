"""Re-score the three stored baselines on the row set: read, pool members, divide by capacity.

Reads `manual_heuristic`, `climatology`, and the production recipe (`xgboost_baseline`, fold
`mid_2025_to_mid_2026`) from the `power_forecasts` Delta table for the 20 series and the 73 runs of
`fold_a_every5d_v1`, takes the 9 quantiles of each forecast's members at `LEVELS`, and divides by
the series' effective capacity. The stored forecasts are in megawatts.

Output: `work/arms/{manual_heuristic,climatology,production}.parquet`, keyed by series, run, and
valid time (no seed), plus `work/arms/stored_checks.json` with the row counts that the score script
prints.

    uv run python ml-experiments/00013-primary-demand-against-stored-baselines/e00013_baselines.py
"""

import json
import time
from pathlib import Path
from typing import Final

import e00013_common as common
import polars as pl

POWER_FORECASTS: Final[Path] = common.ML_EXPERIMENTS_DATA_DIR.parent / "power_forecasts"
FOLD_ID: Final[str] = "mid_2025_to_mid_2026"
STORED: Final[dict[str, str]] = {
    "manual_heuristic": "manual_heuristic",
    "climatology": "climatology",
    "production": "xgboost_baseline",
}
"""Arm name to the stored `experiment_name`."""
ARMS_DIR: Final[Path] = common.WORK_DIR / "arms"


def validation_inits() -> list:
    """Return the 73 runs' `power_fcst_init_time` values, from the validation files."""
    return sorted(
        pl.concat(
            [
                pl.read_parquet(path, columns=["power_fcst_init_time"]).unique()
                for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet"))
            ]
        )["power_fcst_init_time"].to_list()
    )


def stored_quantiles(*, experiment_name: str, series: pl.DataFrame, inits: list) -> pl.DataFrame:
    """Return one stored baseline's capacity-normalised member quantiles on the row set."""
    capacity = pl.DataFrame(
        {
            "time_series_id": series["time_series_id"],
            "capacity": series["effective_capacity_mw"].cast(pl.Float64),
        }
    ).lazy()
    return (
        pl.scan_delta(str(POWER_FORECASTS))
        .filter(
            (pl.col("experiment_name") == experiment_name)
            & (pl.col("fold_id") == FOLD_ID)
            & pl.col("power_fcst_init_time").is_in(inits)
            & pl.col("time_series_id").is_in(series["time_series_id"].to_list())
        )
        .group_by("time_series_id", "power_fcst_init_time", "valid_time")
        .agg(
            *[
                pl.col("power_fcst").cast(pl.Float64).quantile(float(level), "linear").alias(name)
                for level, name in zip(common.LEVELS, common.LEVEL_COLUMNS, strict=True)
            ],
            members=pl.col("ensemble_member").n_unique(),
        )
        .join(capacity, on="time_series_id", how="left", validate="m:1")
        .with_columns(*[pl.col(name) / pl.col("capacity") for name in common.LEVEL_COLUMNS])
        .drop("capacity")
        .collect(engine="streaming")
        .sort("time_series_id", "power_fcst_init_time", "valid_time")
    )


def main() -> None:
    """Write each stored baseline's quantiles, skipping files that exist."""
    ARMS_DIR.mkdir(parents=True, exist_ok=True)
    series = common.series_table()
    inits = validation_inits()
    checks = {}
    for arm, experiment_name in STORED.items():
        target = ARMS_DIR / f"{arm}.parquet"
        if target.exists():
            continue
        started = time.monotonic()
        frame = stored_quantiles(experiment_name=experiment_name, series=series, inits=inits)
        frame.write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
        checks[arm] = {
            "rows": frame.height,
            "members_min": int(frame.select(pl.col("members").min()).item()),
            "members_max": int(frame.select(pl.col("members").max()).item()),
        }
        print(f"{arm}: {frame.height:,} rows in {time.monotonic() - started:.0f} s", flush=True)
    if checks:
        (ARMS_DIR / "stored_checks.json").write_text(json.dumps(checks, indent=2) + "\n")


if __name__ == "__main__":
    main()
