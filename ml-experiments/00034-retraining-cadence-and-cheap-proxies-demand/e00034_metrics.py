"""Compute every per-case metric of every arm of experiment 00034.

Reads the arm tables in `work/arms/` (this experiment's arms) and `work/reference/` (experiment
00033's stack and baselines), each keyed by series, run, valid time, and seed with the nine
quantiles and the observed power. An arm is scored on the cases it forecasts that also lie in
experiment 00033's shared case set; a contrast later keeps the cases its two arms share. The
thresholds are each series' `historical_p99` and `historical_p95` (fractions of capacity).

Output: `work/metrics/<arm>.parquet` with the run keys, `seed`, `group`, `lead_hours`, and the
metrics of `common.case_metrics`.

    uv run python ml-experiments/00034-*/e00034_metrics.py
"""

from pathlib import Path
from typing import Final

import e00034_common as common
import numpy as np
import polars as pl

METRICS_DIR: Final[Path] = common.WORK_DIR / "metrics"
REFERENCE_DIR: Final[Path] = common.WORK_DIR / "reference"
FINITE: Final = pl.all_horizontal(pl.col(*common.LEVEL_COLUMNS).is_finite())


def shared_keys() -> pl.DataFrame:
    """Return the cases every arm of experiment 00033 holds, for the three demand groups.

    Scoring on these cases keeps absolute scores comparable with experiment 00033's, whose
    `manual_heuristic` and production arms lack about 5% of the cases the N1 arms hold.
    """
    return pl.concat(
        [
            pl.read_parquet(common.WORK_DIR / f"reference_shared_keys_{group}.parquet")
            for group in ("Primary", "BSP", "GSP")
        ]
    ).select(*common.KEYS)


def arm_files() -> dict[str, Path]:
    """Return every arm table by name; this experiment's arms override a reference of equal name."""
    files = {p.stem: p for p in sorted(REFERENCE_DIR.glob("*.parquet"))}
    files.update({p.stem: p for p in sorted(common.ARMS_DIR.glob("*.parquet"))})
    return files


def main() -> None:
    """Write every arm's metrics."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    series = common.series_table()
    arrays = common.load_series_arrays(series=series)
    groups = series.select("time_series_id", group=pl.col("substation_type").cast(pl.String))
    thresholds = {
        int(row["time_series_id"]): common.historical_thresholds(power=arrays[row["site"]].power)
        for row in series.iter_rows(named=True)
    }
    keys = shared_keys()
    for name, path in arm_files().items():
        target = METRICS_DIR / f"{name}.parquet"
        if target.exists() and target.stat().st_mtime > path.stat().st_mtime:
            continue
        frame = (
            pl.read_parquet(path)
            .join(groups, on="time_series_id", how="left", validate="m:1")
            .filter(FINITE & pl.col("actual").is_finite() & (pl.col("valid_time") < common.HOLDOUT))
            .join(keys, on=list(common.KEYS), how="semi")
            .sort(*common.KEYS, "seed")
        )
        quantiles = np.sort(
            frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64), axis=1
        )
        actual = frame["actual"].to_numpy().astype(np.float64)
        ids = frame["time_series_id"].to_numpy()
        levels = {
            label: np.array([thresholds[int(i)][label] for i in ids]) for label in ("p99", "p95")
        }
        metrics = common.case_metrics(quantiles=quantiles, actual=actual, thresholds=levels)
        table = frame.select(
            *common.KEYS,
            "seed",
            "group",
            lead_hours=(
                (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
            ),
        ).with_columns(
            *[pl.Series(label, values.astype(np.float32)) for label, values in metrics.items()]
        )
        table.write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
        print(f"{name}: {table.height:,} case-seeds", flush=True)


if __name__ == "__main__":
    main()
