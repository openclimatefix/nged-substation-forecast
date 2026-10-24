"""Wind cross-breed: apply the recalibration table to experiment 259's stored wind forecasts.

Descriptive only (the wind family has no research default). Farms W1 to W3, experiment 259's kept
rows (the half-hours it did not mask), the 53 Tuesday runs, 2025-07 to 2026-06. The table is the one
of arm 1 (`recal`): offsets per farm, lead day, and quantile level fitted rolling-origin on the runs
of earlier months. Reports, before and after, the share of actual values strictly below and at or
below the 0.01, 0.05, 0.95, and 0.99 quantiles and the quantile score, per lead-day group.

    PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring:ml-experiments/00321-... \
        uv run --no-sync python ml-experiments/00321-*/e00321_wind.py
"""

from typing import Final

import e00321_common as common
import e00321_recal as recal
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final = ML_EXPERIMENTS_DATA_DIR / "00259-wind-plant-plus-quantile-offset-head" / "work"
QCOLS: Final[list[str]] = recal.QCOLS
LEVELS: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
GROUPS: Final[dict[str, tuple[int, int]]] = {
    "day 1": (1, 1),
    "day 2": (2, 2),
    "days 3 to 4": (3, 4),
    "day 5 and later": (5, 99),
}
ARMS: Final[tuple[str, ...]] = ("R1", "R0", "H5c")


def load_arm(*, arm: str) -> pl.DataFrame:
    """Return a stored wind arm's seed-averaged quantiles on the kept rows, with actual and lead."""
    head = pl.read_parquet(WORK / "heads" / "H5c.parquet").filter(~pl.col("masked"))
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    meta = head.group_by(keys).agg(
        pl.col("site").first(), pl.col("lead_hours").first(), pl.col("actual_norm").first()
    )
    if arm == "H5c":
        source = head
    elif arm == "R1":
        source = pl.read_parquet(WORK / "base" / "R1_scored_seed0.parquet")
    else:
        source = pl.concat(
            [pl.read_parquet(WORK / "base" / f"R0_scored_seed{s}.parquet") for s in range(5)]
        )
    averaged = source.group_by(keys).agg(pl.col(QCOLS).cast(pl.Float64).mean())
    return (
        meta.join(averaged, on=keys, how="inner")
        .rename(
            {
                "site": "farm",
                "power_fcst_init_time": "init_time",
                "valid_time": "time",
                "actual_norm": "actual",
            }
        )
        .with_columns(time_series_id=pl.col("farm"))
        .select("time_series_id", "time", "init_time", "lead_hours", "actual", *QCOLS)
    )


def summarise(*, frame: pl.DataFrame, arm: str, stage: str) -> list[dict[str, object]]:
    """Return coverage and score rows per lead-day group for a frame of quantiles."""
    q = np.sort(frame.select(QCOLS).to_numpy(), axis=1)
    actual = frame["actual"].to_numpy()
    day = frame["day"].to_numpy()
    error = actual[:, None] - q
    pin = np.maximum(common_levels() * error, (common_levels() - 1.0) * error)
    score = 200.0 * pin.mean(axis=1)
    out = []
    for name, (lo, hi) in GROUPS.items():
        rows = (day >= lo) & (day <= hi)
        if not rows.any():
            continue
        record: dict[str, object] = {
            "arm": arm,
            "stage": stage,
            "group": name,
            "rows": int(rows.sum()),
            "score": float(score[rows].mean()),
        }
        for column, level in LEVELS.items():
            i = QCOLS.index(column)
            record[f"strict_{column}"] = float((actual[rows] < q[rows, i]).mean())
            record[f"atmost_{column}"] = float((actual[rows] <= q[rows, i]).mean())
            record[f"nominal_{column}"] = level
        out.append(record)
    return out


def common_levels() -> np.ndarray:
    """Return the 11 quantile levels."""
    from e00310_common import LEVELS  # ty: ignore[unresolved-import]

    return LEVELS


def main() -> None:
    """Write and print the before and after coverage table."""
    rows: list[dict[str, object]] = []
    for arm in ARMS:
        frame = load_arm(arm=arm)
        before = recal.add_day(frame=frame, lead_origin_hours=0.0).sort(recal.KEYS)
        after = recal.recalibrate(frame=frame, variant="recal", lead_origin_hours=0.0)
        rows += summarise(frame=before, arm=arm, stage="before")
        rows += summarise(frame=after, arm=arm, stage="after")
    table = pl.DataFrame(rows)
    table.write_csv(common.DATA_DIR / "wind_recalibration.csv")
    show = table.select(
        "arm",
        "stage",
        "group",
        "score",
        "strict_q01",
        "atmost_q01",
        "strict_q05",
        "strict_q95",
        "strict_q99",
    ).with_columns(pl.col(pl.Float64).round(4))
    with pl.Config(tbl_rows=60, tbl_cols=12, tbl_width_chars=200, fmt_str_lengths=20):
        print(show.sort("arm", "group", "stage"))


if __name__ == "__main__":
    main()
