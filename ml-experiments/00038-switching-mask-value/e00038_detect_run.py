"""Run D1 (window 8, h 4) and D0 (window 48, h 4), build intervals, and write the Q3 tables.

Reads experiment 00037's residual panel. Reads the logs-derived `oracle_real.parquet` only to count
how many detected intervals overlap a logged closure (a report, never an input to the intervals).
"""

import numpy as np
import polars as pl
from e00038_base import GRID_LENGTH, SOURCE_DIR, WORK_DIR, alarms, prepare
from e00038_intervals import (
    build_intervals,
    intervals_frame,
    relative_residual,
    split_fractions,
)

D1_WINDOW, D0_WINDOW, THRESHOLD = 8, 48, 4.0


def load_arrays(*, panel: pl.DataFrame, column: str) -> dict[int, np.ndarray]:
    """Return one float array per series on the grid, NaN where missing."""
    out: dict[int, np.ndarray] = {}
    for (series,), group in panel.group_by(["series"]):
        array = np.full(GRID_LENGTH, np.nan)
        array[group["stamp"].to_numpy()] = group[column].to_numpy()
        out[int(series)] = array
    return out


def main() -> None:
    """Write the alarm, interval, split-fraction, and Q3 tables."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    panel = pl.read_parquet(SOURCE_DIR / "panel.parquet")
    residual = load_arrays(panel=panel, column="res_mw")
    expected = load_arrays(panel=panel, column="expected_mw")
    power = load_arrays(panel=panel, column="obs_mw")
    relative = {s: relative_residual(residual=residual[s], expected=expected[s]) for s in residual}
    oracle = pl.read_parquet(SOURCE_DIR / "oracle_real.parquet").filter("primary")
    tables = {}
    variants = (
        ("d1", "D1", D1_WINDOW, False),
        ("d0", "D0", D0_WINDOW, False),
        ("d1fx", "D1", D1_WINDOW, True),
        ("d0fx", "D0", D0_WINDOW, True),
    )
    for name, detector, window, fixed in variants:
        table = alarms(
            inputs=prepare(residuals=residual, window=window),
            detector=detector,
            threshold=THRESHOLD,
        )
        table.write_parquet(WORK_DIR / f"alarms_{name[:2]}.parquet")
        intervals = build_intervals(
            alarm_table=table, relative=relative, window=window, detector=detector, fixed=fixed
        )
        frame = intervals_frame(intervals=intervals)
        frame.write_parquet(WORK_DIR / f"intervals_{name}.parquet")
        tables[name] = (table, intervals, frame)
        print(f"{name}: {table.height} alarms, {frame.height} intervals")
    for name in ("d1", "d1fx"):
        _, intervals, frame = tables[name]
        split_fractions(power=power, interval_set=intervals).write_parquet(
            WORK_DIR / f"split_fractions_{name}.parquet"
        )
        q3 = frame.with_columns(
            hours=(pl.col("end") - pl.col("onset")) / 2,
            overlaps_logged=pl.struct("series", "onset", "end").map_elements(
                lambda r: bool(
                    oracle.filter(
                        (pl.col("series_a") == r["series"]) | (pl.col("series_b") == r["series"]),
                        pl.col("first_stamp") < r["end"],
                        pl.col("last_stamp_exclusive") > r["onset"],
                    ).height
                ),
                return_dtype=pl.Boolean,
            ),
        )
        q3.write_parquet(WORK_DIR / f"q3_table_{name}.parquet")
        print(name, q3.group_by("role").agg(n=pl.len(), overlap=pl.col("overlaps_logged").mean()))
        print(q3["hours"].describe())


if __name__ == "__main__":
    main()
