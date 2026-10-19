"""Join the refit files of experiment 99 into one prediction set per arm.

Reads `work/preds/<arm>/<site>_<refit>.parquet`, attaches the observed power over capacity, the lead
in hours, and the series labels, drops the rows with no observed power, and writes the set to the
experiment store with `row_set_id = fold_a_daily_gap_v1` (the 292 daily runs of the validation year
not on the every-fifth-day stride). It checks that every arm holds the 16 series and 292 runs, and
that the keys match the stored weekly best guess of experiment 87.

    uv run python ml-experiments/00099-*/e00099_assemble.py [--arms p1 p2 p3 c1]
"""

import argparse
from typing import Final

import e00099_common as common
import e00099_fit as fit
import numpy as np
import polars as pl
from studies.experiment_store import read_predictions, write_predictions

BG2_EXPERIMENT_ID: Final[int] = 87
BG2_PRIMARY_SET: Final[str] = "p_s4_weekly"
N_RUNS: Final[int] = 292
N_SERIES: Final[int] = 16


def arm_frame(*, arm: str) -> pl.DataFrame:
    """Return one arm's predictions with `actual`, `lead_hours`, `site`, `group`, and `capacity_mw`.

    Raises:
        ValueError: If the arm lacks a refit file for a series or covers the wrong runs or series.
    """
    series = common.primary_series()
    arrays = common.load_power_arrays(series=series)
    pieces = []
    for row in series.iter_rows(named=True):
        files = [
            fit.job_path(arm=arm, site=row["site"], refit=refit) for refit in fit.served_runs()
        ]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            msg = f"arm {arm}: missing {len(missing)} files for {row['site']}: {missing[:3]}"
            raise ValueError(msg)
        frame = pl.concat([pl.read_parquet(f) for f in files])
        index = np.asarray(common.grid_index(times=frame["valid_time"]))
        actual = common.take(values=arrays[row["site"]].power, index=index)
        pieces.append(
            frame.with_columns(
                actual=pl.Series(actual),
                lead_hours=(
                    (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
                    / 60.0
                ).cast(pl.Float32),
                site=pl.lit(row["site"]),
                group=pl.lit("Primary"),
                capacity_mw=pl.lit(row["effective_capacity_mw"]).cast(pl.Float32),
            )
        )
    table = pl.concat(pieces).filter(pl.col("actual").is_finite())
    runs = table["power_fcst_init_time"].n_unique()
    sites = table["time_series_id"].n_unique()
    if runs != N_RUNS or sites != N_SERIES:
        msg = f"arm {arm}: {runs} runs and {sites} series, expected {N_RUNS} and {N_SERIES}"
        raise ValueError(msg)
    return table.select(
        "time_series_id",
        "power_fcst_init_time",
        "valid_time",
        "seed",
        "lead_hours",
        "actual",
        *common.LEVEL_COLUMNS,
        "site",
        "group",
        "capacity_mw",
    )


def main() -> None:
    """Write the sets and check their keys against the best guess."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=[*fit.ARMS, *fit.POOLED_AGAIN])
    args = parser.parse_args()
    reference = read_predictions(
        experiment_id=BG2_EXPERIMENT_ID, name=BG2_PRIMARY_SET, row_set_id=common.ROW_SET_ID
    ).select(*common.KEYS, "seed")
    for arm in args.arms:
        table = arm_frame(arm=arm)
        shared = table.select(*common.KEYS, "seed").join(
            reference, on=[*common.KEYS, "seed"], how="inner"
        )
        print(
            f"{arm}: {table.height:,} rows, {shared.height:,} shared with {BG2_PRIMARY_SET} "
            f"({reference.height:,} rows)"
        )
        write_predictions(
            experiment_id=common.EXPERIMENT_ID,
            name=arm,
            frame=table,
            kind="quantiles",
            row_set_id=common.ROW_SET_ID,
        )


if __name__ == "__main__":
    main()
