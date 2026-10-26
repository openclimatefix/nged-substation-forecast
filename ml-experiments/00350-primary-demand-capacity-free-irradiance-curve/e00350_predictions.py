"""Write the scored-year predictions of arms A, B, and C with `experiment_store.write_predictions`.

Columns: `time_series_id`, `power_fcst_init_time`, `valid_time`, `lead_hours`, `q01` to `q99`,
`actual` (points of effective capacity). A set that already exists is skipped.

    PYTHONPATH=ml-experiments/00350-primary-demand-capacity-free-irradiance-curve \
        uv run --no-sync python ml-experiments/00350-*/e00350_predictions.py
"""

import e00350_common as common
import e00350_rows as rows
import polars as pl

from studies import experiment_store

ROW_SET_ID = "fold_a_daily_gap_v1"


def main() -> None:
    """Write one prediction set per arm."""
    for arm in common.ARMS:
        frame = rows.load_arm(arm=arm, period="scored").select(
            "time_series_id",
            "power_fcst_init_time",
            "valid_time",
            "lead_hours",
            *common.QUANTILES,
            "actual",
        )
        try:
            folder = experiment_store.write_predictions(
                experiment_id=common.EXPERIMENT_ID,
                name=f"arm-{arm.lower()}",
                frame=frame.with_columns(pl.col("lead_hours").cast(pl.Float64)),
                kind="quantiles",
                row_set_id=ROW_SET_ID,
            )
            print(f"arm {arm}: {frame.height:,} rows written to {folder}")
        except FileExistsError:
            print(f"arm {arm}: set exists, skipped")


if __name__ == "__main__":
    main()
