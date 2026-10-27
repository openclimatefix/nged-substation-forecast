"""Step 3 of experiment 00356: one table of every daily case with its plant summary and default 1.

Cases are experiment 00347's hindcast cases (daily runs of 2024-04-01 to 2025-06-16, in its order)
followed by the daily runs of 2025-06-17 to 2026-10-23 (farms W1 to W3 in turn, each in key order),
keeping only cases with an observed power. Columns: the key columns, `site`, `lead_hours`,
`lead_day`, `actual_norm`, `masked` (experiment 210's mask; rows after the mask's end count as not
masked in the table, and `kept` drops them), `kept`, `month` (the run's month), `tuesday`, and
`scored` (the 53 Tuesday runs of `fold_a_tuesdays_v1`). Stores `frame.parquet`, `summary.npy`
(cases, 6), and `d1.npy` (3 seeds, cases, 11).

    PYTHONPATH=<see e00356_common> uv run --no-sync python e00356_frame.py
"""

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_common as c303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00336_common as c336  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00347_s4_data as data  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00356_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

COLUMNS = (*c303.KEY_COLUMNS, "site", "lead_hours", "lead_day", "actual_norm", "masked")


def new_cases() -> tuple[pl.DataFrame, np.ndarray]:
    """Return the usable daily cases after the hindcast and their plant summaries."""
    from datetime import UTC, datetime

    sites = c303.site_table().select("time_series_id", "site")
    c303.TRAIN_END = datetime(2027, 1, 1, tzinfo=UTC)  # observed power beyond the hindcast
    actual = c303.observed_training_power(sites=c303.site_table()).rename({"target": "actual_norm"})
    mask = c336.idle_mask_frame()
    frames, sums = [], []
    for farm in common.FARMS:
        keys = pl.read_parquet(common.INPUTS_DIR / f"keys_{farm}.parquet").join(
            sites, on="time_series_id", validate="m:1"
        )
        summary = np.load(common.INPUTS_DIR / f"summary_{farm}.npy")
        frame = (
            keys.join(actual, on=["time_series_id", "valid_time"], how="left", validate="m:1")
            .join(mask, on=["time_series_id", "valid_time"], how="left", validate="m:1")
            .with_columns(
                masked=pl.col("masked").fill_null(value=False),
                lead_hours=(
                    pl.col("valid_time") - pl.col("power_fcst_init_time")
                ).dt.total_minutes()
                / 60.0,
            )
            .with_columns(
                lead_day=c303.lead_day(
                    power_fcst_init_time=pl.col("power_fcst_init_time"),
                    valid_time=pl.col("valid_time"),
                ).clip(0, common.MAX_LEAD_DAY)
            )
        )
        usable = frame["actual_norm"].is_finite().fill_null(value=False).to_numpy()
        frames.append(frame.filter(pl.Series(usable)).select(COLUMNS))
        sums.append(summary[usable])
    return pl.concat(frames), np.concatenate(sums)


def main() -> None:
    """Assemble the table and join default 1."""
    hind = data.hindcast_frame().select(COLUMNS)
    sum_h = data.summaries(arm_plant="y2", set_name="hindcast")
    if len(sum_h) != hind.height:
        msg = f"hindcast summaries {len(sum_h)} do not match cases {hind.height}"
        raise ValueError(msg)
    new, sum_n = new_cases()
    frame = pl.concat([hind, new])
    summary = np.concatenate([sum_h, sum_n])
    frame = frame.with_columns(
        kept=~pl.col("masked") & (pl.col("valid_time") <= common.MASK_END),
        month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
        tuesday=pl.col("power_fcst_init_time").dt.weekday() == 2,
        row=pl.int_range(pl.len()),
    )
    # Default 1, joined on the key columns.
    d1_keys = pl.read_parquet(common.DATA_DIR / "d1_keys.parquet").with_columns(
        d1_row=pl.int_range(pl.len())
    )
    joined = frame.join(d1_keys, on=list(c303.KEY_COLUMNS), how="left", validate="1:1")
    if joined["d1_row"].null_count():
        msg = f"{joined['d1_row'].null_count()} cases have no default 1 forecast"
        raise ValueError(msg)
    index = joined["d1_row"].to_numpy()
    d1 = np.stack([np.load(common.DATA_DIR / f"d1_s{s}.npy")[index] for s in range(3)])
    # The scored year: experiment 33's Tuesday runs, by key.
    import e00303_score as score303  # isort: skip  # ty: ignore[unresolved-import]

    scored_keys = (
        score303.ordered_cases().select(*c303.KEY_COLUMNS).with_columns(scored=pl.lit(value=True))
    )
    joined = joined.join(scored_keys, on=list(c303.KEY_COLUMNS), how="left", validate="1:1")
    joined = joined.with_columns(scored=pl.col("scored").fill_null(value=False))
    print("scored cases:", int(joined["scored"].sum()), "of", scored_keys.height)
    out = joined.drop("d1_row")
    out.write_parquet(common.DATA_DIR / "frame.parquet")
    np.save(common.DATA_DIR / "summary.npy", summary.astype(np.float32))
    np.save(common.DATA_DIR / "d1.npy", d1.astype(np.float32))
    print(out.height, out["power_fcst_init_time"].min(), out["power_fcst_init_time"].max())


if __name__ == "__main__":
    main()
