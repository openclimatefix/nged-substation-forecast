# ruff: noqa: E501
"""Build the master frame of experiment 266: experiment 251's rows with weather and per-row scores.

Joins experiment 251's scored rows to experiment 178's realised (CAMS) and ensemble-mean
clear-sky information, then adds each stored arm's per-row quantile score and median.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00266-*/e00266_build.py
"""

import numpy as np
import polars as pl

import e00266_common as common  # isort: skip


def main() -> None:
    """Write `work/master.parquet`."""
    rows = pl.read_parquet(common.E251_SCORED / "rows.parquet")
    weather = pl.read_parquet(
        common.E178_ROWS,
        columns=["site", "power_fcst_init_time", "valid_time", "ghi", "clear", "csi", "hour"],
    )
    keys = ["site", "power_fcst_init_time", "valid_time"]
    master = rows.join(weather, on=keys, how="left", validate="1:1")
    assert master["ghi"].null_count() == 0, "weather join lost rows"
    actual = master["actual"].to_numpy()
    frame = master.select(
        "row", "site", "time_series_id", "run", "power_fcst_init_time", "valid_time", "lead_hours",
        "actual", "unit", "sunny", "daylight", "ghi", "clear", "csi", "hour", "month", "valid_month", "mean_d",
    )  # fmt: skip
    extra: dict[str, np.ndarray] = {}
    for arm in (*common.STACK_ARMS, "manual_heuristic"):
        q = np.load(common.E251_SCORED / f"{arm}.npy").astype(np.float64)
        assert q.shape == (master.height, 11), arm
        extra[f"qs_{arm}"] = common.qs_per_row(actual=actual, q=q)
        extra[f"med_{arm}"] = q[:, 5]
    realised = np.where(
        master["clear"].to_numpy() > 20.0,
        master["ghi"].to_numpy()
        / np.where(master["clear"].to_numpy() > 20.0, master["clear"].to_numpy(), 1.0),
        np.nan,
    )
    frame = frame.with_columns(
        [pl.Series(k, v) for k, v in extra.items()],
        realised_csi=pl.Series(realised),
        init_month=master["power_fcst_init_time"].dt.strftime("%Y-%m"),
        init_day=master["power_fcst_init_time"].dt.strftime("%Y-%m-%d"),
        valid_local_month=master["valid_time"]
        .dt.convert_time_zone("Europe/London")
        .dt.strftime("%Y-%m"),
    )
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(common.WORK_DIR / "master.parquet")
    print(frame.shape)
    for col in ("month", "init_month", "valid_local_month"):
        gap = (
            frame.group_by(col)
            .agg((pl.col("qs_arm1") - pl.col("qs_bg2")).mean().alias("gap"), pl.len())
            .sort(col)
        )
        print(col, gap.head(12).to_dicts() if col == "month" else "")
    print(
        frame.group_by("init_month")
        .agg((pl.col("qs_arm1") - pl.col("qs_bg2")).mean().alias("gap"))
        .sort("init_month")
    )


if __name__ == "__main__":
    main()
