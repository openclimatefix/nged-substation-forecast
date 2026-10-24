# ruff: noqa: E501
"""Write the 168-to-240-hour rows and arm-2 input matrix of experiment 337 as arrays for the workers.

Rows are experiment 317's stored chain at valid times more than 168 hours after the origin, outside
hand-filtered events and with a reading. The core is the chain's mean (arm D). The level is that of
the same run and hour band at 168 hours or less (the filter's level is constant per run and band),
and the anomaly is zero (mean absolute value 0.009 points at 144 to 168 hours). The weather is the
51-member ensemble mean of experiment 111's store, and the sun's geometry comes from experiment
201's feature cache. Output: `work/prep/*.npy` and `work/prep/rows.parquet`.

    uv run --no-sync python ml-experiments/00337-*/e00337_prep.py
"""

import e00201_common as c201  # ty: ignore[unresolved-import]
import e00201_features as f201  # ty: ignore[unresolved-import]
import e00278_h_common as c278  # ty: ignore[unresolved-import]
import e00278_h_data as data  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.dp_calendar import bank_holiday_names

import e00337_common as own  # isort: skip

STEPS_PER_DAY = 48


def site_rows(*, site: str, cache) -> pl.DataFrame:  # noqa: ANN001
    """Return one substation's long-lead rows with every input column of arm 2."""
    chain = (
        pl.read_parquet(own.CHAIN_DIR / f"{site}.parquet")
        .filter(~pl.col("switching_mask") & pl.col("actual").is_finite())
        .filter(pl.col("lead_hours") > own.STORED_HORIZON_HOURS)
        .with_columns(site=pl.lit(site))
    )
    short = (
        pl.read_parquet(c278.E201_FEATURES / f"{site}.parquet", columns=["run", "hour_band", "level", "lead_hours", "era"])
        .filter((pl.col("era") == "ens") & (pl.col("lead_hours") <= own.STORED_HORIZON_HOURS))
        .group_by("run", "hour_band")
        .agg(pl.col("level").first())
    )  # fmt: skip
    chain = chain.join(short, on=["run", "hour_band"], how="left", validate="m:1")
    assert chain["level"].null_count() == 0, f"{site}: a long-lead row has no level"
    features = cache.sites[site]
    valid = chain["valid_time"].dt.replace_time_zone(None).dt.cast_time_unit("us").to_numpy()
    pos = np.searchsorted(cache.end_times, valid)
    assert (cache.end_times[np.minimum(pos, len(cache.end_times) - 1)] == valid).all(), f"{site}: a valid time is off the cache grid"  # fmt: skip
    step = np.rint(chain["lead_hours"].to_numpy() * 2).astype(np.int64) - 1
    run = chain["run"].to_numpy().astype(np.int64)
    ens_t, ens_sw = f201.ens_means(site=site)
    drv_temp, drv_sw = ens_t[run, step], ens_sw[run, step]
    assert np.isfinite(drv_temp).all() and np.isfinite(drv_sw).all(), f"{site}: ensemble weather has gaps"  # fmt: skip
    return chain.with_columns(
        anomaly=pl.lit(0.0, dtype=pl.Float32),
        drv_temp_c=pl.Series(drv_temp.astype(np.float32)),
        drv_sw_w_m2=pl.Series(drv_sw.astype(np.float32)),
        elevation_deg=pl.Series(features.elevation_deg[pos].astype(np.float32)),
        dark=pl.Series(features.dark[pos]),
    )


def add_calendar(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the frame with experiment 278's calendar columns."""
    holiday_days = pl.Series(sorted(bank_holiday_names())).cast(pl.Date)
    local = pl.col("valid_time").dt.convert_time_zone(data.LONDON)
    return (
        frame.with_columns(
            local_hh=(local.dt.hour().cast(pl.Int32) * 2 + local.dt.minute().cast(pl.Int32) // 30),
            dow=local.dt.weekday().cast(pl.Int32) - 1,
            doy=local.dt.ordinal_day().cast(pl.Int32),
            bank_holiday=local.dt.date().is_in(holiday_days),
            month_key=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
        )
        .with_columns(
            christmas=(local.dt.month() == 12) & (local.dt.day() >= 24)
            | (local.dt.month() == 1) & (local.dt.day() <= 1),
            e7=pl.col("local_hh").is_between(1, 14),
        )
        .sort(["site", "power_fcst_init_time", "valid_time"])
    )


def main() -> None:
    """Build the arrays."""
    cache = c201.load_cache()
    frame = add_calendar(frame=pl.concat([site_rows(site=s, cache=cache) for s in c278.SCORED_SITES]))
    frame = frame.unique(subset=["site", "power_fcst_init_time", "valid_time"], keep="first", maintain_order=True)  # fmt: skip
    matrix = data.design_columns(frame=frame, arm=own_input_arm())
    width = (frame["q95"] - frame["q05"]).to_numpy().astype(np.float64) / 3.29
    scale = np.maximum(width, 0.15)
    own.PREP_DIR.mkdir(parents=True, exist_ok=True)
    np.save(own.PREP_DIR / "x.npy", np.column_stack([matrix, np.log(scale)]).astype(np.float32))
    np.save(own.PREP_DIR / "mean.npy", frame["mean"].to_numpy().astype(np.float64))
    np.save(own.PREP_DIR / "scale.npy", scale)
    np.save(own.PREP_DIR / "actual.npy", frame["actual"].to_numpy().astype(np.float64))
    frame.select("site", "run", "power_fcst_init_time", "valid_time", "lead_hours", "month_key", "period").write_parquet(own.PREP_DIR / "rows.parquet")  # fmt: skip
    print("rows", frame.height, "inputs", matrix.shape[1] + 1)
    print(frame.group_by("period").agg(pl.len(), pl.col("month_key").n_unique().alias("months")))


def own_input_arm() -> int:
    """Return experiment 311's input arm, which fixes the column positions the adapter reads."""
    import e00311_common as c311  # ty: ignore[unresolved-import]

    return c311.INPUT_ARM


if __name__ == "__main__":
    main()
