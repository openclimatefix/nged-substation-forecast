"""Checks 2, 5, and 6 of experiment 282: flat-top level, free orientation, and seasonal shape error.

Run under `systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue` with
`uv run --no-sync python`. Four worker processes, one thread each.
"""

import itertools
import multiprocessing
import os
from typing import Final

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import e00282_common as common
import polars as pl
from studies.pv_physics import plane_of_array_w_m2
from studies.pv_plant_fit import (
    fit_scale_and_clip,
    fraction_of_output_fitted,
    plant_output,
    unclipped_unit_power,
)

N_PROCESSES: Final[int] = 4
TILTS_DEG: Final[tuple[float, ...]] = tuple(float(t) for t in range(0, 61, 5))
AZIMUTHS_DEG: Final[tuple[float, ...]] = tuple(float(a) for a in range(90, 271, 10))
EAST_WEST_TILT_DEG: Final[float] = 15.0
FLAT_TOP_POA_W_M2: Final[float] = 900.0
SEASONAL_POA_RANGE_W_M2: Final[tuple[float, float]] = (400.0, 900.0)


def flat_top_table() -> pl.DataFrame:
    """Return check 2: the 99th percentile of output over capacity where the plane is above 900."""
    frame = pl.read_parquet(common.FRAME_PATH).filter(
        pl.col("scored") & (pl.col("poa_fleet_w_m2") >= FLAT_TOP_POA_W_M2)
    )
    columns = ["output", *common.MODELS]
    return (
        frame.group_by("site")
        .agg([pl.col(c).quantile(0.99).alias(f"p99_{c}") for c in columns] + [pl.len().alias("n")])
        .sort("site")
    )


def _orientation_task(farm: str) -> list[dict[str, float | str]]:
    """Fit the plant at every tilt and azimuth for one farm and return the fit and its score."""
    frame = pl.read_parquet(common.FRAME_PATH).filter(pl.col("site") == farm).sort("time")
    tuning = frame["tuning"].to_numpy()
    scored = frame["scored"].to_numpy()
    sky = common.sky_of(frame)
    output = frame["output"].to_numpy()
    temperature = frame["temperature_c"].to_numpy()
    candidates = {
        (tilt, azimuth): plane_of_array_w_m2(
            sky=sky, tilt_deg=tilt, azimuth_deg=azimuth, tracker=False
        )
        for tilt, azimuth in itertools.product(TILTS_DEG, AZIMUTHS_DEG)
    }
    east_west = 0.5 * (
        plane_of_array_w_m2(sky=sky, tilt_deg=EAST_WEST_TILT_DEG, azimuth_deg=90.0, tracker=False)
        + plane_of_array_w_m2(
            sky=sky, tilt_deg=EAST_WEST_TILT_DEG, azimuth_deg=270.0, tracker=False
        )
    )
    rows = []
    for (tilt, azimuth), poa in [*candidates.items(), ((-1.0, -1.0), east_west)]:
        unclipped = unclipped_unit_power(
            poa_w_m2=poa, temperature_c=temperature, heat_per_kelvin=0.0
        )
        fit = fit_scale_and_clip(unclipped=unclipped[tuning], output=output[tuning])
        predicted = plant_output(unclipped=unclipped, scale=fit.scale, ac_limit=fit.ac_limit)
        rows.append(
            {
                "site": farm,
                "tilt_deg": tilt,
                "azimuth_deg": azimuth,
                "tuning_sse": fit.squared_error,
                "ac_limit": fit.ac_limit,
                "scale": fit.scale,
                "scored_fraction_fitted": fraction_of_output_fitted(
                    output=output[scored], predicted=predicted[scored]
                ),
            }
        )
    return rows


def orientation_table() -> pl.DataFrame:
    """Return check 5: every orientation per farm; `tilt_deg == -1` is the east-west pair."""
    with multiprocessing.Pool(N_PROCESSES) as pool:
        results = pool.map(_orientation_task, list(common.FARMS))
    return pl.DataFrame([row for part in results for row in part])


def seasonal_ratio_table() -> pl.DataFrame:
    """Return check 6: measured over modelled output by calendar month on clear half-hours.

    For each engine (the stack's term and the per-farm plant fit) the ratio is the sum of measured
    output over the sum of the engine's output, on half-hours with a clear-sky index of at least
    0.9 and the fleet plane's irradiance in 400 to 900 W/m², in all years of data (tuning and
    scored), then divided by the farm's all-month ratio. A value below 1 means the engine
    over-predicts in that month relative to the rest of the year.
    """
    low, high = SEASONAL_POA_RANGE_W_M2
    frame = pl.read_parquet(common.FRAME_PATH).filter(
        (pl.col("clear_index") >= common.CLEAR_INDEX)
        & (pl.col("poa_fleet_w_m2") >= low)
        & (pl.col("poa_fleet_w_m2") < high)
    )
    parts = []
    for model in ("stack_term", "plant_per_farm"):
        monthly = frame.group_by("site", "calendar_month").agg(
            ratio=pl.col("output").sum() / pl.col(model).sum(), n=pl.len()
        )
        overall = frame.group_by("site").agg(overall=pl.col("output").sum() / pl.col(model).sum())
        parts.append(
            monthly.join(overall, on="site", validate="m:1").with_columns(
                relative_ratio=pl.col("ratio") / pl.col("overall"), model=pl.lit(model)
            )
        )
    return pl.concat(parts).sort("model", "site", "calendar_month")


def main() -> None:
    """Write the three tables and print their headlines."""
    flat = flat_top_table()
    orientation = orientation_table()
    seasonal = seasonal_ratio_table()
    flat.write_csv(common.DATA_DIR / "check2_flat_top.csv")
    orientation.write_csv(common.DATA_DIR / "check5_orientation.csv")
    seasonal.write_csv(common.DATA_DIR / "check6_seasonal_ratio.csv")
    pl.Config.set_tbl_rows(40)
    print(flat)
    print(orientation.sort("tuning_sse").group_by("site", maintain_order=True).first().sort("site"))
    print(
        seasonal.group_by("model", "calendar_month")
        .agg(pl.col("relative_ratio").median(), pl.col("n").min())
        .sort("model", "calendar_month")
    )


if __name__ == "__main__":
    main()
