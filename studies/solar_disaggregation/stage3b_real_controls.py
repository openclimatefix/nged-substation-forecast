"""Stage 3b: run the stage 3 protocol on real BMUs whose technology is known.

The aggregate BMUs have no answer key, so the same protocol runs on two groups that do. The
positive controls are the 11 single-site solar BMUs. The negative controls are the largest BMUs of
other fuel types by 99th-percentile output (wind, gas, pumped storage, hydro, nuclear, biomass) and
the largest batteries. No control gets its position: each uses the mean CAMS irradiance of all 18
grid points and the sun at the centre of Great Britain, which is all a real aggregate offers. The
26 aggregate BMUs run under the same protocol, so the three groups share one scale of comparison.

For each BMU the script records the difference separation's capacity, the fitted plant (AC
capacity, DC:AC ratio, tilt, azimuth), and the share of the variance of four-hour output changes
that the fitted plant explains. A second fit of the same kind uses CAMS's clear-sky irradiance
instead of the all-sky irradiance, so it can follow the daily shape of the output but not cloud.
The cloud increment is the all-sky share minus the clear-sky share.

Each aggregate BMU with output also gets a calendar replica: for every month, half-hour of the day,
and day type (weekday, Saturday, Sunday), the mean output, laid back on the BMU's own time grid.
A replica has the BMU's daily shape and no information about cloud, so it sets the cloud increment
that daily shape alone produces.

Run: `uv run python studies/solar_disaggregation/stage3b_real_controls.py`.
"""

from concurrent.futures import ProcessPoolExecutor
from typing import Final

import numpy as np
import polars as pl
from inputs import (
    OUTPUT_DIR,
    REFERENCE_LATITUDE,
    REFERENCE_LONGITUDE,
    VALIDATION_BMUS,
    bmu_register,
    census_classes,
    census_table,
    clear_sky_variance_explained,
    grid_index,
    grid_point_ids,
    hourly_cams,
    on_grid,
    output_on_grid,
    sky_at,
    window_half_hours,
)
from studies.pv_fit import change_variance_explained, fit_plant_to_changes
from studies.pv_physics import plant_power_mw
from studies.pv_separation import (
    DIFFERENCE_LAG_HALF_HOURS,
    baseline_design,
    separate_by_differences,
    solar_basis,
)

CONTROLS_PER_FUEL: Final[int] = 12
NEGATIVE_FUELS: Final[tuple[str, ...]] = ("WIND", "CCGT", "PS", "NPSHYD", "NUCLEAR", "BIOMASS")
BATTERY_MIN_SWING_MW: Final[float] = 5.0
"""A battery is a BMU of fuel type OTHER or none whose 1st percentile is below minus this and whose
99th percentile is above it. Only a BMU whose identifier starts `T_`, `E_`, or `M_` (one site)
counts, because a supplier or virtual BMU can hold solar."""
START_SHARE: Final[float] = 0.15


def _controls() -> pl.DataFrame:
    """Choose the negative controls: the largest BMUs of each fuel type, and batteries."""
    register = bmu_register().select("elexonBmUnit", "fuelType")
    classes = census_classes().filter(
        (pl.col("behaviour") == "not_solar") & (pl.col("half_hours") > 17000)
    )
    joined = classes.join(register, left_on="elexon_bmu_id", right_on="elexonBmUnit")
    rows = []
    for row in joined.iter_rows(named=True):
        output = output_on_grid(bmu_id=row["elexon_bmu_id"])
        rows.append(
            {
                "bmu": row["elexon_bmu_id"],
                "fuel": row["fuelType"] or "none",
                "p01": float(np.nanquantile(output, 0.01)),
                "p99": float(np.nanquantile(output, 0.99)),
            }
        )
    frame = pl.DataFrame(rows)
    chosen = [
        frame.filter(pl.col("fuel") == fuel).sort("p99", descending=True).head(CONTROLS_PER_FUEL)
        for fuel in NEGATIVE_FUELS
    ]
    batteries = (
        frame.filter(
            pl.col("fuel").is_in(["OTHER", "none"])
            & pl.col("bmu").str.contains("^(T_|E_|M_)")
            & (pl.col("p01") < -BATTERY_MIN_SWING_MW)
            & (pl.col("p99") > BATTERY_MIN_SWING_MW)
        )
        .sort("p99", descending=True)
        .head(CONTROLS_PER_FUEL)
        .with_columns(fuel=pl.lit("BATTERY"))
    )
    return pl.concat([*chosen, batteries]).select("bmu", "fuel")


def calendar_replica(*, output: np.ndarray) -> np.ndarray:
    """Return the mean output of each month, half-hour of the day, and day type, on the same grid.

    Months, half-hours, and day types follow UK local time, because demand follows the clock.

    Args:
        output: The output in megawatts on the window grid. NaN marks a missing half-hour.

    Returns:
        A series on the same grid. Each half-hour holds the mean of the finite outputs that share
        its month, local half-hour of the day, and day type. Half-hours with no output stay NaN.
    """
    local = window_half_hours().dt.offset_by("-15m").dt.convert_time_zone("Europe/London")
    frame = pl.DataFrame(
        {
            "month": local.dt.month(),
            "half_hour": local.dt.hour() * 2 + local.dt.minute() // 30,
            "weekday": local.dt.weekday(),
            "output": output,
        },
        nan_to_null=True,
    )
    frame = frame.with_columns(
        day_type=pl.when(pl.col("weekday") >= 6).then(pl.col("weekday")).otherwise(0)
    )
    mean = pl.col("output").mean().over("month", "half_hour", "day_type")
    replica = frame.select(pl.when(pl.col("output").is_not_null()).then(mean))["output"]
    return replica.fill_null(float("nan")).to_numpy()


def analyse(task: tuple[str, str, str]) -> dict | None:
    """Run the protocol on one BMU: its identifier, its group, and its fuel type.

    Args:
        task: The BMU's identifier, its group, and its fuel type. The group `calendar replica`
            runs the protocol on the calendar replica of the BMU's output.

    Returns:
        A row describing the fitted plant and the share of changes it explains, or None when no
        plant can be fitted.
    """
    bmu_id, group, fuel = task
    grid = window_half_hours()
    output = output_on_grid(bmu_id=bmu_id)
    if group == "calendar replica":
        output = calendar_replica(output=output)
    sky = sky_at(
        hourly=hourly_cams(point_ids=grid_point_ids()),
        latitude=REFERENCE_LATITUDE,
        longitude=REFERENCE_LONGITUDE,
    )
    basis = on_grid(
        half_hour_end_time=sky.half_hour_end_time, values=solar_basis(sky=sky, dc_ac_ratio=1.3)
    )
    design = baseline_design(half_hour_end_time=grid, flexibility="seasonal")
    separation = separate_by_differences(output_mw=output, basis=basis, design=design)
    index = grid_index(half_hour_end_time=sky.half_hour_end_time)
    y_sky = output[index]
    p99 = float(np.nanquantile(np.abs(output), 0.99))
    if separation is None or p99 <= 0:
        return None
    lag = DIFFERENCE_LAG_HALF_HOURS
    fit = fit_plant_to_changes(
        sky=sky,
        output_mw=y_sky,
        orientation="free",
        ac_guess_mw=max(separation.total_capacity_mw, START_SHARE * p99),
        lag_half_hours=lag,
    )
    if fit is None:
        return None
    plant = fit.parameters
    power = plant_power_mw(sky=sky, parameters=plant)
    all_sky = change_variance_explained(
        sky=sky, output_mw=y_sky, power_mw=power, lag_half_hours=lag
    )
    clear_sky = clear_sky_variance_explained(
        point_ids=grid_point_ids(),
        latitude=REFERENCE_LATITUDE,
        longitude=REFERENCE_LONGITUDE,
        output=output,
        ac_guess_mw=max(separation.total_capacity_mw, START_SHARE * p99),
        lag_half_hours=lag,
    )
    return {
        "bmu": bmu_id,
        "group": group,
        "fuel": fuel,
        "p99_abs_output_mw": p99,
        "separation_capacity_mw": separation.total_capacity_mw,
        "ac_mw": plant.ac_capacity_mw,
        "dc_ac_ratio": plant.dc_ac_ratio,
        "tilt_deg": plant.tilt_deg,
        "azimuth_deg": plant.azimuth_deg,
        "variance_explained": all_sky,
        "variance_explained_clear_sky": clear_sky,
        "cloud_increment": all_sky - clear_sky,
    }


def main() -> None:
    """Run every BMU and write `stage3b_real_controls.parquet`."""
    tasks = [(b, "positive control", "SOLAR") for b in VALIDATION_BMUS]
    tasks += [(b, "negative control", f) for b, f in _controls().iter_rows()]
    aggregates = census_table().filter(pl.col("scope") == "aggregate")["elexon_bmu_id"].to_list()
    tasks += [(b, "aggregate", "unknown") for b in aggregates]
    tasks += [(b, "calendar replica", "aggregate replica") for b in aggregates]
    with ProcessPoolExecutor(max_workers=16) as pool:
        results = [r for r in pool.map(analyse, tasks) if r is not None]
    frame = pl.DataFrame(results, infer_schema_length=None)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(OUTPUT_DIR / "stage3b_real_controls.parquet")
    print(
        frame.group_by("group", "fuel")
        .agg(
            pl.len(),
            pl.col("variance_explained").median().alias("median"),
            pl.col("variance_explained").max().alias("largest"),
            pl.col("cloud_increment").median().alias("median_increment"),
            pl.col("cloud_increment").max().alias("largest_increment"),
        )
        .sort("group", "fuel")
    )


if __name__ == "__main__":
    main()
