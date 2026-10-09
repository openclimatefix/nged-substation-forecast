"""Rung 7: how the fitted solar capacity of the 25 real aggregate BMUs changes with a battery.

The solar study could not tell solar from batteries in its 25 aggregate BMUs (supplier and virtual
BMUs that hold embedded generation). This rung fits rung 6's joint linear programme to each BMU's
real output for a sweep of assumed battery powers. The solar part is a non-negative sum of the four
fleet curves under the same regional sky as the solar study's stage 3 (the three CAMS grid points
nearest to the centroid of the BMU's GSP group). The battery has power `P`, energy `E = 2 h x P`,
and a one-way efficiency of 0.92, with a free starting state of charge in each window of about 4
weeks. `P = 0` is solar only. No ground truth exists for any of these BMUs, so the rung reports how
far the fitted solar moves with the assumed battery, not how close it is to the truth.

Each BMU's calendar replica (the mean output of its month, local half-hour, and day type, laid back
on the BMU's own grid) is fitted the same way. A replica has no cloud, so a fitted solar capacity
that grows with `P` on a replica is the joint model fitting solar that is not there.

Writes `rung7_fits.parquet` (one row per BMU, source, and assumed power) and `rung7_series.parquet`
(the fitted series of `SERIES_BMU`).

Run: `OMP_NUM_THREADS=2 uv run python studies/battery_pv_separation/battery_rung7.py`.
"""

import json
from concurrent.futures import ProcessPoolExecutor
from typing import Final

import numpy as np
import polars as pl
from battery_inputs import OUTPUT_DIR
from battery_synthetic import (
    NEAREST_POINTS,
    hourly_cams,
    nearest_grid_points,
    on_grid,
    output_on_grid,
    sky_at,
    stage1_ratio,
    window_half_hours,
)
from pyproj import Transformer
from shapely.geometry import shape
from studies.battery_joint_lp import fit_joint_solar_battery
from studies.pv_separation import solar_basis
from studies.sources import SOLAR_BMU_CENSUS_INPUTS_DIR, SOLAR_BMU_DISAGGREGATION_DIR

WINDOW_HALF_HOURS: Final[int] = 4 * 7 * 48
ONE_WAY_EFFICIENCY: Final[float] = 0.92
DURATION_HOURS: Final[float] = 2.0
POWERS_MW: Final[tuple[float, ...]] = (0.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)
"""The assumed battery powers. Zero is solar only."""
SOURCES: Final[tuple[str, ...]] = ("real", "replica")
SERIES_BMU: Final[str] = "2__ATGPL000"
"""The big supplier BMU whose fitted series Figure 18 draws."""
SERIES_POWERS_MW: Final[tuple[float, float]] = (0.0, 50.0)
CLOUD_SIGNAL_BMUS: Final[tuple[str, ...]] = (
    "2__ATGPL000",
    "2__HTGPL000",
    "2__BTGPL000",
    "2__LTGPL000",
    "2__KTGPL000",
    "2__DRWED000",
    "V__HFLEX002",
    "2__BECOT003",
    "2__JAXPO000",
)
"""The nine BMUs the solar study found a cloud-correlated component in."""
MAX_WORKERS: Final[int] = 4
HOURS_PER_HALF_HOUR: Final[float] = 0.5


def aggregate_bmus() -> list[str]:
    """Return the 25 aggregate BMUs that the solar study separated.

    Returns:
        Their identifiers, sorted.
    """
    path = SOLAR_BMU_DISAGGREGATION_DIR / "stage3_separations.parquet"
    return sorted(pl.read_parquet(path)["bmu"].to_list())


def gsp_group_centroids() -> dict[str, tuple[float, float]]:
    """Return the latitude and longitude of the centroid of each GSP group's licence area.

    Returns:
        The (latitude, longitude) in degrees, keyed by GSP group name.
    """
    raw = json.loads((SOLAR_BMU_CENSUS_INPUTS_DIR / "raw" / "dno_licence_areas.json").read_text())
    collection = json.loads(raw["body"])
    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    result: dict[str, tuple[float, float]] = {}
    for feature in collection["features"]:
        centroid = shape(feature["geometry"]).centroid
        longitude, latitude = transformer.transform(centroid.x, centroid.y)
        result[feature["properties"]["Name"]] = (latitude, longitude)
    return result


def gsp_group(*, bmu_id: str) -> str:
    """Return a BMU's GSP group from the census's BMU register.

    Args:
        bmu_id: The BMU's identifier.

    Returns:
        The GSP group, such as `_A`.
    """
    raw = json.loads((SOLAR_BMU_CENSUS_INPUTS_DIR / "raw" / "bmunits_all.json").read_text())
    register = pl.DataFrame(json.loads(raw["body"]), infer_schema_length=None)
    return register.filter(pl.col("elexonBmUnit") == bmu_id)["gspGroupId"][0]


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


def run_bmu(bmu_id: str) -> tuple[list[dict], list[dict]]:
    """Fit the sweep of assumed battery powers to one BMU's output and to its calendar replica.

    Args:
        bmu_id: The aggregate BMU's identifier.

    Returns:
        One row per source and power, and the fitted series of `SERIES_BMU` if this is that BMU.
    """
    latitude, longitude = gsp_group_centroids()[gsp_group(bmu_id=bmu_id)]
    points = nearest_grid_points(latitude=latitude, longitude=longitude, count=NEAREST_POINTS)
    sky = sky_at(hourly=hourly_cams(point_ids=points), latitude=latitude, longitude=longitude)
    basis = on_grid(
        half_hour_end_time=sky.half_hour_end_time,
        values=solar_basis(sky=sky, dc_ac_ratio=stage1_ratio(excluded=())),
    )
    real = output_on_grid(bmu_id=bmu_id)
    outputs = {"real": real, "replica": calendar_replica(output=real)}
    rows: list[dict] = []
    series: list[dict] = []
    for source in SOURCES:
        aggregate = outputs[source]
        finite = np.isfinite(aggregate)
        baseline_residual = float("nan")
        for power in POWERS_MW:
            fit = fit_joint_solar_battery(
                aggregate_mw=aggregate,
                solar_basis=basis,
                power_mw=power,
                energy_mwh=DURATION_HOURS * power,
                one_way_efficiency=ONE_WAY_EFFICIENCY,
                window_half_hours=WINDOW_HALF_HOURS,
            )
            residual = float(np.abs(fit.residual_mw[finite]).sum())
            if power == 0.0:
                baseline_residual = residual
            rows.append(
                {
                    "bmu": bmu_id,
                    "source": source,
                    "power_mw": power,
                    "fitted_ac_mw": float(fit.solar_weights.sum()),
                    "solar_energy_mwh": float(fit.solar_mw[finite].sum() * HOURS_PER_HALF_HOUR),
                    "battery_throughput_mwh": float(
                        np.abs(fit.battery_mw[finite]).sum() * HOURS_PER_HALF_HOUR
                    ),
                    "output_energy_mwh": float(
                        np.abs(aggregate[finite]).sum() * HOURS_PER_HALF_HOUR
                    ),
                    "residual_mean_abs_mw": residual / finite.sum(),
                    "residual_relative_to_no_battery": residual / baseline_residual,
                    "p99_abs_output_mw": float(np.quantile(np.abs(aggregate[finite]), 0.99)),
                }
            )
            if bmu_id == SERIES_BMU and source == "real" and power in SERIES_POWERS_MW:
                series.extend(
                    pl.DataFrame(
                        {
                            "power_mw": power,
                            "time": window_half_hours().dt.replace_time_zone(None),
                            "aggregate_mw": aggregate,
                            "fitted_solar_mw": fit.solar_mw,
                            "fitted_battery_mw": fit.battery_mw,
                        }
                    ).to_dicts()
                )
    return rows, series


def main() -> None:
    """Fit all 25 BMUs and write the fits and the series."""
    bmus = aggregate_bmus()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(run_bmu, bmus))
    fits = pl.DataFrame([r for rows, _ in results for r in rows], infer_schema_length=None)
    fits.write_parquet(OUTPUT_DIR / "rung7_fits.parquet")
    pl.DataFrame([r for _, s in results for r in s], infer_schema_length=None).write_parquet(
        OUTPUT_DIR / "rung7_series.parquet"
    )
    print(
        fits.group_by("source", "power_mw")
        .agg(pl.col("fitted_ac_mw").sum())
        .sort("source", "power_mw")
    )


if __name__ == "__main__":
    main()
