"""Stages 0 and 3: check the decision rule on real aggregate BMUs, then separate their solar part.

Stage 0 measures how well each aggregate BMU's output fits the envelope model of a solar part: the
share of its half-hours with negative output and its night-time output. A BMU is *clean* when under
5% of its half-hours are negative and its 99th-percentile absolute night-time output is under 5%
of its 99th-percentile output. Any other BMU is *structured*. Both limits were fixed in the plan
before any result.

Stage 3 separates each BMU's solar part with the stage 2 method: three orientations plus a seasonal
calendar baseline, with CAMS irradiance averaged over the three grid points nearest to the BMU's
GSP group centroid. A bootstrap of whole calendar months gives an interval on the total capacity.
There is no ground truth for any aggregate.

Run: `uv run python studies/solar_disaggregation/stage3_real_aggregates.py`.
"""

import json
from concurrent.futures import ProcessPoolExecutor
from typing import Final

import numpy as np
import polars as pl
from envelope import envelope_ac_capacity_mw
from inputs import (
    OUTPUT_DIR,
    REFERENCE_LATITUDE,
    REFERENCE_LONGITUDE,
    bmu_register,
    census_table,
    clear_sky_variance_explained,
    grid_index,
    grid_point_ids,
    hourly_cams,
    nearest_grid_points,
    on_grid,
    output_on_grid,
    sky_at,
    window_half_hours,
)
from pyproj import Transformer
from scipy.optimize import lsq_linear
from shapely.geometry import shape
from studies.pv_fit import change_variance_explained, fit_plant_to_changes
from studies.pv_physics import plant_power_mw, solar_position
from studies.pv_separation import (
    DIFFERENCE_LAG_HALF_HOURS,
    baseline_design,
    separate,
    separate_by_differences,
    solar_basis,
)
from studies.sources import SOLAR_BMU_CENSUS_INPUTS_DIR

MAX_NEGATIVE_SHARE: Final[float] = 0.05
MAX_NIGHT_RATIO: Final[float] = 0.05
NEGATIVE_FLOOR_MW: Final[float] = -0.02
BOOTSTRAPS: Final[int] = 500
START_SHARE: Final[float] = 0.15
"""The physical fit starts from this share of the BMU's 99th-percentile absolute output as AC
capacity, or from the difference separation's capacity when that is larger."""
SEED: Final[int] = 20261008
NEAREST_POINTS: Final[int] = 3


def _separation_ratio() -> float:
    """Return the median DC:AC ratio that stage 1 fitted over the 11 single-site BMUs."""
    fits = pl.read_parquet(OUTPUT_DIR / "stage1_fits.parquet").filter(
        (pl.col("arm") == "free") & (pl.col("fold") == -1)
    )
    return float(fits["dc_ac_ratio"].median())  # ty: ignore[invalid-argument-type]


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


def stage0_row(
    *,
    bmu_id: str,
    generation_capacity_mw: float,
    output: np.ndarray,
    night: np.ndarray,
    cosine: np.ndarray,
    cams_shape: np.ndarray,
) -> dict:
    """Measure one aggregate BMU's output against the envelope model's assumptions.

    Args:
        bmu_id: The aggregate BMU's identifier.
        generation_capacity_mw: The BMU's registered Generation Capacity, in megawatts.
        output: The BMU's output in megawatts on the window grid.
        night: Whether each half-hour is at night.
        cosine: The cosine of the solar zenith angle, clipped at zero, at each half-hour.
        cams_shape: CAMS irradiance as a share of its clear-sky peak at each half-hour.

    Returns:
        The BMU's behaviour class and the statistics behind it.
    """
    finite = np.isfinite(output)
    if finite.sum() == 0 or np.nanquantile(output, 0.99) <= 0:
        return {
            "bmu": bmu_id,
            "generation_capacity_mw": generation_capacity_mw,
            "behaviour": "no positive output",
        }
    p99 = float(np.quantile(output[finite], 0.99))
    negative_share = float((output[finite] < NEGATIVE_FLOOR_MW).mean())
    night_ratio = float(np.quantile(np.abs(output[finite & night]), 0.99) / p99)
    clean = negative_share < MAX_NEGATIVE_SHARE and night_ratio < MAX_NIGHT_RATIO
    ok = finite & ~night
    cams_ok = ok & np.isfinite(cams_shape)
    return {
        "bmu": bmu_id,
        "generation_capacity_mw": generation_capacity_mw,
        "behaviour": "clean" if clean else "structured",
        "p99_output_mw": p99,
        "negative_share": negative_share,
        "night_ratio": night_ratio,
        "envelope_cosine_mw": envelope_ac_capacity_mw(shape=cosine[ok], output_mw=output[ok]),
        "envelope_cams_mw": envelope_ac_capacity_mw(
            shape=cams_shape[cams_ok], output_mw=output[cams_ok]
        ),
    }


def _bootstrap_difference_capacity(
    *, output: np.ndarray, basis: np.ndarray, month: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    """Return a 95% interval on the difference separation's total capacity.

    Each draw picks 12 calendar months with replacement and refits the capacities to the changes
    over `DIFFERENCE_LAG_HALF_HOURS` that end in those months.
    """
    lag = DIFFERENCE_LAG_HALF_HOURS
    usable = np.isfinite(output) & np.isfinite(basis).all(axis=1)
    later = np.flatnonzero(usable[lag:] & usable[:-lag]) + lag
    changes = output[later] - output[later - lag]
    basis_changes = basis[later] - basis[later - lag]
    months = month[later]
    totals = []
    for _ in range(BOOTSTRAPS):
        rows = np.concatenate([np.flatnonzero(months == m) for m in rng.integers(0, 12, size=12)])
        weights = lsq_linear(
            basis_changes[rows], changes[rows], bounds=(0.0, np.inf), method="bvls"
        ).x
        totals.append(float(weights.sum()))
    return float(np.quantile(totals, 0.025)), float(np.quantile(totals, 0.975))


def separate_bmu(bmu_id: str) -> tuple[dict, pl.DataFrame] | None:
    """Separate one aggregate BMU's solar part, by differences and by calendar levels.

    Args:
        bmu_id: The aggregate BMU's identifier.

    Returns:
        The BMU's row of separation results and its half-hourly series, or None when a separation
        cannot be fitted.
    """
    register = bmu_register().filter(pl.col("elexonBmUnit") == bmu_id)
    group = register["gspGroupId"][0]
    latitude, longitude = gsp_group_centroids()[group]
    grid = window_half_hours()
    output = output_on_grid(bmu_id=bmu_id)
    points = nearest_grid_points(latitude=latitude, longitude=longitude, count=NEAREST_POINTS)
    design = baseline_design(half_hour_end_time=grid, flexibility="seasonal")
    month = ((grid.dt.offset_by("-1m").dt.month().to_numpy() + 3) % 12).astype(int)
    ratio = _separation_ratio()
    row: dict[str, object] = {"bmu": bmu_id, "gsp_group": group}
    series: dict[str, object] = {"bmu": bmu_id, "half_hour_end_time": grid, "output_mw": output}
    for name, point_ids in (("regional", points), ("gb_mean", grid_point_ids())):
        sky = sky_at(
            hourly=hourly_cams(point_ids=point_ids), latitude=latitude, longitude=longitude
        )
        basis = on_grid(
            half_hour_end_time=sky.half_hour_end_time,
            values=solar_basis(sky=sky, dc_ac_ratio=ratio),
        )
        usable_basis = np.isfinite(basis).all(axis=1)
        difference = separate_by_differences(output_mw=output, basis=basis, design=design)
        levels = separate(output_mw=output, basis=basis, design=design)
        if difference is None or levels is None:
            return None
        low, high = _bootstrap_difference_capacity(
            output=output, basis=basis, month=month, rng=np.random.default_rng(SEED)
        )
        lag = DIFFERENCE_LAG_HALF_HOURS
        usable = np.isfinite(output) & usable_basis
        later = np.flatnonzero(usable[lag:] & usable[:-lag]) + lag
        changes = output[later] - output[later - lag]
        solar_changes = difference.solar_mw[later] - difference.solar_mw[later - lag]
        solar = np.where(usable_basis, difference.solar_mw, np.nan)
        row |= {
            f"{name}_difference_capacity_mw": difference.total_capacity_mw,
            f"{name}_difference_capacity_low_mw": low,
            f"{name}_difference_capacity_high_mw": high,
            f"{name}_difference_solar_p99_mw": float(np.nanquantile(solar, 0.99)),
            f"{name}_difference_solar_energy_mwh": float(np.nansum(solar) / 2.0),
            f"{name}_variance_share_of_changes": 1.0
            - float(
                ((changes - solar_changes) ** 2).sum() / ((changes - changes.mean()) ** 2).sum()
            ),
            f"{name}_levels_capacity_mw": levels.total_capacity_mw,
        }
        if name == "regional":
            index = grid_index(half_hour_end_time=sky.half_hour_end_time)
            p99 = float(np.nanquantile(np.abs(output), 0.99))
            guess = max(difference.total_capacity_mw, START_SHARE * p99)
            fit = fit_plant_to_changes(
                sky=sky,
                output_mw=output[index],
                orientation="free",
                ac_guess_mw=guess,
                lag_half_hours=lag,
            )
            if fit is not None:
                plant = fit.parameters
                power_sky = plant_power_mw(sky=sky, parameters=plant)
                y_sky = output[index]
                all_sky = change_variance_explained(
                    sky=sky, output_mw=y_sky, power_mw=power_sky, lag_half_hours=lag
                )
                clear_sky = clear_sky_variance_explained(
                    point_ids=point_ids,
                    latitude=latitude,
                    longitude=longitude,
                    output=output,
                    ac_guess_mw=guess,
                    lag_half_hours=lag,
                )
                row |= {
                    "physical_ac_mw": plant.ac_capacity_mw,
                    "physical_dc_mw": plant.dc_capacity_mw,
                    "physical_dc_ac_ratio": plant.dc_ac_ratio,
                    "physical_tilt_deg": plant.tilt_deg,
                    "physical_azimuth_deg": plant.azimuth_deg,
                    "physical_variance_explained": all_sky,
                    "physical_variance_explained_clear_sky": clear_sky,
                    "physical_cloud_increment": all_sky - clear_sky,
                    "physical_solar_p99_mw": float(np.quantile(power_sky, 0.99)),
                    "physical_solar_energy_mwh": float(power_sky.sum() / 2.0),
                }
                series["physical_solar_mw"] = on_grid(
                    half_hour_end_time=sky.half_hour_end_time, values=power_sky
                )
            series |= {
                "solar_mw": solar,
                "baseline_mw": difference.baseline_mw,
                "levels_solar_mw": np.where(usable_basis, levels.solar_mw, np.nan),
            }
    return row, pl.DataFrame(series)


def main() -> None:
    """Write the stage 0 table, then separate every aggregate BMU that has output."""
    aggregates = census_table().filter(pl.col("scope") == "aggregate")
    grid = window_half_hours()
    start_sky = sky_at(
        hourly=hourly_cams(point_ids=grid_point_ids()),
        latitude=REFERENCE_LATITUDE,
        longitude=REFERENCE_LONGITUDE,
    )
    cosine = on_grid(
        half_hour_end_time=start_sky.half_hour_end_time,
        values=np.clip(np.cos(np.radians(start_sky.zenith_deg)), 0.0, None),
    )
    zenith, _ = solar_position(
        stamps=grid.dt.offset_by("-15m"), latitude=REFERENCE_LATITUDE, longitude=REFERENCE_LONGITUDE
    )
    night = zenith > 90.0
    cams_shape = on_grid(
        half_hour_end_time=start_sky.half_hour_end_time, values=start_sky.ghi_w_m2
    ) / float(
        hourly_cams(point_ids=grid_point_ids(), column="clear_sky_ghi_w_m2")["ghi_w_m2"].max()  # ty: ignore[invalid-argument-type]
    )
    stage0 = []
    for row in aggregates.iter_rows(named=True):
        output = output_on_grid(bmu_id=row["elexon_bmu_id"])
        peak = np.nanmax(np.where(np.isfinite(output), cosine, np.nan))
        stage0.append(
            stage0_row(
                bmu_id=row["elexon_bmu_id"],
                generation_capacity_mw=row["generation_capacity_mw"],
                output=output,
                night=night,
                cosine=cosine / peak,
                cams_shape=cams_shape,
            )
        )
    stage0_frame = pl.DataFrame(stage0, infer_schema_length=None)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stage0_frame.write_parquet(OUTPUT_DIR / "stage0_decision_rule.parquet")
    with_output = stage0_frame.filter(pl.col("behaviour") != "no positive output")["bmu"].to_list()
    with ProcessPoolExecutor(max_workers=16) as pool:
        results = [r for r in pool.map(separate_bmu, with_output) if r is not None]
    pl.DataFrame([r[0] for r in results], infer_schema_length=None).write_parquet(
        OUTPUT_DIR / "stage3_separations.parquet"
    )
    pl.concat([r[1] for r in results]).write_parquet(OUTPUT_DIR / "stage3_series.parquet")
    print(stage0_frame)


if __name__ == "__main__":
    main()
