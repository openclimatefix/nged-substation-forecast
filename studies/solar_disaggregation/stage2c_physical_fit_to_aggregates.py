"""Stage 2c: infer a plant's tilt, azimuth, DC:AC ratio, and capacity from a synthetic aggregate.

The aggregates are the stage 2 sums of real series. The difference separation first gives a total
solar capacity, which seeds `studies.pv_fit.fit_plant_to_changes`. That fit adjusts tilt, azimuth,
DC capacity, and AC capacity of one plant so that the plant's changes in power over four hours match
the aggregate's changes in output. The reference for a single-site solar half is the direct fit of
the same forward model to that site's own real output, seen through the same regional irradiance, so
the comparison isolates what the aggregate costs. A fit's capacity is compared with the reference
capacity times the solar half's scale factor.

The recovered series is the fitted plant's power. It is scored against the true solar half in
megawatts as a share of the solar half's 99th percentile.

Run after `stage1_single_sites.py`:
`uv run python studies/solar_disaggregation/stage2c_physical_fit_to_aggregates.py`.
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
    clear_sky_variance_explained,
    grid_index,
    grid_point_ids,
    hourly_cams,
    nearest_grid_points,
    on_grid,
    output_on_grid,
    site_coordinates,
    sky_at,
    window_half_hours,
)
from stage2_synthetic_separation import (
    AGGREGATE_P99_MW,
    DEMAND_SWING_MW,
    NEAREST_POINTS,
    NON_SOLAR_BMUS,
    SOLAR_SETS,
    demand_term,
    stage1_ratio,
)
from studies.pv_fit import (
    LOOSE_PRIORS,
    TIGHT_PRIORS,
    FitResult,
    PlantPriors,
    change_variance_explained,
    fit_plant,
    fit_plant_to_changes,
)
from studies.pv_physics import SunAndSky, daylight, plant_power_mw
from studies.pv_separation import (
    DIFFERENCE_LAG_HALF_HOURS,
    baseline_design,
    separate_by_differences,
    solar_basis,
)

SCENARIOS: Final[tuple[tuple[float, bool], ...]] = (
    (0.0, False),
    (0.1, False),
    (0.25, False),
    (0.5, False),
    (0.0, True),
    (0.25, True),
)
"""Solar share of the aggregate, and whether a demand term that rises on dull days is added."""
PRIORS: Final[dict[str, PlantPriors | None]] = {
    "none": None,
    "loose": LOOSE_PRIORS,
    "tight": TIGHT_PRIORS,
}
"""The priors each aggregate is fitted with. `none` is the fit every other table uses."""
REGRESSORS: Final[tuple[str, ...]] = ("regional", "gb_mean")
"""The skies each aggregate is fitted with. `regional` is the mean irradiance of the 3 grid points
nearest to the solar half's centroid, seen from that centroid: more accurate than a real aggregate
can have. `gb_mean` is the mean of all 18 grid points seen from the reference position, the sky
`stage3b_real_controls.py` gives every real BMU."""
START_SHARE: Final[float] = 0.15
"""The fit starts from this share of the aggregate's 99th percentile as AC capacity, or from the
difference separation's capacity when that is larger, so that an aggregate with no solar still
gets a fit and a significance statistic."""


def _fit_columns(
    *,
    fit: FitResult,
    sky: SunAndSky,
    day: np.ndarray,
    aggregate_on_sky: np.ndarray,
    solar_on_sky: np.ndarray,
    share: float,
) -> dict[str, float]:
    """Describe a fitted plant: its parameters, the variance of the change it explains, its error.

    Args:
        fit: The plant fitted to the aggregate's changes.
        sky: The sun and sky the plant is fitted on.
        day: Whether each half-hour of `sky` is in daylight.
        aggregate_on_sky: The aggregate's output at each half-hour of `sky`.
        solar_on_sky: The true solar part at each half-hour of `sky`.
        share: The solar share of the aggregate. The error against the truth is only reported
            when it is above zero.

    Returns:
        The fitted parameters and variance explained, plus the error against the true solar part
        when `share` is above zero.
    """
    plant = fit.parameters
    power = plant_power_mw(sky=sky, parameters=plant)
    lag = DIFFERENCE_LAG_HALF_HOURS
    columns: dict[str, float] = {
        "fitted_tilt_deg": plant.tilt_deg,
        "fitted_azimuth_deg": plant.azimuth_deg,
        "fitted_dc_ac_ratio": plant.dc_ac_ratio,
        "fitted_ac_mw": plant.ac_capacity_mw,
        "variance_explained": change_variance_explained(
            sky=sky, output_mw=aggregate_on_sky, power_mw=power, lag_half_hours=lag
        ),
    }
    if share > 0:
        truth = solar_on_sky
        valid = np.isfinite(truth) & day
        truth_p99 = float(np.nanquantile(truth, 0.99))
        columns |= {
            "nmae_of_solar_p99": float(np.abs(power - truth)[valid].mean() / truth_p99),
            "energy_ratio": float(
                power[np.isfinite(truth)].sum() / truth[np.isfinite(truth)].sum()
            ),
            "correlation": float(np.corrcoef(power[valid], truth[valid])[0, 1]),
        }
    return columns


def run_set(set_name: str) -> list[dict]:
    """Fit every non-solar half and share for one solar set.

    Args:
        set_name: A key of `SOLAR_SETS`.

    Returns:
        One row per non-solar half and scenario, describing the fit and its score.
    """
    members = SOLAR_SETS[set_name]
    grid = window_half_hours()
    coordinates = site_coordinates()
    solar_p99 = {b: float(np.nanquantile(output_on_grid(bmu_id=b), 0.99)) for b in VALIDATION_BMUS}
    weights = np.array([solar_p99[b] for b in members])
    latitude = float(np.average([coordinates[b][0] for b in members], weights=weights))
    longitude = float(np.average([coordinates[b][1] for b in members], weights=weights))
    points = nearest_grid_points(latitude=latitude, longitude=longitude, count=NEAREST_POINTS)
    ratio = stage1_ratio(excluded=members)
    design = baseline_design(half_hour_end_time=grid, flexibility="seasonal")
    solar_raw = np.nansum([output_on_grid(bmu_id=b) for b in members], axis=0)
    solar_raw[np.any([~np.isfinite(output_on_grid(bmu_id=b)) for b in members], axis=0)] = np.nan
    regional_sky = sky_at(
        hourly=hourly_cams(point_ids=points), latitude=latitude, longitude=longitude
    )
    regional_index = grid_index(half_hour_end_time=regional_sky.half_hour_end_time)
    reference = fit_plant(
        sky=regional_sky,
        output_mw=solar_raw[regional_index],
        orientation="free",
        usable=np.isfinite(solar_raw[regional_index]),
    )
    if reference is None:
        raise RuntimeError(f"No reference fit for {set_name}")
    clear_sky = sky_at(
        hourly=hourly_cams(point_ids=points, column="clear_sky_ghi_w_m2"),
        latitude=latitude,
        longitude=longitude,
    )
    index_values = np.clip(
        regional_sky.ghi_w_m2 / np.where(clear_sky.ghi_w_m2 > 1.0, clear_sky.ghi_w_m2, np.nan),
        0.0,
        1.0,
    )
    clear_sky_index = np.nan_to_num(
        on_grid(half_hour_end_time=regional_sky.half_hour_end_time, values=index_values), nan=1.0
    )
    skies = {
        "regional": (points, latitude, longitude),
        "gb_mean": (grid_point_ids(), REFERENCE_LATITUDE, REFERENCE_LONGITUDE),
    }
    prepared = {}
    for regressor, (point_ids, sun_latitude, sun_longitude) in skies.items():
        sky = sky_at(
            hourly=hourly_cams(point_ids=point_ids), latitude=sun_latitude, longitude=sun_longitude
        )
        prepared[regressor] = (
            sky,
            grid_index(half_hour_end_time=sky.half_hour_end_time),
            on_grid(
                half_hour_end_time=sky.half_hour_end_time,
                values=solar_basis(sky=sky, dc_ac_ratio=ratio),
            ),
            daylight(sky=sky),
        )
    rows: list[dict] = []
    for other_name, other_bmu in NON_SOLAR_BMUS.items():
        other_raw = output_on_grid(bmu_id=other_bmu)
        for share, weather_demand in SCENARIOS:
            factor = share * AGGREGATE_P99_MW / np.nanquantile(solar_raw, 0.99)
            solar = solar_raw * factor
            other = other_raw * (
                (1.0 - share) * AGGREGATE_P99_MW / np.nanquantile(np.abs(other_raw), 0.99)
            )
            if weather_demand:
                other = other + demand_term(
                    clear_sky_index=clear_sky_index, grid=grid, swing_mw=DEMAND_SWING_MW
                )
            aggregate = solar + other
            row: dict[str, object] = {
                "solar_set": set_name,
                "non_solar": other_name,
                "share": share,
                "weather_demand": weather_demand,
                "reference_tilt_deg": reference.parameters.tilt_deg,
                "reference_azimuth_deg": reference.parameters.azimuth_deg,
                "reference_dc_ac_ratio": reference.parameters.dc_ac_ratio,
                "reference_ac_mw": reference.parameters.ac_capacity_mw * factor,
            }
            for regressor, (sky, index, basis, day) in prepared.items():
                separation = separate_by_differences(
                    output_mw=aggregate, basis=basis, design=design
                )
                separation_capacity = 0.0 if separation is None else separation.total_capacity_mw
                guess = max(separation_capacity, START_SHARE * AGGREGATE_P99_MW)
                regressor_row = {
                    **row,
                    "regressor": regressor,
                    "separation_capacity_mw": separation_capacity,
                }
                for prior_name, priors in PRIORS.items():
                    if regressor != "regional" and prior_name != "none":
                        continue
                    fit = fit_plant_to_changes(
                        sky=sky,
                        output_mw=aggregate[index],
                        orientation="free",
                        ac_guess_mw=guess,
                        lag_half_hours=DIFFERENCE_LAG_HALF_HOURS,
                        priors=priors,
                    )
                    prior_row = {**regressor_row, "prior": prior_name}
                    if fit is not None:
                        fit_columns = _fit_columns(
                            fit=fit,
                            sky=sky,
                            day=day,
                            aggregate_on_sky=aggregate[index],
                            solar_on_sky=solar[index],
                            share=share,
                        )
                        prior_row |= fit_columns
                        if prior_name == "none":
                            clear = clear_sky_variance_explained(
                                point_ids=skies[regressor][0],
                                latitude=skies[regressor][1],
                                longitude=skies[regressor][2],
                                output=aggregate,
                                ac_guess_mw=guess,
                                lag_half_hours=DIFFERENCE_LAG_HALF_HOURS,
                            )
                            prior_row |= {
                                "variance_explained_clear_sky": clear,
                                "cloud_increment": fit_columns["variance_explained"] - clear,
                            }
                    rows.append(prior_row)
    return rows


def main() -> None:
    """Run the four planned solar sets in parallel and write `stage2c_physical_fit.parquet`."""
    with ProcessPoolExecutor(max_workers=len(SOLAR_SETS)) as pool:
        results = list(pool.map(run_set, list(SOLAR_SETS)))
    frame = pl.DataFrame([row for rows in results for row in rows], infer_schema_length=None)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(OUTPUT_DIR / "stage2c_physical_fit.parquet")
    frame = frame.filter(pl.col("prior") == "none")
    no_solar = frame.filter((pl.col("share") == 0) & ~pl.col("weather_demand"))
    thresholds = no_solar.group_by("regressor").agg(
        pl.col("variance_explained").max().alias("threshold"),
        pl.col("cloud_increment").max().alias("increment_threshold"),
    )
    print("detection thresholds (largest no-solar value):", thresholds)
    print(
        frame.filter((pl.col("share") > 0) & ~pl.col("weather_demand"))
        .join(thresholds, on="regressor")
        .group_by("regressor", "share")
        .agg(
            (pl.col("variance_explained") > pl.col("threshold")).mean().alias("detected"),
            (pl.col("cloud_increment") > pl.col("increment_threshold"))
            .mean()
            .alias("detected_by_increment"),
            (pl.col("fitted_tilt_deg") - pl.col("reference_tilt_deg"))
            .abs()
            .mean()
            .alias("tilt_mae"),
            (pl.col("fitted_azimuth_deg") - pl.col("reference_azimuth_deg"))
            .abs()
            .mean()
            .alias("az_mae"),
            (pl.col("fitted_dc_ac_ratio") - pl.col("reference_dc_ac_ratio"))
            .abs()
            .mean()
            .alias("ratio_mae"),
            (pl.col("fitted_ac_mw") / pl.col("reference_ac_mw") - 1).abs().mean().alias("ac_rel"),
            pl.col("nmae_of_solar_p99").mean(),
            pl.col("energy_ratio").mean(),
        )
        .sort("regressor", "share")
    )


if __name__ == "__main__":
    main()
