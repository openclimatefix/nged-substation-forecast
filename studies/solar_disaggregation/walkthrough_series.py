"""Save the half-hourly series that the walk-through figures draw.

Two groups of series. First, for two pure-PV BMUs, the forward model's steps: CAMS irradiance, the
direct and diffuse split, the irradiance on the fitted plane, and the predicted and measured power.
Second, for four synthetic aggregates, every piece of the separation: the real solar BMUs that are
summed (the truth), the real non-solar BMU they are summed with, the aggregate, the solar curves
the separation fits to, and the separation's recovered solar and calendar baseline, beside the
census-style envelope curve and the clear-sky separation.

Run after `stage1_single_sites.py`:
`uv run python studies/solar_disaggregation/walkthrough_series.py`.
"""

from typing import Final

import numpy as np
import polars as pl
from envelope import envelope_ac_capacity_mw
from inputs import (
    OUTPUT_DIR,
    VALIDATION_BMUS,
    align_output,
    clear_sky_peak_w_m2,
    grid_index,
    hourly_cams,
    nearest_grid_points,
    on_grid,
    output_by_bmu,
    output_on_grid,
    settled_mask,
    site_cams_point,
    site_coordinates,
    sky_at,
    window_half_hours,
)
from stage2_synthetic_separation import (
    AGGREGATE_P99_MW,
    NEAREST_POINTS,
    NON_SOLAR_BMUS,
    SOLAR_SETS,
    centroid_of,
    regressor_for,
    stage1_ratio,
)
from studies.pv_fit import fit_plant, fit_plant_to_changes
from studies.pv_physics import plane_of_array_w_m2, plant_power_mw
from studies.pv_separation import (
    DIFFERENCE_LAG_HALF_HOURS,
    FLEET_ORIENTATIONS,
    baseline_design,
    separate,
    separate_by_differences,
)

WALKTHROUGH_SITES: Final[tuple[str, ...]] = ("T_BURWS-1", "C__ESTAT019")
"""The two pure-PV BMUs whose forward-model steps are drawn."""
WALKTHROUGH_SCENARIOS: Final[dict[str, tuple[str, str, float]]] = {
    "three_pure_pv_plus_offshore_wind_25": ("pure_three_sites", "offshore_wind", 0.25),
    "three_pure_pv_plus_offshore_wind_10": ("pure_three_sites", "offshore_wind", 0.10),
    "three_pure_pv_plus_battery_25": ("pure_three_sites", "battery_lakeside", 0.25),
    "three_pure_pv_plus_gas_25": ("pure_three_sites", "gas_baseload", 0.25),
}
"""Scenario name: (solar set, non-solar half, solar share of the aggregate's 99th percentile)."""


def site_series(*, bmu_id: str) -> pl.DataFrame:
    """Return one BMU's forward-model steps at every half-hour, fitted with the free arm.

    Args:
        bmu_id: The validation BMU's identifier.

    Returns:
        One row per half-hour: irradiance, the plane's irradiance, and the predicted and measured
        power.
    """
    latitude, longitude = site_coordinates()[bmu_id]
    point = site_cams_point(bmu_id=bmu_id)
    hourly = hourly_cams(point_ids=[point])
    sky = sky_at(hourly=hourly, latitude=latitude, longitude=longitude)
    output = align_output(sky=sky, output=output_by_bmu(bmu_id=bmu_id))
    settled = settled_mask(sky=sky, output_mw=output)
    fit = fit_plant(sky=sky, output_mw=output, orientation="free", usable=settled)
    if fit is None:
        raise RuntimeError(f"No fit for {bmu_id}")
    plant = fit.parameters
    poa = plane_of_array_w_m2(
        sky=sky, tilt_deg=plant.tilt_deg, azimuth_deg=plant.azimuth_deg, tracker=False
    )
    return pl.DataFrame(
        {
            "bmu": bmu_id,
            "half_hour_end_time": pl.Series(sky.half_hour_end_time)
            .cast(pl.Datetime("us"))
            .dt.replace_time_zone("UTC"),
            "ghi_w_m2": sky.ghi_w_m2,
            "dni_w_m2": sky.dni_w_m2,
            "dhi_w_m2": sky.dhi_w_m2,
            "zenith_deg": sky.zenith_deg,
            "poa_w_m2": poa,
            "fitted_mw": plant_power_mw(sky=sky, parameters=plant),
            "measured_mw": output,
            "used_in_fit": settled,
            "tilt_deg": plant.tilt_deg,
            "azimuth_deg": plant.azimuth_deg,
            "dc_capacity_mw": plant.dc_capacity_mw,
            "ac_capacity_mw": plant.ac_capacity_mw,
        }
    )


def scenario_series(*, name: str, solar_set: str, other_name: str, share: float) -> pl.DataFrame:
    """Return every piece of one synthetic aggregate and its separation, on the window grid.

    Args:
        name: The scenario's name.
        solar_set: A key of `SOLAR_SETS`.
        other_name: A key of `NON_SOLAR_BMUS`.
        share: The solar share of the aggregate.

    Returns:
        One row per half-hour: the aggregate, its two halves, the solar curves, and the separations.
    """
    grid = window_half_hours()
    members = SOLAR_SETS[solar_set]
    solar_p99 = {b: float(np.nanquantile(output_on_grid(bmu_id=b), 0.99)) for b in VALIDATION_BMUS}
    ratio = stage1_ratio(excluded=members)
    latitude, longitude = centroid_of(bmus=members, weights=solar_p99)
    points = nearest_grid_points(latitude=latitude, longitude=longitude, count=NEAREST_POINTS)
    regional = regressor_for(point_ids=points, latitude=latitude, longitude=longitude, ratio=ratio)
    clear = regressor_for(
        point_ids=points, latitude=latitude, longitude=longitude, ratio=ratio, clear_sky=True
    )
    members_output = {b: output_on_grid(bmu_id=b) for b in members}
    solar_raw = np.nansum(list(members_output.values()), axis=0)
    solar_raw[np.any([~np.isfinite(o) for o in members_output.values()], axis=0)] = np.nan
    solar_factor = share * AGGREGATE_P99_MW / np.nanquantile(solar_raw, 0.99)
    other_raw = output_on_grid(bmu_id=NON_SOLAR_BMUS[other_name])
    other_factor = (1.0 - share) * AGGREGATE_P99_MW / np.nanquantile(np.abs(other_raw), 0.99)
    solar, other = solar_raw * solar_factor, other_raw * other_factor
    aggregate = solar + other
    design = baseline_design(half_hour_end_time=grid, flexibility="seasonal")
    columns: dict[str, object] = {
        "scenario": name,
        "half_hour_end_time": grid,
        "solar_truth_mw": solar,
        "non_solar_mw": other,
        "aggregate_mw": aggregate,
        "regional_ghi_w_m2": on_grid(
            half_hour_end_time=regional.sky.half_hour_end_time, values=regional.sky.ghi_w_m2
        ),
    }
    for index, (orientation, _) in enumerate(FLEET_ORIENTATIONS):
        columns[f"basis_{orientation}"] = regional.basis[:, index]
    for label, regressor in (("separation", regional), ("separation_clear_sky", clear)):
        usable = np.isfinite(regressor.basis).all(axis=1)
        fit = separate(output_mw=aggregate, basis=regressor.basis, design=design)
        if fit is not None:
            columns[f"{label}_solar_mw"] = np.where(usable, fit.solar_mw, np.nan)
        diff = separate_by_differences(output_mw=aggregate, basis=regressor.basis, design=design)
        if diff is None:
            continue
        columns[f"difference{label.removeprefix('separation')}_solar_mw"] = np.where(
            usable, diff.solar_mw, np.nan
        )
        if label == "separation":
            columns["difference_baseline_mw"] = diff.baseline_mw
            columns["difference_residual_mw"] = diff.residual_mw
            for index, (orientation, _) in enumerate(FLEET_ORIENTATIONS):
                columns[f"weight_{orientation}_mw"] = float(diff.capacities_mw[index])
    index = grid_index(half_hour_end_time=regional.sky.half_hour_end_time)
    guess = max(
        float(columns.get("weight_south_mw", 0.0))  # ty: ignore[invalid-argument-type]
        + sum(
            float(columns.get(f"weight_{o}_mw", 0.0))  # ty: ignore[invalid-argument-type]
            for o in ("east", "west", "tracker")
        ),
        0.15 * AGGREGATE_P99_MW,
    )
    fit = fit_plant_to_changes(
        sky=regional.sky,
        output_mw=aggregate[index],
        orientation="free",
        ac_guess_mw=guess,
        lag_half_hours=DIFFERENCE_LAG_HALF_HOURS,
    )
    reference = fit_plant(
        sky=regional.sky,
        output_mw=solar_raw[index],
        orientation="free",
        usable=np.isfinite(solar_raw[index]),
    )
    if fit is not None and reference is not None:
        columns["physical_solar_mw"] = on_grid(
            half_hour_end_time=regional.sky.half_hour_end_time,
            values=plant_power_mw(sky=regional.sky, parameters=fit.parameters),
        )
        for prefix, plant, factor in (
            ("fitted", fit.parameters, 1.0),
            ("reference", reference.parameters, solar_factor),
        ):
            columns[f"{prefix}_tilt_deg"] = plant.tilt_deg
            columns[f"{prefix}_azimuth_deg"] = plant.azimuth_deg
            columns[f"{prefix}_dc_ac_ratio"] = plant.dc_ac_ratio
            columns[f"{prefix}_ac_mw"] = plant.ac_capacity_mw * factor
    shape = on_grid(
        half_hour_end_time=regional.sky.half_hour_end_time, values=regional.sky.ghi_w_m2
    )
    shape = shape / clear_sky_peak_w_m2(point_id=site_cams_point(bmu_id=members[0]))
    ok = np.isfinite(shape) & np.isfinite(aggregate)
    capacity = envelope_ac_capacity_mw(shape=shape[ok], output_mw=aggregate[ok], dc_ac_ratio=ratio)
    columns["envelope_solar_mw"] = (
        np.nan if capacity is None else capacity * np.minimum(ratio * shape, 1.0)
    )
    return pl.DataFrame(columns)


def main() -> None:
    """Write `walkthrough_sites.parquet` and `walkthrough_scenarios.parquet`."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.concat([site_series(bmu_id=b) for b in WALKTHROUGH_SITES]).write_parquet(
        OUTPUT_DIR / "walkthrough_sites.parquet"
    )
    pl.concat(
        [
            scenario_series(name=name, solar_set=s, other_name=o, share=share)
            for name, (s, o, share) in WALKTHROUGH_SCENARIOS.items()
        ],
        how="diagonal",
    ).write_parquet(OUTPUT_DIR / "walkthrough_scenarios.parquet")


if __name__ == "__main__":
    main()
