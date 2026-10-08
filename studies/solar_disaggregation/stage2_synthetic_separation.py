"""Stage 2: separate solar from a synthetic sum of real series, where the solar part is known.

Each aggregate is the real output of one or more solar BMUs plus the real output of a BMU that has
no PV behind it (wind, gas, pumped storage, or a battery), each scaled to a stated share of the
aggregate's 99th-percentile output. The solar share is the solar part's 99th-percentile output
divided by 100 MW, and the non-solar part is scaled to a 99th-percentile absolute output of the
rest. Methods try to recover the solar part from the sum, using irradiance regressors that do not
include the solar sites' own CAMS points (except the labelled oracle).

Methods (the column `method`):

- `separation_cams_regional`: the separation, with the mean of the three grid points nearest to
  the solar sites' capacity-weighted centroid. The primary method.
- `separation_cams_gb_mean`: the same, with the mean of all 18 grid points.
- `separation_cams_own_sites`: the same, with the solar sites' own CAMS points (an oracle).
- `separation_clear_sky`: the same, with CAMS clear-sky irradiance in place of all-sky irradiance,
  which has the sun's position but no cloud.
- `envelope_scaled`: the census's envelope fit to the whole aggregate with the regional CAMS shape.
- `oracle_baseline`: the separation with the non-solar part known exactly.

Run: `uv run python studies/solar_disaggregation/stage2_synthetic_separation.py`.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from envelope import envelope_ac_capacity_mw
from inputs import (
    OUTPUT_DIR,
    VALIDATION_BMUS,
    clear_sky_peak_w_m2,
    grid_point_ids,
    hourly_cams,
    nearest_grid_points,
    on_grid,
    output_on_grid,
    site_cams_point,
    site_coordinates,
    sky_at,
    weather_covariates_on_grid,
    window_half_hours,
)
from studies.pv_physics import SunAndSky
from studies.pv_separation import (
    BaselineFlexibilityType,
    baseline_design,
    separate,
    separate_by_differences,
    solar_basis,
)

SOLAR_SETS: Final[dict[str, tuple[str, ...]]] = {
    "pure_burwell": ("T_BURWS-1",),
    "pure_bishampton": ("C__ESTAT019",),
    "pure_litchardon": ("C__LSTAT020",),
    "pure_three_sites": ("T_BURWS-1", "C__ESTAT019", "C__LSTAT020"),
}
"""The solar halves: BMUs the census calls pure PV, singly and together. These are planned."""
EXPLORATORY_SOLAR_SETS: Final[dict[str, tuple[str, ...]]] = {
    "hybrid_breach": ("T_BRCHS-1",),
    "hybrid_larks_green": ("T_LARKS-1",),
}
"""Hybrid BMUs, whose output includes a battery: exploratory."""
NON_SOLAR_BMUS: Final[dict[str, str]] = {
    "offshore_wind": "T_HOWBO-1",
    "gas_peaker": "T_PEHE-1",
    "gas_baseload": "T_HUMR-1",
    "pumped_storage": "T_DINO-4",
    "battery_lakeside": "T_LKSDB-1",
    "battery_dollymans": "E_DOLLB-1",
}
"""BMUs with no PV behind them, chosen by type before any result: the largest transmission-connected
wind farm, the two largest gas units by 99th-percentile output, the largest pumped-storage unit,
and two of the largest batteries."""
SUPPLIER_BMU: Final[str] = "2__HTGPL000"
"""A supplier BMU for the labelled extra scenario: its true solar share is unknown."""
SHARES: Final[tuple[float, ...]] = (0.1, 0.25, 0.5)
"""Planned solar shares: the solar part's 99th-percentile output over 100 MW."""
AGGREGATE_P99_MW: Final[float] = 100.0
DEMAND_SWING_MW: Final[float] = 15.0
"""The injected weather-correlated demand term's size on a fully dull day, at its daily peak."""
NEAREST_POINTS: Final[int] = 3
PLANNED_FLEXIBILITY: Final[BaselineFlexibilityType] = "seasonal"


@dataclass(frozen=True)
class Regressor:
    """An irradiance regressor on the window grid."""

    sky: SunAndSky
    basis: np.ndarray


def centroid_of(*, bmus: tuple[str, ...], weights: dict[str, float]) -> tuple[float, float]:
    """Return the weighted mean latitude and longitude of the BMUs' sites.

    Args:
        bmus: The BMUs whose sites are averaged.
        weights: Each BMU's weight, keyed by BMU id.

    Returns:
        The weighted mean latitude and the weighted mean longitude, in degrees.
    """
    coordinates = site_coordinates()
    total = sum(weights[b] for b in bmus)
    latitude = sum(coordinates[b][0] * weights[b] for b in bmus) / total
    longitude = sum(coordinates[b][1] * weights[b] for b in bmus) / total
    return latitude, longitude


def regressor_for(
    *,
    point_ids: list[str],
    latitude: float,
    longitude: float,
    ratio: float,
    clear_sky: bool = False,
) -> Regressor:
    """Build the irradiance regressor for a set of CAMS points, as seen from one centroid.

    Args:
        point_ids: The CAMS grid points whose irradiance is averaged.
        latitude: The centroid latitude, in degrees.
        longitude: The centroid longitude, in degrees.
        ratio: The DC:AC ratio assumed in the solar basis.
        clear_sky: Use CAMS clear-sky irradiance instead of all-sky irradiance.

    Returns:
        The sun and sky at the centroid and the solar basis on the window grid.
    """
    hourly = hourly_cams(
        point_ids=point_ids, column="clear_sky_ghi_w_m2" if clear_sky else "ghi_w_m2"
    )
    sky = sky_at(hourly=hourly, latitude=latitude, longitude=longitude)
    basis = on_grid(
        half_hour_end_time=sky.half_hour_end_time, values=solar_basis(sky=sky, dc_ac_ratio=ratio)
    )
    return Regressor(sky=sky, basis=basis)


def stage1_ratio(*, excluded: tuple[str, ...]) -> float:
    """Return the median fitted DC:AC ratio over the stage 1 sites that are not in the aggregate.

    Args:
        excluded: The BMUs in the aggregate, whose stage 1 fits are left out.

    Returns:
        The median fitted DC:AC ratio of the free-orientation stage 1 fits to all the data.
    """
    fits = pl.read_parquet(OUTPUT_DIR / "stage1_fits.parquet").filter(
        (pl.col("arm") == "free") & (pl.col("fold") == -1) & ~pl.col("bmu").is_in(list(excluded))
    )
    return float(fits["dc_ac_ratio"].median())  # ty: ignore[invalid-argument-type]


def _local_hour(*, grid: pl.Series) -> np.ndarray:
    local = grid.dt.convert_time_zone("Europe/London").dt.offset_by("-30m")
    return (local.dt.hour() + local.dt.minute() / 60.0).to_numpy()


def demand_term(*, clear_sky_index: np.ndarray, grid: pl.Series, swing_mw: float) -> np.ndarray:
    """Return a demand that rises on dull days: minus `swing_mw` * (1 - index) * a daytime bump.

    Args:
        clear_sky_index: All-sky over clear-sky irradiance on the window grid, 1 where unknown.
        grid: The end time of each half-hour of the window.
        swing_mw: The size of the term on a fully dull day, at the daily peak, in megawatts.

    Returns:
        The demand term in megawatts at each half-hour of the window; zero or negative.
    """
    hour = _local_hour(grid=grid)
    bump = np.where((hour >= 7) & (hour <= 19), np.sin(np.pi * (hour - 7) / 12.0), 0.0)
    return -swing_mw * np.clip(1.0 - clear_sky_index, 0.0, 1.0) * bump


def _score_rows(
    *, truth: np.ndarray, estimate: np.ndarray, daylight: np.ndarray, month: np.ndarray
) -> tuple[dict, list[dict]]:
    valid = np.isfinite(truth) & np.isfinite(estimate)
    day = valid & daylight
    p99 = float(np.quantile(truth[valid], 0.99))
    errors = np.abs(estimate - truth)
    energy_ok = day.any() and float(estimate[day].sum()) > 0 and float(truth[day].sum()) > 0
    summary = {
        "nmae_of_solar_p99": float(errors[day].mean() / p99) if day.any() and p99 > 0 else None,
        "energy_ratio": float(estimate[valid].sum() / truth[valid].sum())
        if truth[valid].sum() > 0
        else None,
        "estimate_p99_mw": float(np.quantile(estimate[valid], 0.99)),
        "truth_p99_mw": p99,
        "half_hours": int(valid.sum()),
        "correlation": float(np.corrcoef(estimate[day], truth[day])[0, 1])
        if energy_ok and estimate[day].std() > 0
        else None,
        "nmae_rescaled": float(
            np.abs(estimate[day] * truth[day].sum() / estimate[day].sum() - truth[day]).mean() / p99
        )
        if energy_ok
        else None,
    }
    monthly = [
        {
            "month_index": int(m),
            "abs_error_sum": float(errors[day & (month == m)].sum()),
            "rows": int((day & (month == m)).sum()),
            "truth_p99_mw": p99,
        }
        for m in range(12)
    ]
    return summary, monthly


def _solar_set(*, set_name: str) -> tuple[tuple[str, ...], bool]:
    """Return the BMUs of a named solar set, and whether the set is planned, not exploratory."""
    if set_name in SOLAR_SETS:
        return SOLAR_SETS[set_name], True
    return EXPLORATORY_SOLAR_SETS[set_name], False


def _clear_sky_index(*, all_sky: Regressor, clear_sky: Regressor) -> np.ndarray:
    """Return all-sky over clear-sky irradiance on the grid, clipped to 0..1 (1 if unknown)."""
    all_sky_ghi = on_grid(
        half_hour_end_time=all_sky.sky.half_hour_end_time, values=all_sky.sky.ghi_w_m2
    )
    clear_sky_ghi = on_grid(
        half_hour_end_time=clear_sky.sky.half_hour_end_time, values=clear_sky.sky.ghi_w_m2
    )
    index = np.clip(all_sky_ghi / np.where(clear_sky_ghi > 1, clear_sky_ghi, np.nan), 0.0, 1.0)
    return np.nan_to_num(index, nan=1.0)


def _on_basis_rows(*, basis: np.ndarray, solar_mw: np.ndarray) -> np.ndarray:
    """Return the fitted solar part, NaN at the half-hours where the regressor is missing."""
    return np.where(np.isfinite(basis).all(axis=1), solar_mw, np.nan)


def _estimates_for_aggregate(
    *,
    y: np.ndarray,
    other: np.ndarray,
    regressors: dict[str, Regressor],
    designs: dict[str, np.ndarray],
    weather_design: np.ndarray,
    cams_shape_peak: float,
    ratio: float,
) -> dict[str, np.ndarray]:
    """Return every method's estimate of the solar part, keyed `method|flexibility`."""
    estimates: dict[str, np.ndarray] = {}
    for method, regressor in regressors.items():
        flexibilities = (
            ("daily", "seasonal", "monthly")
            if method == "separation_cams_regional"
            else ("seasonal",)
        )
        for flexibility in flexibilities:
            fit = separate(output_mw=y, basis=regressor.basis, design=designs[flexibility])
            if fit is None:
                continue
            estimates[f"{method}|{flexibility}"] = _on_basis_rows(
                basis=regressor.basis, solar_mw=fit.solar_mw
            )
    for method, regressor in regressors.items():
        diff_fit = separate_by_differences(
            output_mw=y, basis=regressor.basis, design=designs["seasonal"]
        )
        if diff_fit is not None:
            estimates[f"{method.replace('separation', 'difference')}|lag8"] = _on_basis_rows(
                basis=regressor.basis, solar_mw=diff_fit.solar_mw
            )
    regional = regressors["separation_cams_regional"]
    weather_fit = separate(output_mw=y, basis=regional.basis, design=weather_design)
    if weather_fit is not None:
        estimates["separation_cams_regional_weather|seasonal"] = _on_basis_rows(
            basis=regional.basis, solar_mw=weather_fit.solar_mw
        )
    oracle = separate(output_mw=y - other, basis=regional.basis, design=np.zeros((len(y), 1)))
    if oracle is not None:
        estimates["oracle_baseline|none"] = _on_basis_rows(
            basis=regional.basis, solar_mw=oracle.solar_mw
        )
    shape = (
        on_grid(half_hour_end_time=regional.sky.half_hour_end_time, values=regional.sky.ghi_w_m2)
        / cams_shape_peak
    )
    ok = np.isfinite(shape) & np.isfinite(y)
    capacity = envelope_ac_capacity_mw(shape=shape[ok], output_mw=y[ok], dc_ac_ratio=ratio)
    if capacity is not None:
        estimates["envelope_scaled|none"] = capacity * np.minimum(ratio * shape, 1.0)
    return estimates


def _scored_rows(
    *,
    scenario: dict,
    truth: np.ndarray,
    estimates: dict[str, np.ndarray],
    daylight: np.ndarray,
    month_index: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    """Score every estimate of one scenario, returning its summary rows and monthly rows."""
    summaries: list[dict] = []
    monthly_rows: list[dict] = []
    for key, estimate in estimates.items():
        method, flexibility = key.split("|")
        summary, monthly = _score_rows(
            truth=truth, estimate=estimate, daylight=daylight, month=month_index
        )
        summaries.append({**scenario, "method": method, "flexibility": flexibility, **summary})
        monthly_rows.extend(
            {**scenario, "method": method, "flexibility": flexibility, **m} for m in monthly
        )
    return summaries, monthly_rows


def run_set(set_name: str) -> tuple[list[dict], list[dict]]:
    """Run every scenario of one solar set, and return its summary rows and monthly rows.

    Args:
        set_name: A key of `SOLAR_SETS` or `EXPLORATORY_SOLAR_SETS`.

    Returns:
        The summary rows and the monthly rows of every scenario of the set.
    """
    grid = window_half_hours()
    month_index = ((grid.dt.offset_by("-1m").dt.month().to_numpy() + 3) % 12).astype(int)
    solar_p99 = {b: float(np.nanquantile(output_on_grid(bmu_id=b), 0.99)) for b in VALIDATION_BMUS}
    members, planned = _solar_set(set_name=set_name)
    others = {**NON_SOLAR_BMUS, "supplier_totalenergies": SUPPLIER_BMU}
    summaries: list[dict] = []
    monthly_rows: list[dict] = []
    designs = {
        flexibility: baseline_design(half_hour_end_time=grid, flexibility=flexibility)
        for flexibility in ("daily", "seasonal", "monthly")
    }
    weather_design = np.hstack([designs["seasonal"], weather_covariates_on_grid()])
    ratio = stage1_ratio(excluded=members)
    latitude, longitude = centroid_of(bmus=members, weights=solar_p99)
    own_points = [site_cams_point(bmu_id=b) for b in members]
    regional_points = nearest_grid_points(
        latitude=latitude, longitude=longitude, count=NEAREST_POINTS
    )
    point_sets = {
        "separation_cams_regional": regional_points,
        "separation_cams_gb_mean": grid_point_ids(),
        "separation_cams_own_sites": own_points,
        "separation_clear_sky": regional_points,
    }
    regressors = {
        method: regressor_for(
            point_ids=point_ids,
            latitude=latitude,
            longitude=longitude,
            ratio=ratio,
            clear_sky=method == "separation_clear_sky",
        )
        for method, point_ids in point_sets.items()
    }
    regional = regressors["separation_cams_regional"]
    sun_up = on_grid(
        half_hour_end_time=regional.sky.half_hour_end_time,
        values=((90.0 - regional.sky.zenith_deg) > 5.0).astype(float),
    )
    daylight = np.nan_to_num(sun_up) > 0
    clear_sky_index = _clear_sky_index(
        all_sky=regional, clear_sky=regressors["separation_clear_sky"]
    )
    cams_shape_peak = clear_sky_peak_w_m2(point_id=own_points[0])
    solar_raw = np.nansum([output_on_grid(bmu_id=b) for b in members], axis=0)
    solar_raw[np.any([~np.isfinite(output_on_grid(bmu_id=b)) for b in members], axis=0)] = np.nan
    for other_name, other_bmu in others.items():
        other_raw = output_on_grid(bmu_id=other_bmu)
        for share in (0.0, *SHARES):
            for weather_demand in (False, True):
                solar = solar_raw * (share * AGGREGATE_P99_MW / np.nanquantile(solar_raw, 0.99))
                scale = (1.0 - share) * AGGREGATE_P99_MW / np.nanquantile(np.abs(other_raw), 0.99)
                other = other_raw * scale
                if weather_demand:
                    other = other + demand_term(
                        clear_sky_index=clear_sky_index, grid=grid, swing_mw=DEMAND_SWING_MW
                    )
                y = solar + other
                truth = solar
                scenario = {
                    "solar_set": set_name,
                    "planned": planned,
                    "non_solar": other_name,
                    "share": share,
                    "weather_demand": weather_demand,
                    "ratio": ratio,
                }
                if share == 0.0:
                    truth = np.where(np.isfinite(y), 0.0, np.nan)
                estimates = _estimates_for_aggregate(
                    y=y,
                    other=other,
                    regressors=regressors,
                    designs=designs,
                    weather_design=weather_design,
                    cams_shape_peak=cams_shape_peak,
                    ratio=ratio,
                )
                scenario_summaries, scenario_monthly = _scored_rows(
                    scenario=scenario,
                    truth=truth,
                    estimates=estimates,
                    daylight=daylight,
                    month_index=month_index,
                )
                summaries.extend(scenario_summaries)
                monthly_rows.extend(scenario_monthly)
    return summaries, monthly_rows


def main() -> None:
    """Run every solar set in its own process and write the score tables."""
    set_names = [*SOLAR_SETS, *EXPLORATORY_SOLAR_SETS]
    with ProcessPoolExecutor(max_workers=len(set_names)) as pool:
        results = list(pool.map(run_set, set_names))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = [row for result in results for row in result[0]]
    monthly = [row for result in results for row in result[1]]
    pl.DataFrame(summaries, infer_schema_length=None).write_parquet(
        OUTPUT_DIR / "stage2_scores.parquet"
    )
    pl.DataFrame(monthly, infer_schema_length=None).write_parquet(
        OUTPUT_DIR / "stage2_monthly.parquet"
    )


if __name__ == "__main__":
    main()
