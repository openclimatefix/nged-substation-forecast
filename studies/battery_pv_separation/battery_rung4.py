"""Rung 4: does a price-based battery regressor help recover solar from a solar-plus-battery sum?

Each synthetic aggregate is a real solar half plus a real battery half (see `battery_synthetic`).
Every arm fits the solar study's physical plant (free tilt, azimuth, DC:AC ratio, and AC capacity)
to the aggregate's four-hour changes, with the same pairs, soft-L1 loss, and starting orientations.
The arms differ only in the extra regressors, whose coefficients are free and signed:

- `A0` solar only (the solar study's physical fit).
- `A1` plus the day-ahead price minus its day's mean, and the within-day price rank.
- `A2` plus the system price minus the day-ahead price, on top of `A1`'s two.
- `A3` plus the true battery half (an oracle, an upper bound).
- `A4` the two regressors of `A1` taken from seven days earlier (a negative control).

Two skies are fitted: `regional` (the mean of the 3 CAMS grid points nearest to the solar centroid,
seen from the centroid) and `gb_mean` (all 18 points, seen from a reference position).

Writes `rung4_fits.parquet` (one row per aggregate, sky, and arm), `rung4_series.parquet` (the
fitted series of the aggregate drawn in the figures), `rung4_intervals.parquet`, and
`report_rung4.md`.

Run: `OMP_NUM_THREADS=2 uv run python studies/battery_pv_separation/battery_rung4.py`.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from battery_inputs import BATTERIES, OUTPUT_DIR
from battery_synthetic import (
    AGGREGATE_P99_MW,
    NEAREST_POINTS,
    REFERENCE_LATITUDE,
    REFERENCE_LONGITUDE,
    SHARES,
    SOLAR_SETS,
    grid_index,
    grid_point_positions,
    hourly_cams,
    nearest_grid_points,
    on_grid,
    output_on_grid,
    price_regressors,
    site_coordinates,
    sky_at,
    stage1_ratio,
    window_half_hours,
)
from studies.pv_fit import (
    RegressorFitResult,
    change_variance_explained,
    fit_plant,
    fit_plant_to_changes_with_regressors,
    lagged_pairs,
)
from studies.pv_physics import SunAndSky, daylight, plant_power_mw
from studies.pv_separation import (
    DIFFERENCE_LAG_HALF_HOURS,
    baseline_design,
    separate_by_differences,
    solar_basis,
)

ARMS: Final[tuple[str, ...]] = ("A0", "A1", "A2", "A3", "A4")
ARM_LABELS: Final[dict[str, str]] = {
    "A0": "Solar only",
    "A1": "Plus price level and rank",
    "A2": "Plus imbalance spread",
    "A3": "Plus the true battery (oracle)",
    "A4": "Plus prices from 7 days earlier (control)",
}
SKIES: Final[tuple[str, ...]] = ("regional", "gb_mean")
START_SHARE: Final[float] = 0.15
"""The fit starts from this share of the aggregate's 99th percentile as AC capacity, or from the
difference separation's capacity when that is larger."""
SHIFT_HALF_HOURS: Final[int] = 7 * 48
"""The negative control takes its prices from this many half-hours earlier."""
SERIES_AGGREGATE: Final[tuple[str, str, float, str]] = (
    "pure_burwell",
    "T_LKSDB-1",
    0.25,
    "regional",
)
"""The aggregate whose fitted series the week figures draw: solar set, battery, share, sky."""
MIN_CHANGE_ENERGY: Final[float] = 1e-3
"""A corrected series whose squared changes sum to less than this has no variance to explain."""
N_RESAMPLES: Final[int] = 2000
RESAMPLE_SEED: Final[int] = 20260
PERCENT: Final[float] = 100.0


def arm_regressors(*, prices: np.ndarray, battery: np.ndarray) -> dict[str, np.ndarray]:
    """Return each arm's regressor matrix on the window grid.

    Args:
        prices: `price_regressors`' array (level, rank, spread).
        battery: The scaled battery half in megawatts, NaN where B1610 has none.

    Returns:
        One array with one row per window half-hour for each arm; `A0` has no columns.
    """
    base = prices[:, :2]
    return {
        "A0": np.empty((len(prices), 0)),
        "A1": base,
        "A2": prices,
        "A3": np.nan_to_num(battery)[:, None],
        "A4": np.roll(base, SHIFT_HALF_HOURS, axis=0),
    }


def _plant_variance_explained(*, sky: SunAndSky, corrected: np.ndarray, power: np.ndarray) -> float:
    """Return the share of the corrected changes' variance the plant explains, NaN if none."""
    lag = DIFFERENCE_LAG_HALF_HOURS
    pairs = lagged_pairs(sky=sky, output_mw=corrected, lag_half_hours=lag)
    change = corrected[pairs] - corrected[pairs - lag]
    if float((change**2).sum()) < MIN_CHANGE_ENERGY:
        return float("nan")
    return change_variance_explained(
        sky=sky, output_mw=corrected, power_mw=power, lag_half_hours=lag
    )


def _fit_arm(
    *,
    sky: SunAndSky,
    aggregate: np.ndarray,
    regressors: np.ndarray,
    guess: float,
) -> tuple[RegressorFitResult, np.ndarray, np.ndarray, float] | None:
    """Fit one arm on one sky and describe it.

    Args:
        sky: The sun and sky.
        aggregate: The aggregate on the window grid.
        regressors: The arm's regressors on the window grid.
        guess: The starting AC capacity.

    Returns:
        The result, the plant's power on `sky`, the regressors' contribution on `sky`, and the
        plant's variance explained on the corrected changes; None if no fit exists.
    """
    index = grid_index(half_hour_end_time=sky.half_hour_end_time)
    output = aggregate[index]
    x_sky = regressors[index]
    result = fit_plant_to_changes_with_regressors(
        sky=sky,
        output_mw=output,
        regressors=np.nan_to_num(x_sky),
        orientation="free",
        ac_guess_mw=guess,
        lag_half_hours=DIFFERENCE_LAG_HALF_HOURS,
    )
    if result is None:
        return None
    power = plant_power_mw(sky=sky, parameters=result.fit.parameters)
    contribution = np.nan_to_num(x_sky) @ result.coefficients
    explained = _plant_variance_explained(sky=sky, corrected=output - contribution, power=power)
    return result, power, contribution, explained


@dataclass(frozen=True)
class SkyInputs:
    """One sky's all-sky and clear-sky sun and sky, and the solar basis the guess uses."""

    sky: SunAndSky
    clear: SunAndSky
    basis: np.ndarray


def _prepare_skies(*, members: tuple[str, ...], solar_raw: np.ndarray) -> dict[str, SkyInputs]:
    """Build the regional and GB-mean skies for a solar set.

    Args:
        members: The set's solar BMUs.
        solar_raw: The set's unscaled solar output on the window grid.

    Returns:
        The inputs of each sky in `SKIES`.
    """
    coordinates = site_coordinates(bmu_ids=members)
    weights = np.array([float(np.nanquantile(output_on_grid(bmu_id=b), 0.99)) for b in members])
    latitude = float(np.average([coordinates[b][0] for b in members], weights=weights))
    longitude = float(np.average([coordinates[b][1] for b in members], weights=weights))
    skies = {
        "regional": (
            nearest_grid_points(latitude=latitude, longitude=longitude, count=NEAREST_POINTS),
            latitude,
            longitude,
        ),
        "gb_mean": (
            grid_point_positions()["point_id"].to_list(),
            REFERENCE_LATITUDE,
            REFERENCE_LONGITUDE,
        ),
    }
    ratio = stage1_ratio(excluded=members)
    prepared = {}
    for name, (point_ids, sun_latitude, sun_longitude) in skies.items():
        sky = sky_at(
            hourly=hourly_cams(point_ids=point_ids), latitude=sun_latitude, longitude=sun_longitude
        )
        clear = sky_at(
            hourly=hourly_cams(point_ids=point_ids, column="clear_sky_ghi_w_m2"),
            latitude=sun_latitude,
            longitude=sun_longitude,
        )
        basis = on_grid(
            half_hour_end_time=sky.half_hour_end_time,
            values=solar_basis(sky=sky, dc_ac_ratio=ratio),
        )
        prepared[name] = SkyInputs(sky=sky, clear=clear, basis=basis)
    return prepared


def _arm_row(
    *,
    fitted: tuple[RegressorFitResult, np.ndarray, np.ndarray, float],
    clear_explained: float,
    truth: np.ndarray,
    valid: np.ndarray,
    reference_ac_mw: float,
) -> dict[str, float]:
    """Describe one fitted arm: parameters, coefficients, variance explained, and solar error.

    Args:
        fitted: `_fit_arm`'s output for the all-sky fit.
        clear_explained: The plant's variance explained in the clear-sky fit of the same arm.
        truth: The true solar half at the sky's half-hours; all NaN-free zero when the share is 0.
        valid: The half-hours of daylight with a true solar value.
        reference_ac_mw: The direct fit's AC capacity, scaled to the aggregate's solar half.

    Returns:
        The columns of the arm's row. The solar-error columns are present when `truth` has output.
    """
    result, power, _, explained = fitted
    plant = result.fit.parameters
    coefficients = [float(c) for c in result.coefficients] + [float("nan")] * 3
    row: dict[str, float] = {
        "fitted_tilt_deg": plant.tilt_deg,
        "fitted_azimuth_deg": plant.azimuth_deg,
        "fitted_dc_ac_ratio": plant.dc_ac_ratio,
        "fitted_ac_mw": plant.ac_capacity_mw,
        "loss": result.fit.loss,
        "variance_explained": explained,
        "variance_explained_clear_sky": clear_explained,
        "cloud_increment": explained - clear_explained,
        "coefficient_1": coefficients[0],
        "coefficient_2": coefficients[1],
        "coefficient_3": coefficients[2],
    }
    finite = np.isfinite(truth)
    truth_p99 = float(np.nanquantile(truth, 0.99))
    if truth_p99 > 0:
        row |= {
            "nmae_of_solar_p99": float(np.abs(power - truth)[valid].mean() / truth_p99),
            "energy_ratio": float(power[finite].sum() / truth[finite].sum()),
            "ac_ratio": plant.ac_capacity_mw / reference_ac_mw,
            "correlation": float(np.corrcoef(power[valid], truth[valid])[0, 1]),
        }
    return row


def _fit_aggregate(
    *,
    inputs: SkyInputs,
    aggregate: np.ndarray,
    solar: np.ndarray,
    regressors: dict[str, np.ndarray],
    guess: float,
    reference_ac_mw: float,
    arms: tuple[str, ...] = ARMS,
) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    """Fit the arms to one aggregate on one sky.

    Args:
        inputs: The sky's inputs.
        aggregate: The aggregate on the window grid.
        solar: The true solar half on the window grid.
        regressors: `arm_regressors`' output.
        guess: The starting AC capacity.
        reference_ac_mw: The direct fit's AC capacity, scaled to the aggregate's solar half.
        arms: The arms to fit, each a key of `regressors`.

    Returns:
        Each arm's row columns, and each arm's fitted plant power and regressor contribution.
    """
    sky = inputs.sky
    truth = solar[grid_index(half_hour_end_time=sky.half_hour_end_time)]
    valid = np.isfinite(truth) & daylight(sky=sky)
    columns, series = {}, {}
    for arm in arms:
        fitted = _fit_arm(sky=sky, aggregate=aggregate, regressors=regressors[arm], guess=guess)
        clear_fit = _fit_arm(
            sky=inputs.clear, aggregate=aggregate, regressors=regressors[arm], guess=guess
        )
        if fitted is None:
            continue
        columns[arm] = _arm_row(
            fitted=fitted,
            clear_explained=float("nan") if clear_fit is None else clear_fit[3],
            truth=truth,
            valid=valid,
            reference_ac_mw=reference_ac_mw,
        )
        series[arm] = np.column_stack([fitted[1], fitted[2]])
    return columns, series


def run_set(set_name: str) -> tuple[list[dict], list[dict]]:
    """Fit every battery, share, sky, and arm for one solar set.

    Args:
        set_name: A key of `SOLAR_SETS`.

    Returns:
        One row per aggregate, sky, and arm, and the fitted series of `SERIES_AGGREGATE`.
    """
    members = SOLAR_SETS[set_name]
    solar_raw = np.nansum([output_on_grid(bmu_id=b) for b in members], axis=0)
    solar_raw[np.any([~np.isfinite(output_on_grid(bmu_id=b)) for b in members], axis=0)] = np.nan
    prepared = _prepare_skies(members=members, solar_raw=solar_raw)
    regional = prepared["regional"].sky
    regional_index = grid_index(half_hour_end_time=regional.half_hour_end_time)
    reference = fit_plant(
        sky=regional,
        output_mw=solar_raw[regional_index],
        orientation="free",
        usable=np.isfinite(solar_raw[regional_index]),
    )
    if reference is None:
        raise RuntimeError(f"No reference fit for {set_name}")
    design = baseline_design(half_hour_end_time=window_half_hours(), flexibility="seasonal")
    prices, _ = price_regressors()
    rows: list[dict] = []
    series: list[dict] = []
    for battery_id in BATTERIES:
        battery_raw = output_on_grid(bmu_id=battery_id)
        for share in SHARES:
            factor = share * AGGREGATE_P99_MW / np.nanquantile(solar_raw, 0.99)
            solar = solar_raw * factor
            battery = battery_raw * (
                (1.0 - share) * AGGREGATE_P99_MW / np.nanquantile(np.abs(battery_raw), 0.99)
            )
            aggregate = solar + battery
            regressors = arm_regressors(prices=prices, battery=battery)
            for sky_name, inputs in prepared.items():
                separation = separate_by_differences(
                    output_mw=aggregate, basis=inputs.basis, design=design
                )
                capacity = 0.0 if separation is None else separation.total_capacity_mw
                reference_ac_mw = reference.parameters.ac_capacity_mw * factor
                columns, fitted_series = _fit_aggregate(
                    inputs=inputs,
                    aggregate=aggregate,
                    solar=solar,
                    regressors=regressors,
                    guess=max(capacity, START_SHARE * AGGREGATE_P99_MW),
                    reference_ac_mw=reference_ac_mw,
                )
                key = {
                    "solar_set": set_name,
                    "battery": battery_id,
                    "share": share,
                    "sky": sky_name,
                }
                rows.extend(
                    {
                        **key,
                        "arm": arm,
                        "separation_capacity_mw": capacity,
                        "reference_ac_mw": reference_ac_mw,
                        **values,
                    }
                    for arm, values in columns.items()
                )
                if (set_name, battery_id, share, sky_name) == SERIES_AGGREGATE:
                    index = grid_index(half_hour_end_time=inputs.sky.half_hour_end_time)
                    series.extend(
                        {
                            "arm": arm,
                            "time": time,
                            "solar_truth_mw": truth,
                            "battery_truth_mw": battery_mw,
                            "aggregate_mw": total,
                            "recovered_solar_mw": plant[0],
                            "regressor_contribution_mw": plant[1],
                        }
                        for arm, parts in fitted_series.items()
                        for time, truth, battery_mw, total, plant in zip(
                            inputs.sky.half_hour_end_time.astype("datetime64[us]").tolist(),
                            solar[index].tolist(),
                            battery[index].tolist(),
                            aggregate[index].tolist(),
                            parts.tolist(),
                            strict=True,
                        )
                    )
    return rows, series


def main() -> None:
    """Fit all aggregates and write the fits and series."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=len(SOLAR_SETS)) as pool:
        results = list(pool.map(run_set, list(SOLAR_SETS)))
    fits = pl.DataFrame([r for rows, _ in results for r in rows], infer_schema_length=None)
    fits.write_parquet(OUTPUT_DIR / "rung4_fits.parquet")
    series = pl.DataFrame([r for _, s in results for r in s], infer_schema_length=None)
    series.write_parquet(OUTPUT_DIR / "rung4_series.parquet")
    print(fits.group_by("sky", "arm").len())


if __name__ == "__main__":
    main()
