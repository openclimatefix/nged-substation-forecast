"""Rung 6: a joint solar and state-of-charge battery model, fitted by one linear programme.

The aggregates, skies, and scores are rung 4's. The aggregate is modelled as solar plus a battery:
solar is a non-negative weighted sum of the four fleet curves (east, south, west, and tracker) of
`pv_separation.solar_basis`, and the battery is a free charge and discharge schedule bounded by a
power limit `P`, an energy capacity `E`, and a one-way efficiency of 0.92 (see
`studies.battery_joint_lp`). The fit minimises the absolute residual in 4-week windows, each with
a free starting state of charge, with the solar weights shared by all windows.

- `A8` the true battery's sizes: `P` is its 99th-percentile power and `E` is rung 3's fitted
  energy capacity, both scaled by the share's scale factor (an oracle on size).
- `A8h` the same with half the energy capacity.
- `A8d` the same with double the energy capacity.
- `A9` a battery of 2 hours at its true power, the typical duration.
- `A0b` the same solar model (the four fleet curves) fitted to the aggregate with no battery
  (power 0): the solar-only baseline of the joint model's own solar model.
- `A1b` the same solar model fitted to the aggregate with no battery, plus rung 4's two price
  columns (the day-ahead price minus its day's mean, and the within-day rank) as free signed
  regressors: rung 4's `A1` with the joint model's own solar model, so the joint model and its
  price-regressor rival differ only in the battery's state-of-charge constraints.
- `A3b` the same solar model fitted to the aggregate minus the true battery: the oracle of the joint
  model's own solar model.
- `A8_lam1`, `A8_lam2`, `A8_lam3` `A8` with a penalty on the half-hour change of the battery's
  output (regional sky only).

Rung 4's A0, A1, and A3 are read from `rung4_fits.parquet`, not refitted.

Writes `rung6_fits.parquet` (one row per aggregate, sky, and arm) and `rung6_series.parquet` (the
fitted series of the aggregate the figures draw).

Run: `OMP_NUM_THREADS=2 uv run python studies/battery_pv_separation/battery_rung6.py`.
"""

from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from typing import Final

import numpy as np
import polars as pl
from battery_inputs import BATTERIES, OUTPUT_DIR
from battery_rung4 import (
    SERIES_AGGREGATE,
    SKIES,
    SkyInputs,
    _plant_variance_explained,
    _prepare_skies,
)
from battery_synthetic import (
    AGGREGATE_P99_MW,
    SHARES,
    SOLAR_SETS,
    grid_index,
    output_on_grid,
    price_regressors,
    window_half_hours,
)
from studies.battery_joint_lp import JointFit, fit_joint_solar_battery
from studies.pv_fit import fit_plant
from studies.pv_physics import daylight

WINDOW_HALF_HOURS: Final[int] = 4 * 7 * 48
"""The state of charge starts free in each window of about 4 weeks."""
ONE_WAY_EFFICIENCY: Final[float] = 0.92
TYPICAL_DURATION_HOURS: Final[float] = 2.0
FALSE_ALARM_SHARE: Final[float] = 0.02
"""Capacity rule for B4 (post hoc): a fitted solar capacity above this share of the aggregate's 99th
percentile counts as detected solar."""
ARMS: Final[tuple[str, ...]] = ("A8", "A8h", "A8d", "A9", "A0b", "A1b", "A3b")
CONTROL_ARMS: Final[tuple[str, ...]] = ("A0b", "A1b", "A3b")
"""Arms that fit the solar model with no battery, so they have no battery scores."""
SMOOTHNESS_ARMS: Final[dict[str, float]] = {"A8_lam1": 0.02, "A8_lam2": 0.1, "A8_lam3": 0.5}
"""The `A8` arms with a penalty on the battery's half-hour change, by penalty per megawatt."""
SMOOTHNESS_SKY: Final[str] = "regional"
ARM_LABELS: Final[dict[str, str]] = {
    "A0": "Solar only",
    "A1": "Plus price level and rank",
    "A3": "Plus the true battery (oracle)",
    "A8": "Joint model, true battery size",
    "A8h": "Joint model, half the energy capacity",
    "A8d": "Joint model, double the energy capacity",
    "A9": "Joint model, 2-hour battery",
    "A0b": "Fleet-curve solar only",
    "A1b": "Fleet-curve solar plus price level and rank",
    "A3b": "Fleet-curve solar, true battery removed",
}
ENERGY_FACTORS: Final[dict[str, float]] = {"A8": 1.0, "A8h": 0.5, "A8d": 2.0}
"""Each arm's energy capacity as a multiple of the true battery's, for the arms that use it."""
PERCENT: Final[float] = 100.0
PRICE_REGRESSOR_COLUMNS: Final[int] = 2
"""The price level and the within-day rank, which are `A1b`'s signed regressors."""


def true_battery_sizes() -> dict[str, tuple[float, float]]:
    """Return each battery's raw 99th-percentile power and rung 3's fitted energy capacity.

    Returns:
        For each battery, the 99th percentile of its absolute output in megawatts and the fitted
        energy capacity in megawatt-hours, both before scaling to the aggregate.
    """
    paths = pl.read_parquet(OUTPUT_DIR / "rung3_soc_paths.parquet")
    return {
        bmu_id: (
            float(np.nanquantile(np.abs(output_on_grid(bmu_id=bmu_id)), 0.99)),
            float(paths.filter(pl.col("bmu_id") == bmu_id)["capacity_mwh"][0]),
        )
        for bmu_id in BATTERIES
    }


def true_soc_on_grid(*, bmu_id: str) -> np.ndarray:
    """Return rung 3's state-of-charge path of a battery at every window half-hour.

    Args:
        bmu_id: The battery.

    Returns:
        The path in megawatt-hours after each half-hour, before scaling; NaN where there is none.
    """
    paths = (
        pl.read_parquet(OUTPUT_DIR / "rung3_soc_paths.parquet")
        .filter(pl.col("bmu_id") == bmu_id)
        .select(
            half_hour_end_time=pl.col("time").dt.offset_by("30m").dt.cast_time_unit("us"),
            soc=pl.col("soc_mwh"),
        )
    )
    grid = pl.DataFrame({"half_hour_end_time": window_half_hours()})
    return grid.join(paths, on="half_hour_end_time", how="left")["soc"].to_numpy()


def arm_sizes(*, arm: str, power_mw: float, true_energy_mwh: float) -> tuple[float, float]:
    """Return the power and energy capacity an arm gives the fitted battery.

    Args:
        arm: An arm of `ARMS` or a key of `SMOOTHNESS_ARMS`.
        power_mw: The true battery's 99th-percentile power after scaling.
        true_energy_mwh: The true battery's fitted energy capacity after scaling.

    Returns:
        The power limit and the energy capacity.
    """
    if arm == "A9":
        return power_mw, TYPICAL_DURATION_HOURS * power_mw
    return power_mw, true_energy_mwh * ENERGY_FACTORS.get(arm, 1.0)


def _fit_arm(
    *,
    arm: str,
    penalty: float,
    inputs: SkyInputs,
    aggregate: np.ndarray,
    battery: np.ndarray,
    prices: np.ndarray,
    power_mw: float,
    true_energy_mwh: float,
) -> tuple[float, float, JointFit]:
    """Fit one arm to one aggregate.

    Args:
        arm: An arm of `ARMS` or a key of `SMOOTHNESS_ARMS`.
        penalty: The smoothness penalty.
        inputs: The sky's inputs.
        aggregate: The aggregate on the window grid.
        battery: The true battery half, which only `A3b` uses.
        prices: `price_regressors`' array, whose first two columns only `A1b` uses.
        power_mw: The true battery's 99th-percentile power after scaling.
        true_energy_mwh: The true battery's fitted energy capacity after scaling.

    Returns:
        The power limit, the energy capacity, and the fit.
    """
    if arm in CONTROL_ARMS:
        limit, energy = 0.0, 0.0
        target = aggregate - battery if arm == "A3b" else aggregate
    else:
        limit, energy = arm_sizes(arm=arm, power_mw=power_mw, true_energy_mwh=true_energy_mwh)
        target = aggregate
    fit = fit_joint_solar_battery(
        aggregate_mw=target,
        solar_basis=inputs.basis,
        power_mw=limit,
        energy_mwh=energy,
        one_way_efficiency=ONE_WAY_EFFICIENCY,
        smoothness_penalty=penalty,
        window_half_hours=WINDOW_HALF_HOURS,
        signed_columns=prices[:, :PRICE_REGRESSOR_COLUMNS] if arm == "A1b" else None,
    )
    return limit, energy, fit


def window_correlation(*, fitted: np.ndarray, truth: np.ndarray, starts: np.ndarray) -> float:
    """Return the mean over windows of the correlation between two paths.

    The correlation ignores each window's offset, which the fit leaves free.

    Args:
        fitted: The fitted path.
        truth: The true path, NaN where unknown.
        starts: The first half-hour of each window.

    Returns:
        The mean correlation, over the windows in which both paths vary; NaN if none do.
    """
    edges = [*starts.tolist(), len(fitted)]
    values = []
    for first, last in pairwise(edges):
        a, b = fitted[first:last], truth[first:last]
        keep = np.isfinite(a) & np.isfinite(b)
        if keep.sum() > 2 and a[keep].std() > 0 and b[keep].std() > 0:
            values.append(float(np.corrcoef(a[keep], b[keep])[0, 1]))
    return float(np.mean(values)) if values else float("nan")


def score_fit(
    *,
    fit: JointFit,
    inputs: SkyInputs,
    solar: np.ndarray,
    battery: np.ndarray,
    true_soc: np.ndarray,
    reference_ac_mw: float,
    battery_scores: bool = True,
) -> dict[str, float]:
    """Score one joint fit against the true solar and the true battery.

    Args:
        fit: The fit.
        inputs: The sky's inputs.
        solar: The true solar half on the window grid.
        battery: The true battery half on the window grid.
        true_soc: The true battery's rung 3 state-of-charge path, scaled to the aggregate.
        reference_ac_mw: The direct fit's AC capacity, scaled to the aggregate's solar half.
        battery_scores: Whether to score the fitted battery, which a fit with no battery lacks.

    Returns:
        The scores. The solar-error columns are present when the true solar has output.
    """
    index = grid_index(half_hour_end_time=inputs.sky.half_hour_end_time)
    truth = solar[index]
    power = fit.solar_mw[index]
    valid = np.isfinite(truth) & daylight(sky=inputs.sky)
    fitted_ac = float(fit.solar_weights.sum())
    # The aggregate minus every fitted part but the solar: what the solar is asked to explain.
    corrected = (fit.residual_mw + fit.solar_mw)[index]
    row: dict[str, float] = {
        "fitted_ac_mw": fitted_ac,
        "variance_explained": _plant_variance_explained(
            sky=inputs.sky, corrected=corrected, power=power
        ),
        "weight_east": float(fit.solar_weights[0]),
        "weight_south": float(fit.solar_weights[1]),
        "weight_west": float(fit.solar_weights[2]),
        "weight_tracker": float(fit.solar_weights[3]),
        "residual_mean_abs_mw": float(np.nanmean(np.abs(fit.residual_mw))),
    }
    if battery_scores:
        seen = np.isfinite(battery)
        battery_p99 = float(np.quantile(np.abs(battery[seen]), 0.99))
        row["battery_mae_pct_of_p99"] = float(
            np.abs(fit.battery_mw[seen] - battery[seen]).mean() / battery_p99 * PERCENT
        )
        row["battery_correlation"] = float(np.corrcoef(fit.battery_mw[seen], battery[seen])[0, 1])
        row["soc_correlation"] = window_correlation(
            fitted=fit.soc_mwh, truth=true_soc, starts=fit.window_starts
        )
    finite = np.isfinite(truth)
    truth_p99 = float(np.nanquantile(truth, 0.99))
    if truth_p99 > 0:
        row |= {
            "nmae_of_solar_p99": float(np.abs(power - truth)[valid].mean() / truth_p99),
            "energy_ratio": float(power[finite].sum() / truth[finite].sum()),
            "ac_ratio": fitted_ac / reference_ac_mw,
            "correlation": float(np.corrcoef(power[valid], truth[valid])[0, 1]),
        }
    return row


def _series_rows(
    *,
    arm: str,
    fit: JointFit,
    solar: np.ndarray,
    battery: np.ndarray,
    aggregate: np.ndarray,
    true_soc: np.ndarray,
) -> list[dict]:
    """Return one arm's fitted and true series as rows, one per half-hour end time.

    Args:
        arm: The arm.
        fit: The arm's fit.
        solar: The true solar half.
        battery: The true battery half.
        aggregate: The aggregate.
        true_soc: The true battery's rung 3 state of charge, scaled to the aggregate.

    Returns:
        The rows.
    """
    frame = pl.DataFrame(
        {
            "arm": arm,
            "time": window_half_hours().dt.replace_time_zone(None),
            "solar_truth_mw": solar,
            "battery_truth_mw": battery,
            "aggregate_mw": aggregate,
            "recovered_solar_mw": fit.solar_mw,
            "recovered_battery_mw": fit.battery_mw,
            "recovered_soc_mwh": fit.soc_mwh,
            "true_soc_mwh": true_soc,
        }
    )
    return frame.to_dicts()


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
    sizes = true_battery_sizes()
    prices, _ = price_regressors()
    rows: list[dict] = []
    series: list[dict] = []
    for battery_id in BATTERIES:
        battery_raw = output_on_grid(bmu_id=battery_id)
        raw_p99, raw_energy = sizes[battery_id]
        raw_soc = true_soc_on_grid(bmu_id=battery_id)
        for share in SHARES:
            factor = share * AGGREGATE_P99_MW / np.nanquantile(solar_raw, 0.99)
            solar = solar_raw * factor
            battery_scale = (1.0 - share) * AGGREGATE_P99_MW / raw_p99
            battery = battery_raw * battery_scale
            aggregate = solar + battery
            power_mw = (1.0 - share) * AGGREGATE_P99_MW
            true_energy = raw_energy * battery_scale
            for sky_name, inputs in prepared.items():
                reference_ac_mw = reference.parameters.ac_capacity_mw * factor
                arms = dict.fromkeys(ARMS, 0.0)
                if sky_name == SMOOTHNESS_SKY:
                    arms |= {arm: SMOOTHNESS_ARMS[arm] for arm in SMOOTHNESS_ARMS}
                for arm, penalty in arms.items():
                    limit, energy, fit = _fit_arm(
                        arm=arm,
                        penalty=penalty,
                        inputs=inputs,
                        aggregate=aggregate,
                        battery=battery,
                        prices=prices,
                        power_mw=power_mw,
                        true_energy_mwh=true_energy,
                    )
                    rows.append(
                        {
                            "solar_set": set_name,
                            "battery": battery_id,
                            "share": share,
                            "sky": sky_name,
                            "arm": arm,
                            "power_mw": limit,
                            "energy_mwh": energy,
                            "reference_ac_mw": reference_ac_mw,
                            **score_fit(
                                fit=fit,
                                inputs=inputs,
                                solar=solar,
                                battery=battery,
                                true_soc=raw_soc * battery_scale,
                                reference_ac_mw=reference_ac_mw,
                                battery_scores=arm not in CONTROL_ARMS,
                            ),
                        }
                    )
                    if (set_name, battery_id, share, sky_name) == SERIES_AGGREGATE:
                        series.extend(
                            _series_rows(
                                arm=arm,
                                fit=fit,
                                solar=solar,
                                battery=battery,
                                aggregate=aggregate,
                                true_soc=raw_soc * battery_scale,
                            )
                        )
    return rows, series


def main() -> None:
    """Fit all aggregates and write the fits and series."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=len(SOLAR_SETS)) as pool:
        results = list(pool.map(run_set, list(SOLAR_SETS)))
    fits = pl.DataFrame([r for rows, _ in results for r in rows], infer_schema_length=None)
    fits.write_parquet(OUTPUT_DIR / "rung6_fits.parquet")
    series = pl.DataFrame([r for _, s in results for r in s], infer_schema_length=None)
    series.write_parquet(OUTPUT_DIR / "rung6_series.parquet")
    print(fits.group_by("sky", "arm").len().sort("sky", "arm"))
    print(f"Skies: {SKIES}")


if __name__ == "__main__":
    main()
