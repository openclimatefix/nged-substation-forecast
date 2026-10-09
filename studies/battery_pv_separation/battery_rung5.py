"""Rung 5: does a structured battery schedule recover solar better than price regressors?

Rung 4's unconstrained price regressors (A1) cut the solar error but a wrong-week control (A4)
gained much of it and many no-solar aggregates were falsely detected. Here the extra regressor is
ONE column: a battery schedule in per-unit power built from the day-ahead price alone, with one
free signed coefficient (the battery's power in megawatts). The aggregates, skies, and fit are
rung 4's.

- `A5` the rank-rule schedule of a 2-hour battery.
- `A6` the linear-programme schedule of a 2-hour battery (one-way efficiency 0.92, state of charge
  5% to 95%, at most 1 cycle a day).
- `A5c`, `A6c` the same schedules built from prices seven days earlier (wrong-week controls).
- `A5_1h`, `A5_4h`, `A6_1h`, `A6_4h` the same schedules for 1-hour and 4-hour batteries
  (exploratory).

Rung 4's A0, A1, A3, and A4 are read from `rung4_fits.parquet`, not refitted.

Writes `rung5_fits.parquet` (one row per aggregate, sky, and arm) and `rung5_series.parquet` (the
fitted series of the aggregate the figures draw).

Run: `OMP_NUM_THREADS=2 uv run python studies/battery_pv_separation/battery_rung5.py`.
"""

from concurrent.futures import ProcessPoolExecutor
from typing import Final

import numpy as np
import polars as pl
from battery_inputs import BATTERIES, OUTPUT_DIR
from battery_rung4 import (
    SERIES_AGGREGATE,
    SHIFT_HALF_HOURS,
    START_SHARE,
    _fit_aggregate,
    _prepare_skies,
)
from battery_synthetic import (
    AGGREGATE_P99_MW,
    SHARES,
    SOLAR_SETS,
    day_ahead_on_grid,
    grid_index,
    output_on_grid,
    window_half_hours,
)
from studies.battery_dispatch import lp_schedule, rank_rule_schedule
from studies.pv_fit import fit_plant
from studies.pv_separation import baseline_design, separate_by_differences

ARMS: Final[tuple[str, ...]] = (
    "A5",
    "A6",
    "A5c",
    "A6c",
    "A5_1h",
    "A5_4h",
    "A6_1h",
    "A6_4h",
)
ARM_LABELS: Final[dict[str, str]] = {
    "A5": "Rank-rule schedule, 2 h",
    "A6": "Linear-programme schedule, 2 h",
    "A5c": "Rank-rule schedule, prices from 7 days earlier (control)",
    "A6c": "Linear-programme schedule, prices from 7 days earlier (control)",
    "A5_1h": "Rank-rule schedule, 1 h",
    "A5_4h": "Rank-rule schedule, 4 h",
    "A6_1h": "Linear-programme schedule, 1 h",
    "A6_4h": "Linear-programme schedule, 4 h",
}
HALF_HOURS_PER_HOUR: Final[int] = 2
SCHEDULE_ARMS: Final[dict[str, tuple[str, float, bool]]] = {
    "A5": ("rank", 2.0, False),
    "A6": ("lp", 2.0, False),
    "A5c": ("rank", 2.0, True),
    "A6c": ("lp", 2.0, True),
    "A5_1h": ("rank", 1.0, False),
    "A5_4h": ("rank", 4.0, False),
    "A6_1h": ("lp", 1.0, False),
    "A6_4h": ("lp", 4.0, False),
}
"""For each arm: the schedule kind, the battery's duration in hours, and whether the prices are
taken from seven days earlier."""
LP_CYCLES_PER_DAY: Final[float] = 1.0


def schedule_for(*, prices: np.ndarray, kind: str, duration_hours: float) -> np.ndarray:
    """Return a per-unit schedule from day-ahead prices.

    Args:
        prices: The day-ahead price at every window half-hour.
        kind: `rank` for the rank rule or `lp` for the linear programme.
        duration_hours: The battery's energy in hours at full power.

    Returns:
        The per-unit power at every window half-hour.
    """
    if kind == "rank":
        return rank_rule_schedule(
            prices=prices, duration_half_hours=round(duration_hours * HALF_HOURS_PER_HOUR)
        )
    return lp_schedule(
        prices=prices, energy_hours=duration_hours, cycles_per_day_cap=LP_CYCLES_PER_DAY
    )


def arm_regressors() -> dict[str, np.ndarray]:
    """Return each arm's single regressor column on the window grid.

    Returns:
        One array of shape (half-hours, 1) per arm in `ARMS`.
    """
    prices = day_ahead_on_grid()
    shifted = np.roll(prices, SHIFT_HALF_HOURS)
    return {
        arm: schedule_for(
            prices=shifted if wrong_week else prices, kind=kind, duration_hours=hours
        )[:, None]
        for arm, (kind, hours, wrong_week) in SCHEDULE_ARMS.items()
    }


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
    regressors = arm_regressors()
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
                    arms=ARMS,
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
                            "schedule_per_unit": unit,
                            "recovered_solar_mw": plant[0],
                            "regressor_contribution_mw": plant[1],
                        }
                        for arm, parts in fitted_series.items()
                        for time, truth, battery_mw, total, unit, plant in zip(
                            inputs.sky.half_hour_end_time.astype("datetime64[us]").tolist(),
                            solar[index].tolist(),
                            battery[index].tolist(),
                            aggregate[index].tolist(),
                            regressors[arm][index, 0].tolist(),
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
    fits.write_parquet(OUTPUT_DIR / "rung5_fits.parquet")
    series = pl.DataFrame([r for _, s in results for r in s], infer_schema_length=None)
    series.write_parquet(OUTPUT_DIR / "rung5_series.parquet")
    print(fits.group_by("sky", "arm").len().sort("sky", "arm"))


if __name__ == "__main__":
    main()
