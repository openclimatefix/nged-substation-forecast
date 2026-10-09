"""Rung 7b: the joint model with a calendar baseline on the 25 real aggregate BMUs.

Rung 7's joint model had no calendar baseline, so its solar part had to absorb every regular daily
and seasonal shape of a BMU's output. The solar study's 1,089 MW (difference separation) and 1,163
MW (physical fit) both include a seasonal calendar baseline, so rung 7 was not a test of those two
totals. This rung adds the same baseline to the joint model:

    output = solar (four fleet curves) + calendar baseline + battery.

The baseline is `pv_separation.baseline_design(flexibility="seasonal")`: an indicator per local
half-hour of the day and day type, plus two annual harmonics for each half-hour. Its coefficients
are free and signed, with the small absolute penalty of `studies.battery_joint_lp` standing in for
the solar study's ridge. The fit minimises the absolute residual on half-hourly levels in windows
of about 4 weeks, with the solar weights and the baseline coefficients shared by all windows. The
battery has power `P`, energy `E = 2 h x P`, and a one-way efficiency of 0.92. `P = 0` has no
battery and is a levels fit with an L1 loss, where the solar study's `separate` used least squares.

The skies, the BMUs, the sweep of `P`, and the calendar replicas are rung 7's. A replica has no
cloud, so a fitted solar capacity on a replica is the joint model fitting solar that is not there.

Writes `rung7b_fits.parquet` (one row per BMU, source, and assumed power).

Run: `OMP_NUM_THREADS=2 uv run python studies/battery_pv_separation/battery_rung7b.py`.
"""

from concurrent.futures import ProcessPoolExecutor
from typing import Final

import numpy as np
import polars as pl
from battery_inputs import OUTPUT_DIR
from battery_rung7 import (
    DURATION_HOURS,
    HOURS_PER_HALF_HOUR,
    MAX_WORKERS,
    ONE_WAY_EFFICIENCY,
    SOURCES,
    WINDOW_HALF_HOURS,
    aggregate_bmus,
    gsp_group,
    gsp_group_centroids,
)
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
from studies.battery_capacity import calendar_replica
from studies.battery_joint_lp import fit_joint_solar_battery
from studies.pv_separation import baseline_design, solar_basis

POWERS_MW: Final[tuple[float, ...]] = (0.0, 5.0, 20.0, 50.0, 100.0, 200.0)
"""The assumed battery powers. Zero is solar and baseline only."""
SOLVER_METHOD: Final[str] = "highs-ipm"
"""The interior-point method solves these fits in seconds, where the default takes minutes."""
AT_LIMIT_TOLERANCE_MW: Final[float] = 1e-4
"""A half-hour is at the battery's limit when its output is within this of the power limit."""


def bmu_inputs(*, bmu_id: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return the solar basis, the baseline design, and the outputs of one aggregate BMU.

    Args:
        bmu_id: The aggregate BMU's identifier.

    Returns:
        The four fleet curves under the regional sky of the BMU's GSP group, the seasonal baseline
        design, and the BMU's real output and calendar replica on the window grid, keyed by
        source.
    """
    latitude, longitude = gsp_group_centroids()[gsp_group(bmu_id=bmu_id)]
    points = nearest_grid_points(latitude=latitude, longitude=longitude, count=NEAREST_POINTS)
    sky = sky_at(hourly=hourly_cams(point_ids=points), latitude=latitude, longitude=longitude)
    basis = on_grid(
        half_hour_end_time=sky.half_hour_end_time,
        values=solar_basis(sky=sky, dc_ac_ratio=stage1_ratio(excluded=())),
    )
    design = baseline_design(half_hour_end_time=window_half_hours(), flexibility="seasonal")
    real = output_on_grid(bmu_id=bmu_id)
    return (
        basis,
        design,
        {
            "real": real,
            "replica": calendar_replica(output=real, half_hour_end_time=window_half_hours()),
        },
    )


def run_bmu(bmu_id: str) -> list[dict]:
    """Fit the sweep of assumed battery powers to one BMU's output and to its calendar replica.

    Args:
        bmu_id: The aggregate BMU's identifier.

    Returns:
        One row per source and power.
    """
    basis, design, outputs = bmu_inputs(bmu_id=bmu_id)
    rows: list[dict] = []
    for source in SOURCES:
        aggregate = outputs[source]
        finite = np.isfinite(aggregate)
        no_battery_residual = float("nan")
        for power in POWERS_MW:
            fit = fit_joint_solar_battery(
                aggregate_mw=aggregate,
                solar_basis=basis,
                power_mw=power,
                energy_mwh=DURATION_HOURS * power,
                one_way_efficiency=ONE_WAY_EFFICIENCY,
                window_half_hours=WINDOW_HALF_HOURS,
                signed_columns=design,
                method=SOLVER_METHOD,
            )
            residual = float(np.abs(fit.residual_mw[finite]).sum())
            if power == 0.0:
                no_battery_residual = residual
            at_limit = np.abs(fit.battery_mw[finite]) >= power - AT_LIMIT_TOLERANCE_MW
            rows.append(
                {
                    "bmu": bmu_id,
                    "source": source,
                    "power_mw": power,
                    "fitted_ac_mw": float(fit.solar_weights.sum()),
                    "weight_east": float(fit.solar_weights[0]),
                    "weight_south": float(fit.solar_weights[1]),
                    "weight_west": float(fit.solar_weights[2]),
                    "weight_tracker": float(fit.solar_weights[3]),
                    "solar_energy_mwh": float(fit.solar_mw[finite].sum() * HOURS_PER_HALF_HOUR),
                    "battery_throughput_mwh": float(
                        np.abs(fit.battery_mw[finite]).sum() * HOURS_PER_HALF_HOUR
                    ),
                    "baseline_mean_mw": float(fit.signed_mw[finite].mean()),
                    "residual_mean_abs_mw": residual / finite.sum(),
                    "residual_relative_to_no_battery": residual / no_battery_residual,
                    "share_at_battery_limit": float(at_limit.mean()) if power > 0 else 0.0,
                    "p99_abs_output_mw": float(np.quantile(np.abs(aggregate[finite]), 0.99)),
                }
            )
    return rows


def main() -> None:
    """Fit all 25 BMUs and write the fits."""
    bmus = aggregate_bmus()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(run_bmu, bmus))
    fits = pl.DataFrame([r for rows in results for r in rows], infer_schema_length=None)
    fits.write_parquet(OUTPUT_DIR / "rung7b_fits.parquet")
    print(
        fits.group_by("source", "power_mw")
        .agg(pl.col("fitted_ac_mw").sum())
        .sort("source", "power_mw")
    )


if __name__ == "__main__":
    main()
