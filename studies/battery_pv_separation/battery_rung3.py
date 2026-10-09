"""Rung 3: integrate a battery's output into a state of charge.

The state of charge (SoC) in megawatt-hours follows the settled output: a half-hour that exports
`x` MWh removes `x / eta` from the cells, and a half-hour that imports `y` MWh adds `y * eta`,
where `eta` is the one-way efficiency. With the starting SoC free, the path is
`SoC(t) = start + c(t)` where `c` is the cumulative cell energy. The path stays inside `[0, E]`
exactly when `E >= max(c) - min(c)`, and the best start is `-min(c)`. So for a given `eta` the
smallest possible energy capacity is the range of `c`, and the fit is a one-dimensional search for
the `eta` in `[ETA_MIN, ETA_MAX]` that makes the range smallest.

Saves the fitted path of every battery, the range-against-efficiency curves, and the three-
efficiency drift paths for one battery, and writes `report_rung3.md`.

Run: `OMP_NUM_THREADS=2 uv run python studies/battery_pv_separation/battery_rung3.py`.
"""

from typing import Final

import numpy as np
import polars as pl
from battery_inputs import BATTERIES, NAMES, OUTPUT_DIR, battery_frame, p99_output_mw

ETA_MIN: Final[float] = 0.80
ETA_MAX: Final[float] = 0.98
ETA_STEP: Final[float] = 0.001
DRIFT_ETAS: Final[tuple[float, float, float]] = (0.85, 0.92, 0.98)
"""The one-way efficiencies of the drift paths saved to `rung3_drift_paths.parquet`."""
DRIFT_BATTERY: Final[str] = "T_LKSDB-1"
BOUND_FRACTION: Final[float] = 0.02
"""A half-hour is at a bound when its SoC is within this fraction of the capacity of 0 or `E`."""
DAYS_IN_WINDOW: Final[int] = 365
SHORT_WINDOW_HALF_HOURS: Final[int] = 7 * 48
"""The length of the short windows of the post hoc drift check: one week."""


def cell_energy_path(*, output_mwh: np.ndarray, eta: float) -> np.ndarray:
    """Return the cumulative energy added to the cells, in megawatt-hours.

    Args:
        output_mwh: The half-hourly output, positive for export.
        eta: The one-way efficiency.

    Returns:
        The running sum of `-x / eta` for export and `-x * eta` for import.
    """
    step = np.where(output_mwh > 0, -output_mwh / eta, -output_mwh * eta)
    return np.cumsum(step)


def smallest_capacity(*, output_mwh: np.ndarray, eta: float) -> tuple[float, np.ndarray]:
    """Return the smallest capacity that holds the path, and the path that starts it at zero.

    Args:
        output_mwh: The half-hourly output, positive for export.
        eta: The one-way efficiency.

    Returns:
        The capacity `max(c) - min(c)` and the SoC path `c - min(c)`.
    """
    path = cell_energy_path(output_mwh=output_mwh, eta=eta)
    return float(path.max() - path.min()), path - path.min()


def fit_efficiency(*, output_mwh: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Find the efficiency that minimises the capacity.

    Args:
        output_mwh: The half-hourly output, positive for export.

    Returns:
        The best `eta`, the capacity at that `eta`, the grid of efficiencies, and the capacity at
        each.
    """
    grid = np.arange(ETA_MIN, ETA_MAX + ETA_STEP / 2, ETA_STEP)
    capacities = np.array([smallest_capacity(output_mwh=output_mwh, eta=eta)[0] for eta in grid])
    best = int(np.argmin(capacities))
    return float(grid[best]), float(capacities[best]), grid, capacities


def main() -> None:
    """Fit every battery and write the paths, the curves, and the report."""
    lines = [
        "## Rung 3: integrate to a state of charge",
        "",
        (
            f"One-way efficiency searched on [{ETA_MIN}, {ETA_MAX}] in steps of {ETA_STEP}. "
            f"A half-hour is 'at a bound' when SoC is within {BOUND_FRACTION:.0%} of the capacity "
            "of 0 or the capacity."
        ),
        "",
    ]
    paths, curves = [], []
    for bmu_id in BATTERIES:
        frame = battery_frame(bmu_id=bmu_id)
        output = frame["output_mwh"].to_numpy()
        p99 = p99_output_mw(frame=frame)
        eta, capacity, grid, capacities = fit_efficiency(output_mwh=output)
        _, path = smallest_capacity(output_mwh=output, eta=eta)
        discharged = float(output[output > 0].sum())
        charged = float(-output[output < 0].sum())
        balance_eta = float(np.sqrt(discharged / charged))
        at_bound = float(
            ((path <= BOUND_FRACTION * capacity) | (path >= (1 - BOUND_FRACTION) * capacity)).mean()
        )
        cycles = discharged / eta / capacity / DAYS_IN_WINDOW
        lines += [
            f"### {NAMES[bmu_id]} ({bmu_id})",
            "",
            (
                f"- Fitted one-way efficiency {eta:.3f} (round trip {eta**2:.3f}); "
                f"at the search bound: {eta <= ETA_MIN + 1e-9 or eta >= ETA_MAX - 1e-9}."
            ),
            (
                f"- Fitted energy capacity {capacity:.0f} MWh; p99 absolute output {p99:.1f} MW, "
                f"so {capacity / p99:.2f} hours at p99 power."
            ),
            (
                f"- Efficiency from the year's energy balance, sqrt(exported / imported): "
                f"{balance_eta:.3f}; exported {discharged:.0f} MWh, imported {charged:.0f} MWh."
            ),
            f"- Fraction of the year at a bound: {at_bound:.1%}.",
            f"- Implied cycles per day (cell energy removed / capacity / 365): {cycles:.2f}.",
            "- Capacity if eta were 0.85, 0.92, 0.98: "
            + ", ".join(
                f"{smallest_capacity(output_mwh=output, eta=e)[0]:.0f} MWh" for e in DRIFT_ETAS
            ),
        ]
        cell = cell_energy_path(output_mwh=output, eta=eta)
        n_weeks = len(cell) // SHORT_WINDOW_HALF_HOURS
        weekly = cell[: n_weeks * SHORT_WINDOW_HALF_HOURS].reshape(n_weeks, SHORT_WINDOW_HALF_HOURS)
        week_ranges = weekly.max(axis=1) - weekly.min(axis=1)
        lines.append(
            "- Post hoc: range of the fitted-eta path within each of the year's "
            f"{n_weeks} whole weeks (MWh): median {np.median(week_ranges):.0f}, "
            f"95th percentile {np.percentile(week_ranges, 95):.0f}, "
            f"maximum {week_ranges.max():.0f}; "
            f"the maximum is {week_ranges.max() / p99:.2f} hours at p99 power."
        )
        # Stability: fit each three-month fold alone.
        fold_lines = []
        for fold in range(4):
            mask = (frame["fold"] == fold).to_numpy()
            fold_eta, fold_capacity, _, _ = fit_efficiency(output_mwh=output[mask])
            fold_lines.append(f"fold {fold}: eta {fold_eta:.3f}, E {fold_capacity:.0f} MWh")
        lines += ["- Each fold fitted alone: " + "; ".join(fold_lines), ""]
        paths.append(
            pl.DataFrame(
                {
                    "bmu_id": bmu_id,
                    "time": frame["time"],
                    "output_mw": frame["output_mw"],
                    "soc_mwh": path,
                    "capacity_mwh": capacity,
                    "eta": eta,
                }
            )
        )
        curves.append(pl.DataFrame({"bmu_id": bmu_id, "eta": grid, "capacity_mwh": capacities}))
        if bmu_id == DRIFT_BATTERY:
            drift = []
            for drift_eta in DRIFT_ETAS:
                _, drift_path = smallest_capacity(output_mwh=output, eta=drift_eta)
                drift.append(
                    pl.DataFrame({"time": frame["time"], "eta": drift_eta, "soc_mwh": drift_path})
                )
            pl.concat(drift).write_parquet(OUTPUT_DIR / "rung3_drift_paths.parquet")
    pl.concat(paths).write_parquet(OUTPUT_DIR / "rung3_soc_paths.parquet")
    pl.concat(curves).write_parquet(OUTPUT_DIR / "rung3_capacity_curves.parquet")
    (OUTPUT_DIR / "report_rung3.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
