"""Tune the convex model's ridge penalty, drift step, and step penalty on 2024-07 to 2025-06.

Every fit uses data before 2024-07-01 only and is scored on 2024-07-01 to 2025-06-30, by the mean
absolute error of the mean prediction in points of effective capacity, on rows with a reading and
outside the switching masks. D9 holds 9.0 months at 2024-07-01, so it is left out of the tuning
score. The scored year (2025-07 onward) is never touched here.

The three grids are tuned in order: the ridge penalty on the arm with no drift (G-all), then the
drift step on G-all+D1 at the chosen ridge, then the step penalty on G-all+D1+steps at the chosen
ridge and drift step. Each grid is one fit per substation, because a batch of penalties shares the
Gram matrix.

    uv run python ml-experiments/00104-*/e00104_tune.py
"""

import json
from dataclasses import replace
from typing import Final

import e00104_common as common
import numpy as np
from studies.dp_gam import GamSpec, TrainingWindow, fit_gam, predict_gam

RIDGE_GRID: Final[tuple[float, ...]] = (0.1, 1.0, 10.0)
DRIFT_STEP_GRID: Final[tuple[float, ...]] = (3e3, 3e4, 3e5)
STEP_PENALTY_GRID: Final[tuple[float, ...]] = (100.0, 300.0, 1000.0)
TUNING_SITES: Final[tuple[str, ...]] = common.ELIGIBLE_SITES
TIE_POINTS: Final[float] = 0.01
"""A larger value that beats a smaller one by less than this many points is not chosen.

The ridge grid turned out flat to within 0.003 points, and a smaller ridge shrinks the capacity and
heating coefficients less. The rule was written after the first tuning run and applies to all
three grids.
"""


def grid_errors(*, specs: list[GamSpec], cache) -> np.ndarray:  # noqa: ANN001
    """Return the mean absolute error of each spec at each tuning site, shape (specs, sites).

    Args:
        specs: The penalty sets, sharing a drift form.
        cache: The feature cache.

    Returns:
        Errors in points on the tuning year's rows outside the switching masks.
    """
    errors = np.zeros((len(specs), len(TUNING_SITES)))
    for position, site in enumerate(TUNING_SITES):
        fit = fit_gam(
            cache=cache,
            site=site,
            specs=specs,
            cut=common.TUNING_CUT,
            window=TrainingWindow(),
            season_years=common.SEASON_YEARS,
        )
        index = common.scoring_index(
            cache=cache, site=site, after=common.TUNING_CUT, through=common.TUNING_END
        )
        index = index[~cache.sites[site].switching_mask[index]]
        actual = cache.sites[site].y[index]
        for member in range(len(specs)):
            predicted = predict_gam(cache=cache, fit=fit, index=index, member=member)
            errors[member, position] = float(np.abs(actual - predicted).mean())
    return errors


def chosen(*, grid: tuple[float, ...], errors: np.ndarray) -> float:
    """Return the smallest grid value whose mean error is within `TIE_POINTS` of the lowest.

    Args:
        grid: The values tried, in ascending order.
        errors: Errors of shape (len(grid), sites).

    Returns:
        The chosen value.
    """
    means = errors.mean(axis=1)
    return next(
        value for value, mean in zip(grid, means, strict=True) if mean <= means.min() + TIE_POINTS
    )


def main() -> None:
    """Tune the three penalties in order and write `work/tuned.json`."""
    cache = common.load_cache()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {"sites": list(TUNING_SITES)}

    base = GamSpec(drift="none")
    ridge_specs = [replace(base, ridge=value) for value in RIDGE_GRID]
    ridge_errors = grid_errors(specs=ridge_specs, cache=cache)
    best_ridge = chosen(grid=RIDGE_GRID, errors=ridge_errors)
    result["ridge"] = {"grid": RIDGE_GRID, "mae": ridge_errors.tolist(), "chosen": best_ridge}

    drift_specs = [
        GamSpec(drift="d1", ridge=best_ridge, drift_step=value) for value in DRIFT_STEP_GRID
    ]
    drift_errors = grid_errors(specs=drift_specs, cache=cache)
    best_drift = chosen(grid=DRIFT_STEP_GRID, errors=drift_errors)
    result["drift_step"] = {
        "grid": DRIFT_STEP_GRID,
        "mae": drift_errors.tolist(),
        "chosen": best_drift,
    }

    step_specs = [
        GamSpec(drift="d1_steps", ridge=best_ridge, drift_step=best_drift, step_penalty=value)
        for value in STEP_PENALTY_GRID
    ]
    step_errors = grid_errors(specs=step_specs, cache=cache)
    best_step = chosen(grid=STEP_PENALTY_GRID, errors=step_errors)
    result["step_penalty"] = {
        "grid": STEP_PENALTY_GRID,
        "mae": step_errors.tolist(),
        "chosen": best_step,
    }

    drift_mae = float(drift_errors.mean(axis=1).min())
    steps_mae = float(step_errors.mean(axis=1).min())
    result["steps_gain_over_d1"] = drift_mae - steps_mae
    result["default_drift_for_part_4"] = "d1_steps" if drift_mae - steps_mae > 0.1 else "d1"
    (common.WORK_DIR / "tuned.json").write_text(json.dumps(result, indent=2) + "\n")

    print("== tuning year, mean absolute error in points, by site ==")
    for name, grid, errors in (
        ("ridge (G-all)", RIDGE_GRID, ridge_errors),
        ("drift step (G-all+D1)", DRIFT_STEP_GRID, drift_errors),
        ("step penalty (G-all+D1+steps)", STEP_PENALTY_GRID, step_errors),
    ):
        print(f"\n{name}")
        print("  value " + " ".join(f"{site:>7}" for site in TUNING_SITES) + "    mean")
        for value, row in zip(grid, errors, strict=True):
            print(f"  {value:>5g} " + " ".join(f"{e:7.3f}" for e in row) + f" {row.mean():7.3f}")
    print(
        f"\nchosen ridge {best_ridge:g}, drift step {best_drift:g}, step penalty {best_step:g}; "
        f"steps lower the error by {result['steps_gain_over_d1']:.3f} points over D1; "
        f"Part 4's default drift: {result['default_drift_for_part_4']}"
    )


if __name__ == "__main__":
    main()
