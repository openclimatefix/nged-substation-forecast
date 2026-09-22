"""Measure whether the physical model's objective has one basin or one lucky starting point.

`run_physics_experiment._fit` runs eight starting points, and the first is a fixed vector of zeros
that every seed shares. If that fixed start wins every fit, the agreement between seeds the physics
results report says only that every seed inherited the same winner, not that the objective has a
single minimum. This script separates the two by fitting each arm at each site twice: once from the
fixed start alone, and once from many independent random starts the fixed start never touches.

Run it from this directory as:

```bash
uv run python restart_basins.py --source cams
```
"""

import argparse
import concurrent.futures
import logging
import sys
from typing import Final

import numpy as np
import polars as pl
from commissioning import drop_commissioning_ramp
from export_cap import with_export_cap
from run_experiment import _add_time_features, _assign_folds, dataset_path_for
from run_physics_experiment import (
    ARM_SPLITS,
    MAX_ITERATIONS,
    START_SPREAD,
    _n_parameters,
    _predict,
)
from scipy.optimize import minimize
from sources import SOURCE_CHOICES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("restart_basins")

N_RANDOM_STARTS: Final[int] = 64
"""How many independent random starting points each (arm, site) fit is run from.

The experiment uses eight starts, seven of them random. Sixty-four is enough that a basin reached
by even a twentieth of starts shows up several times rather than once by luck.
"""

RANDOM_START_SEED: Final[int] = 20260921
"""Seeds the random starting points, so the diagnostic reproduces."""

TIED_LOSS_TOLERANCE: Final[float] = 1e-6
"""How close to the best loss a restart must come to count as having reached the same minimum.

The losses are mean absolute errors in megawatts, of order 1, and Powell stops on its own tolerance
rather than at an exact minimum, so two starts that found the same basin differ in the sixth decimal
rather than agreeing bit for bit.
"""


def _losses_for(*, rows: pl.DataFrame, arm: str, site: str) -> dict[str, object]:
    """Fit one arm at one site from the fixed start and from many random starts.

    Args:
        rows: Every uncurtailed row for the site.
        arm: The key into `ARM_SPLITS`.
        site: The site's anonymised label, carried into the result.

    Returns:
        One record holding the fixed start's loss and the spread of the random starts' losses.
    """
    actual = rows["power_mw"].to_numpy().astype(np.float64)
    capacity_guess = float(rows["effective_capacity_mw"][0])

    def objective(parameters: np.ndarray) -> float:
        modelled = _predict(
            rows=rows, parameters=parameters, arm=arm, capacity_guess=capacity_guess
        )
        return float(np.abs(modelled - actual).mean())

    n_parameters = _n_parameters(arm=arm)
    fixed = minimize(
        objective,
        np.zeros(n_parameters),
        method="Powell",
        options={"maxiter": MAX_ITERATIONS},
    )
    fixed_loss = float(fixed.fun)

    generator = np.random.default_rng(RANDOM_START_SEED)
    random_losses: list[float] = []
    for _ in range(N_RANDOM_STARTS):
        start = generator.normal(scale=START_SPREAD, size=n_parameters)
        result = minimize(objective, start, method="Powell", options={"maxiter": MAX_ITERATIONS})
        random_losses.append(float(result.fun))

    losses = np.array(random_losses)
    best_loss = min(fixed_loss, float(losses.min()))
    reaching_best = int((losses <= best_loss + TIED_LOSS_TOLERANCE).sum())
    _LOG.info("%s / site %s done", arm, site)
    return {
        "arm": arm,
        "site": site,
        "fixed_start_loss": fixed_loss,
        "best_random_loss": float(losses.min()),
        "median_random_loss": float(np.median(losses)),
        "worst_random_loss": float(losses.max()),
        "best_loss": best_loss,
        "random_starts_reaching_best": reaching_best,
        "fixed_start_reaches_best": bool(fixed_loss <= best_loss + TIED_LOSS_TOLERANCE),
    }


def main() -> int:
    """Fit every arm at every site from both start schemes and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCE_CHOICES, default="cams")
    arguments = parser.parse_args()

    dataset = with_export_cap(
        dataset=_assign_folds(
            dataset=_add_time_features(
                dataset=drop_commissioning_ramp(
                    dataset=pl.read_parquet(dataset_path_for(source=arguments.source))
                )
            )
        )
    ).filter(~pl.col("constrained"))

    sites = sorted(dataset["site"].unique().to_list())
    records: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as pool:
        futures = [
            pool.submit(
                _losses_for,
                rows=dataset.filter(pl.col("site") == site),
                arm=arm,
                site=site,
            )
            for arm in ARM_SPLITS
            for site in sites
        ]
        records.extend(future.result() for future in concurrent.futures.as_completed(futures))

    results = pl.DataFrame(records).sort("arm", "site")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220):
        print(results)

    reaching = results["random_starts_reaching_best"].to_numpy()
    penalty = results["best_random_loss"] - results["fixed_start_loss"]
    fits = len(results)
    print(f"\n{N_RANDOM_STARTS} random starts per fit, {fits} fits")
    reaches_best = int(results["fixed_start_reaches_best"].sum())
    print(f"fixed start reaches the best loss: {reaches_best}/{fits}")
    print(f"fits no random start reaches the best loss on: {int((reaching == 0).sum())}/{fits}")
    print(
        f"random starts reaching the best loss: min {reaching.min()}, "
        f"median {np.median(reaching)}, max {reaching.max()}"
    )
    print(
        f"best random loss minus fixed start loss: min {penalty.min():+.6f}, "
        f"max {penalty.max():+.6f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
