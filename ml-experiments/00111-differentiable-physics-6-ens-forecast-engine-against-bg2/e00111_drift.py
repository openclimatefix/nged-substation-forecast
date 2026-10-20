"""Tune the drift step of experiment 111's `gam0` arm on the tuning months.

Part 4's drift step of 30 sat at the edge of its grid and did not generalise, so the step is tuned
afresh here over a grid that reaches the smooth end (3000 and above). For each step the script
fits `gam0` at every monthly cut, forecasts the tuning months' runs with the calibrated ensemble
and no correction, fits the arm's spread on those residuals, and scores the 11 quantiles at the
five decision substations (D9 is left out of the tuning score). The step with the lowest mean
score is chosen. The edge check requires the chosen step to have a neighbour on each side of the
grid with a higher score; the script prints and stores whether it passed, and a failure means the
grid must be extended and the script rerun.

    uv run python ml-experiments/00111-*/e00111_drift.py [--workers 4]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Final

import e00111_arms as arms
import e00111_common as common
import e00111_ens as ens
import e00111_fits as fits_module
import e00111_tune as tune
import numpy as np
import torch

STEP_GRID: Final[tuple[float, ...]] = (30.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0)


def tag_of(*, step: float) -> str:
    """Return the folder suffix of a drift-step variant."""
    return f"_step{step:g}"


def score_step(job: tuple[str, float]) -> dict:
    """Fit `gam0` with one drift step at one site and score the tuning months.

    Args:
        job: The site's label and the drift step.

    Returns:
        The site, the step, and the per-lead-band score sums and counts.
    """
    site, step = job
    torch.set_num_threads(1)
    tag = tag_of(step=step)
    fits_module.fit_site(arm="gam0", site=site, drift_step=step, tag=tag)
    cache = common.load_cache()
    setup = ens.load_setup(cache=cache, site=site).with_calibration(
        calibration=arms.load_calibration(site=site)
    )
    fits = fits_module.load_fits(arm="gam0", site=site, tag=tag)
    runs = setup.indices(periods={"tuning"})
    base = ens.forecast_frame(
        cache=cache,
        setup=setup,
        fits=fits,
        spread=ens.ZERO_SPREAD,
        trace=None,
        run_indices=runs,
        arm="gam0",
    )
    spread = ens.fit_spread_from_frame(frame=base)
    frame = ens.forecast_frame(
        cache=cache,
        setup=setup,
        fits=fits,
        spread=spread,
        trace=None,
        run_indices=runs,
        arm="gam0",
    )
    return {"site": site, "step": step, "scores": tune.band_scores(frame=frame).tolist()}


def main() -> None:
    """Score the grid, choose the step, check the edge, and write `work/drift.json`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    started = time.monotonic()
    jobs = [(site, step) for step in STEP_GRID for site in common.DECISION_SITES]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(score_step, jobs))
    mean = {}
    for step in STEP_GRID:
        total = np.sum([np.array(r["scores"]) for r in results if r["step"] == step], axis=0)
        mean[step] = float(total[:, 0].sum() / total[:, 1].sum())
    chosen = min(mean, key=lambda step: mean[step])
    position = STEP_GRID.index(chosen)
    interior = 0 < position < len(STEP_GRID) - 1
    print({step: round(score, 4) for step, score in mean.items()})
    print(f"chosen step {chosen:g}; interior to the grid: {interior}")
    common.write_json_atomic(
        path=common.WORK_DIR / "drift.json",
        content={
            "chosen_step": chosen,
            "grid": list(STEP_GRID),
            "mean_score": {str(step): score for step, score in mean.items()},
            "chosen_is_interior": interior,
        },
    )
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
