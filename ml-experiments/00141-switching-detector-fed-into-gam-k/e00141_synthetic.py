"""Synthetic check: an 8-point step in the residual, filtered with and without a switched regime.

Prints the correction (level plus transient) 12 hours, 1 day, and 2 days after the step starts,
with illustrative filter settings (`q` 0.5 per day, noise 2.0, transient factor 0.88).
"""

from math import log

import e00141_kf as kf
import numpy as np
from studies.dp_assimilation import FilterHyperparameters

STEP_START_DAY = 10
STEP_DAYS = 5
STEP_POINTS = 8.0
HOURS_AFTER = (12, 24, 48)


def main() -> None:
    """Filter a noisy step and print the correction at three times after its start."""
    hyper = FilterHyperparameters(log(0.5), 2.0, log(1.5), log(2.0))
    n = 48 * 30
    rng = np.random.default_rng(0)
    residual = rng.normal(0.0, 2.0, n)
    start = 48 * STEP_START_DAY
    residual[start : start + 48 * STEP_DAYS] += STEP_POINTS
    band = np.zeros(n, dtype=int)
    for label, probability, multiplier in (("K0", 0.0, 1.0), ("oracle, 64 times q", 0.9, 64.0)):
        prior = np.zeros(n)
        if probability:
            prior[start : start + 48 * STEP_DAYS] = probability
        trace, _ = kf.run_two_regime(
            residual=residual, band=band, prior=prior, hyper=hyper, multiplier=multiplier
        )
        total = trace.level[0] + trace.anomaly[0]
        found = [round(float(total[start + 2 * h]), 2) for h in HOURS_AFTER]
        print(f"{label}: correction {found} points at {HOURS_AFTER} hours after the step starts")


if __name__ == "__main__":
    main()
