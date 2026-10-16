"""Known-answer self-tests of the arms and the detector; the vectors script runs them first.

(a) Linearity: `c(y + g) - c(y) - c(g)` is under 1e-8 for arm M2h and a confound shape `g`.
(b) Injecting exactly `R * C` with no noise gives `T` under 1e-10 and `C` equal to the truth.
(c) The flat shape fitted by arm M1L leaks under 1e-8, because the arm contains it.
(d) A load equal to `m * u` leaks `-m` into M1 within 1e-6, so the blind spot has the right sign.
(e) With a pure-PV null, the threshold built from 2,000 draws gives a false-positive rate of 0.05
    within 0.02 on 2,000 fresh draws.
"""

import e00040_arms as arms
import e00040_common as common
import e00040_detector as detector
import e00040_model as model
import numpy as np
import polars as pl

M1 = common.ALL_FITS.index("M1")
M1L = common.ALL_FITS.index("M1L")
M2H = common.ALL_FITS.index("M2h")


def _example_window(*, prepared: pl.DataFrame) -> model.Window:
    rows = prepared.filter(
        (pl.col("site") == "D16")
        & pl.col("power").is_not_null()
        & pl.col("hdd").is_not_null()
        & (pl.col("season_year") == 2024)
    ).with_columns(day_cloud=pl.col("cloud").mean().over("date"))
    return model.window_from(rows=rows)


def run(*, prepared: pl.DataFrame) -> None:
    """Run the self-tests on one Primary-year and raise `AssertionError` on a breach."""
    window = _example_window(prepared=prepared)
    shapes = arms.confound_shapes(window=window)
    y = window.power

    shape = shapes["lowsun"] * 2.0
    gap = abs(
        arms.arm_capacities(y=y + shape, window=window)[M2H]
        - arms.arm_capacities(y=y, window=window)[M2H]
        - arms.arm_capacities(y=shape, window=window)[M2H]
    )
    assert gap < 1e-8, f"(a) linearity {gap}"

    farm_like = np.clip(np.sin(np.linspace(0, 200, len(y))), 0.0, None)
    ratios = arms.arm_capacities(y=-farm_like, window=window)
    exact = 2.7 * ratios[None, : len(common.ARMS)]
    capacity, misfit = detector.capacity_and_misfit(vectors=exact, ratios=ratios)
    assert misfit[0] < 1e-10, f"(b) T {misfit[0]}"
    assert abs(capacity[0] - 2.7) < 1e-10, f"(b) C {capacity[0]}"

    leak = arms.arm_capacities(y=shapes["flat"], window=window)[M1L]
    assert abs(leak) < 1e-8, f"(c) flat leakage into M1L {leak}"

    sun = arms.arm_capacities(y=3.0 * window.u, window=window)[M1]
    assert abs(sun + 3.0) < 1e-6, f"(d) sunload leakage {sun}"

    rng = np.random.default_rng(5)
    cov = 0.04**2 * (0.8 * np.ones((6, 6)) + 0.2 * np.eye(6))

    def misfits(*, n: int) -> np.ndarray:
        noise = rng.multivariate_normal(np.zeros(6), cov, size=n)
        return detector.capacity_and_misfit(vectors=ratios[None, :6] * 1.0 + noise, ratios=ratios)[
            1
        ]

    threshold = np.quantile(misfits(n=2000), common.FLAG_QUANTILE)
    rate = float(np.mean(misfits(n=2000) > threshold))
    assert abs(rate - 0.05) < 0.02, f"(e) false-positive rate {rate}"
    print(f"self-tests passed: (a) {gap:.1e}, (b) {misfit[0]:.1e}, (c) {leak:.1e}, (e) {rate:.3f}")
