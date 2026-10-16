"""Known-answer self-tests of the fitting machinery. A breach raises and stops the run."""

import e00073_common as common
import e00073_fit as fit
import numpy as np


def run() -> None:
    """Check exact recovery, the all-south reduction, and ratio recovery on noise-free data."""
    rng = np.random.default_rng(0)
    n = 3000
    hour = np.tile(np.arange(30), n // 30)
    cells = hour * 10
    nuisance = np.column_stack([rng.normal(size=n)])
    design = fit.Design.build(cells=cells, nuisance=nuisance)
    cloud = np.repeat(rng.uniform(0.3, 1.0, n // 30), 30)
    sun = np.sin(np.pi * hour / 29)
    g = np.column_stack(
        [
            cloud * np.roll(sun, 4) * 1.0,
            cloud * sun * 1.0,
            cloud * np.roll(sun, -4) * 1.0,
        ]
    )
    shapes = np.minimum(1.15 * g, 1.0)
    base = 5.0 + 2.0 * hour + 3.0 * nuisance[:, 0]
    truth = np.array([0.6, 1.2, 0.2])
    y = base - shapes @ truth
    estimate = fit.fit_orientation(design=design, y=y, shapes=shapes)
    assert np.allclose(estimate, truth, atol=1e-6), f"orientation recovery {estimate}"
    south = fit.fit_orientation(design=design, y=base - shapes[:, 1] * 2.0, shapes=shapes)
    assert np.allclose(south, [0, 2.0, 0], atol=1e-6), f"all-south {south}"
    pooled = fit.fit_pooled(designs=[design], ys=[y], shapes=[shapes], ridge=0.0)
    assert abs(pooled.sum() - truth.sum()) < 0.3, f"pooled total {pooled}"
    east_truth = np.array([1.0, 0.0, 0.0])
    y_east = base - shapes @ east_truth
    pooled_east = fit.fit_pooled(designs=[design], ys=[y_east], shapes=[shapes], ridge=0.0)
    assert pooled_east[0] > 0.8 * pooled_east.sum(), f"pooled east-only {pooled_east}"
    # A linear increment: fit(y + injection) - fit(y) returns the injection.
    inj = shapes @ np.array([0.3, 0.3, 0.3])
    inc = fit.fit_orientation(design=design, y=y - inj, shapes=shapes) - estimate
    assert np.allclose(inc, 0.3, atol=1e-6), f"increment {inc}"
    # The ratio search returns the true ratio on noise-free clipped output.
    wide = 1.4 * np.maximum(g[:, 1], 0.0) * (0.6 + 0.8 * rng.uniform(size=n))
    for true_ratio in (1.1, 1.3, 1.5):
        series = fit.clip_shape(g=wide, ratio=true_ratio) * 2.0
        profile = fit.ratio_profile(design=None, y=series, g=wide, grid=common.RATIO_GRID)
        assert abs(fit.best_ratio(profile=profile, grid=common.RATIO_GRID) - true_ratio) < 0.03
        with_design = fit.ratio_profile(
            design=design, y=base - series, g=wide, grid=common.RATIO_GRID
        )
        found = fit.best_ratio(profile=with_design, grid=common.RATIO_GRID)
        assert abs(found - true_ratio) < 0.03, f"ratio with design {found} vs {true_ratio}"
    print("self-tests passed")


if __name__ == "__main__":
    run()
