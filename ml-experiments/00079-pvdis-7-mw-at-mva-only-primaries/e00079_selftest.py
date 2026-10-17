"""Known-answer self-tests of the unfolding fit. A breach raises and stops the run."""

import e00079_fit as fit
import numpy as np


def _series(*, capacity: float, base_level: float, n_days: int = 100) -> tuple:
    """Return a noise-free window: design, unit shape, signed power, and calendar cells."""
    rng = np.random.default_rng(0)
    hour = np.tile(np.arange(30), n_days)
    cells = hour * 10
    nuisance = rng.normal(size=(30 * n_days, 1))
    design = fit.Design.build(cells=cells, nuisance=nuisance)
    cloud = np.repeat(rng.uniform(0.2, 1.0, n_days), 30)
    u = cloud * np.sin(np.pi * hour / 29)
    base = base_level * (1.0 + 0.5 * np.sin(np.pi * hour / 29)) + 0.3 * nuisance[:, 0]
    return design, u, base - capacity * u


def run() -> None:
    """Check recovery through zero, no flip at zero or small capacity, and the tie rule."""
    design, u, y = _series(capacity=4.0, base_level=1.0)
    assert (y < 0).sum() > 100, "the test series must cross zero"
    c_signed, _ = fit.signed_fit(design=design, y=y, u=u)
    assert abs(c_signed - 4.0) < 1e-6, f"signed fit {c_signed}"
    result = fit.unfold_fit(design=design, a=np.abs(y), u=u, scale=float(np.median(np.abs(y))))
    assert abs(result.capacity - 4.0) < 0.04, f"unfold recovered {result.capacity}"
    accuracy = fit.balanced_accuracy(true_sign=y, fitted_mean=result.mean, floor=0.0)
    assert accuracy > 0.99, f"sign accuracy {accuracy}"
    naive, _ = fit.naive_fit(design=design, a=np.abs(y), u=u)
    assert abs(naive - 4.0) > 0.5, f"the naive fit should miss, got {naive}"
    for capacity in (0.0, 0.3):
        design, u, y = _series(capacity=capacity, base_level=3.0)
        assert (y > 0).all()
        result = fit.unfold_fit(design=design, a=np.abs(y), u=u, scale=3.0)
        assert (result.mean >= 0).all(), f"capacity {capacity}: flipped signs"
        assert abs(result.capacity - capacity) < 1e-6, f"capacity {capacity}: {result.capacity}"


if __name__ == "__main__":
    run()
    print("self-tests passed")
