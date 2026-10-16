"""Known-answer self-tests of the separation models; the injection script runs them first.

(a) Injecting the model's own `k * cap * u` recovers a capacity ratio of 1 within 1e-6.
(b) Adding Gaussian noise of 5% of the injected capacity leaves the mean ratio within 0.01 of 1.
(c) The model is linear: the fitted increment equals the fit of the injection alone, within 1e-8.
(d) PV plus demand plus residual adds back to the observed series, for M1 and M1L.
(e) The unconstrained and constrained fits agree when `c` is positive.
"""

import e00039_model as model
import numpy as np
import polars as pl


def _example_window(*, prepared: pl.DataFrame) -> model.Window:
    rows = prepared.filter(
        (pl.col("site") == "D16")
        & pl.col("power").is_not_null()
        & pl.col("hdd").is_not_null()
        & (pl.col("season_year") == 2024)
    )
    return model.window_from(rows=rows)


def run(*, prepared: pl.DataFrame) -> None:
    """Run the self-tests on one Primary-year and raise `AssertionError` on a breach."""
    window = _example_window(prepared=prepared)
    capacity = 0.2 * 9.0
    base = model.fit_half_hourly(y=window.power, window=window, constrain=False)
    injected = window.power - capacity * window.u
    own = model.fit_half_hourly(y=injected, window=window, constrain=False)
    ratio = (own.c - base.c) / capacity
    assert abs(ratio - 1.0) < 1e-6, f"(a) own-shape ratio {ratio}"

    rng = np.random.default_rng(1)
    ratios = []
    for _ in range(200):
        noisy = window.power - capacity * window.u + rng.normal(0.0, 0.05 * capacity, len(window.u))
        ratios.append(
            (model.fit_half_hourly(y=noisy, window=window, constrain=False).c - base.c) / capacity
        )
    assert abs(np.mean(ratios) - 1.0) < 0.01, f"(b) noisy mean ratio {np.mean(ratios)}"

    farm_like = capacity * np.clip(np.sin(np.linspace(0, 200, len(window.u))), 0, None)
    increment = model.fit_half_hourly(y=window.power - farm_like, window=window, constrain=False).c
    alone = model.fit_half_hourly(y=-farm_like, window=window, constrain=False).c
    assert abs(increment - base.c - alone) < 1e-8, f"(c) linearity {increment - base.c - alone}"

    for fit in (
        base,
        model.fit_half_hourly(y=window.power, window=window, extra=[window.cloud]),
    ):
        gap = np.max(np.abs(fit.demand - fit.pv + fit.residual - window.power))
        assert gap < 1e-8, f"(d) components do not add to the series: {gap}"
    constrained = model.fit_half_hourly(y=window.power, window=window, constrain=True)
    assert not constrained.constrained, "(e)"
    assert abs(constrained.c - base.c) < 1e-12, "(e)"
    print(f"self-tests passed: (a) ratio-1 = {ratio - 1:.1e}, (b) mean ratio {np.mean(ratios):.4f}")
