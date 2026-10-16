"""Self-tests of experiment 00045's weekly design and ADMM solver.

Prints PASS or FAIL with the numbers for each check and exits non-zero if any fails:

    POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
        uv run python ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1/e00045_selftest.py
"""

import math
import sys
import time
from typing import Final

import e00045_common as common
import e00045_model as model
import numpy as np
import polars as pl
import scipy.optimize

TIMING_SITE: Final[str] = "D16"
"""Donington, the Primary the plan draws its examples from."""
TIMING_GRID: Final[tuple[tuple[float, float], ...]] = ((0.5, 10), (0.5, 30), (4, 10), (4, 30))
GO_NO_GO_SECONDS: Final[float] = 3.0
"""Ten times the budgeted 0.3 s per fit."""

_failures: list[str] = []


def report(*, name: str, passed: bool, detail: str) -> None:
    """Print one check's outcome and remember a failure."""
    print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")
    if not passed:
        _failures.append(name)


def synthetic_design(
    *,
    capacity: np.ndarray,
    noise: float,
    seed: int,
    perturb_from: int | None = None,
) -> model.Design:
    """Build a design whose power follows the model with the given weekly capacity and noise.

    Args:
        capacity: True weekly capacity, MW.
        noise: Standard deviation of the Gaussian noise, MW.
        seed: Random seed.
        perturb_from: When set, add a large offset to every row of weeks from this one on.

    Returns:
        The design, with 12 daylight half-hours on each of 7 days per week.
    """
    rng = np.random.default_rng(seed)
    n_weeks = len(capacity)
    per_week = 7 * 12
    week = np.repeat(np.arange(n_weeks), per_week)
    half_hour = np.tile(np.arange(per_week) % 12, n_weeks)
    month = 1 + (week // 4) % 12
    cells = 2019 * 1_000_000 + month * 10_000 + half_hour * 10
    day_effect = rng.uniform(0.2, 1.0, size=n_weeks * 7).repeat(12)
    u = day_effect * np.sin(np.pi * (half_hour + 0.5) / 12) * rng.uniform(0.6, 1.0, size=len(week))
    hdd = rng.uniform(0.0, 8.0, size=len(week))
    cell_effect = {c: 5.0 + 0.1 * (c % 1000) for c in np.unique(cells)}
    base = np.array([cell_effect[c] for c in cells])
    power = base + 0.3 * hdd - capacity[week] * u + noise * rng.standard_normal(len(week))
    if perturb_from is not None:
        power = power + np.where(week >= perturb_from, 25.0 + 3 * u, 0.0)
    rows = model.Rows(power=power, u=u, hdd=hdd, cells=cells, week=week)
    return model.Design(rows=rows, n_weeks=n_weeks, week_season_year=np.zeros(n_weeks, dtype=int))


def reference_qp(*, gram: np.ndarray, rhs: np.ndarray, rise: float, fall: float) -> np.ndarray:
    """Solve the same problem with scipy's SLSQP on the split increments (rise, fall >= 0)."""
    n = len(rhs)

    def objective(z: np.ndarray) -> float:
        x = z[:n]
        return float(
            0.5 * x @ gram @ x
            - rhs @ x
            + rise * z[n : 2 * n - 1].sum()
            + fall * z[2 * n - 1 :].sum()
        )

    def constraint(z: np.ndarray) -> np.ndarray:
        return np.diff(z[:n]) - z[n : 2 * n - 1] + z[2 * n - 1 :]

    start = np.zeros(3 * n - 2)
    result = scipy.optimize.minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(0, None)] * (3 * n - 2),
        constraints=[{"type": "eq", "fun": constraint}],
        options={"ftol": 1e-14, "maxiter": 2000},
    )
    return result.x[:n]


def test_admm_against_reference() -> None:
    """The ADMM objective and path match scipy's SLSQP on five random problems."""
    rng = np.random.default_rng(1)
    worst_objective, worst_path = 0.0, 0.0
    for k in range(5):
        n = 12
        factor = rng.standard_normal((n + 3, n))
        gram = factor.T @ factor / n
        truth = np.maximum(0.0, 2 + np.cumsum(rng.standard_normal(n)))
        rhs = gram @ truth + 0.3 * rng.standard_normal(n)
        rise, fall = 0.05 + 0.1 * k, (0.05 + 0.1 * k) * (1 + 7 * (k % 2))
        x, _ = model.asymmetric_l1_admm(
            gram=gram, rhs=rhs, rise_weight=rise, fall_weight=fall, tolerance=1e-9
        )
        reference = reference_qp(gram=gram, rhs=rhs, rise=rise, fall=fall)

        def value(
            path: np.ndarray,
            gram: np.ndarray = gram,
            rhs: np.ndarray = rhs,
            rise: float = rise,
            fall: float = fall,
        ) -> float:
            steps = np.diff(path)
            return float(
                0.5 * path @ gram @ path
                - rhs @ path
                + rise * np.maximum(steps, 0).sum()
                + fall * np.maximum(-steps, 0).sum()
            )

        worst_objective = max(worst_objective, value(x) - value(reference))
        worst_path = max(worst_path, float(np.abs(x - reference).max()))
    report(
        name="ADMM vs SLSQP on 5 random problems (n=12)",
        passed=worst_objective < 1e-8 and worst_path < 1e-4,
        detail=(
            f"largest ADMM-minus-reference objective {worst_objective:.2e}, "
            f"largest path gap {worst_path:.2e}"
        ),
    )


def test_large_penalty_constant() -> None:
    """With a huge symmetric penalty the path is the constant least-squares capacity."""
    capacity = np.full(40, 4.0)
    design = synthetic_design(capacity=capacity, noise=2.0, seed=5)
    path = model.fit_path(design=design, a=1e5, r=1.0)
    problem = design.problem()
    ones = np.ones(len(problem.rhs))
    constant = (ones @ problem.rhs) / (ones @ problem.gram @ ones)
    spread = float(np.ptp(path))
    gap = float(abs(path.mean() - constant))
    report(
        name="a very large symmetric penalty gives the constant least-squares path",
        passed=spread < 1e-4 and gap < 1e-4,
        detail=f"path spread {spread:.1e} MW, gap to the closed-form constant {gap:.1e} MW",
    )


def test_noise_free_recovery() -> None:
    """A noise-free known path is recovered to 1e-6 with a = 0."""
    capacity = 4 + 2 * np.sin(np.arange(40) / 5.0) + (np.arange(40) > 25) * 1.5
    design = synthetic_design(capacity=capacity, noise=0.0, seed=3)
    path = model.fit_path(design=design, a=0.0, r=1.0)
    error = float(np.abs(path - capacity).max())
    report(
        name="noise-free recovery with a=0",
        passed=error < 1e-6,
        detail=f"largest error {error:.2e} MW on a 40-week path of 2.5 to 7 MW",
    )


def _fall_share(*, weeks: int, a: float, r: float, seed: int) -> float:
    """Return the share of a 30% fall of `weeks` weeks that the fit recovers (mean over seeds)."""
    capacity = np.full(120, 4.0)
    capacity[50 : 50 + weeks] = 4.0 * 0.7
    design = synthetic_design(capacity=capacity, noise=3.0, seed=seed)
    path = model.fit_path(design=design, a=a, r=r)
    inside = path[50 : 50 + weeks].mean()
    outside = np.r_[path[:50], path[50 + weeks :]].mean()
    return float((outside - inside) / (4.0 * 0.3))


def test_asymmetric_fall() -> None:
    """A 30% fall of 26 weeks is less shrunk than one of 8 weeks."""
    short = np.mean([_fall_share(weeks=8, a=1.0, r=10, seed=s) for s in range(6)])
    long = np.mean([_fall_share(weeks=26, a=1.0, r=10, seed=s) for s in range(6)])
    report(
        name="asymmetric penalty shrinks an 8-week fall more than a 26-week fall",
        passed=long > short,
        detail=f"share recovered: 26 weeks {long:.2f}, 8 weeks {short:.2f} (a=1, r=10, 6 seeds)",
    )


def test_strict_never_falls() -> None:
    """The strictly non-decreasing fit has no negative increment."""
    capacity = np.full(80, 4.0)
    capacity[30:50] = 2.5
    worst = 0.0
    for seed in range(5):
        design = synthetic_design(capacity=capacity, noise=2.0, seed=seed)
        path = model.fit_path(design=design, a=0.5, r=math.inf)
        worst = min(worst, float(np.diff(path).min()))
    report(
        name="strict fit never falls",
        passed=worst >= -1e-12,
        detail=f"most negative weekly increment over 5 noisy paths with a fall: {worst:.2e}",
    )


def test_origin_truncation() -> None:
    """Perturbing data from the origin on leaves the path before the origin unchanged."""
    origin = 70
    capacity = 3 + np.linspace(0, 2, 100)
    clean = synthetic_design(capacity=capacity, noise=1.5, seed=11)
    # Same noise draw, but later data shifted: rebuild with the perturbation.
    perturbed = synthetic_design(capacity=capacity, noise=1.5, seed=11, perturb_from=origin)
    worst = 0.0
    for a, r in ((0.0, 1.0), (1.0, 1.0), (1.0, 10.0), (1.0, math.inf)):
        left = model.fit_path(design=clean, a=a, r=r, origin_week=origin)
        right = model.fit_path(design=perturbed, a=a, r=r, origin_week=origin)
        worst = max(worst, float(np.nanmax(np.abs(left - right))))
    full_left = model.fit_path(design=clean, a=1.0, r=10.0)
    full_right = model.fit_path(design=perturbed, a=1.0, r=10.0)
    moved = float(np.nanmax(np.abs(full_left[:origin] - full_right[:origin])))
    after = int(
        np.isnan(model.fit_path(design=clean, a=1.0, r=10.0, origin_week=origin)[origin:]).all()
    )
    report(
        name="origin truncation uses no data after the origin",
        passed=worst < 1e-10 and moved > 1e-3 and after == 1,
        detail=(
            f"largest change before the origin with truncation {worst:.1e}; without truncation "
            f"{moved:.2f} MW (so the perturbation is felt); weeks from the origin are NaN: "
            f"{bool(after)}"
        ),
    )


def test_timing() -> bool:
    """Time the fits on a real Primary; return True when the go/no-go passes."""
    half_hourly = pl.read_parquet(common.HALF_HOURLY_PATH)
    weeks = pl.read_parquet(common.INPUTS_PATH)
    started = time.perf_counter()
    design = model.design_from_half_hourly(half_hourly=half_hourly, weeks=weeks, site=TIMING_SITE)
    problem = design.problem()
    build = time.perf_counter() - started
    print(
        f"INFO  {common.PRIMARY_NAMES[TIMING_SITE]}: {len(design.rows.power):,} rows, "
        f"{int(problem.observed.sum())} observed weeks, projection {build:.2f} s, "
        f"s_P {problem.scale:.3f} MW, sigma {math.sqrt(problem.sigma2):.3f} MW"
    )
    free = pl.read_parquet(common.WEEKLY_FREE_PATH)
    raw = free.filter(pl.col("site") == TIMING_SITE).sort("week")
    mine = model.fit_path(design=design, a=0.0, r=1.0)
    both = np.isfinite(mine[raw["week"].to_numpy()])
    gap = np.abs(mine[raw["week"].to_numpy()][both] - raw["capacity_mw"].to_numpy()[both])
    print(
        f"INFO  free fit vs U1's raw free weekly capacity at {both.sum()} weeks: "
        f"median gap {np.median(gap):.3f} MW, 95th percentile {np.percentile(gap, 95):.3f} MW"
    )
    print("a      r    seconds  weeks  mean_path_MW  max_fall_MW")
    slowest = 0.0
    for a, r in TIMING_GRID:
        started = time.perf_counter()
        path = model.fit_path(design=design, a=a, r=r)
        seconds = time.perf_counter() - started
        slowest = max(slowest, seconds) if (a, r) == (0.5, 30) else slowest
        steps = np.diff(path[~np.isnan(path)])
        print(
            f"{a:<6} {r:<4} {seconds:7.3f}  {np.isfinite(path).sum():5d}  "
            f"{np.nanmean(path):12.3f}  {-steps.min():11.3f}"
        )
    for name in ("free", "season_year"):
        started = time.perf_counter()
        model.FITS[name](design=design, a=1.0)
        print(f"{name:<18} {time.perf_counter() - started:7.3f} s")
    report(
        name="go/no-go: ADMM at a=0.5, r=30 within 3 s",
        passed=slowest < GO_NO_GO_SECONDS,
        detail=f"{slowest:.3f} s per fit against a limit of {GO_NO_GO_SECONDS} s",
    )
    return slowest < GO_NO_GO_SECONDS


def main() -> None:
    """Run every self-test."""
    test_admm_against_reference()
    test_large_penalty_constant()
    test_noise_free_recovery()
    test_asymmetric_fall()
    test_strict_never_falls()
    test_origin_truncation()
    test_timing()
    print("ALL PASS" if not _failures else f"FAILED: {_failures}")
    sys.exit(1 if _failures else 0)


if __name__ == "__main__":
    main()
