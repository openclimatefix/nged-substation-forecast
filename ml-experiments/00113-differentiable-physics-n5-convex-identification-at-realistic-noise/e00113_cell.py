"""Run experiment 113's synthetic cells for one substation: truths, noises, arms, and intervals.

A cell is one (truth, noise, arm, window) combination run on every draw of a substation. For each
draw the script builds a truth, a noise series, and a set of 201 month-count weightings (the full
sample, 100 for I1, and 100 for I1u), solves the arm for every weighting, and writes one row per
scored functional with the point estimate and every interval.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00113_common as common
import e00113_design as design_module
import e00113_noise as noise_module
import numpy as np
import polars as pl
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import N_BANDS

TruthKind = str
"""`hd` (the real fit's harmonics and drift), `t0` (both zero), or `zero` (no signal at all)."""
NoiseKind = str
"""`real`, `iid5`, `iid1`, or `none`."""
SPECTRAL_PERIOD_DAYS: Final[float] = 10.0


@dataclass(frozen=True)
class CellSpec:
    """One cell.

    Attributes:
        truth: The truth kind.
        noise: The noise kind.
        arm: A key of `design_module.ARMS`.
        window: `all` or `2y`.
    """

    truth: TruthKind
    noise: NoiseKind
    arm: str
    window: str = "all"


def draw_cells() -> list[CellSpec]:
    """Return the cells run on every ordinary draw, in the order of the plan's arm table."""
    cells = []
    for truth in ("hd", "t0"):
        for noise in ("real", "iid5"):
            cells.append(CellSpec(truth, noise, "g_full"))
            cells.append(CellSpec(truth, noise, "g_full_30"))
    for noise in ("real", "iid5"):
        cells.append(CellSpec("t0", noise, "g_0"))
        cells.append(CellSpec("hd", noise, "g_full_unbounded"))
    cells += [
        CellSpec("hd", "real", "g_0"),
        CellSpec("hd", "real", "g_full_pin"),
        CellSpec("hd", "real", "g_full_pin80"),
        CellSpec("t0", "real", "g_0_pin"),
        CellSpec("t0", "real", "g_h"),
        CellSpec("t0", "real", "g_d"),
        CellSpec("zero", "real", "g_full"),
        CellSpec("hd", "real", "g_full", "2y"),
        CellSpec("hd", "iid5", "g_full", "2y"),
    ]
    return cells


def control_cells() -> list[CellSpec]:
    """Return the cells of the one noise-free draw and the one iid sd 1.0 draw per substation."""
    arms = ("g_full", "g_full_30", "g_full_pin", "g_full_pin80", "g_0", "g_0_pin", "g_h", "g_d")
    cells = []
    for noise in ("none", "iid1"):
        for arm in arms:
            cells.append(CellSpec("hd", noise, arm))
            cells.append(CellSpec("t0", noise, arm))
    return cells


CONTROL_DRAW: Final[int] = 100
"""The draw index used for the noise-free and iid sd 1.0 controls."""


def contains_truth(*, cell: CellSpec) -> bool:
    """Return True if the arm's columns contain every column of the truth, with no wrong pin."""
    arm = design_module.ARMS[cell.arm]
    if arm.pin is not None and arm.pin != 1.0:
        return False
    if cell.truth == "hd":
        return not arm.drop
    return True


def real_fits_path() -> Path:
    """Return the file holding the real fits and their residuals."""
    return common.WORK_DIR / "real_fits.npz"


def build_real_fits(*, cache: FeatureCache) -> None:
    """Fit the full model to each decision substation's real target and store the fits.

    The residuals are the training rows' `y - X theta` on the grid, `NaN` elsewhere.
    """
    arrays: dict[str, np.ndarray] = {}
    arm = design_module.ARMS["g_full"]
    for site in common.DECISION_SITES:
        design = design_module.build_design(
            cache=cache, site=site, cut=common.REPLICATE_CUT, window="all"
        )
        target = cache.sites[site].y[design.index]
        counts = np.ones((1, len(design.months)))
        gram = design_module.weighted_gram(design=design, counts=counts)
        rhs = design_module.weighted_rhs(
            counts=counts,
            month_rhs=design_module.month_cross_products(design=design, target=target),
        )
        theta = design_module.solve_arm(
            design=design, arm=arm, gram=gram, rhs=rhs, pinned_increments=None
        )[0]
        residual = np.full(len(cache.end_times), np.nan)
        residual[design.index] = target - design.x @ theta
        arrays[f"theta_{site}"] = theta
        arrays[f"residual_{site}"] = residual
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(real_fits_path(), **arrays)  # ty: ignore[invalid-argument-type]


def make_truth(
    *,
    design: design_module.Design,
    theta_real: np.ndarray,
    kind: TruthKind,
    rng: np.random.Generator,
    parameters: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build the truth's coefficients on the all-years layout.

    The band slopes and capacity increments are drawn once per synthetic substation (pass the
    returned `parameters` to reuse them for another truth kind).

    Args:
        design: The all-years design.
        theta_real: The real fit's coefficients.
        kind: The truth kind.
        rng: Draws the slopes and increments when `parameters` is `None`.
        parameters: Previously drawn `band_total` and `increments`.

    Returns:
        `(theta, parameters)`.
    """
    layout = design.layout
    heating = layout.slices["heating"]
    if parameters is None:
        real_total = np.array(
            [
                sum(theta_real[heating.start + s * N_BANDS + b] for s in range(4))
                for b in range(N_BANDS)
            ]
        )
        total = np.maximum(common.MIN_BAND_SLOPE, real_total * rng.uniform(0.7, 1.3, size=N_BANDS))
        n_years = layout.slices["capacity"].stop - layout.slices["capacity"].start
        increments = rng.uniform(*common.CAPACITY_INCREMENT_RANGE, size=n_years)
        increments[0] = rng.uniform(*common.CAPACITY_FIRST_RANGE)
        parameters = {"band_total": total, "increments": increments}
    if kind == "zero":
        return np.zeros(design.p), parameters
    theta = theta_real.copy()
    for band in range(N_BANDS):
        for smoothing, share in enumerate(common.HEATING_SPLIT):
            theta[heating.start + smoothing * N_BANDS + band] = (
                parameters["band_total"][band] * share
            )
    theta[layout.slices["capacity"]] = parameters["increments"]
    if kind == "t0":
        theta[layout.slices["harmonics"]] = 0.0
        theta[layout.slices["drift"]] = 0.0
    return theta, parameters


@dataclass(frozen=True)
class ArmAnalytics:
    """The unweighted, noise-independent variance terms of one arm on one design.

    Attributes:
        posterior: `v^T A^-1 v` for each functional.
        sandwich: `u^T G u` with `u = A^-1 v`, for each functional.
        edf: The effective number of parameters, `tr(A^-1 G)`.
    """

    posterior: np.ndarray
    sandwich: np.ndarray
    edf: float


def arm_analytics(*, design: design_module.Design, arm: design_module.Arm) -> ArmAnalytics:
    """Return the posterior and sandwich variance terms of an arm."""
    keep = design_module.selection(design=design, arm=arm)
    chosen = np.flatnonzero(keep)
    gram = design_module.summed_gram(design=design).numpy()[np.ix_(chosen, chosen)]
    matrix = gram + design.penalties[arm.drift_step][np.ix_(chosen, chosen)]
    v = design.functionals[chosen]
    solved = np.linalg.solve(matrix, v)
    return ArmAnalytics(
        posterior=np.einsum("pf,pf->f", v, solved),
        sandwich=np.einsum("pf,pq,qf->f", solved, gram, solved),
        edf=float(np.trace(np.linalg.solve(matrix, gram))),
    )


def months_per_stratum(*, design: design_module.Design) -> float:
    """Return the mean number of months in a calendar quarter of the window."""
    return len(design.months) / 4.0


def run_cells_for_draw(
    *,
    design: design_module.Design,
    cells: list[CellSpec],
    truth: np.ndarray,
    noise: np.ndarray,
    counts: np.ndarray,
    analytics: dict[str, ArmAnalytics],
    site: str,
    draw: int,
    truth_kind: str,
    noise_kind: str,
    cell: CellSpec,
) -> list[dict]:
    """Return the result rows of one cell on one draw."""
    arm = design_module.ARMS[cell.arm]
    month_gram_truth = torch.einsum("mpq,q->mp", design.month_gram, torch.as_tensor(truth))
    month_noise = design_module.month_cross_products(design=design, target=noise)
    gram = design_module.weighted_gram(design=design, counts=counts)
    rhs = design_module.weighted_rhs(counts=counts, month_rhs=month_gram_truth + month_noise)
    capacity_slice = design.layout.slices["capacity"]
    increments = truth[capacity_slice]
    theta = design_module.solve_arm(
        design=design, arm=arm, gram=gram, rhs=rhs, pinned_increments=increments
    )
    noise_free = design_module.solve_arm(
        design=design,
        arm=arm,
        gram=gram[:1],
        rhs=design_module.weighted_rhs(counts=counts[:1], month_rhs=month_gram_truth),
        pinned_increments=increments,
    )[0]
    values = design_module.functional_values(design=design, theta=theta)
    truth_values = design_module.functional_values(design=design, theta=truth)
    nf_values = design_module.functional_values(design=design, theta=noise_free)
    analytic_b = (
        design_module.analytic_bias(design=design, arm=arm, theta_truth=truth)
        if contains_truth(cell=cell)
        else np.full(design_module.N_FUNCTIONALS, np.nan)
    )
    analytic = analytics[cell.arm]
    target = design.x @ truth + noise
    residual = target - design.x @ theta[0]
    sigma2 = float(residual @ residual) / (design.n - analytic.edf)
    n_boot = common.N_BOOTSTRAP
    boot = values[1 : 1 + n_boot]
    boot_u = values[1 + n_boot :]
    median = np.median(boot, axis=0)
    q_low, q_high = np.quantile(boot, [0.1, 0.9], axis=0)
    median_u = np.median(boot_u, axis=0)
    qu_low, qu_high = np.quantile(boot_u, [0.1, 0.9], axis=0)
    heating = design.layout.slices["heating"]
    active_boot = (theta[1 : 1 + n_boot, heating] <= 1e-12).any(axis=1).mean()
    active_point = bool((theta[0, heating] <= 1e-12).any())
    rows = []
    n_months = months_per_stratum(design=design)
    for functional in range(design_module.N_FUNCTIONALS):
        if arm.pin is not None and functional == design_module.CAPACITY_FUNCTIONAL:
            continue
        m = float(
            common.CAPACITY_MONTHS if functional == design_module.CAPACITY_FUNCTIONAL else n_months
        )
        inflate = float(np.sqrt(m / (m - 1.0)))
        support = design_module.band_supporting_months(design=design, functional=functional)
        drawn = counts[1 : 1 + n_boot][:, support].sum(axis=1)
        sd_posterior = float(np.sqrt(analytic.posterior[functional] * sigma2))
        sd_sandwich = float(np.sqrt(analytic.sandwich[functional] * sigma2))
        rows.append(
            {
                "site": site,
                "draw": draw,
                "window": design.window,
                "truth": truth_kind,
                "noise": noise_kind,
                "arm": cell.arm,
                "functional": design_module.FUNCTIONAL_NAMES[functional],
                "truth_value": float(truth_values[functional]),
                "noise_free_value": float(nf_values[functional]),
                "analytic_bias": float(analytic_b[functional]),
                "point": float(values[0, functional]),
                "i1_median": float(median[functional]),
                "i1_lo": float(
                    median[functional] - inflate * (median[functional] - q_low[functional])
                ),
                "i1_hi": float(
                    median[functional] + inflate * (q_high[functional] - median[functional])
                ),
                "i1_raw_lo": float(q_low[functional]),
                "i1_raw_hi": float(q_high[functional]),
                "i1_inflate": inflate,
                "i1u_lo": float(qu_low[functional]),
                "i1u_hi": float(qu_high[functional]),
                "i1u_median": float(median_u[functional]),
                "i2_sd": sd_posterior,
                "sandwich_sd": sd_sandwich,
                "boot_sd": float(boot[:, functional].std(ddof=1)),
                "active_bound_share": float(active_boot),
                "active_bound_point": active_point,
                "n_support_months": int(support.sum()),
                "share_below_half_support": float((drawn < 0.5 * support.sum()).mean()),
                "sigma2": sigma2,
                "n_rows": design.n,
            }
        )
    return rows


def noise_for(
    *,
    kind: NoiseKind,
    design: design_module.Design,
    pool: noise_module.ResidualPool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, noise_module.RealNoise | None]:
    """Draw a noise series of the given kind on the design's training rows."""
    if kind == "real":
        real = pool.draw(design=design, rng=rng)
        return real.values, real
    if kind == "iid5":
        return noise_module.iid_noise(n=design.n, sd=common.IID_SD, rng=rng), None
    if kind == "iid1":
        return noise_module.iid_noise(n=design.n, sd=1.0, rng=rng), None
    return np.zeros(design.n), None


def run_site_draw(
    *,
    cache: FeatureCache,
    site: str,
    draw: int,
    cells: list[CellSpec],
    designs: dict[str, design_module.Design],
    pool: noise_module.ResidualPool,
    theta_real: np.ndarray,
    analytics: dict[tuple[str, str], ArmAnalytics],
) -> tuple[list[dict], dict]:
    """Run every cell on one draw and return the result rows and the noise statistics."""
    site_position = common.DECISION_SITES.index(site)
    sequence = np.random.SeedSequence([common.EXPERIMENT_ID, site_position, draw])
    rng = np.random.default_rng(sequence)
    big = designs["all"]
    truths: dict[str, np.ndarray] = {}
    parameters = None
    for kind in ("hd", "t0", "zero"):
        truths[kind], parameters = make_truth(
            design=big, theta_real=theta_real, kind=kind, rng=rng, parameters=parameters
        )
    noises: dict[str, np.ndarray] = {}
    stats: dict = {"site": site, "draw": draw}
    for kind in sorted({cell.noise for cell in cells}):
        values, real = noise_for(kind=kind, design=big, pool=pool, rng=rng)
        noises[kind] = values
        if real is not None:
            stats["real_zero_share"] = real.zero_share
            stats["real_sd"] = float(values.std())
            for lag in (1, 48, 336):
                stats[f"real_autocorr_{lag}"] = noise_module.autocorrelation(
                    design=big, noise=values, lag=lag
                )
        if kind == "iid5":
            for lag in (1, 48, 336):
                stats[f"iid5_autocorr_{lag}"] = noise_module.autocorrelation(
                    design=big, noise=values, lag=lag
                )
    weightings = {
        window: design_module.weighting_set(design=designs[window], rng=rng) for window in designs
    }
    rows: list[dict] = []
    for cell in cells:
        design = designs[cell.window]
        if cell.window == "all":
            truth = truths[cell.truth]
            noise = noises[cell.noise]
        else:
            truth = design_module.restrict_theta(theta=truths[cell.truth], full=big, sub=design)
            noise = noises[cell.noise][design_module.rows_of(big=big, small=design)]
        local = {cell.arm: analytics[(cell.window, cell.arm)]}
        rows += run_cells_for_draw(
            design=design,
            cells=cells,
            truth=truth,
            noise=noise,
            counts=weightings[cell.window],
            analytics=local,
            site=site,
            draw=draw,
            truth_kind=cell.truth,
            noise_kind=cell.noise,
            cell=cell,
        )
    return rows, stats


def load_real_fits() -> dict[str, np.ndarray]:
    """Return the stored real fits as a plain dict of arrays."""
    with np.load(real_fits_path()) as stored:
        return {key: stored[key] for key in stored.files}


def run_site(*, site: str, draws: range, cells: list[CellSpec], tag: str) -> None:
    """Run the cells for one substation over a range of draws, writing one file per draw.

    Args:
        site: The substation's label.
        draws: The draw indices.
        cells: The cells to run on every draw.
        tag: A label for the files (`main` or `control`).
    """
    common.configure_threads(threads=1)
    cache = common.load_cache()
    stored = load_real_fits()
    pool = noise_module.ResidualPool(
        cache=cache,
        residuals={s: stored[f"residual_{s}"] for s in common.DECISION_SITES},
    )
    windows = sorted({cell.window for cell in cells})
    designs = {
        window: design_module.build_design(
            cache=cache, site=site, cut=common.REPLICATE_CUT, window=window
        )
        for window in windows
    }
    analytics = {
        (window, arm): arm_analytics(design=designs[window], arm=design_module.ARMS[arm])
        for window in windows
        for arm in {cell.arm for cell in cells}
    }
    common.CELLS_DIR.mkdir(parents=True, exist_ok=True)
    for draw in draws:
        path = common.CELLS_DIR / f"{tag}_{site}_{draw:03d}.parquet"
        if path.exists():
            continue
        rows, stats = run_site_draw(
            cache=cache,
            site=site,
            draw=draw,
            cells=cells,
            designs=designs,
            pool=pool,
            theta_real=stored[f"theta_{site}"],
            analytics=analytics,
        )
        temporary = path.with_suffix(".tmp")
        pl.DataFrame(rows).write_parquet(temporary)
        pl.DataFrame([stats]).write_parquet(path.with_name(path.stem + "_noise.parquet"))
        temporary.replace(path)
