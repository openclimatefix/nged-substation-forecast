# ruff: noqa: E501
r"""Stage A: refit experiment 184's T-all model with a flexible sunshine response, per cut.

For every monthly cut of experiment 184's stored T-all fits and every Primary, this script builds
the same training rows, target, and design as experiment 184 (`e00184_design.build_base`), appends
the 30 sunshine-response columns of `e00323_basis`, and solves the same bounded least-squares
problem with the sign bounds of the other columns unchanged and the response columns free in sign.

- **F1.** One response surface `gamma_k` per Primary.
- **F2.** One response surface `g` for all six Primaries and a per-Primary non-negative scale
  `s_k`, so `gamma_k = s_k * g`. Fitted by alternating the Primaries' own columns and scales (the
  bounded solver) with the shared surface (a ridge-penalised normal equation), for
  `F2_ITERATIONS` rounds, entirely from Gram matrices.

The gate refits each Primary without the response columns and compares the coefficients with
experiment 184's stored fit.

    PYTHONPATH=ml-experiments/00184-dp-ens-daylight-calibration \
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00323-*/e00323_fits.py [--workers 4]
"""

import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime

import e00184_common as c184  # ty: ignore[unresolved-import]
import e00184_design as design  # ty: ignore[unresolved-import]
import e00184_tall as tall  # ty: ignore[unresolved-import]
import e00323_basis as sun
import e00323_common as common
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.dp_features import FeatureCache
from studies.dp_gam import design_matrix, pinned_solar_points, solve_bounded_quadratic
from studies.dp_pinned_capacity import read_vintages


@dataclass(frozen=True)
class Grams:
    """One Primary's Gram blocks at one cut: arm columns `X`, sunshine columns `S`, target `y`."""

    xx: np.ndarray
    xs: np.ndarray
    ss: np.ndarray
    xy: np.ndarray
    sy: np.ndarray
    penalty: np.ndarray
    constrained: np.ndarray
    n_rows: int


def build_grams(*, cache: FeatureCache, vintages, site: str, cut: datetime) -> Grams:  # noqa: ANN001
    """Return the Gram blocks of the T-all design plus the sunshine columns at one cut."""
    base = design.build_base(cache=cache, vintages=vintages, site=site, cut=cut)
    arm = design.interaction_arm(name="t_all", penalty_weight=c184.T_ALL_LAMBDA)
    x = design.arm_matrix(
        cache=cache, layout=base.layout, site=site, index=base.index, x0=base.x0, arm=arm
    )
    features = cache.sites[site]
    s = sun.basis(
        ghi_w_m2=features.ghi_w_m2[base.index], elevation_deg=features.elevation_deg[base.index]
    )
    if not (np.isfinite(x).all() and np.isfinite(s).all()):
        msg = f"{site} {cut}: a non-finite training input"
        raise ValueError(msg)
    arm_penalty = design.arm_penalty(layout=base.layout, arm=arm, n_rows=base.n)
    return Grams(
        xx=x.T @ x,
        xs=x.T @ s,
        ss=s.T @ s,
        xy=x.T @ base.target,
        sy=s.T @ base.target,
        penalty=arm_penalty.matrix,
        constrained=arm_penalty.constrained,
        n_rows=base.n,
    )


def solve(*, matrix: np.ndarray, rhs: np.ndarray, constrained: np.ndarray) -> np.ndarray:
    """Solve the bounded quadratic for one problem."""
    theta, _ = solve_bounded_quadratic(
        matrix=torch.as_tensor(matrix)[None],
        rhs=torch.as_tensor(rhs)[None],
        constrained=torch.as_tensor(constrained),
    )
    return theta.numpy()[0]


def fit_baseline(*, g: Grams) -> np.ndarray:
    """Return the arm's coefficients without any response column (experiment 184's fit)."""
    return solve(matrix=g.xx + g.penalty, rhs=g.xy, constrained=g.constrained)


def fit_f1(*, g: Grams) -> tuple[np.ndarray, np.ndarray]:
    """Return `(theta, gamma)` with one free response surface."""
    p = g.xx.shape[0]
    matrix = np.block([[g.xx + g.penalty, g.xs], [g.xs.T, g.ss + sun.penalty(n_rows=g.n_rows)]])
    constrained = np.concatenate([g.constrained, np.zeros(sun.N_COLUMNS, dtype=bool)])
    both = solve(matrix=matrix, rhs=np.concatenate([g.xy, g.sy]), constrained=constrained)
    return both[:p], both[p:]


def fit_f2(
    *, grams: dict[str, Grams], baseline: dict[str, np.ndarray]
) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    """Return `(theta, gamma, scale)` per Primary with one shared response surface."""
    sites = list(grams)
    scale = dict.fromkeys(sites, 1.0)
    total_rows = sum(g.n_rows for g in grams.values())
    shared_penalty = sun.penalty(n_rows=total_rows)
    theta = dict(baseline)

    def shared_surface() -> np.ndarray:
        matrix = shared_penalty.copy()
        rhs = np.zeros(sun.N_COLUMNS)
        for site, g in grams.items():
            matrix += scale[site] ** 2 * g.ss
            rhs += scale[site] * (g.sy - g.xs.T @ theta[site])
        return np.linalg.solve(matrix, rhs)

    surface = shared_surface()
    for _ in range(common.F2_ITERATIONS):
        for site, g in grams.items():
            p = g.xx.shape[0]
            column = g.xs @ surface
            matrix = np.block(
                [[g.xx + g.penalty, column[:, None]], [column[None, :], np.array([[surface @ g.ss @ surface + 1e-9]])]]
            )  # fmt: skip
            rhs = np.concatenate([g.xy, [surface @ g.sy]])
            constrained = np.concatenate([g.constrained, [True]])
            both = solve(matrix=matrix, rhs=rhs, constrained=constrained)
            theta[site], scale[site] = both[:p], float(both[p])
        surface = shared_surface()
    return {site: (theta[site], scale[site] * surface, scale[site]) for site in sites}


def cut_job(cut: datetime) -> str:
    """Fit baseline (gate), F1, and F2 for every Primary at one cut and write the cut's file."""
    torch.set_num_threads(1)
    started = time.monotonic()
    target = common.FITS_DIR / f"{cut:%Y-%m-%d}.pkl"
    if target.exists():
        return f"{cut:%Y-%m-%d}: exists"
    cache = c184.load_cache()
    vintages = read_vintages()
    grams: dict[str, Grams] = {}
    for site in c184.SCORED_SITES:
        if cut in tall.monthly_cuts(site=site):
            grams[site] = build_grams(cache=cache, vintages=vintages, site=site, cut=cut)
    baseline = {site: fit_baseline(g=g) for site, g in grams.items()}
    out: dict = {"cut": cut, "gate": {}, "f1": {}, "f2": {}, "n_rows": {}}
    for site, g in grams.items():
        stored = next(f for f in tall.load_fits(cadence="monthly", site=site) if f.cut == cut)
        out["gate"][site] = float(np.abs(baseline[site] - stored.fit.theta).max())
        out["n_rows"][site] = g.n_rows
        theta, gamma = fit_f1(g=g)
        out["f1"][site] = {"theta": theta, "gamma": gamma}
    for site, (theta, gamma, scale) in fit_f2(grams=grams, baseline=baseline).items():
        out["f2"][site] = {"theta": theta, "gamma": gamma, "scale": scale}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(pickle.dumps(out))
    temporary.rename(target)
    worst = max(out["gate"].values())
    return f"{cut:%Y-%m-%d}: {len(grams)} sites, gate max |theta diff| {worst:.2e}, {time.monotonic() - started:.0f} s"


def load_cut(*, cut: datetime) -> dict:
    """Return the stored F1 and F2 coefficients of one cut."""
    return pickle.loads((common.FITS_DIR / f"{cut:%Y-%m-%d}.pkl").read_bytes())


def predict_sun(
    *,
    cache: FeatureCache,
    fit: tall.TallFit,
    theta: np.ndarray,
    gamma: np.ndarray,
    index: np.ndarray,
) -> np.ndarray:
    """Predict net power in points: T-all's design times `theta`, plus the response, minus the pinned solar."""
    features = cache.sites[fit.site]
    x0 = design_matrix(cache=cache, features=features, layout=fit.layout, index=index)
    x = design.arm_matrix(
        cache=cache, layout=fit.layout, site=fit.site, index=index, x0=x0, arm=fit.arm
    )
    s = sun.basis(ghi_w_m2=features.ghi_w_m2[index], elevation_deg=features.elevation_deg[index])
    if not (np.isfinite(x).all() and np.isfinite(s).all()):
        msg = f"{fit.site} has a non-finite input among the rows to predict"
        raise ValueError(msg)
    return (
        x @ theta
        + s @ gamma
        - pinned_solar_points(cache=cache, features=features, pinned=fit.pinned, index=index)
    )


def main() -> None:
    """Fit every monthly cut in parallel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--first", type=int, default=0, help="Skip this many cuts (smoke test).")
    parser.add_argument("--count", type=int, default=0, help="Fit only this many cuts.")
    args = parser.parse_args()
    cuts = sorted({c for site in c184.SCORED_SITES for c in tall.monthly_cuts(site=site)})
    cuts = cuts[args.first :]
    if args.count:
        cuts = cuts[: args.count]
    print(len(cuts), "cuts", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(cut_job, cuts):
            print(line, flush=True)


if __name__ == "__main__":
    main()
