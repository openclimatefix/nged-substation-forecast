# ruff: noqa: E501
r"""Stage A: refit the T-all model with each sunshine-response variant at every monthly cut.

Cuts are the union of experiment 201's early cuts (2019-09 to 2024-07, for the past-weather rows)
and experiment 184's monthly cuts (2024-04 onwards, for the ensemble-era rows). For each Primary
and cut the training design is experiment 184's `build_base` plus the 30 response columns of
`e00325_basis`, solved with the bounded least squares of experiment 323 (the response columns free
in sign). One file per (Primary, cut) holds the coefficients of F1 and F1s. `e00325_chain` gates the rebuilt chain against the
stored one; the fitted coefficients are compared with experiment 323's stored F1 by hand.

    PYTHONPATH=ml-experiments/00184-dp-ens-daylight-calibration:ml-experiments/00201-dp-gam-residual-mlp:ml-experiments/00323-primary-demand-default-3-flexible-sunshine-response-in-the-convex-fit \
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00325-*/e00325_fits.py [--workers 10]
"""

import argparse
import multiprocessing as mp
import os
import pickle
import time
from datetime import datetime

import e00184_common as c184  # ty: ignore[unresolved-import]
import e00184_design as design  # ty: ignore[unresolved-import]
import e00184_tall as tall  # ty: ignore[unresolved-import]
import e00201_base as base201  # ty: ignore[unresolved-import]
import e00323_fits as fits323  # ty: ignore[unresolved-import]
import e00325_basis as sun
import e00325_common as common
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.dp_features import FeatureCache
from studies.dp_pinned_capacity import read_vintages


def cuts_of(*, cache: FeatureCache, site: str) -> list[datetime]:
    """Return the union of the early and the monthly cuts at one Primary, ascending."""
    return sorted(
        set(base201.early_cuts(cache=cache, site=site)) | set(tall.monthly_cuts(site=site))
    )


def fit_path(*, site: str, cut: datetime):  # noqa: ANN201
    """Return the file holding the variants' coefficients at one Primary and cut."""
    return common.FITS_DIR / site / f"{cut:%Y-%m-%d}.pkl"


def job(arguments: tuple[str, datetime]) -> str:
    """Fit F1 and F1s at one Primary and cut and write the file."""
    site, cut = arguments
    torch.set_num_threads(1)
    target = fit_path(site=site, cut=cut)
    if target.exists():
        return f"{site} {cut:%Y-%m-%d}: exists"
    started = time.monotonic()
    cache = c184.load_cache()
    vintages = read_vintages()
    base = design.build_base(cache=cache, vintages=vintages, site=site, cut=cut)
    arm = design.interaction_arm(name="t_all", penalty_weight=c184.T_ALL_LAMBDA)
    x = design.arm_matrix(
        cache=cache, layout=base.layout, site=site, index=base.index, x0=base.x0, arm=arm
    )
    features = cache.sites[site]
    arm_penalty = design.arm_penalty(layout=base.layout, arm=arm, n_rows=base.n)
    xx, xy = x.T @ x, x.T @ base.target
    out: dict = {"cut": cut, "site": site, "n_rows": base.n}
    for variant in common.RESPONSE_VARIANTS:
        s = sun.basis(
            ghi_w_m2=features.ghi_w_m2[base.index],
            elevation_deg=features.elevation_deg[base.index],
            variant=variant,
        )
        if not (np.isfinite(x).all() and np.isfinite(s).all()):
            msg = f"{site} {cut}: a non-finite training input"
            raise ValueError(msg)
        grams = fits323.Grams(
            xx=xx,
            xs=x.T @ s,
            ss=s.T @ s,
            xy=xy,
            sy=s.T @ base.target,
            penalty=arm_penalty.matrix,
            constrained=arm_penalty.constrained,
            n_rows=base.n,
        )
        theta, gamma = fits323.fit_f1(g=grams)
        out[variant] = {"theta": theta, "gamma": gamma}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(pickle.dumps(out))
    temporary.rename(target)
    return f"{site} {cut:%Y-%m-%d}: {base.n:,} rows, {time.monotonic() - started:.0f} s"


def load_table(*, variant: str) -> dict[tuple[str, datetime], tuple[np.ndarray, np.ndarray]]:
    """Return `(theta, gamma)` by `(site, cut)` of a response variant, from the stored files."""
    table = {}
    for path in sorted(common.FITS_DIR.glob("*/*.pkl")):
        stored = pickle.loads(path.read_bytes())
        table[(stored["site"], stored["cut"])] = (
            stored[variant]["theta"],
            stored[variant]["gamma"],
        )
    return table


def main() -> None:
    """Fit every Primary and cut in parallel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--count", type=int, default=0, help="Smoke test: only this many jobs.")
    args = parser.parse_args()
    cache = c184.load_cache()
    jobs = [(site, cut) for site in common.SITES for cut in cuts_of(cache=cache, site=site)]
    if args.count:
        jobs = jobs[: args.count]
    print(len(jobs), "jobs", flush=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for line in pool.imap_unordered(job, jobs, chunksize=1):
            print(line, flush=True)


if __name__ == "__main__":
    main()
