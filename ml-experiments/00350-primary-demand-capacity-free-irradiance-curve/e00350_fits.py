"""Monthly convex fits of the three arms of experiment 350, at the cuts of experiment 136's T-all.

- **A** is experiment 136's T-all fit, refitted: the pinned PV capacity times the derated unit
  solar shape is added back to the target and subtracted from the prediction.
- **C** drops the pinned capacity and the unit shape. Net power gets eight extra columns,
  `-ramp_j(GHI)`, each with a non-negative coefficient, so the PV term is a monotone piecewise
  linear curve of the half-hour's irradiance, fitted separately at each Primary. The fit stays
  convex (a bounded quadratic).
- **B** has one curve shape `c` (non-negative, summing to 1) shared by the six Primaries and one
  non-negative scale `s` per Primary. The product `s * c` is not jointly convex, so the fit
  alternates at each cut: given `c`, each Primary's fit is convex in its scale and its other
  coefficients; given those, `c` is a non-negative least-squares solution pooled over the
  Primaries. The start is the pooled arm C curves. Each round minimises the objective over one
  block, so the penalised loss never rises.

The training rows, the 24-month window, and the cuts are experiment 136's, so no fit sees a row
that ends after its cut.

    PYTHONPATH=ml-experiments/00136-dp-bg-stage1-assembly-on-the-ensemble-mean \
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00350-*/e00350_fits.py [--workers 6]
"""

import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from multiprocessing import get_context

import e00136_common as c136  # ty: ignore[unresolved-import]
import e00136_tall as tall  # ty: ignore[unresolved-import]
import e00350_common as common
import e00350_curve as curve
import torch
from studies.dp_pinned_capacity import read_vintages


def cut_job(cut: datetime) -> tuple[datetime, dict, str]:
    """Fit arms A, B, and C at every Primary at one cut."""
    torch.set_num_threads(1)
    started = time.monotonic()
    cache = c136.load_cache()
    vintages = read_vintages()
    problems = {
        site: curve.SiteProblem.build(cache=cache, site=site, cut=cut, vintages=vintages)
        for site in common.SCORED_SITES
    }
    out: dict = {"A": {}, "B": {}, "C": {}}
    for site in common.SCORED_SITES:
        out["A"][site] = tall.fit_tall(
            cache=cache, vintages=vintages, site=site, cut=cut, penalty_weight=c136.T_ALL_LAMBDA
        )
    fits_c = curve.fit_arm_c(problems=problems)
    fits_b, losses = curve.fit_arm_b(problems=problems)
    for arm, fits in (("B", fits_b), ("C", fits_c)):
        for site, (theta, loss) in fits.items():
            problem = problems[site]
            out[arm][site] = curve.CurveFit(
                site=site,
                cut=cut,
                layout=problem.base.layout,
                arm=problem.arm,
                theta=theta,
                n_rows=problem.base.n,
                penalised_loss=loss,
            )
    out["b_losses"] = losses
    return (
        cut,
        out,
        (
            f"{cut:%Y-%m}: {time.monotonic() - started:.0f} s, "
            f"B losses {losses[0]:.1f} -> {losses[-1]:.1f}"
        ),
    )


def main() -> None:
    """Fit every cut in parallel and write one pickle per arm and site."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    cuts = tall.cuts_of(site=common.SCORED_SITES[0], cadence="monthly")
    for site in common.SCORED_SITES:
        if tall.cuts_of(site=site, cadence="monthly") != cuts:
            msg = f"{site} has other cuts than {common.SCORED_SITES[0]}"
            raise ValueError(msg)
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        for cut, out, line in pool.map(cut_job, cuts):
            results[cut] = out
            print(line, flush=True)
    losses = {f"{cut:%Y-%m}": results[cut]["b_losses"] for cut in cuts}
    common.write_json_atomic(path=common.FITS_DIR / "b_losses.json", content=losses)
    for arm in common.ARMS:
        for site in common.SCORED_SITES:
            target = common.FITS_DIR / arm / f"{site}.pkl"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
            temporary.write_bytes(pickle.dumps([results[cut][arm][site] for cut in cuts]))
            temporary.rename(target)


if __name__ == "__main__":
    main()
