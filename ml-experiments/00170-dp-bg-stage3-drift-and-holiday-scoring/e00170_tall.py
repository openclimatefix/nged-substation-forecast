"""T-all fits at monthly cuts, with or without a drift term, for experiment 170.

T-all is experiment 114's interaction arm: the heating columns split by day type and by half of the
hour band, and the lighting columns by day type, with the difference penalty at experiment 114's
chosen weight. `e00170_design.py` is experiment 167's copy of that design code, which takes the
penalties, the drift form, and the training window as arguments. A `TallFit` keeps what
`predict_tall` needs and drops the training matrix, so a year of fits fits in memory.

Three fit sets differ only in the drift form and the training window: `m24` (no drift, the last 24
months, experiment 136's fits, refitted here as a control), `m24_d1` (drift D1, the last 24 months),
and `all_d1` (drift D1, every year back to late 2019).
"""

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import e00170_common as common
import e00170_design as design
import numpy as np
import polars as pl
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    DriftType,
    GamFit,
    GamLayout,
    GamSpec,
    TrainingWindow,
    design_matrix,
    pinned_solar_points,
)
from studies.dp_pinned_capacity import PinnedCapacity, read_vintages

FITS_DIR: Final[Path] = common.WORK_DIR / "tall_fits"


@dataclass(frozen=True)
class FitSet:
    """One way to fit T-all: the drift form and the training window.

    Attributes:
        name: The fit set's name.
        drift: The drift form.
        months: The training window in months, or `None` for all years.
    """

    name: str
    drift: DriftType
    months: int | None


FIT_SETS: Final[dict[str, FitSet]] = {
    "m24": FitSet(name="m24", drift="none", months=24),
    "m24_d1": FitSet(name="m24_d1", drift="d1", months=24),
    "all_d1": FitSet(name="all_d1", drift="d1", months=None),
}


@dataclass(frozen=True)
class TallFit:
    """A T-all fit at one site and cut.

    Attributes:
        site: The substation's label.
        cut: The cut; no training row ends after it.
        layout: B0's columns.
        pinned: The pinned PV capacity.
        arm: The arm that was fitted.
        fit: The fitted coefficients.
    """

    site: str
    cut: datetime
    layout: GamLayout
    pinned: PinnedCapacity
    arm: design.ArmSpec
    fit: design.ArmFit


def fit_tall(
    *,
    cache: FeatureCache,
    vintages: pl.DataFrame,
    site: str,
    cut: datetime,
    fit_set: FitSet,
    tuned: dict,
) -> TallFit:
    """Fit T-all at one site and cut, with a fit set's drift form and training window."""
    spec = GamSpec(
        drift=fit_set.drift,
        ridge=tuned["ridge"]["chosen"],
        drift_step=tuned["drift_step"]["chosen"],
        capacity="pinned",
        final_knot="free",
    )
    base = design.build_base(
        cache=cache,
        vintages=vintages,
        site=site,
        cut=cut,
        spec=spec,
        window=TrainingWindow(months=fit_set.months),
    )
    arm = design.interaction_arm(name="t_all", penalty_weight=common.T_ALL_LAMBDA)
    assert base.pinned is not None
    return TallFit(
        site=site,
        cut=cut,
        layout=base.layout,
        pinned=base.pinned,
        arm=arm,
        fit=design.fit_arm(cache=cache, base=base, arm=arm),
    )


def predict_tall(*, cache: FeatureCache, fit: TallFit, index: np.ndarray) -> np.ndarray:
    """Predict net power in points at the grid rows `index`: the design times the coefficients.

    Args:
        cache: The cache whose weather drives the prediction.
        fit: The T-all fit.
        index: Grid rows.

    Returns:
        The predictions, one per row.

    Raises:
        ValueError: If a row has a non-finite input.
    """
    features = cache.sites[fit.site]
    x0 = design_matrix(cache=cache, features=features, layout=fit.layout, index=index)
    x = design.arm_matrix(
        cache=cache, layout=fit.layout, site=fit.site, index=index, x0=x0, arm=fit.arm
    )
    if not np.isfinite(x).all():
        msg = f"{fit.site} has a non-finite input among the rows to predict"
        raise ValueError(msg)
    return x @ fit.fit.theta - pinned_solar_points(
        cache=cache, features=features, pinned=fit.pinned, index=index
    )


def monthly_cuts(*, site: str) -> list[datetime]:
    """Return the cuts of experiment 111's stored `gamk` fits at a site."""
    stored: list[GamFit] = pickle.loads((common.FITS_DIR / "gamk" / f"{site}.pkl").read_bytes())
    return [fit.cut for fit in stored]


def fit_path(*, fit_set: str, site: str) -> Path:
    """Return the file that holds a fit set's T-all fits at a site."""
    return FITS_DIR / fit_set / f"{site}.pkl"


def load_fits(*, fit_set: str, site: str) -> list[TallFit]:
    """Return the stored T-all fits of a fit set at a site, ascending by cut."""
    return pickle.loads(fit_path(fit_set=fit_set, site=site).read_bytes())


def _job(job: tuple[str, str]) -> str:
    torch.set_num_threads(1)
    fit_set_name, site = job
    started = time.monotonic()
    cache = common.load_cache()
    vintages = read_vintages()
    tuned = json.loads(common.TUNED_PATH.read_text())
    fit_set = FIT_SETS[fit_set_name]
    fits = [
        fit_tall(cache=cache, vintages=vintages, site=site, cut=cut, fit_set=fit_set, tuned=tuned)
        for cut in monthly_cuts(site=site)
    ]
    target = fit_path(fit_set=fit_set_name, site=site)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(pickle.dumps(fits))
    temporary.rename(target)
    return f"{fit_set_name} {site}: {len(fits)} fits, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Fit T-all at every monthly cut, site, and fit set in parallel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--fit-sets", nargs="+", default=list(FIT_SETS))
    parser.add_argument("--sites", nargs="+", default=list(common.SCORED_SITES))
    args = parser.parse_args()
    jobs = [(f, s) for f in args.fit_sets for s in args.sites]
    # Map the module's own `_job`, so a pickled `TallFit` names `e00170_tall`, not `__main__`.
    import e00170_tall as module

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(module._job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
