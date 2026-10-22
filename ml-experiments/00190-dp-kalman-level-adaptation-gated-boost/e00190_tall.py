"""T-all fits at monthly and weekly cuts, and their predictions, for experiment 136.

T-all is experiment 114's interaction arm: the heating columns split by day type and by half of the
hour band, and the lighting columns by day type, with the difference penalty at experiment 114's
chosen weight. `e00190_design.py` is a copy of experiment 114's design code. A `TallFit` keeps what
`predict_tall` needs and drops the training matrix, so a year of weekly fits fits in memory.
"""

import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00190_common as common
import e00190_design as design
import numpy as np
import polars as pl
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import GamFit, GamLayout, design_matrix, pinned_solar_points
from studies.dp_pinned_capacity import PinnedCapacity, read_vintages

CADENCES: Final[tuple[str, ...]] = ("monthly", "weekly")
WEEKLY_STEP: Final[timedelta] = timedelta(days=7)
LAST_WEEKLY_CUT: Final[datetime] = datetime(2026, 6, 29, tzinfo=UTC)
FITS_DIR: Final[Path] = common.E136_WORK_DIR / "tall_fits"
"""Experiment 136's stored fits, read and never written."""


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
    penalty_weight: float,
) -> TallFit:
    """Fit T-all at one site and cut."""
    base = design.build_base(cache=cache, vintages=vintages, site=site, cut=cut)
    arm = design.interaction_arm(name="t_all", penalty_weight=penalty_weight)
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


def cuts_of(*, site: str, cadence: str) -> list[datetime]:
    """Return a cadence's cuts at a site, starting at the site's first monthly cut."""
    monthly = monthly_cuts(site=site)
    if cadence == "monthly":
        return monthly
    cuts = []
    cut = monthly[0]
    while cut <= LAST_WEEKLY_CUT:
        cuts.append(cut)
        cut += WEEKLY_STEP
    return cuts


def fit_path(*, cadence: str, site: str) -> Path:
    """Return the file that holds a cadence's T-all fits at a site."""
    return FITS_DIR / cadence / f"{site}.pkl"


class _RenamedUnpickler(pickle.Unpickler):
    """Read experiment 136's pickles, whose classes live in this folder's `e00190_*` copies."""

    def find_class(self, module: str, name: str) -> type:
        return super().find_class(module.replace("e00136_", "e00190_"), name)


def load_fits(*, cadence: str, site: str) -> list[TallFit]:
    """Return the stored T-all fits of a cadence at a site, ascending by cut."""
    with fit_path(cadence=cadence, site=site).open("rb") as stored:
        return _RenamedUnpickler(stored).load()


def _job(job: tuple[str, str]) -> str:
    torch.set_num_threads(1)
    cadence, site = job
    started = time.monotonic()
    cache = common.load_cache()
    vintages = read_vintages()
    fits = [
        fit_tall(
            cache=cache, vintages=vintages, site=site, cut=cut, penalty_weight=common.T_ALL_LAMBDA
        )
        for cut in cuts_of(site=site, cadence=cadence)
    ]
    target = fit_path(cadence=cadence, site=site)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(pickle.dumps(fits))
    temporary.rename(target)
    return f"{cadence} {site}: {len(fits)} fits, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Fit T-all at every cut, site, and cadence in parallel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cadences", nargs="+", default=list(CADENCES))
    args = parser.parse_args()
    jobs = [(c, s) for c in args.cadences for s in common.SCORED_SITES]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
