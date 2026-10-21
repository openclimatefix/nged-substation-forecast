"""T-all fits at monthly cuts, and their predictions, for experiment 161.

Copied from experiment 136 with a fitted-capacity option: the BSPs and their remainders have no
capacity vintage, so the PV capacity is fitted for them and pinned for the Primaries.

T-all is experiment 114's interaction arm: the heating columns split by day type and by half of the
hour band, and the lighting columns by day type, with the difference penalty at experiment 114's
chosen weight. `e00161_design.py` is a copy of experiment 114's design code. A `TallFit` keeps what
`predict_tall` needs and drops the training matrix, so a year of weekly fits fits in memory.
"""

import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import e00161_common as common
import e00161_design as design
import numpy as np
import polars as pl
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import GamLayout, design_matrix, pinned_solar_points
from studies.dp_pinned_capacity import PinnedCapacity, read_vintages

FITS_DIR: Final[Path] = common.FITS_DIR


@dataclass(frozen=True)
class TallFit:
    """A T-all fit at one site and cut.

    Attributes:
        site: The substation's label.
        cut: The cut; no training row ends after it.
        layout: B0's columns.
        pinned: The pinned PV capacity, or `None` when the capacity is fitted.
        arm: The arm that was fitted.
        fit: The fitted coefficients.
    """

    site: str
    cut: datetime
    layout: GamLayout
    pinned: PinnedCapacity | None
    arm: design.ArmSpec
    fit: design.ArmFit


def fit_tall(
    *,
    cache: FeatureCache,
    vintages: pl.DataFrame,
    site: str,
    cut: datetime,
    penalty_weight: float,
    fitted: bool = False,
) -> TallFit:
    """Fit T-all at one site and cut."""
    base = design.build_base(cache=cache, vintages=vintages, site=site, cut=cut, fitted=fitted)
    arm = design.interaction_arm(name="t_all", penalty_weight=penalty_weight)
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
    prediction = x @ fit.fit.theta
    if fit.pinned is None:
        return prediction
    return prediction - pinned_solar_points(
        cache=cache, features=features, pinned=fit.pinned, index=index
    )


def cuts_of(*, cache: FeatureCache, site: str) -> list[datetime]:
    """Return the monthly cuts from the first with `MIN_TRAINING_DAYS` days of default rows."""
    rows = cache.sites[site].default_rows
    cuts = []
    for cut in common.monthly_cuts():
        usable = int(rows[cache.end_times <= np.datetime64(cut.replace(tzinfo=None), "us")].sum())
        if usable >= common.MIN_TRAINING_DAYS * 48:
            cuts.append(cut)
    return cuts


def fit_path(*, site: str) -> Path:
    """Return the file that holds the monthly T-all fits at a site."""
    return FITS_DIR / f"{site}.pkl"


def load_fits(*, site: str) -> list[TallFit]:
    """Return the stored monthly T-all fits at a site, ascending by cut."""
    return pickle.loads(fit_path(site=site).read_bytes())


def uses_fitted_capacity(*, site: str) -> bool:
    """Return True for a BSP, a remainder, and any site that has no capacity vintage."""
    return site in common.CHAIN_SITES or site.endswith("f")


def _job(site: str) -> str:
    torch.set_num_threads(1)
    started = time.monotonic()
    cache = common.load_cache()
    vintages = read_vintages()
    fits = [
        fit_tall(
            cache=cache,
            vintages=vintages,
            site=site,
            cut=cut,
            penalty_weight=common.T_ALL_LAMBDA,
            fitted=uses_fitted_capacity(site=site),
        )
        for cut in cuts_of(cache=cache, site=site)
    ]
    target = fit_path(site=site)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(pickle.dumps(fits))
    temporary.rename(target)
    return f"{site}: {len(fits)} fits, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Fit T-all at every monthly cut of every chain series in parallel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sites", nargs="+", default=list(common.CHAIN_SITES))
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, args.sites):
            print(line, flush=True)


if __name__ == "__main__":
    main()
