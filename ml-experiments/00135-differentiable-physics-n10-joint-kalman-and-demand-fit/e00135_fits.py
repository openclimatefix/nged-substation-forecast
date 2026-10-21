"""The convex model's settings and the paths of its stored monthly fits, for experiment 135.

Every arm uses Part 4's window of 24 months, the ridge of 0.1, no drift, and the capacity pinned
to experiment 45's real-time `frozen` vintage (`studies.dp_pinned_capacity`), as experiment 111's
`gamk` does. The two-stage fits (arm `gamk`) are read from experiment 111's folder. The joint fits
(`e00135_joint_fits.py`) are stored in this experiment's folder.

A fit at cut `c` trains on rows that end no later than `c` and serves the forecasts of the runs
whose `power_fcst_init_time` is at or after `c` plus 3.5 hours and before the next cut. A site's
first cut is the first monthly cut with at least `MIN_TRAINING_DAYS` days of usable rows.
"""

import json
import pickle
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Final

import e00135_common as common
import numpy as np
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import GamFit, GamSpec

RIDGE: Final[float] = 0.1
"""Experiment 104's tuned ridge penalty."""
WINDOW_MONTHS: Final[int] = 24
J2_RIDGES: Final[dict[str, float]] = {
    "j2_ridge_0p1": 0.1,
    "j2_ridge_0p3": 0.3,
    "j2_ridge_1": 1.0,
    "j2_ridge_3": 3.0,
}
"""The ridge weight `lambda` of each J2 arm, against the plain Gram matrix of the same rows."""


def tuned() -> dict:
    """Return Part 4's tuned settings (drift step, step penalty, final knot, window)."""
    return json.loads(common.PART_4_TUNED.read_text())


def gamk_spec(*, settings: dict | None = None) -> GamSpec:
    """Return the penalties and drift form of `gamk`, the convex model with no drift.

    Args:
        settings: Part 4's tuned settings; read from disk if not given.

    Returns:
        The spec.
    """
    settings = tuned() if settings is None else settings
    base = GamSpec(
        ridge=RIDGE,
        drift_step=float(settings["drift_step"]),
        step_penalty=settings["step_penalty"],
        capacity="pinned",
        final_knot="free",
    )
    return replace(base, drift="none")


def fit_path(*, arm: str, site: str, tag: str = "") -> Path:
    """Return the file that holds an arm's fits at one site; `tag` names a variant.

    The two-stage arm `gamk` is read from experiment 111's folder, every other arm from this one's.
    """
    folder = common.TWO_STAGE_FITS_DIR if arm == "gamk" else common.FITS_DIR
    return folder / f"{arm}{tag}" / f"{site}.pkl"


def first_cut_index(*, cache: FeatureCache, site: str, cuts: list[datetime]) -> int:
    """Return the index of the first cut with `MIN_TRAINING_DAYS` days of usable rows.

    Args:
        cache: The feature cache.
        site: The substation's label.
        cuts: The monthly cuts, ascending.

    Returns:
        The index into `cuts`.

    Raises:
        ValueError: If no cut has enough rows.
    """
    features = cache.sites[site]
    usable = features.default_rows & np.isfinite(features.y)
    needed = common.MIN_TRAINING_DAYS * 48
    for index, cut in enumerate(cuts):
        if int((usable & (cache.end_times <= naive_utc(moment=cut))).sum()) >= needed:
            return index
    msg = f"{site} never has {common.MIN_TRAINING_DAYS} days of usable rows"
    raise ValueError(msg)


def load_fits(*, arm: str, site: str, tag: str = "") -> list[GamFit]:
    """Return the stored monthly fits of an arm at a site, ascending by cut."""
    return pickle.loads(fit_path(arm=arm, site=site, tag=tag).read_bytes())
