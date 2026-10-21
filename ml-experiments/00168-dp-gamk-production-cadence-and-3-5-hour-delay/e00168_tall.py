"""T-all fits at monthly and weekly cuts, and their predictions, for experiment 168.

T-all is experiment 114's interaction arm: the heating columns split by day type and by half of the
hour band, and the lighting columns by day type, with the difference penalty at experiment 114's
chosen weight. `e00168_design.py` is a copy of experiment 114's design code. A `TallFit` keeps what
`predict_tall` needs and drops the training matrix, so a year of weekly fits fits in memory.
"""

import io
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00168_common as common
import e00168_design as design
import numpy as np
from studies.dp_features import FeatureCache
from studies.dp_gam import GamLayout, design_matrix, pinned_solar_points
from studies.dp_pinned_capacity import PinnedCapacity

CADENCES: Final[tuple[str, ...]] = ("monthly", "weekly")
WEEKLY_STEP: Final[timedelta] = timedelta(days=7)
LAST_WEEKLY_CUT: Final[datetime] = datetime(2026, 6, 29, tzinfo=UTC)
FITS_DIR: Final[Path] = common.E136_WORK_DIR / "tall_fits"


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


def fit_path(*, cadence: str, site: str) -> Path:
    """Return the file that holds a cadence's T-all fits at a site."""
    return FITS_DIR / cadence / f"{site}.pkl"


class _FitUnpickler(pickle.Unpickler):
    """Read fits that experiment 136 pickled under its own module names."""

    def find_class(self, module: str, name: str) -> type:
        """Map experiment 136's module names to this folder's copies."""
        return super().find_class(module.replace("e00136", "e00168"), name)


def load_fits(*, cadence: str, site: str) -> list[TallFit]:
    """Return the stored T-all fits of a cadence at a site, ascending by cut."""
    return _FitUnpickler(io.BytesIO(fit_path(cadence=cadence, site=site).read_bytes())).load()
