"""Read experiment 115's cached member forecasts of GAM-K, as stored.

**The correction is the same for every ensemble member**, so an arm adds its correction to the
cached member means (in points of effective capacity) and builds the mixture quantiles, which takes
seconds. Experiment 115 wrote the cache; this experiment only reads it.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00145_common as common
import numpy as np

MEMBERS_DIR: Final[Path] = common.SRC_115_WORK_DIR / "members"


@dataclass(frozen=True)
class RunMembers:
    """The cached member forecasts of one run.

    Attributes:
        index: The run's index in `common.run_dates()`.
        start: The grid position of the first target.
        means: The members' mean forecasts in points, shape (members, targets).
    """

    index: int
    start: int
    means: np.ndarray


def members_path(*, site: str) -> Path:
    """Return the file that holds a site's cached member forecasts."""
    return MEMBERS_DIR / f"{site}.pkl"


def load_members(*, site: str) -> dict[int, RunMembers]:
    """Return a site's cached member forecasts keyed by run index."""
    stored = pickle.loads(members_path(site=site).read_bytes())
    return {
        index: RunMembers(index=index, start=start, means=means) for index, start, means in stored
    }
