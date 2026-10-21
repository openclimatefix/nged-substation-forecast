"""Experiment 111's stored GAM-K fits and calibrations, and experiment 115's cached members.

**The correction is the same for every ensemble member**, so the DP forecast of each member never
changes between arms. Experiment 115 ran `studies.dp_ens.member_means` once per run and substation
and stored the means in points of effective capacity; this experiment only reads them and adds a
correction to them.
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00143_common as common
import numpy as np
from studies.dp_ens_weather import Calibration
from studies.dp_gam import GamFit

MEMBERS_DIR: Final[Path] = common.SRC_115_WORK_DIR / "members"
GAMK_FITS_DIR: Final[Path] = common.FITS_DIR / "gamk"


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


def load_fits(*, site: str) -> list[GamFit]:
    """Return experiment 111's stored monthly `gamk` fits at a site, ascending by cut."""
    return pickle.loads((GAMK_FITS_DIR / f"{site}.pkl").read_bytes())


def load_calibration(*, site: str) -> Calibration:
    """Return experiment 111's stored calibration at a site, with its tuned solar shrink."""
    with np.load(common.SRC_WORK_DIR / "calibration" / f"{site}.npz") as stored:
        calibration = Calibration(
            temperature_shift_c=stored["shift"],
            irradiance_ratio=stored["ratio"],
            solar_shrink=stored["shrink"],
        )
    shrink = json.loads((common.SRC_WORK_DIR / "tuned.json").read_text())["solar_shrink"]
    return calibration.with_solar_shrink(solar_shrink=shrink)


def load_members(*, site: str) -> dict[int, RunMembers]:
    """Return a site's cached member forecasts keyed by run index."""
    stored = pickle.loads(members_path(site=site).read_bytes())
    return {
        index: RunMembers(index=index, start=start, means=means) for index, start, means in stored
    }
