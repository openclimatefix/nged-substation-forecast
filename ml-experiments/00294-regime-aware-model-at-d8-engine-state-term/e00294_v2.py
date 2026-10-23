"""Experiment 199's night-evidence detector, copied without the edge test and price gate.

Every feature is a function from the frozen detector's posterior to a new posterior, so that the
scores of experiment 196 (`score`, `false_alarms`) apply unchanged: a half-hour that a feature
removes gets a point mass on zero units.
"""

import pickle
from pathlib import Path
from typing import Final

import numpy as np
from e00294_detcommon import Q_FLOOR, times
from e00294_hmm import Noise, Params, Posterior, unit_means
from e00294_pipeline import Series, run_loop
from studies.solar import zenith
from studies.sources import ML_EXPERIMENTS_DATA_DIR

FIT_196_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00196-ps2-quantised-engine-detector-on-d8-with-known-answer-gate"
    / "fits_fixed.pkl"
)  # fmt: skip
SITE_LATLON: Final[dict[str, tuple[float, float]]] = {
    "D3": (52.976, -0.026),  # Sleaford Road Boston, approximate; night changes little over 0.1 deg
    "D5": (52.921421, -0.05731), "D8": (52.96159, -0.01362), "D10": (53.16853, 0.10461),
    "D11": (53.218079, -0.11218), "D14": (52.969059, -0.01441), "D16": (52.915459, -0.18896),
}  # fmt: skip
NIGHT_ZENITH_DEG: Final[float] = 93.0
"""Night is a solar elevation below -3 degrees."""


class _Remap(pickle.Unpickler):
    """Load experiment 196's pickles under this folder's module names."""

    def find_class(self, module: str, name: str) -> type:
        return super().find_class(module.replace("e00196", "e00294"), name)


def load_frozen_fit() -> tuple[Noise, Params]:
    """Experiment 196's D8 noise and wide-prior detector parameters."""
    import io

    obj = _Remap(io.BytesIO(FIT_196_PATH.read_bytes())).load()
    return obj["noise"], obj["fits"]["wide"]["params"]


def night_mask(*, label: str, n: int) -> np.ndarray:
    """True where the sun is more than 3 degrees below the horizon at the half-hour's midpoint."""
    lat, lon = SITE_LATLON[label]
    mid = times(n=n).dt.offset_by("15m")
    return zenith(stamps=mid, latitude=lat, longitude=lon) > NIGHT_ZENITH_DEG


def mask_to_night(*, s: Series, night: np.ndarray) -> Series:
    """Same site with the reading of every non-night half-hour missing."""
    y = np.where(night, s.y, np.nan)
    return s.with_reading(y)


def detect(
    *, s: Series, noise: Noise, params: Params, night: np.ndarray | None
) -> tuple[Posterior, np.ndarray]:
    """Run the frozen loop; with `night`, only night half-hours carry evidence (feature N2).

    Returns the posterior and the zero-unit mean-removed reading (the edge test reads it).
    """
    use = s if night is None else mask_to_night(s=s, night=night)
    offset, post = run_loop(s=use, noise=noise, params=params, nan_to_null=night is not None)
    base = s.y - unit_means(load=s.load, g=0.0, q=Q_FLOOR)[:, 0] - offset
    return post, base


def spells(*, flag: np.ndarray, join_gap: int = 1) -> list[tuple[int, int]]:
    """Maximal runs of True, joined across gaps of up to `join_gap` half-hours (inclusive ends)."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flag.astype(int), [0]])))
    runs = [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2], strict=True)]
    merged: list[tuple[int, int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] - 1 <= join_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged
