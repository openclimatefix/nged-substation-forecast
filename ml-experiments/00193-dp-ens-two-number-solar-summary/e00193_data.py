"""Constants and the usable-row rule for the stored rows of experiments 184 and 187."""

from typing import Final

import numpy as np
import polars as pl

QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
HORIZON_HOURS: Final[float] = 168.0
SUN_UP_UNIT_SHAPE: Final[float] = 0.01
"""Experiment 178's limit on the unit PV shape for a sunny-daylight row."""


def usable_rows(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the rows a score may use: a reading, outside switching events, leads to 168 hours."""
    return (
        ~frame["switching_mask"]
        & frame["actual"].is_finite()
        & (frame["lead_hours"] <= HORIZON_HOURS)
    ).to_numpy()
