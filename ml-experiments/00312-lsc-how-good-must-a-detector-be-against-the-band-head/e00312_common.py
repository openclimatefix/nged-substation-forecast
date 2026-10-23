"""Paths and constants of experiment 312; its scripts import experiment 297's modules.

Run every script with experiment 297's folder on `PYTHONPATH`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00312-lsc-how-good-must-a-detector-be-against-the-band-head"
)
"""Private tables; never committed."""
E297_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00297-lsc-need-a-detector-no-detector-arms-against-the-mask"
)
"""Experiment 297's stored scores and features, read here and never written."""
DEG_DIR: Final[Path] = WORK_DIR / "deg"
EXTRA_DIR: Final[Path] = WORK_DIR / "extra"
BAND_DIR: Final[Path] = WORK_DIR / "bandpos"
EVENTS_DIR: Final[Path] = WORK_DIR / "events"
HEADS_DIR: Final[Path] = WORK_DIR / "heads"
REAL_DIR: Final[Path] = WORK_DIR / "real"
RESULTS_DIR: Final[Path] = WORK_DIR / "results"

RECALLS: Final[tuple[float, ...]] = (0.25, 0.5, 0.75)
DELAYS: Final[tuple[int, ...]] = (6, 12, 24, 48, 96, 144, 192)
"""Half-hours after the true onset at which the shift becomes known."""
BLURS: Final[tuple[float, ...]] = (0.25, 0.5)
BIASES: Final[tuple[tuple[str, float], ...]] = (("P25", 1.25), ("M25", 0.75))
PRECISIONS: Final[tuple[float, ...]] = (0.75, 0.5)
FALSE_PER_TRUE: Final[dict[float, float]] = {0.75: 1.0 / 3.0, 0.5: 1.0}
"""Extra false spans per true shift at each precision."""
TAIL_STAMPS: Final[int] = 144
"""72 hours of target time after a shift's end."""
WIDTH: Final[int] = 48
"""Half-hours in the 24-hour window of band position."""
N_PROBS: Final[int] = 199
Z_LIMIT: Final[float] = 3.5
DEGRADED_ARMS: Final[tuple[str, ...]] = (
    "ADD_PERFECT",
    *(f"ADD_R{int(r * 100)}" for r in RECALLS),
    *(f"ADD_D{d}" for d in DELAYS),
    *(f"ADD_B{int(b * 100)}" for b in BLURS),
    *(f"ADD_{name}" for name, _ in BIASES),
    *(f"ADD_P{int(p * 100)}" for p in PRECISIONS),
    "ADD_MIX",
)
