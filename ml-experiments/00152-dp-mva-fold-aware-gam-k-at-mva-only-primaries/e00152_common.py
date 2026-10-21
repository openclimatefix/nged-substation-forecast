"""Constants shared by the scripts of experiment 152 (the fold-aware GAM-K at the MVA Primaries).

Substations appear by D-label only. Experiment 110's cache, fits, and stored predictions are read
from their private folders and never rewritten.
"""

import sys
from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 152
SLUG: Final[str] = "dp-mva-fold-aware-gam-k-at-mva-only-primaries"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
EXPERIMENT_110_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00110-differentiable-physics-5-embedded-generation-and-capacity"
)
EXPERIMENT_110_CODE: Final[Path] = (
    Path(__file__).resolve().parents[1]
    / "00110-differentiable-physics-5-embedded-generation-and-capacity"
)
EXPERIMENT_124_CODE: Final[Path] = (
    Path(__file__).resolve().parents[1]
    / "00124-differentiable-physics-h-bsp-closure-audit-and-bsp-hint"
)
for _code in (EXPERIMENT_110_CODE, EXPERIMENT_124_CODE):
    if str(_code) not in sys.path:
        sys.path.append(str(_code))

MVA_ONLY_PRIMARIES: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
"""The 8 Primaries whose series is metered only in MVA."""
SUNNY_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
"""April to September, as experiment 56."""
MIDDAY_FIRST_HALF_HOUR_UTC: Final[float] = 10.0
MIDDAY_LAST_HALF_HOUR_UTC: Final[float] = 13.5
"""The half-hours that start between 10:00 and 13:30 UTC, as experiment 56."""
CLEAR_SKY_INDEX_MINIMUM: Final[float] = 0.7
"""A sunny half-hour has CAMS irradiance at least this share of the Haurwitz clear-sky value."""
