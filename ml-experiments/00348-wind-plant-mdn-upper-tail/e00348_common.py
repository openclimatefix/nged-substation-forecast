"""Shared paths and constants of experiment 00348 (upper-tail fixes for the wind plant plus head).

Run every script with `PYTHONPATH` set to the folders of experiments 00348, 00347, 00230, 00303,
00310, 00326, and 00336 (see `e00348_run.py`).
"""

import json
from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 348
SLUG: Final[str] = "wind-plant-mdn-upper-tail"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
PARTS_DIR: Final[Path] = DATA_DIR / "parts"
"""One file per fitted head: `<name>.npy`, written atomically."""
E347_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00347-wind-physics-hybrid-plant-then-mdn"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
REPLICATE_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
FARMS: Final[tuple[str, ...]] = ("W1", "W2", "W3")


def farm_scales() -> dict[str, float]:
    """Return each farm's fitted Stage 3 plant scale (the saturation level) of experiment 347."""
    plant = json.loads((E347_DIR / "plant_stage3.json").read_text())
    return {farm: float(plant[farm]["scale"]) for farm in FARMS}
