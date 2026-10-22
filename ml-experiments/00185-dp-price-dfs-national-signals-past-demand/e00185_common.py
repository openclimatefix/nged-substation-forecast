"""Paths and constants shared by the scripts of experiment 185."""

from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 185
SLUG: Final[str] = "dp-price-dfs-national-signals-past-demand"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
PUBLIC_DIR: Final[Path] = DATA_DIR / "public"
WORK_DIR: Final[Path] = DATA_DIR / "work"
E177_PUBLIC: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00177-dp-worst-days-other-data-sources" / "public"
)
E172_WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00172-dp-error-budget-worst-days" / "work"
E175_WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00175-dp-unlabelled-day-shapes" / "work"
PREDS_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00155-dp-bg-stage2-assembly-night-offset-and-error-correction"
    / "work"
    / "dp_preds"
)
LOCAL_TZ: Final[str] = "Europe/London"
DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
SITES: Final[tuple[str, ...]] = (*DECISION_SITES, "D9")
