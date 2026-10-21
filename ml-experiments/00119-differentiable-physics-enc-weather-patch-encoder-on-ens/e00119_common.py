"""Constants and paths shared by the scripts of experiment 119 (differentiable physics, ENC).

ENC trains a small neural network on the ECMWF ensemble (ENS) grid boxes around each Primary. The
patch frames, the stored GAM-K forecasts it corrects, and the hard-day definitions are private
files under `data/ml-experiments/00119-*/work/`. No ENS run on or after 2026-07-01 is read.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from studies.dp_features import (
    FeatureCache,
    default_input_paths,
    load_or_build_feature_cache,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

EXPERIMENT_ID: Final[int] = 119
SLUG: Final[str] = "differentiable-physics-enc-weather-patch-encoder-on-ens"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
EXP111_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00111-differentiable-physics-6-ens-forecast-engine-against-bg2"
)
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
"""Part 4's feature cache, shared because it is keyed by a hash of its inputs."""
NWP_DIR: Final[Path] = REPO_DATA_DIR / "NWP" / "nwp_model_id=ECMWF_ENS_0_25_degree"

DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
SCORED_SITES: Final[tuple[str, ...]] = (*DECISION_SITES, "D9")

FIRST_RUN: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
LAST_RUN: Final[datetime] = datetime(2026, 6, 30, tzinfo=UTC)
HELD_OUT_START: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""No data on or after this date is read."""
PUBLICATION_DELAY: Final[timedelta] = timedelta(hours=9)

N_MEMBERS: Final[int] = 51
N_RINGS: Final[int] = 3
N_PATCH_CELLS: Final[int] = 37
FIELDS: Final[tuple[str, ...]] = (
    "temperature_2m",
    "dew_point_temperature_2m",
    "downward_short_wave_radiation_flux_surface",
    "precipitation_surface",
    "pressure_reduced_to_mean_sea_level",
)
FIELD_LABELS: Final[tuple[str, ...]] = ("temperature", "dew_point", "shortwave", "precip", "mslp")
STEP_HOURS: Final[tuple[int, ...]] = (*range(0, 145, 3), *range(150, 175, 6))
"""Run-relative lead hours of the stored ENS steps: 3-hourly to 144 h, 6-hourly to 174 h. A run's
half-hours reach 177 h after its 00:00 initialisation (the origin is 9 h later and the horizon
168 h), so the last step a half-hour can floor to is 174 h."""
N_STEPS: Final[int] = len(STEP_HOURS)

QUANTILE_LEVELS: Final[tuple[float, ...]] = (
    0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99
)  # fmt: skip
QUANTILE_COLUMNS: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"
)  # fmt: skip


def load_cache() -> FeatureCache:
    """Return the feature cache that experiment 111 used."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def run_dates() -> list[datetime]:
    """Return every daily 00:00 UTC run from 2024-04-01 to 2026-06-30."""
    return [FIRST_RUN + timedelta(days=i) for i in range((LAST_RUN - FIRST_RUN).days + 1)]


def patch_path() -> Path:
    """Return the private file that lays out each site's 37-cell patch."""
    return WORK_DIR / "patches.json"


def frames_path() -> Path:
    """Return the private file that holds the extracted patch frames."""
    return WORK_DIR / "patch_frames.npz"


def write_json_atomic(*, path: Path, content: object) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True))
    temporary.rename(path)
