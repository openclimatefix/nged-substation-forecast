"""Constants shared by the scripts of experiment 252 (the best-guess stack at 8 Primaries).

The constants, periods, and run dates are experiment 111's, copied so this folder imports only
itself.
The stored fits, ensemble arrays, and feature cache are read from experiment 111's private data
folder; this experiment writes only under its own `WORK_DIR`.

**Three periods, each with one role.** The tuning period is the odd months of 2024-07 to 2025-06
(calibration, solar shrink, filter hyperparameters, spread factors, the XGBoost bars' start
choice), the replicate period is the even months of the same year (the sign of the contrasts), and
the scored year is 2025-07-01 to 2026-06-30 (the 292 runs of `fold_a_daily_gap_v1`). A run belongs
to the month of its `power_fcst_init_time`.
"""

import json
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.dp_ens_weather import SunGeometry, replace_site
from studies.dp_features import (
    FLEET,
    FeatureCache,
    default_input_paths,
    load_or_build_feature_cache,
    naive_utc,
)
from studies.dp_gam import CLIP_BETA_PER_POINT, ClipSpec, GamSpec, TrainingWindow
from studies.dp_pinned_capacity import PinnedCapacity, read_vintages
from studies.pv_physics import PlantParameters
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 252
SLUG: Final[str] = "gam-stack-at-eight-more-primaries"
SOURCE_SLUG: Final[str] = "differentiable-physics-6-ens-forecast-engine-against-bg2"
ROW_SET_ID: Final[str] = "fold_a_daily_gap_v1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work" / os.environ.get("E00252_SOURCE", "mva")
ARM: Final[str] = os.environ.get("E00252_ARM", "a0")
"""The arm this process runs; every fit and forecast path is namespaced by it."""
FITTED: Final[bool] = ARM == "a1"
"""Arm `a1` fits the PV capacity at each cut instead of pinning it."""
ARM_DIR: Final[Path] = WORK_DIR / "arms" / ARM
SPEC_PATH: Final[Path] = WORK_DIR / "arm_specs.json"
"""`{arm: {site: {tilt_deg, azimuth_deg, scale, clip}}}`; a missing arm or site is the fleet."""
SOURCE_WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00111-{SOURCE_SLUG}" / "work"
"""Experiment 111's folder, read for `tuned.json` (the solar shrink) only."""
E152_WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00152-dp-mva-fold-aware-gam-k-at-mva-only-primaries" / "work"
)
"""Experiment 152's folder: the eight sites' ensemble stores, calibrations, and monthly cuts."""
SOURCE: Final[str] = os.environ.get("E00252_SOURCE", "mva")
"""`mva` runs the eight new sites; `control` runs a MW site from experiment 111's stores."""
if SOURCE == "control":
    STORE_DIR: Final[Path] = SOURCE_WORK_DIR / "ens_store"
    CALIBRATION_DIR: Final[Path] = SOURCE_WORK_DIR / "calibration"
    CUT_FILE: Final[str] = "gamk/{site}.pkl"
    FITS_DIR: Final[Path] = SOURCE_WORK_DIR / "fits"
else:
    STORE_DIR: Final[Path] = E152_WORK_DIR / "ens_store"
    CALIBRATION_DIR: Final[Path] = E152_WORK_DIR / "calibration"
    CUT_FILE: Final[str] = "{site}_naive.pkl"
    FITS_DIR: Final[Path] = E152_WORK_DIR / "fits" / "real"
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
"""Part 4's feature cache, shared because it is keyed by a hash of its inputs."""
PART_4_TUNED: Final[Path] = CACHE_DIR.parent / "work" / "tuned.json"

SCORED_SITES: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
"""The eight Primaries metered only in MVA that have no stack forecast yet."""
DONORS: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D19", "D21", "D22")
"""The sites whose tuning-month residuals supply the spread of D8, which has no tuning month."""
D8_FIRST_DAY: Final[datetime] = datetime(2025, 4, 1, tzinfo=UTC)
"""D8 enters training and scoring on this day, when its series changed meaning."""

FIRST_RUN: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
LAST_RUN: Final[datetime] = datetime(2026, 6, 30, tzinfo=UTC)
PUBLICATION_DELAY: Final[timedelta] = timedelta(hours=9)
SCORED_FIRST_RUN: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
EVERY5D_STRIDE_DAYS: Final[int] = 5

TUNING_MONTHS: Final[frozenset[tuple[int, int]]] = frozenset(
    {(2024, 7), (2024, 9), (2024, 11), (2025, 1), (2025, 3), (2025, 5)}
)
REPLICATE_MONTHS: Final[frozenset[tuple[int, int]]] = frozenset(
    {(2024, 8), (2024, 10), (2024, 12), (2025, 2), (2025, 4), (2025, 6)}
)
REPLICATE_SIGN_MONTHS: Final[frozenset[tuple[int, int]]] = frozenset(
    {(2025, 2), (2025, 4), (2025, 6)}
)
"""The replicate months with 10 or more months of ENS rows for XGBoost; the sign is read here."""
START_CHOICE_MONTHS: Final[frozenset[tuple[int, int]]] = frozenset(
    {(2025, 1), (2025, 3), (2025, 5)}
)
"""The tuning months on which the bars' start choice is read."""

MIN_TRAINING_DAYS: Final[int] = 90
"""A site's first fit has at least this many days of usable rows."""
SEASON_YEARS: Final[range] = range(2019, 2027)
HORIZON_HALF_HOURS: Final[int] = 336
PROXY_WARM_UP_RUNS: Final[int] = 9
"""Runs before this index lack a week of analysis proxy before their origin."""


def load_cache() -> FeatureCache:
    """Return the feature cache, building and saving it on the first call."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def monthly_cuts(*, last: datetime = datetime(2026, 6, 1, tzinfo=UTC)) -> list[datetime]:
    """Return the first-of-the-month 00:00 UTC cuts from 2024-04-01 to `last`."""
    cuts = []
    year, month = FIRST_RUN.year, FIRST_RUN.month
    while (cut := datetime(year, month, 1, tzinfo=UTC)) <= last:
        cuts.append(cut)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return cuts


def run_dates() -> list[datetime]:
    """Return every daily 00:00 UTC run from 2024-04-01 to 2026-06-30."""
    return [FIRST_RUN + timedelta(days=i) for i in range((LAST_RUN - FIRST_RUN).days + 1)]


def origin_of(*, run: datetime) -> datetime:
    """Return a run's `power_fcst_init_time`: its initialisation plus the publication delay."""
    return run + PUBLICATION_DELAY


def month_of_origin(*, run: datetime) -> tuple[int, int]:
    """Return the (year, month) of a run's `power_fcst_init_time`."""
    origin = origin_of(run=run)
    return origin.year, origin.month


def is_scored_run(*, run: datetime) -> bool:
    """Return True for the 292 scored runs: daily runs of the scored year off the 5-day stride."""
    if not SCORED_FIRST_RUN <= run <= LAST_RUN:
        return False
    return (run - SCORED_FIRST_RUN).days % EVERY5D_STRIDE_DAYS != 0


def period_of_run(*, run: datetime) -> str:
    """Return `tuning`, `replicate`, `scored`, `stride`, or `warm_up` for a run.

    `stride` is a run of the scored year on the every-fifth-day stride, which chose BG2's blend and
    offset and is never scored; `warm_up` is a run before 2024-07 or in no named period.
    """
    key = month_of_origin(run=run)
    if key in TUNING_MONTHS:
        return "tuning"
    if key in REPLICATE_MONTHS:
        return "replicate"
    if run >= SCORED_FIRST_RUN:
        return "scored" if is_scored_run(run=run) else "stride"
    return "warm_up"


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True))
    temporary.rename(path)


RIDGE: Final[float] = 0.1
"""Experiment 104's tuned ridge penalty, as in experiments 111 and 114."""
WINDOW_MONTHS: Final[int] = 24
BASE_THRESHOLD_C: Final[float] = 15.5
T_ALL_LAMBDA: Final[float] = 1.0
"""T-all's difference-penalty weight, chosen by experiment 114 (`decisions.json`)."""
POOL_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00133-differentiable-physics-vk-learned-noise-and-observation-variance-in-gam-k"
    / "work"
    / "select"
    / "pool.json"
)
"""Experiment 133's single shared filter cell (POOL)."""


def base_spec() -> GamSpec:
    """Return B0's penalties: GAM-K's ridge and no drift, with the capacity pinned or fitted.

    Arm `a0` pins the capacity to the vintage (zero at the eight sites); arm `a1` fits it at each
    cut.
    """
    return GamSpec(
        ridge=RIDGE, drift="none", capacity=("fitted" if FITTED else "pinned"), final_knot="free"
    )


def base_window() -> TrainingWindow:
    """Return GAM-K's training window: the last 24 months before the cut."""
    return TrainingWindow(months=WINDOW_MONTHS)


def arm_spec(*, site: str) -> dict:
    """Return the running arm's solar-term spec at a site; the fleet shape if none is stored."""
    specs = json.loads(SPEC_PATH.read_text()) if SPEC_PATH.exists() else {}
    return specs.get(ARM, {}).get(site, {})


def plant_of(*, site: str) -> PlantParameters:
    """Return the plant whose output is the running arm's unit PV shape at a site.

    A scale multiplies the DC and AC capacities together, so the output scales and the DC:AC ratio
    and the inverter clip stay as they are.
    """
    spec = arm_spec(site=site)
    scale = float(spec.get("scale", 1.0))
    return PlantParameters(
        tilt_deg=float(spec.get("tilt_deg", FLEET.tilt_deg)),
        azimuth_deg=float(spec.get("azimuth_deg", FLEET.azimuth_deg)),
        dc_capacity_mw=FLEET.dc_capacity_mw * scale,
        ac_capacity_mw=FLEET.ac_capacity_mw * scale,
    )


def clip_of(*, site: str, pinned: PinnedCapacity) -> ClipSpec | None:
    """Return the running arm's export clip at a site, or `None`.

    A spec with `clip_table` (cut date to share) clips at that share of the capacity held after the
    cut, learned at each cut by `e00252_learn.py`. A spec with `beta` sets the clip's sharpness.
    A spec with `clip_share` clips at that share of the capacity held after the cut. A spec with
    `clip_points` clips at that fixed number of points at every cut, whatever the held capacity.
    """
    spec = arm_spec(site=site)
    if not spec.get("clip"):
        return None
    beta = float(spec.get("beta", CLIP_BETA_PER_POINT))
    if "clip_table" in spec:
        share = float(spec["clip_table"][pinned.cut.strftime("%Y-%m-%d")])
        return ClipSpec(p_max_share=share, beta=beta)
    if "clip_points" in spec:
        return ClipSpec(
            p_max_share=float(spec["clip_points"]) / float(pinned.held_points), beta=beta
        )
    return ClipSpec(p_max_share=float(spec.get("clip_share", 1.0)), beta=beta)


def masked_cache(*, cache: FeatureCache, site: str) -> FeatureCache:
    """Return the cache with D8's rows before its first day removed; other sites are unchanged."""
    if site != "D8":
        return cache
    features = cache.sites[site]
    early = cache.end_times <= naive_utc(moment=D8_FIRST_DAY)
    return replace_site(
        cache=cache,
        site=site,
        y=np.where(early, np.nan, features.y),
        present=features.present & ~early,
        default_rows=features.default_rows & ~early,
    )


def variant_cache(*, cache: FeatureCache, site: str) -> FeatureCache:
    """Return the cache with the site's training-row unit PV shape from the running arm's plant."""
    cache = masked_cache(cache=cache, site=site)
    features = cache.sites[site]
    n_even = len(cache.end_times) // 2 * 2
    sun = SunGeometry.build(
        end_times=cache.end_times[:n_even],
        latitude=features.table.latitude,
        longitude=features.table.longitude,
    )
    shape, _ = sun.pv_shape_and_ghi(
        ghi_w_m2=features.ghi_w_m2[None, :n_even], plant=plant_of(site=site)
    )
    # The grid ends at 23:30 on a June night, so the unpaired last half-hour has no output.
    shape = np.concatenate([shape[0], np.zeros(len(cache.end_times) - n_even)])[None, :]
    return replace_site(cache=cache, site=site, pv_unit_shape=shape[0])


def arm_file(*, arm: str, period: str, site: str) -> Path:
    """Return the file of a stage-2 arm's forecasts for one period at one site."""
    return ARM_DIR / "dp_preds_s2" / f"{arm}_{period}_{site}.parquet"


def vintages_for(*, cache: FeatureCache) -> pl.DataFrame:
    """Return experiment 45's vintages for the control, and zero-capacity vintages otherwise."""
    return read_vintages() if SOURCE == "control" else zero_vintages(cache=cache)


def zero_vintages(*, cache: FeatureCache) -> pl.DataFrame:
    """Return capacity vintages that hold a PV capacity of zero at every scored site.

    No real-time capacity vintage exists for the eight sites, so the pinned solar term is zero. The
    frame has the columns `pinned_capacity` reads, with one origin Monday and weekly rows from 2019.
    """
    weeks = pl.date_range(date(2019, 1, 7), date(2027, 1, 4), interval="1w", eager=True)
    frames = [
        pl.DataFrame({"week_start": weeks}).with_columns(
            time_series_id=pl.lit(int(cache.sites[site].table.time_series_id), dtype=pl.Int32),
            variant=pl.lit("frozen"),
            origin_monday=pl.lit(date(2023, 8, 7)),
            capacity_mw=pl.lit(0.0),
            unobserved=pl.lit(False),
        )
        for site in SCORED_SITES
    ]
    return pl.concat(frames)
