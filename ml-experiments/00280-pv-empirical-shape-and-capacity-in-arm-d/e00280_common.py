"""Constants and helpers shared by the scripts of experiment 180 (best-guess stage 1 assembly).

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
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00280_shape as shape_module
import numpy as np
from studies.dp_ens_weather import SunGeometry, replace_site
from studies.dp_features import (
    FLEET,
    FeatureCache,
    default_input_paths,
    load_or_build_feature_cache,
)
from studies.dp_gam import ClipSpec, GamSpec, TrainingWindow
from studies.dp_pinned_capacity import PinnedCapacity
from studies.pv_physics import PlantParameters
from studies.pv_plant_fit import ResponseTable
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 280
SLUG: Final[str] = "pv-empirical-shape-and-capacity-in-arm-d"
SOURCE_SLUG: Final[str] = "differentiable-physics-6-ens-forecast-engine-against-bg2"
ROW_SET_ID: Final[str] = "fold_a_daily_gap_v1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
ARM: Final[str] = os.environ.get("E00280_ARM", "c")
"""The arm this process runs; every fit and forecast path is namespaced by it."""
ARM_DIR: Final[Path] = WORK_DIR / "arms" / ARM
SPEC_PATH: Final[Path] = WORK_DIR / "arm_specs.json"
"""`{arm: {site: {tilt_deg, azimuth_deg, scale, clip}}}`; a missing arm or site is the fleet."""
SOURCE_WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00111-{SOURCE_SLUG}" / "work"
STORE_DIR: Final[Path] = SOURCE_WORK_DIR / "ens_store"
FITS_DIR: Final[Path] = SOURCE_WORK_DIR / "fits"
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
"""Part 4's feature cache, shared because it is keyed by a hash of its inputs."""
PART_4_TUNED: Final[Path] = CACHE_DIR.parent / "work" / "tuned.json"

DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
"""The sites counted for the decision: Phase A needs 4 of 5."""
SCORED_SITES: Final[tuple[str, ...]] = (*DECISION_SITES, "D9")
DESCRIPTIVE_SITES: Final[tuple[str, ...]] = ("D18",)
ALL_SITES: Final[tuple[str, ...]] = (*SCORED_SITES, *DESCRIPTIVE_SITES)
COASTAL_SITES: Final[tuple[str, ...]] = ("D5", "D9")

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
    """Return B0's penalties: GAM-K's ridge, no drift, and the capacity pinned to the vintage."""
    return GamSpec(ridge=RIDGE, drift="none", capacity="pinned", final_knot="free")


def base_window() -> TrainingWindow:
    """Return GAM-K's training window: the last 24 months before the cut."""
    return TrainingWindow(months=WINDOW_MONTHS)


def arm_spec(*, site: str) -> dict:
    """Return the running arm's solar-term spec at a site; the fleet shape if none is stored."""
    specs = json.loads(SPEC_PATH.read_text()) if SPEC_PATH.exists() else {}
    return specs.get(ARM, {}).get(site, {})


@dataclass(frozen=True)
class SiteShape(PlantParameters):
    """A plant whose unit PV shape may be the empirical table and may be scaled by season.

    Attributes:
        site: The substation's label, which names its empirical table.
        empirical: True to read the shape from the substation's averaged learned table.
        level: A constant that multiplies the shape.
        april_june: A further factor that multiplies the shape in April, May, and June.
    """

    site: str = ""
    empirical: bool = False
    level: float = 1.0
    april_june: float = 1.0


MAX_EMPIRICAL_SHAPE: Final[float] = 1.3
"""The empirical shape is capped here, in units of a farm's 99th percentile, because the table is
unconstrained in the cells that no farm half-hour reaches."""
APRIL_TO_JUNE: Final[tuple[int, ...]] = (4, 5, 6)


def empirical_table(*, site: str) -> ResponseTable:
    """Return the substation's averaged learned table."""
    return ResponseTable(
        grid=shape_module.RESPONSE_GRID, values=np.load(shape_module.table_path(site=site))
    )


def site_shape(
    *,
    sun: SunGeometry,
    ghi_w_m2: np.ndarray,
    fleet: tuple[np.ndarray, np.ndarray],
    plant: SiteShape,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the shape and irradiance of a `SiteShape` from the fleet's, per `pv_shape_and_ghi`.

    The fleet call has already computed the half-hourly irradiance (`NaN` where the hour is
    missing). The empirical shape reads the table at that irradiance and the sun's height; a
    half-hour with the sun on or below the horizon has no output.
    """
    shape, flux = fleet
    if plant.empirical:
        elevation = 90.0 - np.tile(sun.zenith_deg, (flux.shape[0], 1))
        table = empirical_table(site=plant.site)
        read = table.predict(
            irradiance_w_m2=np.nan_to_num(flux).ravel(), elevation_deg=elevation.ravel()
        ).reshape(flux.shape)
        read = np.clip(read, 0.0, MAX_EMPIRICAL_SHAPE)
        read[elevation <= shape_module.RESPONSE_MIN_ELEVATION_DEG] = 0.0
        shape = np.where(np.isfinite(flux), read, np.nan)
    months = (sun.end_times - np.timedelta64(15, "m")).astype("datetime64[M]").astype(
        np.int64
    ) % 12 + 1
    season = np.where(np.isin(months, APRIL_TO_JUNE), plant.april_june, 1.0)
    return shape * plant.level * season[None, :], flux


_FLEET_SHAPE = SunGeometry.pv_shape_and_ghi


def _pv_shape_and_ghi(
    self: SunGeometry, *, ghi_w_m2: np.ndarray, plant: PlantParameters = FLEET
) -> tuple[np.ndarray, np.ndarray]:
    """Return `SunGeometry.pv_shape_and_ghi`, with `SiteShape` plants handled here."""
    if not isinstance(plant, SiteShape):
        return _FLEET_SHAPE(self, ghi_w_m2=ghi_w_m2, plant=plant)
    fleet = _FLEET_SHAPE(
        self,
        ghi_w_m2=ghi_w_m2,
        plant=PlantParameters(
            tilt_deg=plant.tilt_deg,
            azimuth_deg=plant.azimuth_deg,
            dc_capacity_mw=plant.dc_capacity_mw,
            ac_capacity_mw=plant.ac_capacity_mw,
        ),
    )
    return site_shape(sun=self, ghi_w_m2=ghi_w_m2, fleet=fleet, plant=plant)


SunGeometry.pv_shape_and_ghi = _pv_shape_and_ghi


def plant_of(*, site: str) -> PlantParameters:
    """Return the plant whose output is the running arm's unit PV shape at a site.

    A scale multiplies the DC and AC capacities together, so the output scales and the DC:AC ratio
    and the inverter clip stay as they are. A spec with `empirical`, `level`, or `april_june`
    returns a `SiteShape`, which the patched `SunGeometry.pv_shape_and_ghi` reads.
    """
    spec = arm_spec(site=site)
    scale = float(spec.get("scale", 1.0))
    tilt_deg = float(spec.get("tilt_deg", FLEET.tilt_deg))
    azimuth_deg = float(spec.get("azimuth_deg", FLEET.azimuth_deg))
    dc_capacity_mw = FLEET.dc_capacity_mw * scale
    ac_capacity_mw = FLEET.ac_capacity_mw * scale
    if not {"empirical", "level", "april_june"} & spec.keys():
        return PlantParameters(
            tilt_deg=tilt_deg,
            azimuth_deg=azimuth_deg,
            dc_capacity_mw=dc_capacity_mw,
            ac_capacity_mw=ac_capacity_mw,
        )
    return SiteShape(
        tilt_deg=tilt_deg,
        azimuth_deg=azimuth_deg,
        dc_capacity_mw=dc_capacity_mw,
        ac_capacity_mw=ac_capacity_mw,
        site=site,
        empirical=bool(spec.get("empirical", False)),
        level=float(spec.get("level", 1.0)),
        april_june=float(spec.get("april_june", 1.0)),
    )


def clip_of(*, site: str, pinned: PinnedCapacity) -> ClipSpec | None:
    """Return the running arm's export clip at a site, or `None`.

    A spec with `clip_share` clips at that share of the capacity held after the cut. A spec with
    `clip_points` clips at that fixed number of points at every cut, whatever the held capacity.
    """
    spec = arm_spec(site=site)
    if not spec.get("clip"):
        return None
    if "clip_points" in spec:
        return ClipSpec(p_max_share=float(spec["clip_points"]) / float(pinned.held_points))
    return ClipSpec(p_max_share=float(spec.get("clip_share", 1.0)))


def variant_cache(*, cache: FeatureCache, site: str) -> FeatureCache:
    """Return the cache with the site's training-row unit PV shape from the running arm's plant."""
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
