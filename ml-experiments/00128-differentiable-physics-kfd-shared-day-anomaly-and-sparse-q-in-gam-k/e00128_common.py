"""Constants, paths, and loaders shared by the scripts of experiment 128 (the day-level state, KFD).

The experiment reuses experiment 111's stored inputs read-only: the feature cache of Part 4, the
ensemble (ENS) store, the weather calibration, the tuned solar shrink and filter hyperparameters,
GAM-K's spread, and the monthly convex fits. It writes only under its own data folder.

**Three periods, each with one role.** The tuning period is the odd months of 2024-07 to 2025-06
(the filter hyperparameters, the spread factors), the replicate period is the even
months of the same year (the sign of the contrasts), and the scored year is the 292 runs of
`fold_a_daily_gap_v1`. A run belongs to the month of its `power_fcst_init_time`.
"""

import json
import pickle
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
from studies.dp_ens import LeadHourSpread
from studies.dp_ens_weather import (
    AnalysisProxy,
    Calibration,
    EnsSiteRun,
    WeatherContext,
    ghi_climatology,
)
from studies.dp_features import (
    FeatureCache,
    default_input_paths,
    load_or_build_feature_cache,
)
from studies.dp_gam import GamFit, ResidualSpread
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 128
SLUG: Final[str] = "differentiable-physics-kfd-shared-day-anomaly-and-sparse-q-in-gam-k"
ROW_SET_ID: Final[str] = "fold_a_daily_gap_v1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"

E111_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00111-differentiable-physics-6-ens-forecast-engine-against-bg2"
)
E111_WORK: Final[Path] = E111_DIR / "work"
"""Experiment 111's work folder, read-only."""
E118_WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00118-differentiable-physics-kfe-kalman-filter-on-the-gam-effects-on-ens"
    / "work"
)
"""Experiment 118's work folder, read-only: its KEZ predictions are a reference."""
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
"""Part 4's feature cache, shared because it is keyed by a hash of its inputs."""

DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
"""The sites counted for the decision: 4 of 5 must improve."""
SCORED_SITES: Final[tuple[str, ...]] = (*DECISION_SITES, "D9")

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
SEASON_YEARS: Final[range] = range(2019, 2027)
PROXY_WARM_UP_RUNS: Final[int] = 9
"""Runs before this index lack a week of analysis proxy before their origin."""
N_LEAD_BANDS: Final[int] = 4
SPREAD_FACTOR_GRID: Final[tuple[float, ...]] = (0.8, 0.85, 0.9, 1.0, 1.15, 1.3)
"""The spread factor values tried per lead band."""
HALF_HOURS_PER_HOUR: Final[int] = 2


def gamk_hyper(*, site: str) -> list[float]:
    """Return experiment 111's four GAM-K hyperparameters of a site on the unconstrained scale.

    The values are log q, logit phi, log sigma_z, and log sigma_eps.
    """
    return tuned_111()["filter_hyper"]["gamk"][site]


def local_day_index(*, cache: FeatureCache) -> np.ndarray:
    """Return the local (Europe/London) calendar day of each grid row, in days since the first row.

    The day is the local date the half-hour starts on, so each hour band falls on one day and the
    day lasts 23 or 25 hours at a clock change.
    """
    dates = cache.calendar.local_date
    return (dates - dates[0]).astype(np.int64)


def load_cache() -> FeatureCache:
    """Return the feature cache, building and saving it on the first call."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def run_dates() -> list[datetime]:
    """Return every daily 00:00 UTC run from 2024-04-01 to 2026-06-30."""
    return [FIRST_RUN + timedelta(days=i) for i in range((LAST_RUN - FIRST_RUN).days + 1)]


def origin_of(*, run: datetime) -> datetime:
    """Return a run's `power_fcst_init_time`: its initialisation plus the publication delay."""
    return run + PUBLICATION_DELAY


def period_of_run(*, run: datetime) -> str:
    """Return `tuning`, `replicate`, `scored`, `stride`, or `warm_up` for a run.

    `stride` is a run of the scored year on the every-fifth-day stride, which is never scored;
    `warm_up` is a run before 2024-07 or in no named period.
    """
    origin = origin_of(run=run)
    key = (origin.year, origin.month)
    if key in TUNING_MONTHS:
        return "tuning"
    if key in REPLICATE_MONTHS:
        return "replicate"
    if run >= SCORED_FIRST_RUN:
        return "scored" if (run - SCORED_FIRST_RUN).days % EVERY5D_STRIDE_DAYS != 0 else "stride"
    return "warm_up"


def in_months(*, cache: FeatureCache, months: frozenset[tuple[int, int]]) -> np.ndarray:
    """Return a mask of the grid rows whose half-hour starts in one of the (year, month) pairs."""
    starts = cache.end_times - np.timedelta64(30, "m")
    year = starts.astype("datetime64[Y]").astype(np.int64) + 1970
    month = starts.astype("datetime64[M]").astype(np.int64) % 12 + 1
    mask = np.zeros(len(starts), dtype=bool)
    for y, m in months:
        mask |= (year == y) & (month == m)
    return mask


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True))
    temporary.rename(path)


@dataclass(frozen=True)
class SiteSetup:
    """A substation's ensemble runs and the weather layer built from them.

    Attributes:
        site: The substation's label.
        run_dates: The 00:00 UTC initialisation time of each run.
        origins: Each run's `power_fcst_init_time`.
        runs: Each run's ensemble weather on the cache's grid.
        context: The calibration, climatology, and analysis proxy.
    """

    site: str
    run_dates: list[datetime]
    origins: list[datetime]
    runs: list[EnsSiteRun]
    context: WeatherContext

    def indices(self, *, periods: set[str]) -> list[int]:
        """Return the runs that belong to the named periods."""
        return [
            index for index, run in enumerate(self.run_dates) if period_of_run(run=run) in periods
        ]


def tuned_111() -> dict:
    """Return experiment 111's tuned settings (solar shrink, hyperparameters, spread factors)."""
    return json.loads((E111_WORK / "tuned.json").read_text())


def load_calibration(*, site: str) -> Calibration:
    """Return experiment 111's stored calibration of a site, with its tuned solar shrink."""
    with np.load(E111_WORK / "calibration" / f"{site}.npz") as stored:
        calibration = Calibration(
            temperature_shift_c=stored["shift"],
            irradiance_ratio=stored["ratio"],
            solar_shrink=stored["shrink"],
        )
    return calibration.with_solar_shrink(solar_shrink=tuned_111()["solar_shrink"])


def load_setup(*, cache: FeatureCache, site: str) -> SiteSetup:
    """Load a substation's runs from experiment 111's ensemble store, with its calibration.

    Args:
        cache: The feature cache.
        site: The substation's label.

    Returns:
        The setup.
    """
    with np.load(E111_WORK / "ens_store" / f"{site}.npz") as stored:
        arrays = {name: stored[name] for name in stored.files}
    dates = run_dates()
    origins = [origin_of(run=run) for run in dates]
    positions = np.searchsorted(cache.end_times, arrays["origins"])
    runs = [
        EnsSiteRun(
            origin_position=int(position),
            temperature_c=arrays["temperature"][index],
            shortwave_w_m2=arrays["shortwave"][index],
        )
        for index, position in enumerate(positions)
    ]
    context = WeatherContext(
        site=site,
        calibration=load_calibration(site=site),
        climatology=ghi_climatology(
            cache=cache, site=site, before=datetime(2024, 7, 1, tzinfo=UTC)
        ),
        proxy=AnalysisProxy.build(runs=runs, n_rows=len(cache.end_times)),
    )
    return SiteSetup(site=site, run_dates=dates, origins=origins, runs=runs, context=context)


def load_fits(*, site: str) -> list[GamFit]:
    """Return experiment 111's monthly no-drift convex fits (`gamk`) at a site, ascending by cut."""
    return pickle.loads((E111_WORK / "fits" / "gamk" / f"{site}.pkl").read_bytes())


def load_gamk_spread(*, site: str) -> LeadHourSpread:
    """Return GAM-K's stored spread at a site, with its tuned factors."""
    with np.load(E111_WORK / "spread" / f"gamk_{site}.npz") as stored:
        tables, factors = stored["tables"], stored["factors"]
    spread = LeadHourSpread(
        by_lead_band=tuple(ResidualSpread(table=table) for table in tables), factors=factors
    )
    return spread.with_factors(factors=tuned_111()["spread_factors"]["gamk"])


def with_calibration(*, setup: SiteSetup, calibration: Calibration) -> SiteSetup:
    """Return the setup with another calibration."""
    return replace(setup, context=replace(setup.context, calibration=calibration))
