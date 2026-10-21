"""Constants and helpers shared by the scripts of experiment 161 (BSP top-down against bottom-up).

The chain is experiment 136's arm A2 (T-all, POOL, monthly refits) followed by experiment 155's
arm D (winter night offset, then recipe R, with arm A's spread table), copied so this folder
imports only itself. The constants, periods, and run dates are experiment 111's.

**Three periods, each with one role.** The tuning period is the odd months of 2024-07 to 2025-06
(the spread's fit), the replicate period is the even months of the same year (the sign of the
contrasts), and the scored year is the 292 runs of `fold_a_daily_gap_v1` from 2025-07-02 to
2026-06-29. A run belongs to the month of its `power_fcst_init_time`.
"""

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from scipy.optimize import lsq_linear, nnls
from studies.dp_features import (
    DpInputPaths,
    FeatureCache,
    SiteFeatures,
    default_input_paths,
    load_or_build_feature_cache,
    naive_utc,
)
from studies.dp_gam import GamSpec, TrainingWindow
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 161
SLUG: Final[str] = "dp-bsp-top-down-against-bottom-up"
ROW_SET_ID: Final[str] = "fold_a_daily_gap_v1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
STORE_DIR: Final[Path] = WORK_DIR / "ens_store"
CALIBRATION_DIR: Final[Path] = WORK_DIR / "calibration"
INPUT_DIR: Final[Path] = WORK_DIR / "inputs"
OUT_DIR: Final[Path] = WORK_DIR / "dp_preds"
SOURCE_WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00111-differentiable-physics-6-ens-forecast-engine-against-bg2"
    / "work"
)
"""Experiment 111's private folder: its tuned values and the children's stored fits."""
E155_PREDS_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00155-dp-bg-stage2-assembly-night-offset-and-error-correction"
    / "work"
    / "dp_preds"
)
PANEL_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00103-differentiable-physics-1-data-look" / "panel.parquet"
)
CACHE_DIR: Final[Path] = WORK_DIR / "cache"
OLD_CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
FITS_DIR: Final[Path] = WORK_DIR / "tall_fits"

BSPS: Final[dict[str, tuple[str, ...]]] = {
    "D13": ("D9", "D10", "D11"),
    "D17": ("D5", "D14", "D16"),
}
"""Each BSP and the covered children: the Primaries that report real power, have a long history, and
have an arm D forecast in experiment 155."""
VARIANT_SUFFIX: Final[dict[str, str]] = {"learned": "r", "ones": "s", "free": "f"}
"""The label suffix of a remainder series: bounded weights, all weights 1, unbounded weights."""
WEIGHT_BOUNDS: Final[tuple[float, float]] = (0.8, 1.2)
"""Experiment 124 fitted multipliers of 1.01 to 1.16 on the Primaries beneath a BSP."""
WEIGHT_WINDOW_START: Final[datetime] = datetime(2023, 10, 1, tzinfo=UTC)
WEIGHT_WINDOW_END: Final[datetime] = datetime(2024, 3, 31, 23, 30, tzinfo=UTC)
LOCKDOWN_START: Final[date] = date(2020, 3, 23)
LOCKDOWN_END: Final[date] = date(2021, 7, 19)
CHAIN_SITES: Final[tuple[str, ...]] = tuple(
    f"{bsp}{suffix}" for bsp in BSPS for suffix in ("", *VARIANT_SUFFIX.values())
)
"""The series the chain forecasts: each BSP and its three remainders."""

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


def bsp_rows() -> pl.DataFrame:
    """Return the cache rows of the BSPs, built from experiment 103's panel by its rows rule.

    A half-hour is a default row when the reading is present and unflagged, on or after the series'
    first usable instant, outside the fixed lockdown window, and outside the hand-filtered
    switching events.
    """
    paths = default_input_paths()
    first_usable = {
        r["site"]: r["first_usable"]
        for r in pl.read_parquet(paths.sites, columns=["site", "first_usable"]).iter_rows(
            named=True
        )
    }
    panel = pl.read_parquet(PANEL_PATH).filter(pl.col("site").is_in(list(BSPS)))
    in_lockdown = (pl.col("local_date") >= LOCKDOWN_START) & (pl.col("local_date") <= LOCKDOWN_END)
    return (
        panel.with_columns(
            usable=pl.col("time") >= pl.col("site").replace_strict(first_usable),
            lockdown=in_lockdown,
        )
        .with_columns(in_pool=pl.col("present") & ~pl.col("flagged") & pl.col("usable"))
        .with_columns(
            default_rows=pl.col("in_pool") & ~pl.col("lockdown") & ~pl.col("switching_mask")
        )
        .select(
            "time_series_id", "site", "time", "power", "present", "in_pool", "default_rows",
            "lockdown", "switching_mask", "dark",
        )
    )  # fmt: skip


def extended_paths() -> DpInputPaths:
    """Return the input paths with a rows file that adds the BSPs to the 16 Primaries."""
    paths = default_input_paths()
    target = INPUT_DIR / "rows_with_bsps.parquet"
    if not target.exists():
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        rows = pl.concat([pl.read_parquet(paths.rows), bsp_rows()], how="diagonal_relaxed")
        temporary = target.with_name(f"{target.name}.tmp")
        rows.write_parquet(temporary)
        temporary.rename(target)
    return replace(paths, rows=target)


def load_base_cache() -> FeatureCache:
    """Return the feature cache of the 16 Primaries and the two BSPs, building it on first use."""
    return load_or_build_feature_cache(paths=extended_paths(), cache_dir=CACHE_DIR)


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


def _megawatts(*, cache: FeatureCache, site: str) -> np.ndarray:
    """Return a site's reading in MW on the cache grid, `NaN` where there is none."""
    features = cache.sites[site]
    return features.y * features.table.effective_capacity_mw / 100.0


def weight_rows(*, cache: FeatureCache, bsp: str, end: datetime) -> np.ndarray:
    """Return the grid rows in the weight window, outside flags and switching events."""
    children = BSPS[bsp]
    stamps = cache.end_times
    inside = (stamps >= naive_utc(moment=WEIGHT_WINDOW_START)) & (stamps <= naive_utc(moment=end))
    ok = inside & cache.sites[bsp].in_pool & ~cache.sites[bsp].switching_mask
    for child in children:
        ok &= cache.sites[child].in_pool & ~cache.sites[child].switching_mask
    return np.flatnonzero(ok)


def fit_weights(*, cache: FeatureCache, bsp: str, end: datetime, bounded: bool = True) -> dict:
    """Fit the covered children's weights by bounded least squares, with no intercept.

    With `bounded` False the weights are only non-negative.

    The target is the BSP's reading in MW and the columns are the children's readings in MW, on the
    half-hours of `weight_rows`.

    Args:
        cache: The feature cache holding the BSP and its children.
        bsp: The BSP's label.
        end: The last half-hour of the weight window.
        bounded: Whether to bound the weights to 0.8 to 1.2.

    Returns:
        The weights by child, the number of rows, the in-sample residual standard deviation in MW,
        and the share of the BSP's variance the fit leaves unexplained.
    """
    rows = weight_rows(cache=cache, bsp=bsp, end=end)
    target = _megawatts(cache=cache, site=bsp)[rows]
    columns = np.column_stack([_megawatts(cache=cache, site=c)[rows] for c in BSPS[bsp]])
    if bounded:
        weights = lsq_linear(columns, target, bounds=WEIGHT_BOUNDS).x
    else:
        weights, _ = nnls(columns, target)
    residual = target - columns @ weights
    return {
        "weights": dict(zip(BSPS[bsp], (float(w) for w in weights), strict=True)),
        "rows": len(rows),
        "residual_sd_mw": float(residual.std()),
        "unexplained_share": float(residual.var() / target.var()),
    }


def weights_for(*, cache: FeatureCache, bsp: str, variant: str) -> dict[str, float]:
    """Return the weights of a remainder variant: learned, ones, or free."""
    if variant == "ones":
        return dict.fromkeys(BSPS[bsp], 1.0)
    fit = fit_weights(cache=cache, bsp=bsp, end=WEIGHT_WINDOW_END, bounded=variant == "learned")
    return fit["weights"]


def remainder_features(
    *, cache: FeatureCache, bsp: str, weights: dict[str, float], label: str
) -> SiteFeatures:
    """Return the BSP's features with the reading replaced by the remainder, in the BSP's points.

    The remainder is the BSP's reading minus the weighted children's readings. It is `NaN` where
    any child lacks a reading, and a row is a default row only where the BSP's default row holds
    and every child is present, unflagged, and outside the switching events.
    """
    features = cache.sites[bsp]
    cap = features.table.effective_capacity_mw
    y = features.y.copy()
    default = features.default_rows.copy()
    for child, weight in weights.items():
        other = cache.sites[child]
        y = y - weight * other.y * other.table.effective_capacity_mw / cap
        default &= other.in_pool & ~other.switching_mask
    return replace(
        features,
        table=replace(features.table, site=label),
        y=y,
        present=features.present & np.isfinite(y),
        default_rows=default,
    )


def load_cache() -> FeatureCache:
    """Return the base cache plus every remainder series, each on its own label."""
    cache = load_base_cache()
    extra = {}
    for bsp in BSPS:
        for variant, suffix in VARIANT_SUFFIX.items():
            weights = weights_for(cache=cache, bsp=bsp, variant=variant)
            extra[f"{bsp}{suffix}"] = remainder_features(
                cache=cache, bsp=bsp, weights=weights, label=f"{bsp}{suffix}"
            )
    return replace(cache, sites={**cache.sites, **extra})


PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])

WINTER_MONTHS: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
N_NIGHT_SLOTS: Final[int] = 24
NIGHT_START_MINUTE: Final[int] = 21 * 60 + 30
"""The first night slot is the half-hour that ends at 21:30 UTC; the last ends at 09:00 UTC."""
OFFSET_RIDGE: Final[float] = 100.0
"""Experiment 137's ridge penalty, chosen on its odd tuning months."""
R_KAPPA: Final[float] = 0.75
R_PROFILE_DAYS: Final[int] = 14
R_MIN_DAYS: Final[int] = 3
R_FIRST_SOURCE_BACK: Final[int] = 2
"""The newest source day is `T - 2`: the target day `T - 1` is the origin's own day."""
MIN_DAY_READINGS: Final[int] = 40
D1_FIRST_LEAD: Final[float] = 15.5
D1_LAST_LEAD: Final[float] = 39.0
SHORT_LEAD_LIMIT_HOURS: Final[float] = 48.0
HORIZON_HOURS: Final[float] = 168.0

BASE_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id", "run", "power_fcst_init_time", "valid_time", "lead_hours", "seed",
    *QUANTILES, "actual", "mean", "hour_band", "quarter", "switching_mask",
)  # fmt: skip
