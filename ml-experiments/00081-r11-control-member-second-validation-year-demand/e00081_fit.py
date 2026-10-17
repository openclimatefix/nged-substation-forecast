"""Fit the arms of experiment 00081 and write each refit's quantiles for the daily control runs.

One job is one refit of one configuration. A refit at time `t` trains each series' N1 model on the
control member's rows whose valid time is at least 3.5 hours before `t`, weighted by a recency
half-life anchored at `t`. Rows are not cut by initialisation time: a row that is old enough by
valid time necessarily comes from an earlier run. The job then forecasts the daily runs that the
configuration's regimes assign to it, from the frames of `e00081_build_frames.py`, and writes the
nine quantiles of the control member's forecast to `work/anchor_preds/<fold>/<config>/`, keyed by
series, run, and valid time. Existing files stay, so a stopped run resumes. Seed 0 only.

Configurations (`CONFIGS`):

- `frozen`: trained once on rows to the fold's anchor (the half-hour before its first run),
  half-life 2 years, 500 rounds. The assemble step turns it into the stack (14-day correction and
  spread factor) and into the frozen arm without correction.
- `n1h`: the weekly and the monthly regimes (half-life 0.5 years, 500 rounds, `eta` 0.05).
- `roll12`, `roll12_no_christmas`, `roll12_hl2y`: experiment 00049's P-roll, a refit every month on
  the latest 12 months (150 rounds, `eta` 0.15), with the Christmas column dropped, or with a
  half-life of 2 years.

Two baselines read the observed power alone: `manual_heuristic` (the power at the same weekday and
time of day 1 to 6 weeks and 49 to 55 weeks earlier, as ensemble members) and `climatology` (the 51
pooled quantiles of power in each local month, half-hour, and day type, fitted on valid times before
the fold's first run). Both are reduced to the nine quantile levels.

Folds: `b` (2023-07 to 2024-03), `c` (2022-07 to 2023-03), and `d` (the descriptive 9 km check,
which runs only `frozen` and `manual_heuristic`). With `--smoke` one series is fitted for fold `b`
at its first refit on the three runs 2023-07-01 to 2023-07-03, with 20 rounds, from the six smoke
frames.

The jobs run as parallel worker processes, each with `10 // workers` threads, and each loads only
the series of the job it is working on. GPU memory use is one booster per worker.

    uv run python ml-experiments/00081-*/e00081_fit.py --fold b c --arm frozen n1h [--workers 4]
    uv run python ml-experiments/00081-*/e00081_fit.py --smoke
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import e00081_build_frames as build
import e00081_common as common
import e00081_control_nwp as control
import numpy as np
import polars as pl
import xgboost as xgb
from baseline_forecasters.climatology import (
    _CELL_KEY_COLUMNS,
    CLIMATOLOGY_QUANTILE_COLUMNS,
    _local_calendar_cell_keys,
    _pooled_quantile_lookup,
)
from ml_core.features.feature_engineer import DEFAULT_LOCAL_TIMEZONE

CORES: Final[int] = 10
REFIT_CUTOFF_AGE: Final[timedelta] = timedelta(hours=3.5)
"""A refit sees only rows whose valid time is at least this long before the refit."""
SMOKE_ROUNDS: Final[int] = 20
SMOKE_RUNS: Final[tuple[str, ...]] = ("2023-07-01", "2023-07-02", "2023-07-03")
WEATHER: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
)
"""The weather columns of the frozen feature set; the other feature columns are derived."""
MANUAL_LAG_HOURS: Final[tuple[int, ...]] = (
    168,
    336,
    504,
    672,
    840,
    1008,
    8232,
    8400,
    8568,
    8736,
    8904,
    9072,
    9240,
)
"""`conf/model/manual_heuristic.yaml`'s power lags: weeks 1 to 6 and weeks 49 to 55."""
FIT_ARMS_BY_FOLD: Final[dict[common.FoldNameType, tuple[str, ...]]] = {
    "b": ("frozen", "n1h", "roll12", "roll12_no_christmas", "roll12_hl2y"),
    "c": ("frozen", "n1h", "roll12", "roll12_no_christmas", "roll12_hl2y"),
    "d": ("frozen",),
}
BASELINES_BY_FOLD: Final[dict[common.FoldNameType, tuple[str, ...]]] = {
    "b": ("manual_heuristic", "climatology"),
    "c": ("manual_heuristic", "climatology"),
    "d": ("manual_heuristic",),
}


@dataclass(frozen=True)
class Config:
    """A training configuration.

    Attributes: name: The folder name under `anchor_preds/<fold>`. half_life_days: Recency half-life
    of the row weights, anchored at the refit time. window_days: Keep only rows whose valid time is
    within this many days of the refit, or `None` for the whole history. rounds: Boosting rounds.
    eta: Learning rate. regime_names: The regimes (`weekly`, `monthly`, `frozen`) whose refits the
    configuration fits. frozen: Train once on rows to the fold's anchor, with no later rows. drop:
    Frozen-feature-set columns left out of the input.
    """

    name: str
    half_life_days: float
    window_days: float | None
    rounds: int
    eta: float
    regime_names: tuple[str, ...]
    frozen: bool = False
    drop: tuple[str, ...] = ()


CONFIGS: Final[dict[str, Config]] = {
    "frozen": Config(
        "frozen",
        common.RECENCY_HALF_LIFE_DAYS,
        None,
        common.N_ROUNDS,
        0.05,
        ("frozen",),
        frozen=True,
    ),
    "n1h": Config(
        "n1h", common.SHORT_HALF_LIFE_DAYS, None, common.N_ROUNDS, 0.05, ("weekly", "monthly")
    ),
    "roll12": Config("roll12", common.SHORT_HALF_LIFE_DAYS, 365.0, 150, 0.15, ("monthly",)),
    "roll12_no_christmas": Config(
        "roll12_no_christmas",
        common.SHORT_HALF_LIFE_DAYS,
        365.0,
        150,
        0.15,
        ("monthly",),
        drop=("days_to_christmas",),
    ),
    "roll12_hl2y": Config(
        "roll12_hl2y", common.RECENCY_HALF_LIFE_DAYS, 365.0, 150, 0.15, ("monthly",)
    ),
}
ARM_CHOICES: Final[tuple[str, ...]] = (*CONFIGS, "manual_heuristic", "climatology")


def input_columns(*, drop: tuple[str, ...] = ()) -> list[str]:
    """Return the input columns in matrix order, without the dropped ones."""
    return [
        *(column for column in common.feature_columns() if column not in drop),
        *common.N1_EXTRA_COLUMNS,
        *common.LAG_COLUMNS,
    ]


def _midnight(*, day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def refit_times(*, fold: common.Fold, regime: str) -> tuple[datetime, ...]:
    """Return a regime's refit times for a fold.

    Args:
        fold: The fold whose runs the refits serve.
        regime: `monthly` (the first of each month from the fold's first run), `weekly` (every 7
            days from the fold's first run), or `frozen` (the fold's anchor alone).

    Returns:
        The refit times in ascending order, all at 00:00 UTC except the anchor at 23:30 the day
        before.

    Raises:
        ValueError: If the regime is unknown.
    """
    first, last = _midnight(day=fold.first_run), _midnight(day=fold.last_run)
    if regime == "frozen":
        return (fold.anchor,)
    if regime == "weekly":
        return tuple(first + timedelta(days=7 * i) for i in range((last - first).days // 7 + 1))
    if regime == "monthly":
        months = (last.year - first.year) * 12 + last.month - first.month + 1
        return tuple(
            datetime(
                first.year + (first.month - 1 + i) // 12,
                (first.month - 1 + i) % 12 + 1,
                1,
                tzinfo=UTC,
            )
            for i in range(months)
        )
    msg = f"unknown regime {regime!r}"
    raise ValueError(msg)


def chosen_refit(*, refits: tuple[datetime, ...], origin: datetime) -> datetime:
    """Return the latest refit at or before an origin.

    Raises:
        ValueError: If every refit is later than the origin.
    """
    earlier = [refit for refit in refits if refit <= origin]
    if not earlier:
        msg = f"no refit at or before {origin}"
        raise ValueError(msg)
    return max(earlier)


def served(
    *, fold: common.Fold, regime: str, origins: list[datetime]
) -> dict[datetime, list[datetime]]:
    """Return, for a regime, each refit time and the origins whose latest refit it is."""
    out: dict[datetime, list[datetime]] = {}
    refits = refit_times(fold=fold, regime=regime)
    for origin in sorted(origins):
        out.setdefault(chosen_refit(refits=refits, origin=origin), []).append(origin)
    return out


def jobs_for(
    *, fold: common.Fold, regime_names: tuple[str, ...], origins: list[datetime]
) -> dict[datetime, list[datetime]]:
    """Return each refit time and the union of origins that any of the named regimes assigns it.

    The weekly and monthly regimes share the fold's first refit and assign it different origins, so
    one fit there forecasts the union and a file holds every origin a regime may ask for.
    """
    jobs: dict[datetime, set[datetime]] = {}
    for name in regime_names:
        for refit, assigned in served(fold=fold, regime=name, origins=origins).items():
            jobs.setdefault(refit, set()).update(assigned)
    return {refit: sorted(jobs[refit]) for refit in sorted(jobs)}


def add_lags(*, frame: pl.DataFrame, power: np.ndarray) -> pl.DataFrame:
    """Add the three origin lags to a frame of runs."""
    target = np.asarray(common.grid_index(times=frame["valid_time"]))
    origin = np.asarray(common.grid_index(times=frame["power_fcst_init_time"]))
    return frame.with_columns(
        pl.DataFrame(
            common.origin_lags(power=power, target=target, origin=origin),
            schema=list(common.LAG_COLUMNS),
            orient="row",
        ).cast(pl.Float32)
    )


def training_rows(*, row: dict, power: np.ndarray, frames_dir: Path) -> pl.DataFrame:
    """Return one series' control-member rows from every daily run, with derived columns and lags.

    Every row's eligibility for a refit (its valid time) is applied per refit, not here.

    Args:
        row: The series' row of `common.series_table()`.
        power: The series' observed power over capacity on the dense grid.
        frames_dir: The folder of the daily frames.

    Returns:
        The rows keyed by series, run, and valid time, with `power` (a fraction of capacity), the
        weather columns, the derived columns, `lead_hours`, and the three lags. Rows whose power is
        unobserved are dropped.
    """
    rows = (
        pl.scan_parquet(frames_dir / "init=*.parquet")
        .filter(pl.col("time_series_id") == row["time_series_id"])
        .select(*common.KEYS, "lead_hours", *WEATHER)
        .collect()
    )
    index = np.asarray(common.grid_index(times=rows["valid_time"]))
    rows = rows.with_columns(power=pl.Series(common.take(values=power, index=index)))
    rows = rows.filter(pl.col("power").is_finite())
    rows = common.with_derived(frame=rows, latitude=row["latitude"], longitude=row["longitude"])
    return add_lags(frame=rows, power=power)


def known_rows(*, rows: pl.DataFrame, refit: datetime, config: Config) -> pl.DataFrame:
    """Return the rows a refit at `refit` may see, and assert none is too new.

    A refit sees a row only if its valid time is at least 3.5 hours before the refit (the frozen
    model: no later than its anchor), and, for a rolling window, only the latest `window_days`.

    Raises:
        ValueError: If a kept row's valid time is later than the cut-off.
    """
    cutoff = refit if config.frozen else refit - REFIT_CUTOFF_AGE
    known = rows.filter(pl.col("valid_time") <= cutoff)
    if config.window_days is not None:
        known = known.filter(pl.col("valid_time") > refit - timedelta(days=config.window_days))
    if not known.is_empty() and known["valid_time"].max() > cutoff:  # ty: ignore[unsupported-operator]
        msg = f"a training row is later than the cut-off {cutoff}"
        raise ValueError(msg)
    return known


def train_matrix(*, rows: pl.DataFrame, refit: datetime, config: Config) -> xgb.DMatrix:
    """Return the weighted training matrix of one series' rows, weights anchored at `refit`."""
    age_days = (refit - rows["valid_time"]).dt.total_seconds() / 86_400
    return xgb.DMatrix(
        rows.select(input_columns(drop=config.drop)).to_numpy().astype(np.float32),
        label=rows["power"].to_numpy(),
        weight=0.5 ** (age_days.to_numpy() / config.half_life_days),
    )


def fit(*, matrix: xgb.DMatrix, config: Config, rounds: int, nthread: int) -> xgb.Booster:
    """Fit one quantile model with seed 0."""
    return xgb.train(
        {**common.QUANTILE_PARAMETERS, "seed": common.SEED, "eta": config.eta, "nthread": nthread},
        matrix,
        num_boost_round=rounds,
    )


def predict_run(
    *,
    model: xgb.Booster,
    run_rows: pl.DataFrame,
    row: dict,
    power: np.ndarray,
    config: Config,
) -> pl.DataFrame:
    """Return the nine quantiles of one run for one series.

    Args:
        model: The series' fitted booster.
        run_rows: The run's frame rows for the series, one per valid time.
        row: The series' row of `common.series_table()`.
        power: The series' observed power over capacity on the dense grid.
        config: The configuration, which says which columns the booster reads.

    Returns:
        One row per valid time: the key columns and the nine sorted quantile columns.

    Raises:
        ValueError: If a valid time appears more than once.
    """
    prepared = common.with_derived(
        frame=run_rows, latitude=row["latitude"], longitude=row["longitude"]
    )
    prepared = add_lags(frame=prepared, power=power).sort("valid_time")
    if prepared.height != prepared.select(*common.KEYS).n_unique():
        msg = "every valid time must appear once"
        raise ValueError(msg)
    matrix = prepared.select(input_columns(drop=config.drop)).to_numpy().astype(np.float32)
    raw = model.predict(xgb.DMatrix(matrix)).reshape(-1, len(common.LEVELS))
    return prepared.select(*common.KEYS).hstack(
        pl.DataFrame(np.sort(raw, axis=1).astype(np.float32), schema=list(common.LEVEL_COLUMNS))
    )


def job_path(*, root: Path, fold: str, config: str, site: str, refit: datetime | None) -> Path:
    """Return the file one refit's predictions for one series are written to.

    A baseline has no refit, and its file name ends `baseline`.
    """
    stamp = "baseline" if refit is None else f"{refit:%Y%m%dT%H%M}"
    return root / fold / config / f"{site}_s{common.SEED}_{stamp}.parquet"


_CACHE: dict[str, object] = {}


def _series_and_power() -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Return the series table and power arrays, loaded once per worker process."""
    if "series" not in _CACHE:
        series = common.series_table()
        _CACHE["series"] = series
        _CACHE["power"] = common.load_power_arrays(series=series)
        _CACHE["rows"] = {}
    return _CACHE["series"], _CACHE["power"]  # ty: ignore[invalid-return-type]


def _training_rows_cached(*, row: dict, power: np.ndarray, frames_dir: Path) -> pl.DataFrame:
    """Return a series' training rows, kept for the worker's later jobs."""
    cache: dict[tuple[str, Path], pl.DataFrame] = _CACHE["rows"]  # ty: ignore[invalid-assignment]
    key = (row["site"], frames_dir)
    if key not in cache:
        cache[key] = training_rows(row=row, power=power, frames_dir=frames_dir)
    return cache[key]


def run_job(
    *,
    fold_name: common.FoldNameType,
    config_name: str,
    refit: datetime,
    origins: list[datetime],
    sites: list[str],
    out_root: Path,
    frames_dir: Path,
    rounds: int | None,
    nthread: int,
) -> list[str]:
    """Fit one refit for each series that lacks its file, forecast its runs, and write the files.

    Args:
        fold_name: The fold whose runs are forecast.
        config_name: A key of `CONFIGS`.
        refit: The refit time.
        origins: The `power_fcst_init_time` of the runs the refit forecasts.
        sites: The series labels to fit.
        out_root: The folder holding one subfolder per fold.
        frames_dir: The folder of the daily frames.
        rounds: Boosting rounds, or `None` for the configuration's own.
        nthread: Threads for each fit.

    Returns:
        One progress line per series fitted.
    """
    config = CONFIGS[config_name]
    series, power_of = _series_and_power()
    todo = [
        site
        for site in sites
        if not job_path(
            root=out_root, fold=fold_name, config=config.name, site=site, refit=refit
        ).exists()
    ]
    if not todo:
        return [f"{fold_name} {config.name} refit {refit:%Y-%m-%d %H:%M}: all files exist"]
    rows_of = {row["site"]: row for row in series.iter_rows(named=True)}
    models: dict[str, xgb.Booster] = {}
    lines = []
    for site in todo:
        started = time.monotonic()
        known = known_rows(
            rows=_training_rows_cached(
                row=rows_of[site], power=power_of[site], frames_dir=frames_dir
            ),
            refit=refit,
            config=config,
        )
        models[site] = fit(
            matrix=train_matrix(rows=known, refit=refit, config=config),
            config=config,
            rounds=config.rounds if rounds is None else rounds,
            nthread=nthread,
        )
        lines.append(
            f"{fold_name} {config.name} {site} refit {refit:%Y-%m-%d %H:%M}: {known.height:,} rows "
            f"to {known['valid_time'].max():%Y-%m-%d %H:%M}, {len(origins)} runs, "
            f"fit {time.monotonic() - started:.0f} s"
        )
    pieces: dict[str, list[pl.DataFrame]] = {site: [] for site in todo}
    ids = {site: rows_of[site]["time_series_id"] for site in todo}
    for origin in origins:
        frame = pl.read_parquet(
            common.frame_path(run=origin - common.PUBLICATION_DELAY, folder=frames_dir)
        )
        for site in todo:
            pieces[site].append(
                predict_run(
                    model=models[site],
                    run_rows=frame.filter(pl.col("time_series_id") == ids[site]),
                    row=rows_of[site],
                    power=power_of[site],
                    config=config,
                )
            )
    for site in todo:
        common.atomic_write(
            frame=pl.concat(pieces[site]).with_columns(seed=pl.lit(common.SEED).cast(pl.Int8)),
            target=job_path(
                root=out_root, fold=fold_name, config=config.name, site=site, refit=refit
            ),
        )
    return lines


def case_keys(*, runs: list[datetime], site_id: int, frames_dir: Path) -> pl.DataFrame:
    """Return the `(series, power_fcst_init_time, valid_time, lead_hours)` cases of some runs."""
    return pl.concat(
        [
            pl.read_parquet(
                common.frame_path(run=run, folder=frames_dir),
                columns=[*common.KEYS, "lead_hours"],
            ).filter(pl.col("time_series_id") == site_id)
            for run in runs
        ]
    )


def manual_heuristic_quantiles(*, cases: pl.DataFrame, power: np.ndarray) -> pl.DataFrame:
    """Return the nine quantiles of the manual heuristic's members for each case.

    The members are the observed powers at the valid time less each of `MANUAL_LAG_HOURS`; a member
    is absent where its lag is no longer than the lead, and a case with no member is dropped.

    Args:
        cases: `case_keys`' output.
        power: The series' observed power over capacity on the dense grid.

    Returns:
        The key columns and the nine quantile columns.
    """
    target = np.asarray(common.grid_index(times=cases["valid_time"]))
    lead = cases["lead_hours"].to_numpy()
    members = np.stack(
        [
            np.where(
                hours > lead,
                common.take(values=power, index=target - 2 * hours),
                np.nan,
            )
            for hours in MANUAL_LAG_HOURS
        ],
        axis=1,
    ).astype(np.float64)
    kept = np.isfinite(members).any(axis=1)
    quantiles = np.nanquantile(members[kept], common.LEVELS, axis=1).T
    return (
        cases.filter(pl.Series(kept))
        .select(common.KEYS)
        .hstack(pl.DataFrame(quantiles.astype(np.float32), schema=list(common.LEVEL_COLUMNS)))
    )


def climatology_quantiles(
    *, cases: pl.DataFrame, power: np.ndarray, before: datetime
) -> pl.DataFrame:
    """Return the nine quantiles of the climatology of power before `before`, for each case.

    The climatology is `baseline_forecasters`' pooled 51 quantiles of each local calendar cell
    (month, half-hour of day, day type, each pooled with its neighbours), reduced to the nine
    levels.

    Args:
        cases: `case_keys`' output.
        power: The series' observed power over capacity on the dense grid.
        before: The first valid time not used for fitting.

    Returns:
        The key columns and the nine quantile columns; a case in a cell with no sample is dropped.
    """
    grid = common.grid_times()
    usable = (np.arange(grid.len()) < int(common.grid_index(times=before))) & np.isfinite(power)
    samples = pl.DataFrame(
        {
            "time_series_id": pl.Series(
                np.full(int(usable.sum()), int(cases["time_series_id"][0])), dtype=pl.Int32
            ),
            "valid_time": grid.filter(pl.Series(usable)),
            "power": power[usable].astype(np.float64),
        }
    )
    lookup = _pooled_quantile_lookup(samples)
    keyed = cases.with_columns(
        **_local_calendar_cell_keys(pl.col("valid_time"), time_zone=DEFAULT_LOCAL_TIMEZONE)
    ).join(lookup, on=list(_CELL_KEY_COLUMNS), how="inner")
    members = keyed.select(CLIMATOLOGY_QUANTILE_COLUMNS).to_numpy().astype(np.float64)
    return keyed.select(common.KEYS).hstack(
        pl.DataFrame(
            np.quantile(members, common.LEVELS, axis=1).T.astype(np.float32),
            schema=list(common.LEVEL_COLUMNS),
        )
    )


def run_baseline(
    *,
    fold: common.Fold,
    arm: str,
    sites: list[str],
    runs: list[datetime],
    out_root: Path,
    frames_dir: Path,
) -> list[str]:
    """Write one baseline's quantiles for each series that lacks its file; return progress lines."""
    series, power_of = _series_and_power()
    lines = []
    for row in series.filter(pl.col("site").is_in(sites)).iter_rows(named=True):
        target = job_path(root=out_root, fold=fold.name, config=arm, site=row["site"], refit=None)
        if target.exists():
            continue
        cases = case_keys(runs=runs, site_id=row["time_series_id"], frames_dir=frames_dir)
        power = power_of[row["site"]]
        if arm == "manual_heuristic":
            table = manual_heuristic_quantiles(cases=cases, power=power)
        else:
            table = climatology_quantiles(
                cases=cases, power=power, before=_midnight(day=fold.first_run)
            )
        common.atomic_write(
            frame=table.with_columns(seed=pl.lit(common.SEED).cast(pl.Int8)), target=target
        )
        lines.append(f"{fold.name} {arm} {row['site']}: {table.height:,} of {cases.height:,} cases")
    return lines


def _summarise(*, root: Path) -> list[str]:
    """Return one line per written file: its rows and the mean of its median quantile."""
    lines = []
    for path in sorted(root.rglob("*.parquet")):
        frame = pl.read_parquet(path)
        lines.append(
            f"{path.relative_to(root)}: {frame.height:,} rows, "
            f"{frame['power_fcst_init_time'].n_unique()} runs, mean q50 {frame['q50'].mean():.4f}, "
            f"mean q05 {frame['q05'].mean():.4f}, mean q95 {frame['q95'].mean():.4f}"
        )
    return lines


def main() -> None:
    """Fit and write the requested folds' arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", nargs="+", choices=sorted(common.FOLDS), default=["b", "c"])
    parser.add_argument("--arm", nargs="+", choices=ARM_CHOICES, default=list(ARM_CHOICES))
    parser.add_argument("--workers", type=int, default=4, help="processes; each gets 10 // workers")
    parser.add_argument(
        "--smoke", action="store_true", help="one series, first refit, 3 runs, 20 rounds, scratch"
    )
    args = parser.parse_args()
    series = common.series_table()
    sites = series["site"].to_list()
    frames_dir, out_root = common.FRAMES_DIR, common.PREDS_DIR
    rounds, workers = None, args.workers
    folds = common.fold_names(selected=args.fold)
    smoke_runs = [datetime.fromisoformat(text).replace(tzinfo=UTC) for text in SMOKE_RUNS]
    if args.smoke:
        sites = sites[:1]
        frames_dir, out_root = (
            build.SMOKE_FRAMES_DIR,
            common.PREDS_DIR.with_name("anchor_preds_smoke"),
        )
        rounds, workers, folds = SMOKE_ROUNDS, 1, common.fold_names(selected=["b"])
    nthread = max(1, CORES // workers)
    os.environ["POLARS_MAX_THREADS"] = str(nthread)
    started = time.monotonic()
    futures = []
    baseline_lines: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for fold_name in folds:
            fold = common.FOLDS[fold_name]
            runs = (
                smoke_runs
                if args.smoke
                else common.fold_runs(fold=fold, folder=control.BACKFILL_DIR)
            )
            origins = [common.origin_of(run=run) for run in runs]
            for name in args.arm:
                if name in CONFIGS and name in FIT_ARMS_BY_FOLD[fold_name]:
                    config = CONFIGS[name]
                    jobs = jobs_for(fold=fold, regime_names=config.regime_names, origins=origins)
                    ordered = sorted(jobs.items(), key=lambda item: -len(item[1]))
                    # Fewer refits than workers (the frozen model has one): split the series.
                    n_chunks = workers if len(ordered) < workers else 1
                    futures += [
                        pool.submit(
                            run_job,
                            fold_name=fold_name,
                            config_name=config.name,
                            refit=refit,
                            origins=assigned,
                            sites=sites[i::n_chunks],
                            out_root=out_root,
                            frames_dir=frames_dir,
                            rounds=rounds,
                            nthread=nthread,
                        )
                        for refit, assigned in ordered
                        for i in range(n_chunks)
                    ]
                elif name in BASELINES_BY_FOLD[fold_name]:
                    baseline_lines += run_baseline(
                        fold=fold,
                        arm=name,
                        sites=sites,
                        runs=runs,
                        out_root=out_root,
                        frames_dir=frames_dir,
                    )
        print("\n".join(baseline_lines), flush=True)
        for done, future in enumerate(futures, start=1):
            for line in future.result():
                print(f"[{done}/{len(futures)}] {line}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s")
    if args.smoke:
        print("\n".join(_summarise(root=out_root)))


if __name__ == "__main__":
    main()
