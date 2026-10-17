"""Fit the demand arms of experiment 00080 and write each refit's quantiles for the daily runs.

The fitting recipe is experiment 00048's (`e00048_fit.py`), and the only difference between a
configuration and its experiment 00048 twin is the starting value: `n1h_m0` and `frozen_m0` start
every quantile level from zero (`base_score` 0, identical to a base margin of zero) instead of from
the estimate XGBoost makes from the training labels. One job is one refit of a configuration. A
refit at time `t` trains each series' N1 model on the ENS control member's rows whose valid time
is at least 3.5 hours before `t` and whose run began at or before `t`, weighted by a recency
half-life anchored at `t`. It then forecasts the daily runs that the configuration's regimes assign
to it (`e00080_regimes`), from the 365 daily runs of 2025-07-01 to 2026-06-30. A job writes, for
each series, the mixture of the 51 members' nine quantiles for the runs to
`work/anchor_preds/<config>/<site>_s0_<refit>.parquet`, keyed by series, run, and valid time.
Existing files stay, so a stopped run resumes. Seed 0 only.

Configurations:

- `n1h_m0`: the weekly regime (half-life 0.5 years, 500 rounds, `eta` 0.05) with a zero start, for
  the 16 Primary substations and the 4 bulk and grid supply points.
- `n1h_def`: the same with XGBoost's estimated start, only for the reproduction check
  (`--limit-refits 2`): its quantiles must match experiment 00048's stored ones.
- `frozen_m0`: the stack's model, trained once on the control member's rows to 2025-06-30 with a
  half-life of 2 years anchored at 2025-06-30 and 500 rounds, with a zero start.

The jobs run as parallel worker processes, each with `CORES // workers` threads, and each loads only
the runs of the job it is working on.

    uv run python ml-experiments/00080-*/e00080_fit.py --config n1h_m0 [--groups Primary BSP GSP]
        [--workers 3] [--smoke]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import e00080_common as common
import e00080_regimes as regimes
import numpy as np
import polars as pl
import xgboost as xgb

CORES: Final[int] = 6
WEATHER: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
)


@dataclass(frozen=True)
class Config:
    """A training configuration.

    Attributes:
        name: The folder name under `anchor_preds`.
        half_life_days: Recency half-life of the row weights, anchored at the refit time.
        window_days: Keep only rows whose valid time is within this many days of the refit, or
            `None` for the whole history.
        rounds: Boosting rounds.
        eta: Learning rate.
        regime_names: The regimes (keys of `regimes.REGIMES`) whose refits the configuration fits.
        groups: The substation groups the configuration is fitted for.
        frozen: Train once on the pilot's control-member rows to 2025-06-30 only (experiment
            00033's recipe), with no rows from later runs.
        zero_start: Start every quantile level from zero instead of from the estimate.
    """

    name: str
    half_life_days: float
    window_days: float | None
    rounds: int
    eta: float
    regime_names: tuple[str, ...]
    groups: tuple[str, ...]
    frozen: bool = False
    zero_start: bool = False


CONFIGS: Final[dict[str, Config]] = {
    "n1h_def": Config(
        "n1h_def",
        common.SHORT_HALF_LIFE_DAYS,
        None,
        common.N_ROUNDS,
        0.05,
        ("weekly",),
        common.GROUPS,
    ),
    "n1h_m0": Config(
        "n1h_m0",
        common.SHORT_HALF_LIFE_DAYS,
        None,
        common.N_ROUNDS,
        0.05,
        ("weekly",),
        common.GROUPS,
        zero_start=True,
    ),
    "frozen_m0": Config(
        "frozen_m0",
        common.RECENCY_HALF_LIFE_DAYS,
        None,
        common.N_ROUNDS,
        0.05,
        ("frozen",),
        common.GROUPS,
        frozen=True,
        zero_start=True,
    ),
}


def input_columns() -> list[str]:
    """Return the input columns in matrix order."""
    return [*common.feature_columns(), *common.N1_EXTRA_COLUMNS, *common.LAG_COLUMNS]


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


def mixture_quantiles(*, member_quantiles: np.ndarray) -> np.ndarray:
    """Pool members' quantiles into the mixture's nine quantiles, as experiment 00019 does."""
    cases, members, levels = member_quantiles.shape
    values = member_quantiles.reshape(cases, members * levels)
    weights = np.tile(common.ATOM_MASSES, members) / members
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    cumulative = np.cumsum(weights[order], axis=1)
    picks = (cumulative[:, :, None] >= (common.LEVELS - 1e-9)).argmax(axis=1)
    pooled = np.take_along_axis(sorted_values, picks, axis=1)
    return np.sort(pooled, axis=1).astype(np.float32)


def training_rows(*, row: dict, arrays: common.SeriesArrays, frozen: bool) -> pl.DataFrame:
    """Return one series' control-member rows with derived columns and lags.

    The rows run to 2025-06-30 for the frozen configuration and to 2026-06-29 otherwise. The power
    of the rows after 2025-06-30 is filled from the dense grid. Every row's eligibility for a refit
    (valid time and run start) is applied per refit, not here.

    Args:
        row: The series' row of `common.series_table()`.
        arrays: The series' arrays on the dense grid.
        frozen: Whether to keep only the pilot's training rows, which end at 2025-06-30.

    Returns:
        The rows keyed by series, run, and valid time, with `ensemble_member`, `power` (in MW),
        the weather columns, the derived columns, `lead_hours`, and the three lags.
    """
    series_id = row["time_series_id"]
    before = (
        pl.scan_parquet(common.ENS_TRAIN_PATH)
        .filter(pl.col("time_series_id") == series_id)
        .select(*common.KEYS, "ensemble_member", "power", *WEATHER)
        .collect()
    )
    if frozen:
        rows = before
    else:
        after = pl.concat(
            [
                pl.read_parquet(path).filter(pl.col("time_series_id") == series_id)
                for path in sorted(common.AFTER_DIR.glob("init=*.parquet"))
            ]
        )
        padded = after.with_columns(power=pl.lit(None, dtype=pl.Float32)).select(before.columns)
        rows = pl.concat([before, padded.cast(before.schema)])
    index = np.asarray(common.grid_index(times=rows["valid_time"]))
    filled = common.take(values=arrays.power, index=index) * row["effective_capacity_mw"]
    rows = rows.with_columns(
        power=pl.when(pl.col("power").is_null()).then(pl.Series(filled)).otherwise(pl.col("power"))
    )
    rows = rows.filter(
        pl.col("power").is_finite() & (pl.col("valid_time") > pl.col("power_fcst_init_time"))
    )
    rows = common.with_derived(frame=rows, latitude=row["latitude"], longitude=row["longitude"])
    rows = rows.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    )
    return add_lags(frame=rows, power=arrays.power)


def known_rows(*, rows: pl.DataFrame, refit: datetime, config: Config) -> pl.DataFrame:
    """Return the rows a refit at `refit` may see.

    A refit sees a row only if its valid time is at least 3.5 hours before the refit and its run
    began at or before the refit, and, for a rolling window, only the latest `window_days`. The
    frozen model sees every row of the pilot's training table, which ends at its anchor.
    """
    if config.frozen:
        return rows.filter(pl.col("valid_time") <= common.LAST_TRAINING_VALID)
    known = rows.filter(
        (pl.col("valid_time") <= refit - regimes.REFIT_CUTOFF_AGE)
        & (pl.col("power_fcst_init_time") <= refit)
    )
    if config.window_days is not None:
        known = known.filter(pl.col("valid_time") > refit - timedelta(days=config.window_days))
    return known


def train_matrix(
    *, rows: pl.DataFrame, row: dict, refit: datetime, half_life_days: float
) -> xgb.DMatrix:
    """Return the weighted training matrix of one series' rows, weights anchored at `refit`."""
    age_days = (refit - rows["valid_time"]).dt.total_seconds() / 86_400
    return xgb.DMatrix(
        rows.select(input_columns()).to_numpy().astype(np.float32),
        label=(rows["power"] / row["effective_capacity_mw"]).to_numpy(),
        weight=0.5 ** (age_days.to_numpy() / half_life_days),
    )


def fit(*, matrix: xgb.DMatrix, config: Config, rounds: int, nthread: int) -> xgb.Booster:
    """Fit one quantile model with seed 0."""
    parameters = {
        **common.QUANTILE_PARAMETERS,
        "seed": common.SEED,
        "eta": config.eta,
        "nthread": nthread,
    }
    if config.zero_start:
        parameters["base_score"] = 0.0
    return xgb.train(parameters, matrix, num_boost_round=rounds)


def predict_run(
    *,
    model: xgb.Booster,
    run_rows: pl.DataFrame,
    row: dict,
    arrays: common.SeriesArrays,
    prefixes: tuple[int, ...] = (0,),
) -> dict[int, pl.DataFrame]:
    """Return the mixture quantiles of one run for one series, for each number of rounds.

    Args:
        model: The series' fitted booster.
        run_rows: The run's frame rows for the series, every valid time with every member.
        row: The series' row of `common.series_table()`.
        arrays: The series' arrays on the dense grid.
        prefixes: The numbers of boosting rounds to predict from; 0 means all of them.

    Returns:
        For each prefix, one row per valid time: the key columns and the nine quantile columns.

    Raises:
        ValueError: If a valid time lacks a member.
    """
    prepared = common.with_derived(
        frame=run_rows, latitude=row["latitude"], longitude=row["longitude"]
    )
    prepared = add_lags(frame=prepared, power=arrays.power).sort("valid_time", "ensemble_member")
    members = prepared["ensemble_member"].n_unique()
    if prepared.height != prepared.select(*common.KEYS).n_unique() * members:
        msg = "every valid time must hold every member"
        raise ValueError(msg)
    matrix = xgb.DMatrix(prepared.select(input_columns()).to_numpy().astype(np.float32))
    keys = prepared.select(*common.KEYS).unique(maintain_order=True)
    out = {}
    for prefix in prefixes:
        raw = model.predict(matrix, iteration_range=(0, prefix)).reshape(
            -1, members, len(common.LEVELS)
        )
        out[prefix] = keys.hstack(
            pl.DataFrame(mixture_quantiles(member_quantiles=raw), schema=list(common.LEVEL_COLUMNS))
        )
    return out


def job_path(*, root: Path, config: str, site: str, refit: datetime) -> Path:
    """Return the file one refit's predictions for one series are written to."""
    return root / config / f"{site}_s{common.SEED}_{refit:%Y%m%dT%H%M}.parquet"


_CACHE: dict[str, object] = {}


def _series_and_arrays() -> tuple[pl.DataFrame, dict[str, common.SeriesArrays]]:
    """Return the series table and arrays, loaded once per worker process."""
    if "series" not in _CACHE:
        series = common.series_table()
        _CACHE["series"] = series
        _CACHE["arrays"] = common.load_series_arrays(series=series)
        _CACHE["rows"] = {}
    return _CACHE["series"], _CACHE["arrays"]  # ty: ignore[invalid-return-type]


def _training_rows_cached(*, row: dict, arrays: common.SeriesArrays, frozen: bool) -> pl.DataFrame:
    """Return a series' training rows, kept for the worker's later jobs."""
    cache: dict[tuple[str, bool], pl.DataFrame] = _CACHE["rows"]  # ty: ignore[invalid-assignment]
    key = (row["site"], frozen)
    if key not in cache:
        cache[key] = training_rows(row=row, arrays=arrays, frozen=frozen)
    return cache[key]


def run_job(
    *,
    config_name: str,
    refit: datetime,
    origins: list[datetime],
    sites: list[str],
    out_root: Path,
    rounds: int,
    nthread: int,
) -> list[str]:
    """Fit one refit for each series that lacks its file, forecast its runs, and write the files.

    Args:
        config_name: A key of `CONFIGS`.
        refit: The refit time.
        origins: The `power_fcst_init_time` of the runs the refit forecasts.
        sites: The series labels to fit.
        out_root: The folder holding one subfolder per configuration.
        rounds: Boosting rounds.
        nthread: Threads for each fit.

    Returns:
        One progress line per series fitted.
    """
    config = CONFIGS[config_name]
    series, arrays = _series_and_arrays()
    todo = [
        site
        for site in sites
        if not job_path(root=out_root, config=config.name, site=site, refit=refit).exists()
    ]
    if not todo:
        return [f"{config.name} refit {refit:%Y-%m-%d %H:%M}: all files exist"]
    rows_of = {row["site"]: row for row in series.iter_rows(named=True)}
    models: dict[str, xgb.Booster] = {}
    lines = []
    for site in todo:
        started = time.monotonic()
        row = rows_of[site]
        known = known_rows(
            rows=_training_rows_cached(row=row, arrays=arrays[site], frozen=config.frozen),
            refit=refit,
            config=config,
        )
        models[site] = fit(
            matrix=train_matrix(
                rows=known, row=row, refit=refit, half_life_days=config.half_life_days
            ),
            config=config,
            rounds=rounds,
            nthread=nthread,
        )
        lines.append(
            f"{config.name} {site} refit {refit:%Y-%m-%d %H:%M}: {known.height:,} rows, "
            f"{len(origins)} runs, fit {time.monotonic() - started:.0f} s"
        )
    pieces: dict[str, list[pl.DataFrame]] = {site: [] for site in todo}
    ids = {site: rows_of[site]["time_series_id"] for site in todo}
    for origin in origins:
        frame = pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
        for site in todo:
            pieces[site].append(
                predict_run(
                    model=models[site],
                    run_rows=frame.filter(pl.col("time_series_id") == ids[site]),
                    row=rows_of[site],
                    arrays=arrays[site],
                )[0]
            )
    for site in todo:
        common.atomic_write(
            frame=pl.concat(pieces[site]).with_columns(seed=pl.lit(common.SEED).cast(pl.Int8)),
            target=job_path(root=out_root, config=config.name, site=site, refit=refit),
        )
    return lines


def main() -> None:
    """Fit and write the requested configuration's jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--groups", nargs="+", choices=common.GROUPS)
    parser.add_argument("--workers", type=int, default=5, help="processes; each gets 10 // workers")
    parser.add_argument(
        "--limit-refits", type=int, default=0, help="fit only the first N refits (the check)"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="one series, 2 refits, 3 runs, 20 rounds, scratch dir"
    )
    args = parser.parse_args()
    config = CONFIGS[args.config]
    groups = tuple(args.groups) if args.groups else config.groups
    series = common.series_table().filter(pl.col("substation_type").is_in(groups))
    sites = series["site"].to_list()
    jobs = regimes.jobs_for(regime_names=config.regime_names, origins=common.all_origins())
    out_root = common.ANCHOR_PREDS_DIR
    rounds = config.rounds
    workers = args.workers
    if args.limit_refits:
        jobs = dict(list(jobs.items())[: args.limit_refits])
    if args.smoke:
        sites = sites[:1]
        jobs = {refit: origins[:3] for refit, origins in list(jobs.items())[:2]}
        out_root = common.ANCHOR_PREDS_DIR.with_name("anchor_preds_smoke")
        rounds = 20
        workers = 1
    nthread = max(1, CORES // workers)
    started = time.monotonic()
    ordered = sorted(jobs.items(), key=lambda item: -len(item[1]))
    # Fewer refits than workers (the frozen model has one): split the series so every worker works.
    n_chunks = workers if len(ordered) < workers else 1
    site_chunks = [sites[i::n_chunks] for i in range(n_chunks)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                run_job,
                config_name=config.name,
                refit=refit,
                origins=origins,
                sites=chunk,
                out_root=out_root,
                rounds=rounds,
                nthread=nthread,
            )
            for refit, origins in ordered
            for chunk in site_chunks
        ]
        for done, future in enumerate(futures, start=1):
            for line in future.result():
                print(f"[{done}/{len(futures)}] {line}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
