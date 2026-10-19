"""Fit the no-lag single N1 model of experiment 00102 for the demand series.

Copied from experiment 00087's `e00087_fit_demand.py`, keeping only the `n1h` configuration (the
weekly and monthly members of the BG2 blend) with the three origin lags removed from the input
columns. The recipe is otherwise BG2's: the ENS control member's rows, a recency half-life of 0.5
years anchored at the refit, 500 rounds, `eta` 0.05, Primary, BSP, and GSP, 51 ensemble members
pooled into 11 mixture quantiles. No offset, so there is no base margin.

One job is one refit for a chunk of series. A refit at time `t` trains each series' model on the
control member's rows whose valid time is at least 3.5 hours before `t` and whose run began at or
before `t`, and forecasts the daily runs that the regime assigns to it. Files go to
`work/anchor_preds/n1h_nolag/<site>_s<seed>_<refit>.parquet`; existing files stay, so a stopped run
resumes. Every job appends its wall-clock seconds to `work/gpu_hours.jsonl`.

Leakage check, raising `ValueError`: a refit's training rows have a valid time at most 3.5 hours
before the refit and a run that began at or before the refit (`known_rows`).

    uv run python ml-experiments/00102-*/e00102_fit_demand.py --regime monthly [--workers 6]
        [--cores 12] [--smoke]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Final

import e00102_common as common
import e00102_regimes as regimes
import numpy as np
import polars as pl
import xgboost as xgb

CONFIG_NAME: Final[str] = "n1h_nolag"
WEATHER: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
)
ETA: Final[float] = 0.05
SMOKE_ROUNDS: Final[int] = 20
SMOKE_RUNS: Final[int] = 3
SMOKE_REFITS: Final[int] = 2
REGIME_NAMES: Final[tuple[str, ...]] = ("weekly", "monthly")


def input_columns() -> list[str]:
    """Return the input columns in matrix order: the features and `lead_hours`, no power lags."""
    return [*common.feature_columns(), *common.N1_EXTRA_COLUMNS]


def training_rows(*, row: dict, arrays: common.SeriesArrays) -> pl.DataFrame:
    """Return one series' control-member rows with derived columns and `lead_hours`.

    The rows run to 2026-06-29. The power of the rows after 2025-06-30 is filled from the dense
    grid. Every row's eligibility for a refit is applied per refit, not here.

    Args:
        row: The series' row of `common.series_table()`.
        arrays: The series' arrays on the dense grid.

    Returns:
        The rows keyed by series, run, and valid time, with `ensemble_member`, `power` (in MW), the
        weather columns, the derived columns, and `lead_hours`.
    """
    series_id = row["time_series_id"]
    before = (
        pl.scan_parquet(common.ENS_TRAIN_PATH)
        .filter(pl.col("time_series_id") == series_id)
        .select(*common.KEYS, "ensemble_member", "power", *WEATHER)
        .collect()
    )
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
    return rows.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    )


def known_rows(*, rows: pl.DataFrame, refit: datetime) -> pl.DataFrame:
    """Return the rows a refit at `refit` may see, and check that no row comes from its future.

    Args:
        rows: One series' training rows with `valid_time` and `power_fcst_init_time`.
        refit: The refit time.

    Returns:
        The rows whose valid time is at least 3.5 hours before the refit and whose run began at or
        before the refit.

    Raises:
        ValueError: If no row is usable, or a kept row is newer than the refit allows.
    """
    latest_valid = refit - regimes.REFIT_CUTOFF_AGE
    known = rows.filter(
        (pl.col("valid_time") <= latest_valid) & (pl.col("power_fcst_init_time") <= refit)
    )
    too_new = known.filter(
        (pl.col("valid_time") > latest_valid) | (pl.col("power_fcst_init_time") > refit)
    ).height
    if known.is_empty() or too_new:
        msg = f"refit {refit}: no usable rows or a row from after the refit"
        raise ValueError(msg)
    return known


def parameters(*, seed: int, nthread: int) -> dict[str, object]:
    """Return the XGBoost parameters."""
    return {**common.QUANTILE_PARAMETERS, "seed": seed, "eta": ETA, "nthread": nthread}


def mixture_frame(*, keys: pl.DataFrame, raw: np.ndarray, members: int, seed: int) -> pl.DataFrame:
    """Return one run's keys with the mixture of the members' 11 quantiles and the seed."""
    cube = raw.reshape(-1, members, len(common.LEVELS))
    return keys.hstack(
        pl.DataFrame(
            common.mixture_quantiles(member_quantiles=cube), schema=list(common.LEVEL_COLUMNS)
        )
    ).with_columns(seed=pl.lit(seed).cast(pl.Int8))


def job_path(*, root: Path, site: str, seed: int, refit: datetime) -> Path:
    """Return the file one refit's predictions for one series are written to."""
    return root / CONFIG_NAME / f"{site}_s{seed}_{refit:%Y%m%dT%H%M}.parquet"


_CACHE: dict[str, object] = {}


def _series_and_arrays() -> tuple[pl.DataFrame, dict[str, common.SeriesArrays]]:
    """Return the series table and arrays, loaded once per worker process."""
    if "series" not in _CACHE:
        series = common.series_table()
        _CACHE["series"] = series
        _CACHE["arrays"] = common.load_series_arrays(series=series)
        _CACHE["rows"] = {}
    return _CACHE["series"], _CACHE["arrays"]  # ty: ignore[invalid-return-type]


def _training_rows_cached(*, row: dict, arrays: common.SeriesArrays) -> pl.DataFrame:
    """Return a series' training rows, kept for the worker's later jobs."""
    cache: dict[str, pl.DataFrame] = _CACHE["rows"]  # ty: ignore[invalid-assignment]
    if row["site"] not in cache:
        cache[row["site"]] = training_rows(row=row, arrays=arrays)
    return cache[row["site"]]


def prepared_run(*, run_rows: pl.DataFrame, row: dict) -> pl.DataFrame:
    """Return one run's member rows for one series with derived columns, sorted.

    Raises:
        ValueError: If a valid time lacks a member.
    """
    prepared = common.with_derived(
        frame=run_rows, latitude=row["latitude"], longitude=row["longitude"]
    ).with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    )
    prepared = prepared.sort("valid_time", "ensemble_member")
    members = prepared["ensemble_member"].n_unique()
    if prepared.height != prepared.select(*common.KEYS).n_unique() * members:
        msg = "every valid time must hold every member"
        raise ValueError(msg)
    return prepared


def run_job(
    *,
    refit: datetime,
    origins: list[datetime],
    sites: list[str],
    out_root: Path,
    rounds: int,
    nthread: int,
    seed: int,
    workers: int,
    log_hours: bool,
) -> list[str]:
    """Fit one refit for each series that lacks its file, forecast its runs, and write the files.

    Args:
        refit: The refit time.
        origins: The `power_fcst_init_time` of the runs the refit forecasts.
        sites: The series labels to fit.
        out_root: The folder holding the configuration's subfolder.
        rounds: Boosting rounds.
        nthread: Threads for each fit.
        seed: The fitting seed.
        workers: Worker processes sharing the GPU, recorded in the GPU-hour log.
        log_hours: Append the job to the GPU-hour log (false for the smoke test).

    Returns:
        One progress line per series fitted.
    """
    series, arrays = _series_and_arrays()
    todo = [
        site
        for site in sites
        if not job_path(root=out_root, site=site, seed=seed, refit=refit).exists()
    ]
    if not todo:
        return [f"{CONFIG_NAME} refit {refit:%Y-%m-%d %H:%M}: all files exist"]
    started_job = time.monotonic()
    rows_of = {row["site"]: row for row in series.iter_rows(named=True)}
    models: dict[str, xgb.Booster] = {}
    lines = []
    for site in todo:
        started = time.monotonic()
        row = rows_of[site]
        known = known_rows(rows=_training_rows_cached(row=row, arrays=arrays[site]), refit=refit)
        age_days = (refit - known["valid_time"]).dt.total_seconds() / 86_400
        matrix = xgb.DMatrix(
            known.select(input_columns()).to_numpy().astype(np.float32),
            label=(known["power"] / row["effective_capacity_mw"]).to_numpy(),
            weight=0.5 ** (age_days.to_numpy() / common.SHORT_HALF_LIFE_DAYS),
        )
        models[site] = xgb.train(
            parameters(seed=seed, nthread=nthread), matrix, num_boost_round=rounds
        )
        lines.append(
            f"{CONFIG_NAME} {site} refit {refit:%Y-%m-%d %H:%M}: {known.height:,} rows, "
            f"{len(origins)} runs, fit {time.monotonic() - started:.0f} s"
        )
    pieces: dict[str, list[pl.DataFrame]] = {site: [] for site in todo}
    for origin in origins:
        frame = pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
        for site in todo:
            row = rows_of[site]
            prepared = prepared_run(
                run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]), row=row
            )
            members = prepared["ensemble_member"].n_unique()
            raw = models[site].predict(
                xgb.DMatrix(prepared.select(input_columns()).to_numpy().astype(np.float32))
            )
            pieces[site].append(
                mixture_frame(
                    keys=prepared.select(*common.KEYS).unique(maintain_order=True),
                    raw=raw,
                    members=members,
                    seed=seed,
                )
            )
    for site, tables in pieces.items():
        common.atomic_write(
            frame=pl.concat(tables),
            target=job_path(root=out_root, site=site, seed=seed, refit=refit),
        )
    if log_hours:
        common.log_gpu_seconds(
            job=f"{CONFIG_NAME} {refit:%Y-%m-%d %H:%M} {','.join(todo)} s{seed}",
            seconds=time.monotonic() - started_job,
            workers=workers,
        )
    return lines


def main() -> None:
    """Fit and write the requested regime's jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=REGIME_NAMES, required=True)
    parser.add_argument("--groups", nargs="+", choices=common.GROUPS)
    parser.add_argument("--sites", nargs="+", help="fit only these series labels, e.g. D2 D5")
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--workers", type=int, default=6, help="processes sharing the GPU")
    parser.add_argument("--cores", type=int, default=12, help="CPU cores for all workers")
    parser.add_argument(
        "--smoke", action="store_true", help="one series, 2 refits, 3 runs, 20 rounds, scratch dir"
    )
    args = parser.parse_args()
    nthread = max(1, args.cores // args.workers)
    groups = tuple(args.groups) if args.groups else common.GROUPS
    series = common.series_table().filter(pl.col("substation_type").is_in(groups))
    if args.sites:
        series = series.filter(pl.col("site").is_in(args.sites))
    sites = series["site"].to_list()
    jobs = regimes.served(name=args.regime, origins=common.all_origins())
    out_root = common.ANCHOR_PREDS_DIR
    rounds, workers = common.N_ROUNDS, args.workers
    if args.smoke:
        sites = sites[:1]
        jobs = {refit: runs[:SMOKE_RUNS] for refit, runs in list(jobs.items())[:SMOKE_REFITS]}
        out_root = out_root.with_name("anchor_preds_smoke")
        rounds, workers = SMOKE_ROUNDS, 1
        nthread = args.cores
    started = time.monotonic()
    ordered = sorted(jobs.items(), key=lambda item: -len(item[1]))
    # Fewer refits than workers: split the series so every worker works.
    n_chunks = max(1, workers // len(ordered)) if len(ordered) < workers else 1
    site_chunks = [sites[i::n_chunks] for i in range(n_chunks)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                run_job,
                refit=refit,
                origins=runs,
                sites=chunk,
                out_root=out_root,
                rounds=rounds,
                nthread=nthread,
                seed=args.seed,
                workers=workers,
                log_hours=not args.smoke,
            )
            for refit, runs in ordered
            for chunk in site_chunks
            if chunk
        ]
        for done, future in enumerate(futures, start=1):
            for line in future.result():
                print(f"[{done}/{len(futures)}] {line}", flush=True)
    print(
        f"done in {time.monotonic() - started:.0f} s; "
        f"GPU hours logged: {common.gpu_hours_total():.2f}"
    )


if __name__ == "__main__":
    main()
