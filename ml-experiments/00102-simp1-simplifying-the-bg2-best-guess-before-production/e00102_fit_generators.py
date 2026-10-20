"""Fit the no-lag generator model of experiment 00102: 6 solar farms and 3 wind farms.

Copied from experiment 00087's `e00087_fit_generators.py`, keeping only the ENS-trained N1 arm with
the three power lags removed from the inputs. The recipe is otherwise BG2's farm arm (experiment
00033's recipe): the control member's rows, the technology's common columns and `lead_hours`, 500
rounds, `eta` 0.05, recency weights with a half-life of 2 years anchored at the refit. The refits
are monthly, like BG2's farm arms (experiment 00087's `BG2_ENS_MONTHLY=1` run). A refit trains on
control rows whose valid time is at most the refit minus 3.5 hours and whose run started at or
before the refit. The 51 ensemble members' 11 quantiles are pooled into a mixture and written
without any
spread factor.

Output: `work/generator_preds/ens_monthly_nolag/<site>_<refit date>.parquet`, keyed by series, run,
and valid time, with the 11 quantile columns as a fraction of capacity and `seed`. Existing files
stay, so a stopped run resumes. Every fit appends its seconds to `work/gpu_hours.jsonl`.

    uv run python ml-experiments/00102-*/e00102_fit_generators.py [--workers 6] [--cores 12]
        [--smoke]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import cache, lru_cache
from pathlib import Path
from typing import Final

import e00102_common as common
import e00102_gen_common as gen
import e00102_regimes as regimes
import numpy as np
import polars as pl

ARM_NAME: Final[str] = "ens_monthly_nolag"
SMOKE_ROUNDS: Final[int] = 20
SMOKE_REFITS: Final[int] = 2
SMOKE_RUNS: Final[int] = 3
ENS_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id",
    "nwp_init_time",
    "power_fcst_init_time",
    "valid_time",
    *dict.fromkeys([*gen.SOLAR_FEATURES, *gen.WIND_FEATURES]),
)
"""The columns kept from a control-member row, before the target is added."""
NO_LAG_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    technology: (*columns, "lead_hours") for technology, columns in gen.FEATURES.items()
}
"""Each technology's inputs: its common columns and `lead_hours`, with no power lag."""


@dataclass(frozen=True)
class Job:
    """One fit: a refit label and the origins its model forecasts."""

    label: str
    refit: datetime
    origins: tuple[datetime, ...]


def job_path(*, root: Path, site: str, label: str) -> Path:
    """Return the file one model's forecasts for one series are written to."""
    return root / ARM_NAME / f"{site}_{label}.parquet"


def jobs() -> list[Job]:
    """Return the monthly refits and the runs each serves."""
    return [
        Job(label=f"{refit:%Y%m%d}", refit=refit, origins=tuple(served))
        for refit, served in regimes.served(name="monthly", origins=common.all_origins()).items()
    ]


@cache
def _sites() -> pl.DataFrame:
    return gen.site_table()


@cache
def _power() -> dict[int, np.ndarray]:
    return gen.full_power(sites=_sites())


def _with_lead(*, frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    )


@lru_cache(maxsize=1)
def ens_control_rows(*, series: int) -> pl.DataFrame:
    """Return one series' ENS control-member training rows with target and lead.

    The rows are the stage 1 pilot's training table (valid times from 2024-04-01 to 2025-06-30)
    followed by the control member of every daily generator frame (run start 2025-07-01 onwards).
    The cache holds the last series asked for per process.

    Args:
        series: The `time_series_id`.

    Returns:
        Rows sorted by valid time with `target` (a fraction of capacity, finite), the columns of
        `ENS_COLUMNS`, and `lead_hours`.
    """
    row = _sites().filter(pl.col("time_series_id") == series).row(0, named=True)
    power = _power()[series]
    old = (
        pl.scan_parquet(gen.GENERATOR_ENS_TRAIN_PATH)
        .filter(
            (pl.col("time_series_id") == series)
            & (pl.col("ensemble_member") == 0)
            & (pl.col("valid_time") >= gen.ENS_START)
            & pl.col("power").is_finite()
        )
        .select(*ENS_COLUMNS, target=pl.col("power") / row["effective_capacity_mw"])
        .collect()
    )
    pieces = [
        pl.scan_parquet(gen.generator_frame_path(run=run))
        .filter((pl.col("time_series_id") == series) & (pl.col("ensemble_member") == 0))
        .select(*ENS_COLUMNS)
        for run in common.run_dates(subset="all")
    ]
    new = pl.concat(pieces).collect()
    new = new.with_columns(
        target=pl.Series(
            "target", power[np.asarray(common.grid_index(times=new["valid_time"]))]
        ).cast(pl.Float64)
    ).filter(pl.col("target").is_finite())
    rows = pl.concat([old.cast(new.schema), new.select(old.columns)]).filter(
        pl.col("valid_time") > pl.col("power_fcst_init_time")
    )
    return _with_lead(frame=rows.sort("valid_time"))


def eligible_rows(*, rows: pl.DataFrame, job: Job) -> pl.DataFrame:
    """Return the control rows a refit may train on.

    A refit sees rows whose valid time is at most the refit minus 3.5 hours and whose run started
    at or before the refit.
    """
    return rows.filter(
        (pl.col("valid_time") <= job.refit - regimes.REFIT_CUTOFF_AGE)
        & (pl.col("nwp_init_time") <= job.refit)
    )


def recency_weights(*, rows: pl.DataFrame, anchor: datetime) -> np.ndarray:
    """Return recency weights with a half-life of 2 years, anchored at `anchor`."""
    age_days = (anchor - rows["valid_time"]).dt.total_seconds() / 86_400
    return 0.5 ** (age_days.to_numpy() / common.RECENCY_HALF_LIFE_DAYS)


def forecast_runs(
    *, model: gen.FittedQuantiles, row: dict, origins: tuple[datetime, ...]
) -> pl.DataFrame:
    """Return the pooled-mixture quantiles of a series for the runs with the given origins."""
    pieces = []
    for origin in origins:
        frame = _with_lead(
            frame=pl.scan_parquet(gen.generator_frame_path(run=origin - common.PUBLICATION_DELAY))
            .filter(pl.col("time_series_id") == row["time_series_id"])
            .collect()
        )
        pieces.append(gen.pooled_mixture(part=frame, member_quantiles=model.predict(frame=frame)))
    return pl.concat(pieces)


def run_task(
    *,
    site: str,
    task_jobs: list[Job],
    out_root: Path,
    rounds: int,
    limit_runs: int | None,
    nthread: int,
    workers: int,
    log_hours: bool,
) -> list[str]:
    """Fit and write the given jobs of one generator series; a job whose file exists is skipped."""
    row = _sites().filter(pl.col("site") == site).row(0, named=True)
    technology = row["technology"]
    lines = []
    for job in task_jobs:
        target = job_path(root=out_root, site=site, label=job.label)
        if target.exists():
            continue
        rows = eligible_rows(rows=ens_control_rows(series=int(row["time_series_id"])), job=job)
        started = time.monotonic()
        model = gen.FittedQuantiles(
            rows=rows,
            columns=NO_LAG_COLUMNS[technology],
            seed=common.SEED,
            rounds=rounds,
            use_margin=False,
            nthread=nthread,
            weights=recency_weights(rows=rows, anchor=job.refit),
        )
        seconds = time.monotonic() - started
        if log_hours:
            common.log_gpu_seconds(
                job=f"gen {ARM_NAME} {site} {job.label}", seconds=seconds, workers=workers
            )
        used = job.origins[:limit_runs] if limit_runs else job.origins
        forecast = forecast_runs(model=model, row=row, origins=used)
        common.atomic_write(
            frame=forecast.with_columns(seed=pl.lit(common.SEED).cast(pl.Int8)), target=target
        )
        lines.append(
            f"{ARM_NAME} {site} {job.label}: {rows.height:,} training rows, {len(used)} runs, "
            f"fit {seconds:.0f} s"
        )
    return lines


def main() -> None:
    """Fit the no-lag farm models, one series per worker process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", help="fit only these series labels, e.g. A W1")
    parser.add_argument("--workers", type=int, default=6, help="processes sharing the GPU")
    parser.add_argument("--cores", type=int, default=12, help="CPU cores for all workers")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="20 rounds, the first solar and last wind farm, 2 refits, 3 runs, scratch folder",
    )
    args = parser.parse_args()
    sites = _sites()["site"].to_list()
    if args.sites:
        sites = [site for site in sites if site in args.sites]
    nthread = max(1, args.cores // args.workers)
    all_jobs, out_root = jobs(), common.GENERATOR_PREDS_DIR
    rounds, limit_runs, workers = common.N_ROUNDS, None, args.workers
    if args.smoke:
        table = _sites()
        sites = [
            table.filter(pl.col("technology") == "solar")["site"][0],
            table.filter(pl.col("technology") == "wind")["site"][-1],
        ]
        all_jobs = all_jobs[:SMOKE_REFITS]
        out_root = out_root.with_name("generator_preds_smoke")
        rounds, limit_runs, workers, nthread = SMOKE_ROUNDS, SMOKE_RUNS, 1, args.cores
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                run_task,
                site=site,
                task_jobs=all_jobs,
                out_root=out_root,
                rounds=rounds,
                limit_runs=limit_runs,
                nthread=nthread,
                workers=workers,
                log_hours=not args.smoke,
            )
            for site in sites
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
