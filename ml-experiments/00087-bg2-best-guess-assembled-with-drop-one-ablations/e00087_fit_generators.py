"""Fit the generator arms of experiment 00087 (BG2): 6 solar farms and 3 wind farms.

Every arm applies one quantile model (11 levels) to each of the 51 ensemble members of a daily run
and pools the members' quantiles into a mixture, as experiments 00033 and 00048 did. Seed 0 only.
The mixtures are written without any spread factor: the ablation "spread off" is these files as
written, and the headline arms pass them through `e00087_gen_common.apply_spread_factor` when the
arms are assembled.

Arms:

- `ens_weekly`: ENS-trained N1 for generators, refitted every 7 days on the weekly schedule
  (`e00087_regimes.WEEKLY`, 52 refits). Experiment 00033's recipe: the control member's rows, the
  technology's common columns, `lead_hours`, three lags of power from the forecast origin, 500
  rounds, `eta` 0.05, recency weights with a half-life of 2 years anchored at the refit. A refit
  trains on control rows whose valid time is at most the refit minus 3.5 hours and whose run
  started at or before the refit.
- `ens_frozen`: the same recipe trained once, on valid times up to 2025-06-30 23:30.
- `era5_frozen_mixture`: the previous best guess, refitted at 11 levels. ERA5-trained, solar
  `base` (the weather columns) and wind `margin` (the same columns with the power curve as the base
  margin of all 11 levels), trained once on valid times before 2025-06-17. Labelled "refit at 11
  levels, not 33's stored arm".
- `ens_weekly_margin`: the 3 wind farms only. `ens_weekly` with the power curve as the base margin,
  150 rounds and `eta` 0.15 (the P-roll recipe of experiment 00049; ablation only).

Control-member rows after the stage 1 pilot's training table (which ends with valid time
2025-06-30 23:30) are the `ensemble_member == 0` rows of experiment 00048's generator frames
(`work/frames_generators`, all 365 daily runs), the same frames the forecasts are made from. Their
target is the cleaned power on the dense grid, as for the lags.

Output: `work/generator_preds/<arm>/<site>_<label>.parquet`, keyed by series, run, and valid time,
with the 11 quantile columns as a fraction of capacity and `seed`; `<label>` is the refit date, or
`frozen`. Existing files stay, so a stopped run resumes. Every fit appends its seconds to
`work/gpu_hours.jsonl`.

    uv run python ml-experiments/00087-*/e00087_fit_generators.py [--arms ens_weekly ...] \
        [--workers 2] [--cores 2] [--smoke]
    uv run python ml-experiments/00087-*/e00087_fit_generators.py --dry-gpu-hours
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import cache, lru_cache
from pathlib import Path
from typing import Final

import e00087_common as common
import e00087_gen_common as gen
import e00087_regimes as regimes
import numpy as np
import polars as pl

SMOKE_ROUNDS: Final[int] = 20
SMOKE_REFITS: Final[int] = 2
SMOKE_RUNS: Final[int] = 3
GPU_BUDGET_HOURS: Final[int] = 16
ENS_WEEKLY_ROUNDS: Final[int] = common.N_ROUNDS
MARGIN_ROUNDS: Final[int] = 150
MARGIN_ETA: Final[float] = 0.15
FROZEN_LABEL: Final[str] = "frozen"
ENS_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id",
    "nwp_init_time",
    "power_fcst_init_time",
    "valid_time",
    *dict.fromkeys([*gen.SOLAR_FEATURES, *gen.WIND_FEATURES]),
)
"""The columns kept from a control-member row, before the target and the lags are added."""


@dataclass(frozen=True)
class Arm:
    """How one arm is fitted.

    Attributes:
        name: The arm's folder name under `generator_preds`.
        regime: A key of `regimes.REGIMES` for ENS-trained arms, or `None` for the ERA5-trained
            arm (one frozen fit).
        technologies: The technologies the arm covers.
        rounds: Boosting rounds of a full run.
        eta: The learning rate, or `None` for the default.
        use_margin: Whether the wind power curve is the base margin (wind farms only).
    """

    name: str
    regime: str | None
    technologies: tuple[str, ...]
    rounds: int
    eta: float | None = None
    use_margin: bool = False


ARMS: Final[dict[str, Arm]] = {
    "ens_weekly": Arm("ens_weekly", "weekly", ("solar", "wind"), ENS_WEEKLY_ROUNDS),
    "ens_frozen": Arm("ens_frozen", "frozen", ("solar", "wind"), ENS_WEEKLY_ROUNDS),
    "era5_frozen_mixture": Arm("era5_frozen_mixture", None, ("solar", "wind"), common.N_ROUNDS),
    "ens_weekly_margin": Arm(
        "ens_weekly_margin", "weekly", ("wind",), MARGIN_ROUNDS, eta=MARGIN_ETA, use_margin=True
    ),
}


@dataclass(frozen=True)
class Job:
    """One fit: an arm, a refit label, and the origins its model forecasts."""

    arm: str
    label: str
    refit: datetime
    origins: tuple[datetime, ...]


def job_path(*, root: Path, arm: str, site: str, label: str) -> Path:
    """Return the file one model's forecasts for one series are written to."""
    return root / arm / f"{site}_{label}.parquet"


def jobs_of(*, arm: Arm) -> list[Job]:
    """Return an arm's fits: one per refit of its regime, or one frozen fit for the ERA5 arm."""
    origins = common.all_origins()
    if arm.regime is None:
        return [
            Job(
                arm=arm.name,
                label=FROZEN_LABEL,
                refit=gen.ERA5_FROZEN_TRAIN_END,
                origins=tuple(origins),
            )
        ]
    return [
        Job(
            arm=arm.name,
            label=FROZEN_LABEL if arm.regime == "frozen" else f"{refit:%Y%m%d}",
            refit=refit,
            origins=tuple(served),
        )
        for refit, served in regimes.served(name=arm.regime, origins=origins).items()
    ]


@cache
def _sites() -> pl.DataFrame:
    return gen.site_table()


@cache
def _power() -> dict[int, np.ndarray]:
    return gen.full_power(sites=_sites())


def add_lags(*, frame: pl.DataFrame, power: np.ndarray) -> pl.DataFrame:
    """Add `lead_hours` and the three origin lags to rows with an origin and a valid time."""
    lags = common.origin_lags(
        power=power,
        target=np.asarray(common.grid_index(times=frame["valid_time"])),
        origin=np.asarray(common.grid_index(times=frame["power_fcst_init_time"])),
    )
    return frame.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    ).with_columns(pl.DataFrame(lags, schema=list(common.LAG_COLUMNS), orient="row"))


@lru_cache(maxsize=1)
def ens_control_rows(*, series: int) -> pl.DataFrame:
    """Return one series' ENS control-member training rows with target, lead, and lags.

    The rows are the stage 1 pilot's training table (valid times from 2024-04-01 to 2025-06-30)
    followed by the control member of every daily generator frame (run start 2025-07-01 onwards).
    The cache holds the last series asked for per process, because a series' rows are reused by
    every refit.

    Args:
        series: The `time_series_id`.

    Returns:
        Rows sorted by valid time with `target` (a fraction of capacity, finite) and the columns
        of `ENS_COLUMNS`, `lead_hours`, and `common.LAG_COLUMNS`.
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
    return add_lags(frame=rows.sort("valid_time"), power=power)


def eligible_rows(*, rows: pl.DataFrame, job: Job) -> pl.DataFrame:
    """Return the control rows a refit may train on.

    A weekly refit sees rows whose valid time is at most the refit minus 3.5 hours and whose run
    started at or before the refit. The frozen fit trains on valid times up to its anchor,
    2025-06-30 23:30, as experiment 00033 did.
    """
    last_valid = (
        regimes.FROZEN_REFIT if job.label == FROZEN_LABEL else job.refit - regimes.REFIT_CUTOFF_AGE
    )
    return rows.filter(
        (pl.col("valid_time") <= last_valid) & (pl.col("nwp_init_time") <= job.refit)
    )


def recency_weights(*, rows: pl.DataFrame, anchor: datetime) -> np.ndarray:
    """Return recency weights with a half-life of 2 years, anchored at `anchor`."""
    age_days = (anchor - rows["valid_time"]).dt.total_seconds() / 86_400
    return 0.5 ** (age_days.to_numpy() / common.RECENCY_HALF_LIFE_DAYS)


def era5_training_rows(*, row: dict) -> pl.DataFrame:
    """Return one series' ERA5 analysis rows with observed power before the frozen train end."""
    observed = gen.observed_training_power(
        sites=_sites().filter(pl.col("time_series_id") == row["time_series_id"]),
        train_end=gen.ERA5_FROZEN_TRAIN_END,
    )
    return (
        pl.scan_parquet(gen.ANALYSIS_PATH)
        .filter(pl.col("time_series_id") == row["time_series_id"])
        .collect()
        .join(observed, on=["time_series_id", "valid_time"], how="inner", validate="1:1")
        .filter(
            (pl.col("valid_time") >= gen.ERA5_START)
            & (pl.col("valid_time") < gen.ERA5_FROZEN_TRAIN_END)
        )
    )


def forecast_runs(
    *, model: gen.FittedQuantiles, row: dict, origins: tuple[datetime, ...], with_lags: bool
) -> pl.DataFrame:
    """Return the pooled-mixture quantiles of a series for the runs with the given origins.

    Args:
        model: The fitted model.
        row: The series' row of `gen.site_table()`.
        origins: The `power_fcst_init_time` of each run to forecast.
        with_lags: Whether the model's inputs include `lead_hours` and the power lags (ENS-N1).

    Returns:
        One row per case, keyed by series, origin, and valid time, with the 11 quantile columns.
    """
    power = _power()[int(row["time_series_id"])]
    pieces = []
    for origin in origins:
        frame = (
            pl.scan_parquet(gen.generator_frame_path(run=origin - common.PUBLICATION_DELAY))
            .filter(pl.col("time_series_id") == row["time_series_id"])
            .collect()
        )
        if with_lags:
            frame = add_lags(frame=frame, power=power)
        pieces.append(gen.pooled_mixture(part=frame, member_quantiles=model.predict(frame=frame)))
    return pl.concat(pieces)


def fit_job(
    *, arm: Arm, job: Job, row: dict, rounds: int, nthread: int, workers: int, log_hours: bool
) -> tuple[gen.FittedQuantiles, int, float]:
    """Fit one job's model for one series and log its seconds.

    Returns:
        The model, the number of training rows, and the fit's wall-clock seconds.
    """
    technology = row["technology"]
    if arm.regime is None:
        rows = era5_training_rows(row=row)
        columns, weights = gen.FEATURES[technology], None
    else:
        rows = eligible_rows(rows=ens_control_rows(series=int(row["time_series_id"])), job=job)
        columns, weights = gen.N1_COLUMNS[technology], recency_weights(rows=rows, anchor=job.refit)
    started = time.monotonic()
    model = gen.FittedQuantiles(
        rows=rows,
        columns=columns,
        seed=common.SEED,
        rounds=rounds,
        use_margin=technology == "wind" and (arm.use_margin or arm.regime is None),
        nthread=nthread,
        eta=arm.eta,
        weights=weights,
    )
    seconds = time.monotonic() - started
    if log_hours:
        common.log_gpu_seconds(
            job=f"gen {arm.name} {row['site']} {job.label}", seconds=seconds, workers=workers
        )
    return model, rows.height, seconds


def run_task(
    *,
    site: str,
    jobs: list[tuple[str, Job]],
    out_root: Path,
    rounds_override: int | None,
    limit_runs: int | None,
    nthread: int,
    workers: int,
    log_hours: bool,
) -> list[str]:
    """Fit and write the given jobs of one generator series.

    Args:
        site: The series' anonymous label (A to F, W1 to W3).
        jobs: Pairs of arm name and job to run; a job whose file exists is skipped.
        out_root: The folder holding one subfolder per arm.
        rounds_override: Boosting rounds for every arm (smoke test), or `None` for each arm's own.
        limit_runs: Forecast only this many runs per model (smoke test), or `None` for all.
        nthread: Threads for each fit.
        workers: Worker processes sharing the GPU, for the log.
        log_hours: Whether to append the fits' seconds to the GPU-hour log.

    Returns:
        One progress line per model fitted.
    """
    row = _sites().filter(pl.col("site") == site).row(0, named=True)
    lines = []
    for arm_name, job in jobs:
        target = job_path(root=out_root, arm=arm_name, site=site, label=job.label)
        if target.exists():
            continue
        arm = ARMS[arm_name]
        model, n_rows, seconds = fit_job(
            arm=arm,
            job=job,
            row=row,
            rounds=rounds_override or arm.rounds,
            nthread=nthread,
            workers=workers,
            log_hours=log_hours,
        )
        used = job.origins[:limit_runs] if limit_runs else job.origins
        forecast = forecast_runs(
            model=model, row=row, origins=used, with_lags=arm.regime is not None
        )
        common.atomic_write(
            frame=forecast.with_columns(seed=pl.lit(common.SEED).cast(pl.Int8)), target=target
        )
        lines.append(
            f"{arm_name} {site} {job.label}: {n_rows:,} training rows, {len(used)} runs, "
            f"fit {seconds:.0f} s"
        )
    return lines


def all_jobs(*, arm_names: list[str]) -> dict[str, list[tuple[str, Job]]]:
    """Return, for each series label, the (arm, job) pairs the named arms need."""
    sites = _sites()
    out: dict[str, list[tuple[str, Job]]] = {site: [] for site in sites["site"].to_list()}
    technology_of = dict(zip(sites["site"], sites["technology"], strict=True))
    for name in arm_names:
        arm = ARMS[name]
        for job in jobs_of(arm=arm):
            for site, technology in technology_of.items():
                if technology in arm.technologies:
                    out[site].append((name, job))
    return out


def dry_gpu_hours(*, arm_names: list[str], nthread: int, workers: int) -> None:
    """Time real full-round fits and project the GPU hours of the full run.

    For each arm the probe is the first solar farm and the last wind farm (where the arm covers
    them), at the arm's first and last refit, with the arm's full rounds. The projection multiplies
    the mean seconds per fit of each technology by the number of fits. `summed` counts every fit in
    full, as the GPU-hour log does; `wall-clock` divides by the worker count and assumes the
    workers do not slow each other down, which the CPU and GPU contention of a real run can break.
    """
    sites = _sites()
    probes = {
        "solar": sites.filter(pl.col("technology") == "solar").row(0, named=True),
        "wind": sites.filter(pl.col("technology") == "wind").row(-1, named=True),
    }
    grand = 0.0
    for name in arm_names:
        arm = ARMS[name]
        jobs = jobs_of(arm=arm)
        probe_jobs = [jobs[0], jobs[-1]] if len(jobs) > 1 else jobs
        for technology in arm.technologies:
            row = probes[technology]
            seconds = []
            for job in probe_jobs:
                _model, n_rows, took = fit_job(
                    arm=arm,
                    job=job,
                    row=row,
                    rounds=arm.rounds,
                    nthread=nthread,
                    workers=1,
                    log_hours=False,
                )
                seconds.append(took)
                print(f"  {name} {row['site']} {job.label}: {n_rows:,} rows, {took:.0f} s")
            n_series = int((sites["technology"] == technology).sum())
            total = len(jobs) * n_series * float(np.mean(seconds)) / 3600.0
            grand += total
            print(
                f"  {name} {technology}: {len(jobs)} fits x {n_series} series x "
                f"{np.mean(seconds):.0f} s = {total:.2f} GPU-h summed"
            )
    print(
        f"total {grand:.2f} GPU-h summed; about {grand / workers:.2f} h wall-clock "
        f"with {workers} workers (fits only, no loading or forecasting)"
    )


def main() -> None:
    """Fit the requested arms, one chunk of one series per worker process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=list(ARMS))
    parser.add_argument("--sites", nargs="+", help="fit only these series labels, e.g. A W1")
    parser.add_argument("--workers", type=int, default=2, help="processes sharing the GPU")
    parser.add_argument("--cores", type=int, default=2, help="CPU cores for all workers")
    parser.add_argument(
        "--chunks-per-series",
        type=int,
        default=2,
        help="split each series' fits into this many tasks, so that the workers stay busy",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="20 rounds, the first solar and last wind farm, 2 refits, 3 runs, scratch folder",
    )
    parser.add_argument(
        "--dry-gpu-hours", action="store_true", help="time real fits and project the GPU hours"
    )
    args = parser.parse_args()
    nthread = max(1, args.cores // args.workers)
    if args.dry_gpu_hours:
        dry_gpu_hours(arm_names=args.arms, nthread=nthread, workers=args.workers)
        return
    print(f"GPU hours logged so far: {common.gpu_hours_total():.2f} of {GPU_BUDGET_HOURS}")
    per_site = all_jobs(arm_names=args.arms)
    if args.sites:
        per_site = {site: jobs for site, jobs in per_site.items() if site in args.sites}
    out_root = common.GENERATOR_PREDS_DIR
    rounds_override, limit_runs, workers = None, None, args.workers
    if args.smoke:
        sites = _sites()
        keep = {
            sites.filter(pl.col("technology") == "solar")["site"][0],
            sites.filter(pl.col("technology") == "wind")["site"][-1],
        }
        per_site = {
            site: [
                pair
                for name in args.arms
                for pair in _first_refits(jobs=[p for p in pairs if p[0] == name])
            ]
            for site, pairs in per_site.items()
            if site in keep
        }
        out_root = out_root.with_name("generator_preds_smoke")
        rounds_override, limit_runs, workers = SMOKE_ROUNDS, SMOKE_RUNS, 1
    tasks = []
    for site, pairs in per_site.items():
        n_chunks = 1 if args.smoke else args.chunks_per_series
        tasks.extend((site, pairs[i::n_chunks]) for i in range(n_chunks) if pairs[i::n_chunks])
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                run_task,
                site=site,
                jobs=pairs,
                out_root=out_root,
                rounds_override=rounds_override,
                limit_runs=limit_runs,
                nthread=nthread,
                workers=workers,
                log_hours=not args.smoke,
            )
            for site, pairs in tasks
        ]
        for done, future in enumerate(futures, start=1):
            for line in future.result():
                print(f"[{done}/{len(futures)}] {line}", flush=True)
    print(
        f"done in {time.monotonic() - started:.0f} s; "
        f"GPU hours logged: {common.gpu_hours_total():.2f} of {GPU_BUDGET_HOURS}"
    )


def _first_refits(*, jobs: list[tuple[str, Job]]) -> list[tuple[str, Job]]:
    """Return the first `SMOKE_REFITS` jobs of one arm (all of them when it has fewer)."""
    return jobs[:SMOKE_REFITS]


if __name__ == "__main__":
    main()
