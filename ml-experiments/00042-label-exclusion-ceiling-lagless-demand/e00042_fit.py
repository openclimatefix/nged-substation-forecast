"""Fit experiment 00042's lagless demand models, one parquet per (source, arm, seed, series).

The model is experiment 00033's N1 recipe for demand without the three power lag columns: one
XGBoost quantile model per series with nine levels, 500 rounds, and a 2-year recency half-life,
applied to the 51 ENS members of each validation run and pooled into nine mixture quantiles. Two
training sources (`era5`, `ens`) and the arms of `e00042_common.arm_names()`. A series whose
training rows an exclusion does not change reuses the `none` fit, so only affected series are
refitted for the other arms.

Output: `work/fits/<source>/<arm>_seed<seed>/<time_series_id>.parquet`. Existing files stay.

    uv run python ml-experiments/00042-*/e00042_fit.py [--workers N] [--smoke]
    uv run python ml-experiments/00042-*/e00042_fit.py --determinism-check [--smoke]
"""

import argparse
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final

import e00042_common as common
import numpy as np
import polars as pl
import xgboost as xgb

SMOKE_ROUNDS: Final[int] = 20
SMOKE_ROOT: Final[Path] = common.WORK_DIR / "fits_smoke"
DETERMINISM_FOLDER: Final[str] = "determinism_check"
DEFAULT_WORKERS: Final[int] = 3
EXPECTED_MEMBERS: Final[int] = 51


@dataclass(frozen=True)
class Job:
    """One fit: a source, an arm, a model seed, and a series."""

    source: common.SourceType
    arm: str
    seed: int
    series_id: int


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


@cache
def series_rows() -> dict[int, dict]:
    """Return each series' `series_table()` row by `time_series_id`."""
    return {r["time_series_id"]: r for r in common.series_table().iter_rows(named=True)}


@cache
def arrays_of(*, site: str) -> common.SeriesArrays:
    """Return one series' grid arrays (built for that series alone)."""
    table = common.series_table().filter(pl.col("site") == site)
    return common.load_series_arrays(series=table)[site]


def validation_inputs(*, row: dict) -> pl.DataFrame:
    """Return the series' validation rows (all runs and members), building the cache file once.

    Returns:
        Columns `power_fcst_init_time`, `valid_time`, `ensemble_member`, then the lagless inputs
        (`common.input_columns()`, Float32), sorted by run, valid time, and member.
    """
    path = common.VAL_INPUTS_DIR / f"{row['time_series_id']}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    pieces = []
    for file in sorted(common.VALIDATION_DIR.glob("init=*.parquet")):
        run = pl.read_parquet(file).filter(pl.col("time_series_id") == row["time_series_id"])
        pieces.append(
            run.select("power_fcst_init_time", "valid_time", "ensemble_member", *_weather())
        )
    frame = common.with_derived(
        frame=pl.concat(pieces), latitude=row["latitude"], longitude=row["longitude"]
    )
    frame = (
        frame.with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0
        )
        .select(
            "power_fcst_init_time",
            "valid_time",
            "ensemble_member",
            *[pl.col(c).cast(pl.Float32) for c in common.input_columns()],
        )
        .sort("power_fcst_init_time", "valid_time", "ensemble_member")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(f".{os.getpid()}.tmp")
    frame.write_parquet(scratch)
    scratch.rename(path)
    return frame


def _weather() -> list[str]:
    """Return the feature columns that come from the weather files (not computed)."""
    return [c for c in common.feature_columns() if c not in common.DERIVED]


def predict_series(*, booster: xgb.Booster, row: dict) -> pl.DataFrame:
    """Return the mixture quantiles of every validation run for one series.

    Raises:
        ValueError: If a valid time of a run does not hold every ensemble member.
    """
    frame = validation_inputs(row=row)
    members = frame["ensemble_member"].n_unique()
    if members != EXPECTED_MEMBERS:
        msg = f"series {row['time_series_id']}: {members} members, expected {EXPECTED_MEMBERS}"
        raise ValueError(msg)
    keys = frame.select("power_fcst_init_time", "valid_time").unique(maintain_order=True)
    if frame.height != keys.height * members:
        msg = "every valid time must hold every member"
        raise ValueError(msg)
    raw = booster.predict(
        xgb.DMatrix(frame.select(common.input_columns()).to_numpy().astype(np.float32))
    ).reshape(-1, members, len(common.LEVELS))
    return (
        keys.with_columns(time_series_id=pl.lit(row["time_series_id"]).cast(pl.Int32))
        .hstack(
            pl.DataFrame(mixture_quantiles(member_quantiles=raw), schema=list(common.LEVEL_COLUMNS))
        )
        .select(*common.KEYS, *common.LEVEL_COLUMNS)
    )


def run_job(
    *, job: Job, training: pl.DataFrame, rounds: int, root: Path, out_folder: str | None = None
) -> Path:
    """Fit one job on the training rows that its arm keeps, and write its predictions atomically.

    Args:
        job: The fit to run.
        training: The series' `common.training_frame` for the job's source.
        rounds: Boosting rounds.
        root: The folder holding the `fits/` tree.
        out_folder: Write under `root / out_folder` instead of `root` (the determinism check).

    Returns:
        The file written.
    """
    row = series_rows()[job.series_id]
    index = np.asarray(common.grid_index(times=training["valid_time"]))
    mask = common.exclusion_mask(source=job.source, arm=job.arm, series_id=job.series_id)
    kept = training.filter(pl.Series(~mask[index]))
    weight = common.recency_weights(valid_time=kept["valid_time"])
    matrix = xgb.DMatrix(
        kept.select(common.input_columns()).to_numpy().astype(np.float32),
        label=kept["label"].to_numpy(),
        weight=weight,
    )
    booster = xgb.train(
        {**common.XGBOOST_PARAMETERS, "seed": job.seed}, matrix, num_boost_round=rounds
    )
    predictions = predict_series(booster=booster, row=row).with_columns(
        seed=pl.lit(job.seed).cast(pl.Int8)
    )
    target_root = root if out_folder is None else root / out_folder
    target = common.fit_path(
        source=job.source, arm=job.arm, seed=job.seed, series_id=job.series_id, root=target_root
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(target.with_suffix(".tmp"))
    target.with_suffix(".tmp").rename(target)
    return target


def run_group(
    *, source: common.SourceType, series_id: int, jobs: list[Job], rounds: int, root: Path
) -> list[str]:
    """Run every job of one (source, series) in this process, reusing the training rows."""
    row = series_rows()[series_id]
    training = common.training_frame(source=source, row=row, arrays=arrays_of(site=row["site"]))
    messages = []
    for job in jobs:
        started = time.monotonic()
        run_job(job=job, training=training, rounds=rounds, root=root)
        messages.append(
            f"{source} {row['site']} {job.arm} seed {job.seed}: {time.monotonic() - started:.0f} s"
        )
    return messages


def enumerate_jobs(
    *, root: Path, series_ids: list[int], seeds: tuple[int, ...], refit_all: bool
) -> dict[tuple[common.SourceType, int], list[Job]]:
    """Return the missing jobs, grouped by (source, series).

    A labelled arm is fitted only for the series its label set affects, unless `refit_all`.
    """
    groups: dict[tuple[common.SourceType, int], list[Job]] = {}
    for source in common.SOURCES:
        affected = {
            ls: set(common.affected_series(source=source, label_set=ls)) for ls in common.LABEL_SETS
        }
        for series_id in series_ids:
            for arm in common.arm_names():
                label_set = common.arm_label_set(arm=arm)
                if label_set is not None and not refit_all and series_id not in affected[label_set]:
                    continue
                for seed in common.arm_seeds(arm=arm):
                    if seed not in seeds:
                        continue
                    path = common.fit_path(
                        source=source, arm=arm, seed=seed, series_id=series_id, root=root
                    )
                    if not path.exists():
                        groups.setdefault((source, series_id), []).append(
                            Job(source=source, arm=arm, seed=seed, series_id=series_id)
                        )
    return groups


def determinism_job(*, job: Job, folder: str, rounds: int, root: Path) -> Path:
    """Run one fit of the determinism check in this (worker) process, into `root / folder`."""
    row = series_rows()[job.series_id]
    training = common.training_frame(source=job.source, row=row, arrays=arrays_of(site=row["site"]))
    return run_job(job=job, training=training, rounds=rounds, root=root, out_folder=folder)


def determinism_check(*, root: Path, rounds: int, workers: int) -> None:
    """Refit an unaffected series under each labelled arm, at the production worker settings.

    For each source, takes the first series that neither label set changes, makes sure its `none`
    fit exists, then refits it as arm `trusted`, as arm `allvis`, and as arm `none` again, all at
    the same time in `workers` worker processes with the same threads and device as the real run.
    The three refits train on identical rows, so each must equal the stored `none` predictions bit
    for bit. When any differs, a fit is not reproducible under the production settings and the
    plan's fallback applies: refit every series in every arm.

    Args:
        root: The folder holding the `fits/` tree.
        rounds: Boosting rounds.
        workers: Worker processes, as in the real run.

    Raises:
        AssertionError: If any prediction differs.
        RuntimeError: If every series is affected in some source.
    """
    context = multiprocessing.get_context("spawn")
    for source in common.SOURCES:
        changed = {
            i
            for ls in common.LABEL_SETS
            for i in common.affected_series(source=source, label_set=ls)
        }
        free = [i for i in common.series_table()["time_series_id"].to_list() if i not in changed]
        if not free:
            msg = f"every series is affected in source {source}"
            raise RuntimeError(msg)
        series_id = free[0]
        site = series_rows()[series_id]["site"]
        none_path = common.fit_path(
            source=source, arm="none", seed=0, series_id=series_id, root=root
        )
        if not none_path.exists():
            msg = f"{none_path} is missing: fit the `none` arm with the worker pool first"
            raise RuntimeError(msg)
        reference = pl.read_parquet(none_path)
        arms = ("trusted", "allvis", "none")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = {
                arm: pool.submit(
                    determinism_job,
                    job=Job(source=source, arm=arm, seed=0, series_id=series_id),
                    folder=DETERMINISM_FOLDER,
                    rounds=rounds,
                    root=root,
                )
                for arm in arms
            }
            paths = {arm: future.result() for arm, future in futures.items()}
        for arm, path in paths.items():
            refit = pl.read_parquet(path)
            difference = max(
                float(np.abs(reference[c].to_numpy() - refit[c].to_numpy()).max())
                for c in common.LEVEL_COLUMNS
            )
            print(
                f"determinism {source} {site} arm {arm}: max abs difference {difference:.3g}",
                flush=True,
            )
        if any(
            not all(
                np.array_equal(reference[c].to_numpy(), pl.read_parquet(path)[c].to_numpy())
                for c in common.LEVEL_COLUMNS
            )
            for path in paths.values()
        ):
            msg = f"{source} {site}: a refit differs from the no-exclusion fit"
            raise AssertionError(msg)


def smoke_series() -> list[int]:
    """Return one series the all-visible set changes in both sources, and one no set changes."""
    table = common.series_table()["time_series_id"].to_list()
    changed = [
        i
        for i in table
        if all(i in common.affected_series(source=s, label_set="allvis") for s in common.SOURCES)
    ]
    free = [
        i
        for i in table
        if not any(
            i in common.affected_series(source=s, label_set=ls)
            for s in common.SOURCES
            for ls in common.LABEL_SETS
        )
    ]
    return [changed[0], free[0]]


def main() -> None:
    """Fit the missing jobs, or run the determinism check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="parallel workers")
    parser.add_argument("--smoke", action="store_true", help="2 series, seed 0, 20 rounds, scratch")
    parser.add_argument("--determinism-check", action="store_true")
    parser.add_argument(
        "--refit-all",
        action="store_true",
        help="refit every series in every arm (the fallback when the determinism check fails)",
    )
    args = parser.parse_args()
    root = SMOKE_ROOT if args.smoke else common.FITS_DIR
    rounds = SMOKE_ROUNDS if args.smoke else common.N_ROUNDS
    if args.determinism_check:
        determinism_check(root=root, rounds=rounds, workers=args.workers)
        return
    series_ids = smoke_series() if args.smoke else common.series_table()["time_series_id"].to_list()
    seeds = common.SEEDS[:1] if args.smoke else common.SEEDS
    groups = enumerate_jobs(root=root, series_ids=series_ids, seeds=seeds, refit_all=args.refit_all)
    total = sum(len(v) for v in groups.values())
    print(f"{total} fits missing in {len(groups)} (source, series) groups", flush=True)
    context = multiprocessing.get_context("spawn")
    ordered = sorted(groups.items(), key=lambda item: -len(item[1]))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = [
            pool.submit(
                run_group, source=source, series_id=series_id, jobs=jobs, rounds=rounds, root=root
            )
            for (source, series_id), jobs in ordered
        ]
        for future in as_completed(futures):
            for message in future.result():
                print(message, flush=True)


if __name__ == "__main__":
    main()
