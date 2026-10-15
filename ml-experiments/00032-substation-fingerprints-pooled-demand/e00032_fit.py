"""Fit experiment 00032's pooled quantile models with substation fingerprints.

Every model is experiment 00024's by-type pooled model (the 9-level quantile loss on the ENS
control member of every daily run from 2024-04-01, valid times up to 2025-06-30 23:30, a recency
half-life of 0.5 years, 00006's columns, the lead in hours, and the three origin lags). The arms
add fingerprint columns (`e00032_common.WARM_ARMS`), and the cold-start arms refit each pool
without one held-out series (`e00032_common.COLD_ARMS`).

Stage `fit` writes one booster per arm, group (or held-out series), and seed to `work/models/`
and appends the fit seconds to `work/fit_times.jsonl`. Stage `predict` reads those boosters and
writes `work/arms/<arm>_seed<s>.parquet`, the mixture of the 51 members' 9 quantiles. Existing
files are kept.

    uv run python ml-experiments/00032-substation-fingerprints-pooled-demand/e00032_fit.py \
        --stage fit [--smoke]
    uv run python ml-experiments/00032-substation-fingerprints-pooled-demand/e00032_fit.py \
        --stage predict [--smoke]
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00032_common as common
import numpy as np
import polars as pl
import xgboost as xgb

KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
COLD_SEEDS: Final[tuple[int, ...]] = (0,)


@dataclass(frozen=True)
class Block:
    """One series' training rows.

    Attributes:
        features: The input columns of `input_columns()`.
        target: Power over effective capacity.
        weights: Recency weights.
        origin: Grid index of each row's forecast origin.
    """

    features: np.ndarray
    target: np.ndarray
    weights: np.ndarray
    origin: np.ndarray


def input_columns() -> list[str]:
    """Return the input columns in matrix order, without the fingerprint and the identifier."""
    return [
        *common.feature_columns(calendar=True),
        *common.N1_EXTRA_COLUMNS,
        *common.LAG_COLUMNS,
    ]


def add_lags(*, frame: pl.DataFrame, power: np.ndarray) -> pl.DataFrame:
    """Add the three origin lags to a frame with a run's origin and valid times."""
    lags = common.origin_lags(
        power=power,
        target=np.asarray(common.grid_index(times=frame["valid_time"])),
        origin=np.asarray(common.grid_index(times=frame["power_fcst_init_time"])),
    )
    return frame.with_columns(
        pl.DataFrame(lags, schema=list(common.LAG_COLUMNS), orient="row").cast(pl.Float32)
    )


def mixture_quantiles(*, member_quantiles: np.ndarray) -> np.ndarray:
    """Pool members' quantiles into the mixture's 9 quantiles.

    Args:
        member_quantiles: Shape (cases, members, 9), each member's quantiles at `LEVELS`.

    Returns:
        Shape (cases, 9): the weighted quantiles of the pooled values, each member weighted equally
        and each level by its atom mass, sorted along the levels so none cross.
    """
    cases, members, levels = member_quantiles.shape
    values = member_quantiles.reshape(cases, members * levels)
    weights = np.tile(common.ATOM_MASSES, members) / members
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    cumulative = np.cumsum(weights[order], axis=1)
    picks = (cumulative[:, :, None] >= (common.LEVELS - 1e-9)).argmax(axis=1)
    pooled = np.take_along_axis(sorted_values, picks, axis=1)
    return np.sort(pooled, axis=1).astype(np.float32)


def training_blocks(
    *, series: pl.DataFrame, arrays: dict[str, common.SeriesArrays], train: pl.DataFrame
) -> dict[str, Block]:
    """Return each site's training rows, restricted to the training window."""
    if train["ensemble_member"].unique().to_list() != [0]:
        msg = "the training frame must hold the control member (0) only"
        raise ValueError(msg)
    columns = input_columns()
    out = {}
    for row in series.iter_rows(named=True):
        own = train.filter(
            (pl.col("time_series_id") == row["time_series_id"])
            & pl.col("power").is_finite()
            & (pl.col("valid_time") <= common.LAST_TRAINING_VALID)
        )
        own = common.with_derived(frame=own, latitude=row["latitude"], longitude=row["longitude"])
        own = add_lags(frame=own, power=arrays[row["site"]].power)
        age_days = (common.LAST_TRAINING_VALID - own["valid_time"]).dt.total_seconds() / 86_400
        if (own["valid_time"] > common.LAST_TRAINING_VALID).any():
            msg = "a training row is after the training window"
            raise ValueError(msg)
        out[row["site"]] = Block(
            features=own.select(columns).to_numpy().astype(np.float32),
            target=(own["power"] / row["effective_capacity_mw"]).to_numpy().astype(np.float32),
            weights=(0.5 ** (age_days.to_numpy() / common.SHORT_HALF_LIFE_DAYS)).astype(np.float32),
            origin=np.asarray(common.grid_index(times=own["power_fcst_init_time"])),
        )
    return out


def check_recent_window(*, arrays: dict[str, common.SeriesArrays]) -> None:
    """Raise if the recent-window statistics change when power within 2 days of the origin does.

    Poisons every observation at or after the origin minus 2 days and checks that the statistics
    of sample origins (training and validation) are unchanged.
    """
    origins = np.array(
        [
            common.grid_index(times=common.TRAIN_START) + 3000,
            common.grid_index(times=common.LAST_TRAINING_VALID),
            common.grid_index(times=common.VALIDATION_START) + 9000,
        ]
    )
    for site, arr in arrays.items():
        clean = common.recent_fingerprint(power=arr.power, origin=origins)
        for origin in origins:
            poisoned = arr.power.copy()
            poisoned[origin - common.RECENT_GAP :] = 1e6
            after = common.recent_fingerprint(power=poisoned, origin=np.array([origin]))
            before = clean[origins == origin]
            if not np.allclose(before, after, equal_nan=True):
                msg = f"recent-window statistics of {site} use data within 2 days of the origin"
                raise ValueError(msg)


def fingerprint(
    *, kinds: str, row: dict, power: np.ndarray, origin: np.ndarray, cold: bool
) -> np.ndarray:
    """Return a series' fingerprint columns for the cases with the given forecast origins.

    Args:
        kinds: Blocks to include (`S`, `T`, `R`).
        row: The series' `series_table()` row.
        power: The series' power over capacity on the grid.
        origin: Grid index of each case's origin.
        cold: Use the short training-window statistics of a held-out series when true.

    Returns:
        Shape (cases, columns).
    """
    return common.fingerprint_matrix(
        kinds=kinds,
        latitude=row["latitude"],
        longitude=row["longitude"],
        training=common.training_fingerprint(
            power=power, days=common.COLD_FINGERPRINT_DAYS if cold else None
        ),
        recent=common.recent_fingerprint(power=power, origin=origin),
    )


def model_path(*, models_dir: Path, arm: str, group: str, seed: int) -> Path:
    """Return the file a booster is saved in."""
    return models_dir / arm / f"{group}_seed{seed}.ubj"


def fit_model(
    *,
    blocks: list[Block],
    extras: list[np.ndarray],
    codes: list[int] | None,
    seed: int,
) -> xgb.Booster:
    """Fit one pooled booster on the given blocks, with fingerprint and identifier columns.

    Args:
        blocks: The pool's training rows, one block per series.
        extras: Each block's fingerprint columns.
        codes: Each block's series code to append as a categorical column, or None.
        seed: The fitting seed.

    Returns:
        The fitted booster.
    """
    parts = [np.hstack([b.features, e]) for b, e in zip(blocks, extras, strict=True)]
    if codes is not None:
        parts = [
            np.hstack([p, np.full((p.shape[0], 1), c, dtype=np.float32)])
            for p, c in zip(parts, codes, strict=True)
        ]
    matrix = np.vstack(parts)
    dtrain = xgb.DMatrix(
        matrix,
        label=np.concatenate([b.target for b in blocks]),
        weight=np.concatenate([b.weights for b in blocks]),
        feature_types=["q"] * (matrix.shape[1] - 1) + ["c"] if codes is not None else None,
        enable_categorical=codes is not None,
    )
    return xgb.train(
        {**common.QUANTILE_PARAMETERS, "seed": seed}, dtrain, num_boost_round=common.N_ROUNDS
    )


def log_fit(*, work_dir: Path, record: dict) -> None:
    """Append one fit's record to `fit_times.jsonl` and print it."""
    with (work_dir / "fit_times.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(record, flush=True)


def run_fit(*, series: pl.DataFrame, seeds: list[int], work_dir: Path, held_out: list[str]) -> None:
    """Fit every warm and cold-start booster and log the fit seconds."""
    arrays = common.load_series_arrays(series=series)
    blocks = training_blocks(
        series=series, arrays=arrays, train=pl.read_parquet(common.ENS_TRAIN_PATH)
    )
    check_recent_window(arrays=arrays)
    rows = {r["site"]: r for r in series.iter_rows(named=True)}
    codes = {site: position for position, site in enumerate(series["site"].to_list())}
    group_of = common.group_labels(series=series, grouping="type")
    models_dir = work_dir / "models"

    def extras(*, site: str, kinds: str, cold: bool) -> np.ndarray:
        return fingerprint(
            kinds=kinds,
            row=rows[site],
            power=arrays[site].power,
            origin=blocks[site].origin,
            cold=cold,
        )

    for arm in common.NEW_WARM_ARMS:
        use_id, kinds = common.WARM_ARMS[arm]
        for group in sorted(set(group_of.values())):
            sites = [s for s, g in group_of.items() if g == group]
            for seed in seeds:
                path = model_path(models_dir=models_dir, arm=arm, group=group, seed=seed)
                if path.exists():
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                started = time.monotonic()
                model = fit_model(
                    blocks=[blocks[s] for s in sites],
                    extras=[extras(site=s, kinds=kinds, cold=False) for s in sites],
                    codes=[codes[s] for s in sites] if use_id else None,
                    seed=seed,
                )
                model.save_model(path)
                log_fit(
                    work_dir=work_dir,
                    record={
                        "arm": arm,
                        "group": group,
                        "seed": seed,
                        "series": len(sites),
                        "seconds": time.monotonic() - started,
                    },
                )
    for arm, kinds in common.COLD_ARMS.items():
        for site in held_out:
            sites = [s for s in group_of if group_of[s] == group_of[site] and s != site]
            rows_expected = sum(
                blocks[s].target.size for s in group_of if group_of[s] == group_of[site]
            )
            if (
                sum(blocks[s].target.size for s in sites)
                != rows_expected - blocks[site].target.size
            ):
                msg = f"the pool for held-out {site} still holds its rows"
                raise ValueError(msg)
            for seed in COLD_SEEDS:
                path = model_path(models_dir=models_dir, arm=arm, group=site, seed=seed)
                if path.exists():
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                started = time.monotonic()
                model = fit_model(
                    blocks=[blocks[s] for s in sites],
                    extras=[extras(site=s, kinds=kinds, cold=False) for s in sites],
                    codes=None,
                    seed=seed,
                )
                model.save_model(path)
                log_fit(
                    work_dir=work_dir,
                    record={
                        "arm": arm,
                        "group": site,
                        "seed": seed,
                        "series": len(sites),
                        "seconds": time.monotonic() - started,
                    },
                )


def run_predict(
    *, series: pl.DataFrame, seeds: list[int], work_dir: Path, out_dir: Path, held_out: list[str]
) -> None:
    """Predict every validation run with every new arm's booster and write the arm tables."""
    arrays = common.load_series_arrays(series=series)
    rows = {r["site"]: r for r in series.iter_rows(named=True)}
    codes = {site: position for position, site in enumerate(series["site"].to_list())}
    group_of = common.group_labels(series=series, grouping="type")
    columns = input_columns()
    jobs = {
        **{
            arm: (common.WARM_ARMS[arm][1], common.WARM_ARMS[arm][0], False, list(seeds))
            for arm in common.NEW_WARM_ARMS
        },
        **{arm: (kinds, False, True, list(COLD_SEEDS)) for arm, kinds in common.COLD_ARMS.items()},
    }
    todo = [
        (arm, seed)
        for arm, (_, _, cold, arm_seeds) in jobs.items()
        for seed in arm_seeds
        if not (out_dir / f"{arm}_seed{seed}.parquet").exists()
    ]
    pieces: dict[tuple[str, int], list[pl.DataFrame]] = {key: [] for key in todo}
    for site, row in rows.items():
        started = time.monotonic()
        boosters = {}
        for arm, seed in todo:
            cold = jobs[arm][2]
            if cold and site not in held_out:
                continue
            booster = xgb.Booster()
            booster.load_model(
                model_path(
                    models_dir=work_dir / "models",
                    arm=arm,
                    group=site if cold else group_of[site],
                    seed=seed,
                )
            )
            booster.set_param({"device": "cuda"})
            boosters[arm, seed] = booster
        if not boosters:
            continue
        mixtures: dict[tuple[str, int], list[pl.DataFrame]] = {key: [] for key in boosters}
        for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet")):
            run = pl.read_parquet(path).filter(pl.col("time_series_id") == row["time_series_id"])
            run = common.with_derived(
                frame=run, latitude=row["latitude"], longitude=row["longitude"]
            )
            run = add_lags(frame=run, power=arrays[site].power).sort(
                "valid_time", "ensemble_member"
            )
            members = run["ensemble_member"].n_unique()
            if run.height != run.select(*KEYS).n_unique() * members:
                msg = "every valid time must hold every member"
                raise ValueError(msg)
            if run["ensemble_member"].min() != 0:
                msg = "member 0, the control, must be the first member"
                raise ValueError(msg)
            origin = np.asarray(common.grid_index(times=run["power_fcst_init_time"]))
            matrix = run.select(columns).to_numpy().astype(np.float32)
            keys = run.select(*KEYS).unique(maintain_order=True)
            for (arm, seed), booster in boosters.items():
                kinds, use_id, cold, _ = jobs[arm]
                full = np.hstack(
                    [
                        matrix,
                        fingerprint(
                            kinds=kinds, row=row, power=arrays[site].power, origin=origin, cold=cold
                        ),
                    ]
                )
                if use_id:
                    dmatrix = xgb.DMatrix(
                        np.hstack([full, np.full((full.shape[0], 1), codes[site], np.float32)]),
                        feature_types=["q"] * full.shape[1] + ["c"],
                        enable_categorical=True,
                    )
                else:
                    dmatrix = xgb.DMatrix(full)
                raw = booster.predict(dmatrix).reshape(-1, members, len(common.LEVELS))
                mixtures[arm, seed].append(
                    keys.with_columns(
                        pl.DataFrame(
                            mixture_quantiles(member_quantiles=raw),
                            schema=list(common.LEVEL_COLUMNS),
                            orient="row",
                        )
                    )
                )
        for key, frames in mixtures.items():
            pieces[key].append(pl.concat(frames))
        print(f"predicted {site}: {time.monotonic() - started:.0f} s", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for (arm, seed), frames in pieces.items():
        frame = pl.concat(frames).with_columns(seed=pl.lit(seed).cast(pl.Int8))
        tmp = out_dir / f"{arm}_seed{seed}.tmp"
        frame.write_parquet(tmp)
        tmp.rename(out_dir / f"{arm}_seed{seed}.parquet")


def main() -> None:
    """Run the requested stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("fit", "predict"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(common.SEEDS))
    parser.add_argument("--smoke", action="store_true", help="a scratch folder and 4 series")
    args = parser.parse_args()
    work_dir = common.WORK_DIR.with_name("work_smoke") if args.smoke else common.WORK_DIR
    work_dir.mkdir(parents=True, exist_ok=True)
    series = common.series_table()
    held_out = list(common.HELD_OUT)
    if args.smoke:
        series = series.filter(pl.col("site").is_in(["D1", "D13", "D10", "D12", "D5", "D2"]))
        held_out = ["D10", "D12"]
    if args.stage == "fit":
        run_fit(series=series, seeds=args.seeds, work_dir=work_dir, held_out=held_out)
    else:
        run_predict(
            series=series,
            seeds=args.seeds,
            work_dir=work_dir,
            out_dir=work_dir / "arms",
            held_out=held_out,
        )


if __name__ == "__main__":
    main()
