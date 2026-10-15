"""Fit experiment 00024's pooled quantile models, then write each arm's validation quantiles.

Every model is experiment 00019's N1H: the 9-level quantile loss on the ENS control member of every
daily run from 2024-04-01, valid times up to 2025-06-30 23:30, a recency half-life of 0.5 years
anchored there, 00006's columns, the lead in hours, and the three origin lags. The arms differ in
which series a model trains on (`e00024_common.ARMS`) and in whether the series identifier is a
categorical column.

Stage `fit` writes one booster per arm, group, and seed to `work/models/` and appends the fit
seconds to `work/fit_times.jsonl`. Stage `predict` reads those boosters and writes
`work/arms/<arm>_seed<s>.parquet`, the mixture of the 51 members' 9 quantiles, keyed by series,
run, and valid time. Existing files are kept.

    uv run python ml-experiments/00024-pooling-groupings-demand-xgboost/e00024_fit.py \
        --stage fit --seeds 0 1 2 [--smoke]
    uv run python ml-experiments/00024-pooling-groupings-demand-xgboost/e00024_fit.py \
        --stage predict --seeds 0 1 2 [--smoke]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Final

import e00024_common as common
import numpy as np
import polars as pl
import xgboost as xgb

KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")


def input_columns() -> list[str]:
    """Return the input columns in matrix order, without the series identifier."""
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


def with_identifier(*, matrix: np.ndarray, code: int, use: bool) -> xgb.DMatrix:
    """Return a DMatrix, with the series' integer code as a last categorical column if `use`.

    Args:
        matrix: Shape (rows, columns), float32.
        code: The series' integer code, 0 to 19.
        use: Append the identifier column when true.

    Returns:
        The DMatrix (no label).
    """
    if not use:
        return xgb.DMatrix(matrix)
    column = np.full((matrix.shape[0], 1), code, dtype=np.float32)
    types = ["q"] * matrix.shape[1] + ["c"]
    return xgb.DMatrix(np.hstack([matrix, column]), feature_types=types, enable_categorical=True)


def training_blocks(
    *, series: pl.DataFrame, arrays: dict[str, common.SeriesArrays], train: pl.DataFrame
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return each site's training matrix, target, and recency weights."""
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
        out[row["site"]] = (
            own.select(columns).to_numpy().astype(np.float32),
            (own["power"] / row["effective_capacity_mw"]).to_numpy().astype(np.float32),
            (0.5 ** (age_days.to_numpy() / common.SHORT_HALF_LIFE_DAYS)).astype(np.float32),
        )
    return out


def model_path(*, models_dir: Path, arm: str, group: str, seed: int) -> Path:
    """Return the file a group's booster is saved in."""
    return models_dir / arm / f"{group}_seed{seed}.ubj"


def run_fit(*, series: pl.DataFrame, seeds: list[int], work_dir: Path) -> None:
    """Fit every arm's boosters for the seeds and log the fit seconds."""
    arrays = common.load_series_arrays(series=series)
    blocks = training_blocks(
        series=series, arrays=arrays, train=pl.read_parquet(common.ENS_TRAIN_PATH)
    )
    codes = {site: position for position, site in enumerate(series["site"].to_list())}
    models_dir = work_dir / "models"
    for arm, (grouping, use_id) in common.ARMS.items():
        groups = common.group_labels(series=series, grouping=grouping)
        for group in sorted(set(groups.values())):
            sites = [site for site, name in groups.items() if name == group]
            matrix = np.vstack([blocks[s][0] for s in sites])
            target = np.concatenate([blocks[s][1] for s in sites])
            weights = np.concatenate([blocks[s][2] for s in sites])
            identifier = np.concatenate(
                [np.full(blocks[s][0].shape[0], codes[s], dtype=np.float32) for s in sites]
            )
            if use_id:
                matrix = np.hstack([matrix, identifier[:, None]])
            for seed in seeds:
                path = model_path(models_dir=models_dir, arm=arm, group=group, seed=seed)
                if path.exists():
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                started = time.monotonic()
                dtrain = xgb.DMatrix(
                    matrix,
                    label=target,
                    weight=weights,
                    feature_types=["q"] * (matrix.shape[1] - 1) + ["c"] if use_id else None,
                    enable_categorical=use_id,
                )
                model = xgb.train(
                    {**common.QUANTILE_PARAMETERS, "seed": seed},
                    dtrain,
                    num_boost_round=common.N_ROUNDS,
                )
                seconds = time.monotonic() - started
                model.save_model(path)
                with (work_dir / "fit_times.jsonl").open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "arm": arm,
                                "group": group,
                                "seed": seed,
                                "series": len(sites),
                                "rows": int(matrix.shape[0]),
                                "seconds": seconds,
                            }
                        )
                        + "\n"
                    )
                print(
                    f"{arm} {group} seed {seed}: {len(sites)} series, {seconds:.0f} s", flush=True
                )


def run_predict(*, series: pl.DataFrame, seeds: list[int], work_dir: Path, out_dir: Path) -> None:
    """Predict every validation run with every arm's booster and write the arm tables."""
    arrays = common.load_series_arrays(series=series)
    codes = {site: position for position, site in enumerate(series["site"].to_list())}
    columns = input_columns()
    group_of = {
        arm: common.group_labels(series=series, grouping=grouping)
        for arm, (grouping, _) in common.ARMS.items()
    }
    todo = [
        (arm, seed)
        for arm in common.ARMS
        for seed in seeds
        if not (out_dir / f"{arm}_seed{seed}.parquet").exists()
    ]
    pieces: dict[tuple[str, int], list[pl.DataFrame]] = {key: [] for key in todo}
    for row in series.iter_rows(named=True):
        site = row["site"]
        started = time.monotonic()
        boosters = {}
        for arm, seed in todo:
            booster = xgb.Booster()
            booster.load_model(
                model_path(
                    models_dir=work_dir / "models",
                    arm=arm,
                    group=group_of[arm][site],
                    seed=seed,
                )
            )
            booster.set_param({"device": "cuda"})
            boosters[arm, seed] = booster
        mixtures: dict[tuple[str, int], list[pl.DataFrame]] = {key: [] for key in todo}
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
            matrix = run.select(columns).to_numpy().astype(np.float32)
            keys = run.select(*KEYS).unique(maintain_order=True)
            for (arm, seed), booster in boosters.items():
                use_id = common.ARMS[arm][1]
                dmatrix = with_identifier(matrix=matrix, code=codes[site], use=use_id)
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
    if args.smoke:
        series = series.filter(pl.col("site").is_in(["D1", "D13", "D10", "D12"]))
    if args.stage == "fit":
        run_fit(series=series, seeds=args.seeds, work_dir=work_dir)
    else:
        run_predict(series=series, seeds=args.seeds, work_dir=work_dir, out_dir=work_dir / "arms")


if __name__ == "__main__":
    main()
