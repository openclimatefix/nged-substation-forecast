"""Fit experiment 00093's frozen demand models and write their forecasts for the validation year.

The recipe is experiment 00085's P-roll recipe (150 rounds at `eta` 0.15): one quantile model per
series, the ENS control member's rows as training data, with 11 quantile levels (0.01 and 0.99
added). Each arm trains once on the rows of one year (`common.WINDOWS`), with equal weights and no
refit, and forecasts every daily run from 2025-07-01 to 2026-06-30 (51 members mixed). An arm is a
training window and a level rule:

- `raw`: the target and the power lags as they are.
- `all`: the trailing 365-day mean of the series' power over capacity, taken at the newest usable
  observation of each row, is subtracted from the target and the three lags in training and
  added back to the forecast. Half-hours inside a labelled abnormal arrangement do not count.
- `hh`: the same, with a mean over the same half-hour of the day on each of the 365 days (post hoc).

Training drops targets inside a labelled abnormal arrangement. Stage `fit` trains and saves the
boosters. Stage `predict` forecasts the runs in worker processes, each with its own GPU context.
Existing files stay, so a stopped run resumes.

    uv run python ml-experiments/00093-*/e00093_fit.py --stage fit [--smoke]
    uv run python ml-experiments/00093-*/e00093_fit.py --stage predict --workers 4 [--smoke]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00093_common as common
import e00093_labels as labelling
import numpy as np
import polars as pl
import xgboost as xgb

WEATHER: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
)
LEVEL_DAYS: Final[int] = 365
FIT_ROUNDS: Final[int] = 150
FIT_ETA: Final[float] = 0.15
"""The P-roll recipe of `briefs/common.md`: 150 rounds at `eta` 0.15, about 3 times cheaper."""
MIN_LEVEL_COVERAGE: Final[float] = 0.5
"""A trailing level needs at least this share of its 365 days observed and outside an ARA."""
BOOSTER_DIR: Final[Path] = common.WORK_DIR / "boosters"
booster_root: dict[str, Path] = {"path": BOOSTER_DIR}
"""The folder boosters are read from and written to; the smoke test swaps it."""
PREDICT_CHUNK_RUNS: Final[int] = 10
LAG_COLUMNS_ALL: Final[tuple[str, ...]] = common.LAG_COLUMNS


@dataclass(frozen=True)
class Arm:
    """One training window and level rule.

    Attributes:
        name: The folder name under `preds` and `boosters`.
        window: A key of `common.WINDOWS`.
        level: `raw`, `all`, or `hh`.
    """

    name: str
    window: str
    level: str


ARMS: Final[dict[str, Arm]] = {
    f"{window}_{level}": Arm(f"{window}_{level}", window, level)
    for window in common.WINDOWS
    for level in ("raw", "all", "hh")
    if (window, level) != ("old2", "hh")
}


def input_columns() -> list[str]:
    """Return the input columns in matrix order."""
    return [*common.feature_columns(), *common.N1_EXTRA_COLUMNS, *common.LAG_COLUMNS]


def ara_array(*, labels: labelling.SeriesLabels) -> np.ndarray:
    """Return True where the half-hour lies inside a labelled event or a logged closure."""
    return labels.in_event | labels.closed


def trailing_levels(*, power: np.ndarray, ara: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the trailing 365-day level on the grid, over all hours and per half-hour of day.

    Args:
        power: Observed power over capacity on the grid, NaN where unobserved.
        ara: True where the half-hour is inside an abnormal running arrangement.

    Returns:
        `(all_hours, per_half_hour)`. Entry `i` of the first is the mean of the usable values in
        the 365 days ending at `i`; entry `i` of the second is the mean of the usable values at
        `i`'s half-hour of day on the 365 days ending at `i`'s day. A level with under half of its
        values usable is NaN.
    """
    usable = np.where(ara, np.nan, power).astype(np.float64)
    ok = np.isfinite(usable)
    values = np.where(ok, usable, 0.0)
    window = LEVEL_DAYS * common.STEPS_PER_DAY
    total = np.concatenate([[0.0], np.cumsum(values)])
    count = np.concatenate([[0], np.cumsum(ok)])
    size = usable.size
    ends = np.arange(1, size + 1)
    starts = np.maximum(ends - window, 0)
    n_all = count[ends] - count[starts]
    all_hours = np.where(
        n_all >= MIN_LEVEL_COVERAGE * window,
        (total[ends] - total[starts]) / np.maximum(n_all, 1),
        np.nan,
    )
    days = -(-size // common.STEPS_PER_DAY)
    padded = days * common.STEPS_PER_DAY - size
    grid_values = np.concatenate([values, np.zeros(padded)]).reshape(days, common.STEPS_PER_DAY)
    grid_ok = np.concatenate([ok, np.zeros(padded, dtype=bool)]).reshape(days, common.STEPS_PER_DAY)
    cum_v = np.concatenate([np.zeros((1, common.STEPS_PER_DAY)), np.cumsum(grid_values, axis=0)])
    cum_n = np.concatenate([np.zeros((1, common.STEPS_PER_DAY)), np.cumsum(grid_ok, axis=0)])
    day_ends = np.arange(1, days + 1)
    day_starts = np.maximum(day_ends - LEVEL_DAYS, 0)
    n_hh = cum_n[day_ends] - cum_n[day_starts]
    mean_hh = np.where(
        n_hh >= MIN_LEVEL_COVERAGE * LEVEL_DAYS,
        (cum_v[day_ends] - cum_v[day_starts]) / np.maximum(n_hh, 1),
        np.nan,
    )
    return all_hours.astype(np.float32), mean_hh.reshape(-1)[:size].astype(np.float32)


def row_levels(
    *, levels: tuple[np.ndarray, np.ndarray], target: np.ndarray, origin: np.ndarray, rule: str
) -> np.ndarray:
    """Return the level to subtract from each row's target and lags: shape (rows, 1 + 3 lags).

    Column 0 is the target's level, columns 1 to 3 the levels of `LAG_COLUMNS`.

    Args:
        levels: `trailing_levels` output.
        target: Grid index of each row's valid time.
        origin: Grid index of each row's initialisation time.
        rule: `raw` (zeros), `all`, or `hh`.

    Returns:
        A float32 array; NaN where the level is unavailable.
    """
    if rule == "raw":
        return np.zeros((target.size, 4), dtype=np.float32)
    newest = origin - common.NEWEST_AGE
    if rule == "all":
        one = common.take(values=levels[0], index=newest)
        return np.repeat(one[:, None], 4, axis=1)
    same_hour = newest - ((newest - target) % common.STEPS_PER_DAY)
    target_level = common.take(values=levels[1], index=same_hour)
    newest_level = common.take(values=levels[1], index=newest)
    return np.stack([target_level, newest_level, target_level, target_level], axis=1)


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


def feature_matrix(
    *, frame: pl.DataFrame, rule: str, levels: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the input matrix with its lags shifted by the level, and the target's level.

    Args:
        frame: Rows with the weather, derived columns, `lead_hours`, and the three lags.
        rule: `raw`, `all`, or `hh`.
        levels: `trailing_levels` output.

    Returns:
        The float32 matrix in `input_columns()` order and the level (rows,) of the target.
    """
    target = np.asarray(common.grid_index(times=frame["valid_time"]))
    origin = np.asarray(common.grid_index(times=frame["power_fcst_init_time"]))
    shift = row_levels(levels=levels, target=target, origin=origin, rule=rule)
    columns = input_columns()
    matrix = frame.select(columns).to_numpy().astype(np.float32)
    first_lag = columns.index(common.LAG_COLUMNS[0])
    matrix[:, first_lag : first_lag + 3] -= shift[:, 1:4]
    return matrix, shift[:, 0]


def training_rows(
    *, row: dict, arrays: common.SeriesArrays, ara: np.ndarray, window: str
) -> pl.DataFrame:
    """Return one series' control-member rows of a training window, targets outside any ARA.

    Args:
        row: The series' row of `common.series_table()`.
        arrays: The series' arrays on the dense grid.
        ara: The series' ARA flags on the grid.
        window: A key of `common.WINDOWS`.

    Returns:
        Rows keyed by series, run, and valid time, with `power` in MW, the weather columns, the
        derived columns, `lead_hours`, and the three lags.
    """
    start, end = common.WINDOWS[window]
    series_id = row["time_series_id"]
    columns = [*common.KEYS, "ensemble_member", "power", *WEATHER]
    if window == "recent":
        rows = (
            pl.scan_parquet(common.ENS_TRAIN_PATH)
            .filter(pl.col("time_series_id") == series_id)
            .select(columns)
            .collect()
        )
    else:
        days = pl.date_range(start.date(), end.date(), eager=True).to_list()
        pieces = []
        for day in days:
            path = common.E81_FRAMES_DIR / f"init={day:%Y-%m-%d}.parquet"
            if path.exists():
                pieces.append(
                    pl.read_parquet(path)
                    .filter(pl.col("time_series_id") == series_id)
                    .select(columns)
                    .with_columns(pl.col(*WEATHER).cast(pl.Float32))
                )
        rows = pl.concat(pieces).cast({"power": pl.Float32})
    rows = rows.filter(
        (pl.col("power_fcst_init_time") >= start)
        & (pl.col("power_fcst_init_time") <= end + timedelta(days=1))
        & (pl.col("valid_time") >= start)
        & (pl.col("valid_time") <= end)
    )
    index = np.asarray(common.grid_index(times=rows["valid_time"]))
    filled = common.take(values=arrays.power, index=index) * row["effective_capacity_mw"]
    rows = rows.with_columns(power=pl.Series(filled)).filter(
        pl.col("power").is_finite() & (pl.col("valid_time") > pl.col("power_fcst_init_time"))
    )
    in_ara = ara[np.asarray(common.grid_index(times=rows["valid_time"]))]
    rows = rows.filter(pl.Series(~in_ara))
    rows = common.with_derived(frame=rows, latitude=row["latitude"], longitude=row["longitude"])
    rows = rows.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    )
    return add_lags(frame=rows, power=arrays.power)


def fit_booster(*, matrix: np.ndarray, label: np.ndarray, rounds: int, nthread: int) -> xgb.Booster:
    """Fit one quantile model with seed 0 and equal weights."""
    parameters = {
        **common.QUANTILE_PARAMETERS,
        "eta": FIT_ETA,
        "seed": common.SEED,
        "nthread": nthread,
    }
    return xgb.train(parameters, xgb.DMatrix(matrix, label=label), num_boost_round=rounds)


def booster_path(*, arm: str, site: str) -> Path:
    """Return the file a fitted booster is saved to."""
    return booster_root["path"] / arm / f"{site}.ubj"


def load_inputs(
    *, sites: list[str]
) -> tuple[dict[str, dict], dict[str, common.SeriesArrays], dict[str, np.ndarray], dict]:
    """Return the series rows, grid arrays, ARA flags, and trailing levels of the sites."""
    series = common.series_table().filter(pl.col("site").is_in(sites))
    arrays = common.load_series_arrays(series=series)
    labels = labelling.load_labels(series=series)
    ara = {site: ara_array(labels=labels[site]) for site in labels}
    rows = {row["site"]: row for row in series.iter_rows(named=True)}
    levels = {site: trailing_levels(power=arrays[site].power, ara=ara[site]) for site in arrays}
    return rows, arrays, ara, levels


def usable_sites() -> list[str]:
    """Return the Primary series that appear in experiment 81's frames and have power throughout."""
    frame = pl.read_parquet(
        common.E81_FRAMES_DIR / "init=2021-07-05.parquet", columns=["time_series_id"]
    )
    ids = set(frame["time_series_id"].unique().to_list())
    series = common.series_table().filter(
        (pl.col("substation_type") == "Primary") & pl.col("time_series_id").is_in(ids)
    )
    return series["site"].to_list()


def fit_stage(*, sites: list[str], rounds: int, nthread: int) -> None:
    """Train and save the boosters of every arm for every site that lacks one."""
    rows, arrays, ara, levels = load_inputs(sites=sites)
    for site in sites:
        for window in common.WINDOWS:
            train = None
            for arm in (a for a in ARMS.values() if a.window == window):
                path = booster_path(arm=arm.name, site=site)
                if path.exists():
                    continue
                started = time.monotonic()
                if train is None:
                    train = training_rows(
                        row=rows[site], arrays=arrays[site], ara=ara[site], window=window
                    )
                matrix, shift = feature_matrix(frame=train, rule=arm.level, levels=levels[site])
                label = (train["power"] / rows[site]["effective_capacity_mw"]).to_numpy() - shift
                keep = np.isfinite(label)
                booster = fit_booster(
                    matrix=matrix[keep], label=label[keep], rounds=rounds, nthread=nthread
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                booster.save_model(str(path) + ".tmp.ubj")
                Path(str(path) + ".tmp.ubj").rename(path)
                print(
                    f"{arm.name} {site}: {int(keep.sum()):,} rows, "
                    f"{time.monotonic() - started:.0f} s",
                    flush=True,
                )


def mixture_quantiles(*, member_quantiles: np.ndarray) -> np.ndarray:
    """Pool members' quantiles into the mixture's quantiles, as experiment 00019 does."""
    cases, members, levels = member_quantiles.shape
    values = member_quantiles.reshape(cases, members * levels)
    weights = np.tile(common.ATOM_MASSES, members) / members
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    cumulative = np.cumsum(weights[order], axis=1)
    picks = (cumulative[:, :, None] >= (common.LEVELS - 1e-9)).argmax(axis=1)
    pooled = np.take_along_axis(sorted_values, picks, axis=1)
    return np.sort(pooled, axis=1).astype(np.float32)


def predict_chunk(*, chunk: int, origins: list, sites: list[str], arms: list[str]) -> str:
    """Forecast a chunk of runs for every site and arm, and write one file for each pair."""
    todo = [
        (arm, site)
        for arm in arms
        for site in sites
        if not pred_path(arm=arm, site=site, chunk=chunk).exists()
    ]
    if not todo:
        return f"chunk {chunk}: all files exist"
    todo_sites = sorted({site for _, site in todo})
    rows, arrays, ara, levels = load_inputs(sites=todo_sites)
    boosters = {}
    for arm, site in todo:
        booster = xgb.Booster()
        booster.load_model(str(booster_path(arm=arm, site=site)))
        booster.set_param({"device": "cuda", "nthread": 1})
        boosters[arm, site] = booster
    pieces: dict[tuple[str, str], list[pl.DataFrame]] = {pair: [] for pair in todo}
    for origin in origins:
        frame = pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
        frame = frame.filter(pl.col("valid_time") < common.HOLDOUT)
        for site in todo_sites:
            row = rows[site]
            prepared = common.with_derived(
                frame=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
                latitude=row["latitude"],
                longitude=row["longitude"],
            )
            prepared = add_lags(frame=prepared, power=arrays[site].power).sort(
                "valid_time", "ensemble_member"
            )
            members = prepared["ensemble_member"].n_unique()
            if prepared.height != prepared.select(*common.KEYS).n_unique() * members:
                msg = "every valid time must hold every member"
                raise ValueError(msg)
            keys = prepared.select(*common.KEYS).unique(maintain_order=True)
            target = np.asarray(common.grid_index(times=keys["valid_time"]))
            actual = common.take(values=arrays[site].power, index=target)
            extra = keys.with_columns(
                actual=pl.Series(actual),
                lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time"))
                .dt.total_minutes()
                .cast(pl.Float64)
                / 60.0,
                ara=pl.Series(ara[site][target]),
            )
            for arm, pair_site in todo:
                if pair_site != site:
                    continue
                matrix, shift = feature_matrix(
                    frame=prepared, rule=ARMS[arm].level, levels=levels[site]
                )
                raw = (
                    boosters[arm, site]
                    .predict(xgb.DMatrix(matrix))
                    .reshape(-1, members, len(common.LEVELS))
                )
                raw = raw + shift.reshape(-1, members)[:, :, None]
                mixed = mixture_quantiles(member_quantiles=raw)
                pieces[arm, site].append(
                    extra.hstack(pl.DataFrame(mixed, schema=list(common.LEVEL_COLUMNS)))
                )
    for (arm, site), parts in pieces.items():
        common.atomic_write(
            frame=pl.concat(parts).with_columns(seed=pl.lit(common.SEED).cast(pl.Int8)),
            target=pred_path(arm=arm, site=site, chunk=chunk),
        )
    return f"chunk {chunk}: {len(todo)} files"


def pred_path(*, arm: str, site: str, chunk: int) -> Path:
    """Return the file one arm's forecasts for one series and run chunk are written to."""
    return common.PREDS_DIR / arm / f"{site}_c{chunk:03d}.parquet"


def main() -> None:
    """Run the requested stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fit", "predict"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=sorted(ARMS))
    parser.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("I", "N"))
    parser.add_argument("--smoke", action="store_true", help="one series, 20 rounds, 3 runs")
    args = parser.parse_args()
    sites = usable_sites()[args.shard[0] :: args.shard[1]]
    rounds = FIT_ROUNDS
    origins = common.all_origins()
    if args.smoke:
        sites, rounds, origins = sites[:1], 20, origins[:3]
        common.PREDS_DIR = common.PREDS_DIR.with_name("preds_smoke")
        booster_root.update(path=BOOSTER_DIR.with_name("boosters_smoke"))
    started = time.monotonic()
    if args.stage == "fit":
        fit_stage(sites=sites, rounds=rounds, nthread=4)
    else:
        chunks = [
            origins[i : i + PREDICT_CHUNK_RUNS] for i in range(0, len(origins), PREDICT_CHUNK_RUNS)
        ]
        if args.smoke:
            print(predict_chunk(chunk=0, origins=origins, sites=sites, arms=args.arms))
            return
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(predict_chunk, chunk=k, origins=part, sites=sites, arms=args.arms)
                for k, part in enumerate(chunks)
            ]
            for future in futures:
                print(future.result(), flush=True)
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
