"""Fit experiment 00095's frozen demand models and write their forecasts for the validation year.

The recipe is experiment 00093's raw arms (150 rounds at `eta` 0.15, 11 quantile levels, one model
per series, equal weights, no refit), without the level removal. An arm is a training year and a
weather product:

- `era5`: the weather columns are ERA5 at the valid time, for the same (run, valid time) pairs as
  the control member's rows. At inference the same ERA5 columns are used for each run (diagnostic:
  ERA5 arrives about 5 days late, so no live service could do this).
- `nwp`: the weather columns are the ENS control member's forecast for the run. Only the 2023-07
  year (`mid_nwp`) is fitted here; experiment 00093's `old1_raw`, `old2_raw`, and `recent_raw`
  forecasts are read as they are.

Training drops targets inside a labelled abnormal arrangement. Stage `fit` trains and saves the
boosters. Stage `predict` forecasts the runs in worker processes, each with its own GPU context.
Existing files stay, so a stopped run resumes.

    uv run python ml-experiments/00095-*/e00095_fit.py --stage fit [--smoke]
    uv run python ml-experiments/00095-*/e00095_fit.py --stage predict --workers 4 [--smoke]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00095_common as common
import e00095_labels as labelling
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
FIT_ROUNDS: Final[int] = 150
FIT_ETA: Final[float] = 0.15
"""The P-roll recipe of `briefs/common.md`: 150 rounds at `eta` 0.15, about 3 times cheaper."""
BOOSTER_DIR: Final[Path] = common.WORK_DIR / "boosters"
booster_root: dict[str, Path] = {"path": BOOSTER_DIR}
"""The folder boosters are read from and written to; the smoke test swaps it."""
PREDICT_CHUNK_RUNS: Final[int] = 10


@dataclass(frozen=True)
class Arm:
    """One training year and weather product.

    Attributes:
        name: The folder name under `preds` and `boosters`.
        window: A key of `common.WINDOWS`.
        product: `era5` or `nwp` (the ENS control member).
    """

    name: str
    window: str
    product: str


ARMS: Final[dict[str, Arm]] = {
    **{f"{w}_era5": Arm(f"{w}_era5", w, "era5") for w in ("old1", "old2", "mid", "recent")},
    "mid_nwp": Arm("mid_nwp", "mid", "nwp"),
}


def input_columns() -> list[str]:
    """Return the input columns in matrix order."""
    return [*common.feature_columns(), *common.N1_EXTRA_COLUMNS, *common.LAG_COLUMNS]


def ara_array(*, labels: labelling.SeriesLabels) -> np.ndarray:
    """Return True where the half-hour lies inside a labelled event or a logged closure."""
    return labels.in_event | labels.closed


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


def feature_matrix(*, frame: pl.DataFrame, arrays: common.SeriesArrays, product: str) -> np.ndarray:
    """Return the float32 input matrix of a frame in `input_columns()` order.

    Args:
        frame: Rows with the derived columns, `lead_hours`, the three lags, and for `nwp` the
            weather columns.
        arrays: The series' arrays; for `era5` its feature matrix supplies the weather.
        product: `era5` or `nwp`.

    Returns:
        The matrix. For `era5` the columns of `common.feature_columns()` are taken from the
        series' ERA5 features at each row's valid time; the rest are the frame's.
    """
    columns = input_columns()
    matrix = frame.select(columns).to_numpy().astype(np.float32)
    if product == "era5":
        index = np.asarray(common.grid_index(times=frame["valid_time"]))
        n_features = len(common.feature_columns())
        matrix[:, :n_features] = arrays.features[index]
    return matrix


def window_rows(*, window: str, series_id: int) -> pl.DataFrame:
    """Return the control-member rows of one series whose run starts in the window's span."""
    start, end = common.WINDOWS[window]
    columns = [*common.KEYS, "ensemble_member", "power", *WEATHER]
    pieces = []
    last_frame_day = common.PILOT_TRAIN_START - timedelta(days=1)
    if window == "recent" or end >= common.PILOT_TRAIN_START:
        pieces.append(
            pl.scan_parquet(common.ENS_TRAIN_PATH)
            .filter(
                (pl.col("time_series_id") == series_id)
                & (pl.col("power_fcst_init_time") >= max(start, common.PILOT_TRAIN_START))
            )
            .select(columns)
            .collect()
        )
    if start < common.PILOT_TRAIN_START:
        last = min(end, last_frame_day)
        for day in pl.date_range(start.date(), last.date(), eager=True).to_list():
            path = common.E81_FRAMES_DIR / f"init={day:%Y-%m-%d}.parquet"
            if path.exists():
                pieces.append(
                    pl.read_parquet(path)
                    .filter(pl.col("time_series_id") == series_id)
                    .select(columns)
                    .with_columns(pl.col(*WEATHER).cast(pl.Float32))
                )
    return pl.concat(
        [p.cast({"power": pl.Float32, **dict.fromkeys(WEATHER, pl.Float32)}) for p in pieces]
    )


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
    rows = window_rows(window=window, series_id=row["time_series_id"])
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
) -> tuple[dict[str, dict], dict[str, common.SeriesArrays], dict[str, np.ndarray]]:
    """Return the series rows, grid arrays, and ARA flags of the sites."""
    series = common.series_table().filter(pl.col("site").is_in(sites))
    arrays = common.load_series_arrays(series=series)
    labels = labelling.load_labels(series=series)
    ara = {site: ara_array(labels=labels[site]) for site in labels}
    rows = {row["site"]: row for row in series.iter_rows(named=True)}
    return rows, arrays, ara


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


def fit_stage(*, sites: list[str], rounds: int, nthread: int, arms: list[str]) -> None:
    """Train and save the boosters of every requested arm for every site that lacks one."""
    rows, arrays, ara = load_inputs(sites=sites)
    for site in sites:
        for window in common.WINDOWS:
            train = None
            for arm in (ARMS[a] for a in arms if ARMS[a].window == window):
                path = booster_path(arm=arm.name, site=site)
                if path.exists():
                    continue
                started = time.monotonic()
                if train is None:
                    train = training_rows(
                        row=rows[site], arrays=arrays[site], ara=ara[site], window=window
                    )
                matrix = feature_matrix(frame=train, arrays=arrays[site], product=arm.product)
                label = (train["power"] / rows[site]["effective_capacity_mw"]).to_numpy()
                keep = np.isfinite(label) & np.isfinite(matrix[:, 0])
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
    rows, arrays, ara = load_inputs(sites=todo_sites)
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
                product = ARMS[arm].product
                matrix = feature_matrix(frame=prepared, arrays=arrays[site], product=product)
                raw = boosters[arm, site].predict(xgb.DMatrix(matrix))
                raw = raw.reshape(-1, members, len(common.LEVELS))
                if product == "era5":
                    # ERA5 is the same for every member, so the members' quantiles are identical.
                    mixed = np.sort(raw[:, 0, :], axis=1).astype(np.float32)
                else:
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
    parser.add_argument("--smoke", action="store_true", help="one series, 20 rounds, 3 runs")
    args = parser.parse_args()
    sites = usable_sites()
    rounds = FIT_ROUNDS
    origins = common.all_origins()
    if args.smoke:
        sites, rounds, origins = sites[:1], 20, origins[:3]
        common.PREDS_DIR = common.PREDS_DIR.with_name("preds_smoke")
        booster_root.update(path=BOOSTER_DIR.with_name("boosters_smoke"))
    started = time.monotonic()
    if args.stage == "fit":
        fit_stage(sites=sites, rounds=rounds, nthread=4, arms=args.arms)
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
