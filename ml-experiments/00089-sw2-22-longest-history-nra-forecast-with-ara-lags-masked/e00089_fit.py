"""Fit the arms of experiment 00089 and write each monthly refit's quantiles.

One job is one monthly refit (P-roll, experiment 00049). A refit at time `t` trains, for each
Primary series and each arm, an N1 model on the ENS control member's rows that the arm's window
admits (see `ARMS`), with the rows whose valid time lies inside an abnormal running arrangement
(ARA) removed from the training targets, for 150 rounds at a learning rate of 0.15. The arms that
mask lags set to missing every lag whose time lies inside an ARA, in training and in prediction.
It then forecasts the daily runs of the 292 of `fold_a_daily_gap_v1` that the monthly regime
assigns to it, and writes for each arm and series the mixture of the 51 members' nine quantiles to
`work/anchor_preds/<arm>/<site>_s0_<refit>.parquet`. Existing files stay, so a stopped run
resumes. Seed 0 only.

The ARA is the union of the XL detector's intervals, NGED's logged events, and NGED's logged
closures. Training rows come from three places: the control-member backfill frames of experiment
00081 (2021-03-21 to 2024-03-31, every `BACKFILL_STRIDE`-th run), the stage 1 pilot's every-5-days
training runs (2024-04-01 to 2025-06-30), and experiment 00034's daily control-member runs
(2025-07-01 onward).

    uv run python ml-experiments/00089-*/e00089_fit.py [--workers 3] [--smoke]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00089_common as common
import e00089_labels as labelling
import e00089_regimes as regimes
import numpy as np
import polars as pl
import xgboost as xgb

CORES: Final[int] = 6
BACKFILL_FRAMES_DIR: Final[Path] = (
    common.ML_EXPERIMENTS_DATA_DIR
    / "00081-r11-control-member-second-validation-year-demand"
    / "work"
    / "frames"
)
"""Experiment 00081's control-member demand frames, one `init=<date>.parquet` per run, read only."""
BACKFILL_STRIDE: Final[int] = 5
"""Every fifth backfill run is kept, as the stage 1 pilot's training runs are every fifth day."""
ENS_START: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
"""The first day of the Dynamical.org ENS archive, a different product from the backfill."""


@dataclass(frozen=True)
class Arm:
    """How an arm picks its training rows and treats its lags.

    Attributes:
        rolling: True for the 365 days before the refit; False for all rows from `start`.
        start: The earliest valid time of the rows when `rolling` is False.
        half_life_days: Half-life of the recency weight, anchored at the refit.
        mask_lags: True to set to missing every lag whose time lies inside an ARA.
    """

    rolling: bool
    start: datetime | None
    half_life_days: float
    mask_lags: bool


ARMS: Final[dict[str, Arm]] = {
    "r": Arm(rolling=True, start=None, half_life_days=common.SHORT_HALF_LIFE_DAYS, mask_lags=False),
    "m": Arm(rolling=True, start=None, half_life_days=common.SHORT_HALF_LIFE_DAYS, mask_lags=True),
    "l": Arm(
        rolling=False, start=None, half_life_days=common.RECENCY_HALF_LIFE_DAYS, mask_lags=False
    ),
    "lm": Arm(
        rolling=False, start=None, half_life_days=common.RECENCY_HALF_LIFE_DAYS, mask_lags=True
    ),
    "l_ens": Arm(
        rolling=False,
        start=ENS_START,
        half_life_days=common.RECENCY_HALF_LIFE_DAYS,
        mask_lags=False,
    ),
    "lm_ens": Arm(
        rolling=False,
        start=ENS_START,
        half_life_days=common.RECENCY_HALF_LIFE_DAYS,
        mask_lags=True,
    ),
}
"""R and M: the 12-month window (experiment 00086's recipe). L and LM: all rows since 2021-03.
The `_ens` arms: all rows since 2024-04-01, the clean check on one weather product."""
ARA_LAG_COLUMNS: Final[tuple[str, ...]] = tuple(f"ara_{name}" for name in common.LAG_COLUMNS)
"""True where the lag's time lies inside an ARA."""
SITE_CHUNKS: Final[int] = 2
"""Each refit's series are split into this many jobs, so that files are written sooner."""
ROUNDS: Final[int] = 150
ETA: Final[float] = 0.15
WEATHER: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
)


def input_columns() -> list[str]:
    """Return the input columns in matrix order; every arm has the same ones."""
    return [*common.feature_columns(), *common.N1_EXTRA_COLUMNS, *common.LAG_COLUMNS]


def ara_array(*, labels: dict[str, labelling.SeriesLabels]) -> np.ndarray:
    """Return the half-hours inside an ARA: an XL event, a logged event, or a logged closure."""
    return labels["xl"].in_event | labels["oracle"].in_event | labels["oracle"].closed


def add_lags(*, frame: pl.DataFrame, power: np.ndarray, ara: np.ndarray) -> pl.DataFrame:
    """Add the three origin lags, a flag per lag for a lag inside an ARA, and the target's flag."""
    target = np.asarray(common.grid_index(times=frame["valid_time"]))
    origin = np.asarray(common.grid_index(times=frame["power_fcst_init_time"]))
    lags = pl.DataFrame(
        common.origin_lags(power=power, target=target, origin=origin),
        schema=list(common.LAG_COLUMNS),
        orient="row",
    ).cast(pl.Float32)
    inside = common.origin_lags(power=ara.astype(np.float32), target=target, origin=origin) > 0.5
    flags = pl.DataFrame(inside, schema=list(ARA_LAG_COLUMNS), orient="row")
    in_ara = common.take(values=ara.astype(np.float32), index=target) > 0.5
    return frame.hstack(lags).hstack(flags).with_columns(in_ara=pl.Series(in_ara))


def for_arm(*, frame: pl.DataFrame, arm: Arm) -> pl.DataFrame:
    """Return the frame, with a lag inside an ARA set to missing if the arm masks lags."""
    if not arm.mask_lags:
        return frame
    return frame.with_columns(
        pl.when(pl.col(flag)).then(None).otherwise(pl.col(lag)).alias(lag)
        for lag, flag in zip(common.LAG_COLUMNS, ARA_LAG_COLUMNS, strict=True)
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


_BACKFILL: dict[str, pl.DataFrame] = {}


def _backfill_rows() -> pl.DataFrame:
    """Return every `BACKFILL_STRIDE`-th backfill run's rows for all series, kept per worker."""
    if "rows" not in _BACKFILL:
        paths = sorted(BACKFILL_FRAMES_DIR.glob("init=*.parquet"))[::BACKFILL_STRIDE]
        _BACKFILL["rows"] = pl.concat(
            [
                pl.read_parquet(
                    path, columns=[*common.KEYS, "ensemble_member", *WEATHER]
                ).with_columns(pl.col(*WEATHER).cast(pl.Float32))
                for path in paths
            ]
        )
    return _BACKFILL["rows"]


def training_rows(*, row: dict, arrays: common.SeriesArrays, ara: np.ndarray) -> pl.DataFrame:
    """Return one series' control-member rows with derived columns, lags, and ARA flags.

    The rows run from 2021-03-21 to 2026-06-29. The power of the backfill rows and of the rows
    after 2025-06-30 is read from the dense grid. Every row's eligibility for a refit is applied
    per refit, not here.

    Args:
        row: The series' row of `common.series_table()`.
        arrays: The series' arrays on the dense grid.
        ara: The series' ARA array on the dense grid.

    Returns:
        The rows keyed by series, run, and valid time, with `ensemble_member`, `power` (in MW),
        the weather columns, the derived columns, `lead_hours`, the three lags, the three
        lag-inside-an-ARA flags, and `in_ara` (the valid time lies inside an ARA).
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
    backfill = _backfill_rows().filter(pl.col("time_series_id") == series_id)
    unfilled = pl.concat(
        [backfill, after.select(backfill.columns).cast(backfill.schema)]
    ).with_columns(power=pl.lit(None, dtype=pl.Float32))
    rows = pl.concat(
        [
            unfilled.select(before.columns).cast(before.schema),
            before,
        ]
    ).sort("power_fcst_init_time", "valid_time", "ensemble_member")
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
    return add_lags(frame=rows, power=arrays.power, ara=ara)


def known_rows(*, rows: pl.DataFrame, refit: datetime, arm: Arm) -> pl.DataFrame:
    """Return the rows a refit at `refit` may see in an arm's window.

    A refit sees a row only if its valid time is at least 3.5 hours before the refit, its run
    began at or before the refit, and its valid time lies in the arm's window: the 365 days before
    the refit, or on or after the arm's start.
    """
    known = rows.filter(
        (pl.col("valid_time") <= refit - regimes.REFIT_CUTOFF_AGE)
        & (pl.col("power_fcst_init_time") <= refit)
    )
    if arm.rolling:
        return known.filter(pl.col("valid_time") > refit - regimes.TRAIN_WINDOW)
    if arm.start is not None:
        return known.filter(pl.col("valid_time") >= arm.start)
    return known


def train_matrix(*, rows: pl.DataFrame, row: dict, refit: datetime, arm: Arm) -> xgb.DMatrix:
    """Return the weighted training matrix of one series' rows for an arm, anchored at `refit`.

    Rows whose valid time lies inside an ARA are dropped for every arm.
    """
    rows = for_arm(frame=rows.filter(~pl.col("in_ara")), arm=arm)
    age_days = (refit - rows["valid_time"]).dt.total_seconds() / 86_400
    return xgb.DMatrix(
        rows.select(input_columns()).to_numpy().astype(np.float32),
        label=(rows["power"] / row["effective_capacity_mw"]).to_numpy(),
        weight=0.5 ** (age_days.to_numpy() / arm.half_life_days),
    )


def fit(*, matrix: xgb.DMatrix, rounds: int, nthread: int) -> xgb.Booster:
    """Fit one quantile model with seed 0."""
    return xgb.train(
        {**common.QUANTILE_PARAMETERS, "seed": common.SEED, "eta": ETA, "nthread": nthread},
        matrix,
        num_boost_round=rounds,
    )


def predict_run(
    *,
    model: xgb.Booster,
    arm: Arm,
    run_rows: pl.DataFrame,
    row: dict,
    arrays: common.SeriesArrays,
    ara: np.ndarray,
) -> pl.DataFrame:
    """Return the mixture quantiles of one run for one series and arm.

    Args:
        model: The series' fitted booster for the arm.
        arm: The arm.
        run_rows: The run's frame rows for the series, every valid time with every member.
        row: The series' row of `common.series_table()`.
        arrays: The series' arrays on the dense grid.
        ara: The series' ARA array on the dense grid.

    Returns:
        One row per valid time: the key columns and the nine quantile columns.

    Raises:
        ValueError: If a valid time lacks a member.
    """
    prepared = common.with_derived(
        frame=run_rows, latitude=row["latitude"], longitude=row["longitude"]
    )
    prepared = add_lags(frame=prepared, power=arrays.power, ara=ara)
    prepared = for_arm(frame=prepared, arm=arm).sort("valid_time", "ensemble_member")
    members = prepared["ensemble_member"].n_unique()
    if prepared.height != prepared.select(*common.KEYS).n_unique() * members:
        msg = "every valid time must hold every member"
        raise ValueError(msg)
    matrix = prepared.select(input_columns()).to_numpy().astype(np.float32)
    raw = model.predict(xgb.DMatrix(matrix)).reshape(-1, members, len(common.LEVELS))
    return (
        prepared.select(*common.KEYS)
        .unique(maintain_order=True)
        .hstack(
            pl.DataFrame(mixture_quantiles(member_quantiles=raw), schema=list(common.LEVEL_COLUMNS))
        )
    )


def booster_path(*, arm: str, site: str, refit: datetime) -> Path:
    """Return the file one fitted booster is saved to."""
    return common.BOOSTERS_DIR / arm / f"{site}_s{common.SEED}_{refit:%Y%m%dT%H%M}.ubj"


def job_path(*, root: Path, arm: str, site: str, refit: datetime) -> Path:
    """Return the file one refit's predictions for one arm and series are written to."""
    return root / arm / f"{site}_s{common.SEED}_{refit:%Y%m%dT%H%M}.parquet"


_CACHE: dict[str, object] = {}


def _context() -> tuple[
    pl.DataFrame,
    dict[str, common.SeriesArrays],
    dict[str, np.ndarray],
]:
    """Return the Primary series table, arrays, and ARA arrays, loaded once per worker process."""
    if "series" not in _CACHE:
        series = common.series_table().filter(pl.col("substation_type").is_in(common.GROUPS))
        _CACHE["series"] = series
        _CACHE["arrays"] = common.load_series_arrays(series=series)
        oracle = labelling.load_labels(series=series)
        xl = labelling.load_xl_labels(series=series, oracle=oracle)
        _CACHE["ara"] = {
            site: ara_array(labels={"oracle": oracle[site], "xl": xl[site]}) for site in oracle
        }
        _CACHE["rows"] = {}
    return _CACHE["series"], _CACHE["arrays"], _CACHE["ara"]  # ty: ignore[invalid-return-type]


def _training_rows_cached(
    *, row: dict, arrays: common.SeriesArrays, ara: np.ndarray
) -> pl.DataFrame:
    """Return a series' training rows, kept for the worker's later jobs."""
    cache: dict[str, pl.DataFrame] = _CACHE["rows"]  # ty: ignore[invalid-assignment]
    if row["site"] not in cache:
        cache[row["site"]] = training_rows(row=row, arrays=arrays, ara=ara)
    return cache[row["site"]]


def run_job(
    *,
    refit: datetime,
    origins: list[datetime],
    sites: list[str],
    out_root: Path,
    rounds: int,
    nthread: int,
) -> list[str]:
    """Fit one refit for each (series, arm) that lacks its file, forecast its runs, write files.

    Args:
        refit: The refit time.
        origins: The `power_fcst_init_time` of the runs the refit forecasts.
        sites: The series labels to fit.
        out_root: The folder holding one subfolder per arm.
        rounds: Boosting rounds.
        nthread: Threads for each fit.

    Returns:
        One progress line per series and arm fitted.
    """
    series, arrays, ara = _context()
    todo = [
        (site, arm)
        for site in sites
        for arm in ARMS
        if not job_path(root=out_root, arm=arm, site=site, refit=refit).exists()
        or not (
            out_root != common.ANCHOR_PREDS_DIR
            or booster_path(arm=arm, site=site, refit=refit).exists()
        )
    ]
    if not todo:
        return [f"refit {refit:%Y-%m-%d %H:%M}: all files exist"]
    rows_of = {row["site"]: row for row in series.iter_rows(named=True)}
    models: dict[tuple[str, str], xgb.Booster] = {}
    lines = []
    for site, arm in todo:
        started = time.monotonic()
        row = rows_of[site]
        known = known_rows(
            rows=_training_rows_cached(row=row, arrays=arrays[site], ara=ara[site]),
            refit=refit,
            arm=ARMS[arm],
        )
        matrix = train_matrix(rows=known, row=row, refit=refit, arm=ARMS[arm])
        models[site, arm] = fit(matrix=matrix, rounds=rounds, nthread=nthread)
        if out_root == common.ANCHOR_PREDS_DIR:
            target = booster_path(arm=arm, site=site, refit=refit)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f"tmp_{target.name}")
            models[site, arm].save_model(temporary)
            temporary.rename(target)
        lines.append(
            f"{arm} {site} refit {refit:%Y-%m-%d %H:%M}: {matrix.num_row():,} of {known.height:,} "
            f"rows, {len(origins)} runs, fit {time.monotonic() - started:.0f} s"
        )
    pieces: dict[tuple[str, str], list[pl.DataFrame]] = {key: [] for key in todo}
    for origin in origins:
        frame = pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
        for site, arm in todo:
            pieces[site, arm].append(
                predict_run(
                    model=models[site, arm],
                    arm=ARMS[arm],
                    run_rows=frame.filter(
                        pl.col("time_series_id") == rows_of[site]["time_series_id"]
                    ),
                    row=rows_of[site],
                    arrays=arrays[site],
                    ara=ara[site],
                )
            )
    for site, arm in todo:
        common.atomic_write(
            frame=pl.concat(pieces[site, arm]).with_columns(seed=pl.lit(common.SEED).cast(pl.Int8)),
            target=job_path(root=out_root, arm=arm, site=site, refit=refit),
        )
    return lines


def main() -> None:
    """Fit and write every arm's jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3, help="processes; each gets 6 // workers")
    parser.add_argument(
        "--smoke", action="store_true", help="2 series, 2 refits, 3 runs, 20 rounds, scratch dir"
    )
    args = parser.parse_args()
    series = common.series_table().filter(pl.col("substation_type").is_in(common.GROUPS))
    sites = series["site"].to_list()
    gap = set(common.origins_of_subset(subset="gap"))
    jobs = {
        refit: [origin for origin in origins if origin in gap]
        for refit, origins in regimes.served(name="monthly", origins=common.all_origins()).items()
    }
    jobs = {refit: origins for refit, origins in jobs.items() if origins}
    out_root = common.ANCHOR_PREDS_DIR
    rounds = ROUNDS
    workers = args.workers
    if args.smoke:
        sites = sites[:2]
        jobs = {refit: origins[:3] for refit, origins in list(jobs.items())[:2]}
        out_root = common.ANCHOR_PREDS_DIR.with_name("anchor_preds_smoke")
        rounds = 20
        workers = 1
    nthread = max(1, CORES // workers)
    started = time.monotonic()
    ordered = sorted(jobs.items(), key=lambda item: -len(item[1]))
    site_chunks = [sites[i::SITE_CHUNKS] for i in range(min(SITE_CHUNKS, len(sites)))]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                run_job,
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
