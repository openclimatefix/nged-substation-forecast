"""Fit the arms of experiment 00100 and write each monthly refit's quantiles.

One job is one monthly refit (P-roll, experiment 00049). A refit at time `t` trains, for each
Primary series and each arm, an N1 model on the ENS control member's rows from the 365 days before
`t`, with 11 quantile levels, for 150 rounds at a learning rate of 0.15. The arms (see `ARMS`):
`plain11` keeps every row and every lag; `mask` drops the rows whose valid time lies inside a
labelled event or a logged closure and sets to missing every lag inside a labelled event; `drop10`
and `drop30` add, in training only, pseudo-event intervals inside which lags are also set to
missing, so that about 10% and 30% of a series' training rows have at least one pseudo-masked lag.
At test time every arm masks only the lags inside a labelled event. Each refit forecasts the daily
runs of the 292 of `fold_a_daily_gap_v1` that the monthly regime assigns to it, and writes for each
arm and series the mixture of the 51 members' 11 quantiles to
`work/anchor_preds/<arm>/<site>_s0_<refit>.parquet`. Existing files stay, so a stopped run
resumes. Seed 0 only.

Training rows come from three places: the control-member backfill frames of experiment 00081
(2021-03-21 to 2024-03-31, every `BACKFILL_STRIDE`-th run), the stage 1 pilot's every-5-days
training runs (2024-04-01 to 2025-06-30), and experiment 00034's daily control-member runs
(2025-07-01 onward).

    uv run python ml-experiments/00100-*/e00100_fit.py [--workers 4] [--smoke]
"""

import argparse
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00100_common as common
import e00100_labels as labelling
import e00100_regimes as regimes
import numpy as np
import polars as pl
import xgboost as xgb

CORES: Final[int] = 8
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
    """How an arm picks its training rows and masks its lags.

    Attributes:
        mask_lags: True to set to missing every lag inside a labelled event, in training and test.
        drop_targets: True to drop the training rows whose valid time lies inside a labelled
            event or a logged closure.
        pseudo_percent: The target percentage of training rows with a pseudo-masked lag, or None.
    """

    mask_lags: bool
    drop_targets: bool
    pseudo_percent: int | None


ARMS: Final[dict[str, Arm]] = {
    "plain11": Arm(mask_lags=False, drop_targets=False, pseudo_percent=None),
    "mask": Arm(mask_lags=True, drop_targets=True, pseudo_percent=None),
    "drop10": Arm(mask_lags=True, drop_targets=True, pseudo_percent=10),
    "drop30": Arm(mask_lags=True, drop_targets=True, pseudo_percent=30),
}
"""All four arms use the 12-month rolling window and the recency half-life of 0.5 years."""
PSEUDO_PERCENTS: Final[tuple[int, ...]] = (10, 30)
MAX_PSEUDO_INTERVALS: Final[int] = 3_000
SHARE_TOLERANCE: Final[float] = 0.015
EVENT_LAG_COLUMNS: Final[tuple[str, ...]] = tuple(f"event_{name}" for name in common.LAG_COLUMNS)
"""True where the lag's time lies inside a labelled event."""


def pseudo_lag_columns(*, percent: int) -> tuple[str, ...]:
    """Return the names of the flags for a lag inside a pseudo event at a target percentage."""
    return tuple(f"pseudo{percent}_{name}" for name in common.LAG_COLUMNS)


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


def lag_flags(*, mask: np.ndarray, target: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Return, per row and lag, whether the lag's time lies inside the mask."""
    return common.origin_lags(power=mask.astype(np.float32), target=target, origin=origin) > 0.5


def add_lags(
    *,
    frame: pl.DataFrame,
    power: np.ndarray,
    event: np.ndarray,
    drop: np.ndarray | None = None,
    pseudo: dict[int, np.ndarray] | None = None,
) -> pl.DataFrame:
    """Add the three origin lags and the flags for lags inside a labelled or pseudo event.

    Args:
        frame: The rows, with `valid_time` and `power_fcst_init_time`.
        power: The series' power on the dense grid.
        event: Half-hours inside a labelled event.
        drop: Half-hours inside a labelled event or a logged closure; adds `in_drop` (training).
        pseudo: Pseudo-event masks by target percentage; adds their lag flags (training).

    Returns:
        The frame with the lags, `event_*` flags, and when given `in_drop` and `pseudo*_*` flags.
    """
    target = np.asarray(common.grid_index(times=frame["valid_time"]))
    origin = np.asarray(common.grid_index(times=frame["power_fcst_init_time"]))
    lags = pl.DataFrame(
        common.origin_lags(power=power, target=target, origin=origin),
        schema=list(common.LAG_COLUMNS),
        orient="row",
    ).cast(pl.Float32)
    out = frame.hstack(lags).hstack(
        pl.DataFrame(
            lag_flags(mask=event, target=target, origin=origin),
            schema=list(EVENT_LAG_COLUMNS),
            orient="row",
        )
    )
    if drop is not None:
        out = out.with_columns(
            in_drop=pl.Series(common.take(values=drop.astype(np.float32), index=target) > 0.5)
        )
    for percent, mask in (pseudo or {}).items():
        out = out.hstack(
            pl.DataFrame(
                lag_flags(mask=mask, target=target, origin=origin),
                schema=list(pseudo_lag_columns(percent=percent)),
                orient="row",
            )
        )
    return out


def for_arm(*, frame: pl.DataFrame, arm: Arm, training: bool) -> pl.DataFrame:
    """Return the frame with the arm's masked lags set to missing.

    A lag inside a labelled event is masked if the arm masks lags. In training, a lag inside a
    pseudo event of the arm's percentage is masked too.
    """
    if not arm.mask_lags:
        return frame
    flags = [pl.col(flag) for flag in EVENT_LAG_COLUMNS]
    if training and arm.pseudo_percent is not None:
        pseudo = pseudo_lag_columns(percent=arm.pseudo_percent)
        flags = [event | pl.col(flag) for event, flag in zip(flags, pseudo, strict=True)]
    return frame.with_columns(
        pl.when(flag).then(None).otherwise(pl.col(lag)).alias(lag)
        for lag, flag in zip(common.LAG_COLUMNS, flags, strict=True)
    )


def event_durations() -> np.ndarray:
    """Return the durations in half-hours of the labelled visible events."""
    events = pl.read_parquet(labelling.LABELS_PATH).filter(
        pl.col("kind").is_in(labelling.EVENT_KINDS) & pl.col("visibility").is_in(labelling.VISIBLE)
    )
    return (
        ((events["end_utc"] - events["onset_utc"]).dt.total_minutes() / 30.0 + 1.0)
        .round()
        .to_numpy()
        .astype(np.int64)
    )


def pseudo_masks(
    *, site: str, rows: pl.DataFrame
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Return the pseudo-event masks of a series, and the achieved share of rows each masks.

    Intervals start uniformly over the span of the series' training rows and last as long as a
    bootstrap draw from the labelled events' durations. The draw is seeded by the series. Going
    through the draw in order, an interval is accepted unless it would lift the share of the
    series' training rows with at least one pseudo-masked lag above the target plus half the
    tolerance, and drawing stops once the share reaches the target minus half the tolerance. The
    mask for 10% comes first, and the mask for 30% continues from it, so the masks are nested.

    Args:
        site: The series label.
        rows: The series' training rows, with `valid_time` and `power_fcst_init_time`.

    Returns:
        The masks by percentage on the dense grid, and the achieved row shares.

    Raises:
        ValueError: If the draw cannot reach a target share within the tolerance.
    """
    generator = np.random.default_rng(zlib.crc32(f"{common.SEED}:{site}".encode()))
    size = common.grid_times().len()
    target = np.asarray(common.grid_index(times=rows["valid_time"]))
    origin = np.asarray(common.grid_index(times=rows["power_fcst_init_time"]))
    lag_index = common.origin_lags(
        power=np.arange(size, dtype=np.float32), target=target, origin=origin
    )
    first, last = int(target.min()), int(target.max())
    durations = generator.choice(event_durations(), size=MAX_PSEUDO_INTERVALS)
    onsets = generator.integers(first, last + 1, size=MAX_PSEUDO_INTERVALS)
    masked_rows = np.zeros(rows.height, dtype=bool)
    mask = np.zeros(size, dtype=bool)
    position = 0
    masks: dict[int, np.ndarray] = {}
    shares: dict[int, float] = {}
    half = SHARE_TOLERANCE / 2.0
    for percent in PSEUDO_PERCENTS:
        goal = percent / 100.0
        while position < MAX_PSEUDO_INTERVALS and masked_rows.mean() < goal - half:
            start, stop = int(onsets[position]), int(onsets[position] + durations[position])
            position += 1
            inside = ((lag_index >= start) & (lag_index < stop)).any(axis=1)
            if (masked_rows | inside).mean() > goal + half:
                continue
            masked_rows |= inside
            mask[start : min(stop, size)] = True
        masks[percent] = mask.copy()
        shares[percent] = float(masked_rows.mean())
        if abs(shares[percent] - goal) > SHARE_TOLERANCE:
            msg = (
                f"{site}: pseudo share {shares[percent]:.3f} is not within tolerance of {percent}%"
            )
            raise ValueError(msg)
    return masks, shares


def mixture_quantiles(*, member_quantiles: np.ndarray) -> np.ndarray:
    """Pool members' quantiles into the mixture's 11 quantiles, as experiment 00019 does."""
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


def training_rows(
    *, row: dict, arrays: common.SeriesArrays, event: np.ndarray, drop: np.ndarray
) -> pl.DataFrame:
    """Return one series' control-member rows with derived columns, lags, and ARA flags.

    The rows run from 2021-03-21 to 2026-06-29. The power of the backfill rows and of the rows
    after 2025-06-30 is read from the dense grid. Every row's eligibility for a refit is applied
    per refit, not here.

    Args:
        row: The series' row of `common.series_table()`.
        arrays: The series' arrays on the dense grid.
        event: The series' labelled-event array on the dense grid.
        drop: The series' labelled-event or logged-closure array on the dense grid.

    Returns:
        The rows keyed by series, run, and valid time, with `ensemble_member`, `power` (in MW),
        the weather columns, the derived columns, `lead_hours`, the three lags, the three
        lag-inside-an-event flags, and `in_drop` (the valid time lies inside an event or closure).
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
    return add_lags(frame=rows, power=arrays.power, event=event, drop=drop)


def known_rows(*, rows: pl.DataFrame, refit: datetime) -> pl.DataFrame:
    """Return the rows a refit at `refit` may see in its 365-day window.

    A refit sees a row only if its valid time is at least 3.5 hours before the refit, its run
    began at or before the refit, and its valid time lies in the 365 days before the refit.
    """
    return rows.filter(
        (pl.col("valid_time") <= refit - regimes.REFIT_CUTOFF_AGE)
        & (pl.col("power_fcst_init_time") <= refit)
        & (pl.col("valid_time") > refit - regimes.TRAIN_WINDOW)
    )


def train_matrix(*, rows: pl.DataFrame, row: dict, refit: datetime, arm: Arm) -> xgb.DMatrix:
    """Return the weighted training matrix of one series' rows for an arm, anchored at `refit`.

    Rows whose valid time lies inside an event or closure are dropped if the arm drops targets.
    """
    if arm.drop_targets:
        rows = rows.filter(~pl.col("in_drop"))
    rows = for_arm(frame=rows, arm=arm, training=True)
    age_days = (refit - rows["valid_time"]).dt.total_seconds() / 86_400
    return xgb.DMatrix(
        rows.select(input_columns()).to_numpy().astype(np.float32),
        label=(rows["power"] / row["effective_capacity_mw"]).to_numpy(),
        weight=0.5 ** (age_days.to_numpy() / common.SHORT_HALF_LIFE_DAYS),
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
    event: np.ndarray,
) -> pl.DataFrame:
    """Return the mixture quantiles of one run for one series and arm.

    Args:
        model: The series' fitted booster for the arm.
        arm: The arm.
        run_rows: The run's frame rows for the series, every valid time with every member.
        row: The series' row of `common.series_table()`.
        arrays: The series' arrays on the dense grid.
        event: The series' labelled-event array on the dense grid.

    Returns:
        One row per valid time: the key columns, the 11 quantile columns, and `crossing`, the share
        of members whose quantiles cross before the mixture sorts them.

    Raises:
        ValueError: If a valid time lacks a member.
    """
    prepared = common.with_derived(
        frame=run_rows, latitude=row["latitude"], longitude=row["longitude"]
    )
    prepared = add_lags(frame=prepared, power=arrays.power, event=event)
    prepared = for_arm(frame=prepared, arm=arm, training=False).sort(
        "valid_time", "ensemble_member"
    )
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
        .with_columns(crossing=pl.Series((np.diff(raw, axis=2) < 0).any(axis=2).mean(axis=1)))
    )


def booster_path(*, arm: str, site: str, refit: datetime) -> Path:
    """Return the file one fitted booster is saved to."""
    return common.BOOSTERS_DIR / arm / f"{site}_s{common.SEED}_{refit:%Y%m%dT%H%M}.ubj"


def job_path(*, root: Path, arm: str, site: str, refit: datetime) -> Path:
    """Return the file one refit's predictions for one arm and series are written to."""
    return root / arm / f"{site}_s{common.SEED}_{refit:%Y%m%dT%H%M}.parquet"


_CACHE: dict[str, object] = {}


def context() -> tuple[
    pl.DataFrame,
    dict[str, common.SeriesArrays],
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    """Return the Primary series table, arrays, and (event, drop) arrays, loaded once per worker."""
    if "series" not in _CACHE:
        series = common.series_table().filter(pl.col("substation_type").is_in(common.GROUPS))
        _CACHE["series"] = series
        _CACHE["arrays"] = common.load_series_arrays(series=series)
        oracle = labelling.load_labels(series=series)
        _CACHE["labels"] = {
            site: (labels.in_event, labels.in_event | labels.closed)
            for site, labels in oracle.items()
        }
        _CACHE["rows"] = {}
    return _CACHE["series"], _CACHE["arrays"], _CACHE["labels"]  # ty: ignore[invalid-return-type]


def _training_rows_cached(
    *, row: dict, arrays: common.SeriesArrays, labels: tuple[np.ndarray, np.ndarray]
) -> pl.DataFrame:
    """Return a series' training rows with the pseudo-event flags, kept for the worker's jobs."""
    cache: dict[str, pl.DataFrame] = _CACHE["rows"]  # ty: ignore[invalid-assignment]
    site = row["site"]
    if site not in cache:
        event, drop = labels
        rows = training_rows(row=row, arrays=arrays, event=event, drop=drop)
        masks, shares = pseudo_masks(site=site, rows=rows)
        target = np.asarray(common.grid_index(times=rows["valid_time"]))
        origin = np.asarray(common.grid_index(times=rows["power_fcst_init_time"]))
        for percent, mask in masks.items():
            rows = rows.hstack(
                pl.DataFrame(
                    lag_flags(mask=mask, target=target, origin=origin),
                    schema=list(pseudo_lag_columns(percent=percent)),
                    orient="row",
                )
            )
        event_share = rows.select(pl.any_horizontal(EVENT_LAG_COLUMNS).mean()).item()
        print(
            f"{site}: {rows.height:,} training rows; share with a lag in a labelled event "
            f"{event_share:.3f}; pseudo-event share at 10% target {shares[10]:.3f}, at 30% target "
            f"{shares[30]:.3f}",
            flush=True,
        )
        cache[site] = rows
    return cache[site]


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
    series, arrays, labels = context()
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
    counts: dict[tuple[str, str], int] = {}
    for site, arm in todo:
        started = time.monotonic()
        row = rows_of[site]
        known = known_rows(
            rows=_training_rows_cached(row=row, arrays=arrays[site], labels=labels[site]),
            refit=refit,
        )
        matrix = train_matrix(rows=known, row=row, refit=refit, arm=ARMS[arm])
        counts[site, arm] = matrix.num_row()
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
    for site, _ in todo:
        same = {counts[site, arm] for arm in ("mask", "drop10", "drop30") if (site, arm) in counts}
        if len(same) > 1:
            msg = f"{site}: mask and dropout arms must train on the same rows, got {same}"
            raise ValueError(msg)
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
                    event=labels[site][0],
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
