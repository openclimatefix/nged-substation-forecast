"""Fit the arms of experiment 00090 and write each monthly refit's quantiles.

One job is one monthly refit (P-roll, experiment 00049). A refit at time `t` trains, for each
Primary series and each arm, an N1 model on the ENS control member's rows whose valid time lies in
the 365 days ending 3.5 hours before `t` and whose run began at or before `t`, with the rows whose
valid time lies inside a switching event or closure dropped (hindsight labels), weighted by a
recency half-life anchored at `t`, for 150 rounds at a learning rate of 0.15 and 11 quantile
levels. It then forecasts the daily runs of the 292 of `fold_a_daily_gap_v1` that the monthly
regime assigns to it, and writes for each arm and series the mixture of the 51 members' eleven
quantiles to `work/anchor_preds/<arm>/<site>_s0_<refit>.parquet`. Existing files stay, so a
stopped run resumes. Seed 0 only.

Arms:

- `plain`: the N1 inputs, lags left raw.
- `clim_pit`: the same plus the PIT of each of the three lags against the series' probabilistic
  climatology (`e00090_pit.py`), built from the same training window.

    uv run python ml-experiments/00090-*/e00090_fit.py [--workers 2] [--smoke]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Final

import e00090_common as common
import e00090_labels as labelling
import e00090_pit as pit
import e00090_regimes as regimes
import numpy as np
import polars as pl
import xgboost as xgb

CORES: Final[int] = 8
ARMS: Final[tuple[str, ...]] = ("plain", "clim_pit")
LAG_INDEX_COLUMNS: Final[tuple[str, ...]] = ("lag_index_0", "lag_index_1", "lag_index_2")
SITE_CHUNKS: Final[int] = 2
"""Each refit's series are split into this many jobs, so that files are written sooner."""
ROUNDS: Final[int] = 150
ETA: Final[float] = 0.15
HALF_LIFE_DAYS: Final[float] = common.SHORT_HALF_LIFE_DAYS
WEATHER: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
)


def input_columns(*, arm: str) -> list[str]:
    """Return an arm's input columns in matrix order."""
    base = [*common.feature_columns(), *common.N1_EXTRA_COLUMNS, *common.LAG_COLUMNS]
    return [*base, *pit.PIT_COLUMNS] if arm == "clim_pit" else base


def ara_mask(*, labels: dict[str, labelling.SeriesLabels]) -> np.ndarray:
    """Return the grid half-hours inside an oracle event, a logged closure, or an XL event."""
    return labels["oracle"].in_event | labels["oracle"].closed | labels["xl"].in_event


def add_lags(*, frame: pl.DataFrame, power: np.ndarray) -> pl.DataFrame:
    """Add the three origin lags and their grid indices (`lag_index_0` to `lag_index_2`)."""
    target = np.asarray(common.grid_index(times=frame["valid_time"]))
    origin = np.asarray(common.grid_index(times=frame["power_fcst_init_time"]))
    lags = pl.DataFrame(
        common.origin_lags(power=power, target=target, origin=origin),
        schema=list(common.LAG_COLUMNS),
        orient="row",
    ).cast(pl.Float32)
    index = pl.DataFrame(
        common.origin_lag_index(target=target, origin=origin),
        schema=list(LAG_INDEX_COLUMNS),
        orient="row",
    ).cast(pl.Int32)
    return frame.hstack(lags).hstack(index)


def with_pits(*, frame: pl.DataFrame, grid: np.ndarray) -> pl.DataFrame:
    """Add the three lag PITs from a PIT grid, using the frame's lag indices."""
    return frame.hstack(pit.lag_pits(grid=grid, index=frame.select(LAG_INDEX_COLUMNS).to_numpy()))


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


def training_rows(
    *, row: dict, arrays: common.SeriesArrays, labels: dict[str, labelling.SeriesLabels]
) -> pl.DataFrame:
    """Return one series' control-member rows with derived columns, lags, and the ARA flag.

    The rows run to 2026-06-29. The power of the rows after 2025-06-30 is filled from the dense
    grid. Every row's eligibility for a refit is applied per refit, not here.

    Args:
        row: The series' row of `common.series_table()`.
        arrays: The series' arrays on the dense grid.
        labels: The series' label arrays by source (`oracle`, `xl`).

    Returns:
        The rows keyed by series, run, and valid time, with `ensemble_member`, `power` (in MW),
        the weather columns, the derived columns, `lead_hours`, the three lags and their grid
        indices, and `ara_target` (the valid time lies inside an oracle event, a logged closure,
        or an XL event).
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
    rows = rows.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    )
    rows = add_lags(frame=rows, power=arrays.power)
    ara = ara_mask(labels=labels)[np.asarray(common.grid_index(times=rows["valid_time"]))]
    return rows.with_columns(ara_target=pl.Series(ara))


def known_rows(*, rows: pl.DataFrame, refit: datetime) -> pl.DataFrame:
    """Return the rows a refit at `refit` may see.

    A refit sees a row only if its valid time is at least 3.5 hours before the refit and within
    the 365 days before it, and its run began at or before the refit.
    """
    return rows.filter(
        (pl.col("valid_time") <= refit - regimes.REFIT_CUTOFF_AGE)
        & (pl.col("valid_time") > refit - regimes.TRAIN_WINDOW)
        & (pl.col("power_fcst_init_time") <= refit)
    )


def train_matrix(
    *, rows: pl.DataFrame, row: dict, refit: datetime, arm: str, grid: np.ndarray
) -> xgb.DMatrix:
    """Return the weighted training matrix of one series' rows for an arm, anchored at `refit`.

    Rows whose valid time lies inside an ARA are dropped for every arm.
    """
    rows = rows.filter(~pl.col("ara_target"))
    if arm == "clim_pit":
        rows = with_pits(frame=rows, grid=grid)
    age_days = (refit - rows["valid_time"]).dt.total_seconds() / 86_400
    return xgb.DMatrix(
        rows.select(input_columns(arm=arm)).to_numpy().astype(np.float32),
        label=(rows["power"] / row["effective_capacity_mw"]).to_numpy(),
        weight=0.5 ** (age_days.to_numpy() / HALF_LIFE_DAYS),
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
    arm: str,
    run_rows: pl.DataFrame,
    row: dict,
    arrays: common.SeriesArrays,
    grid: np.ndarray,
) -> pl.DataFrame:
    """Return the mixture quantiles of one run for one series and arm.

    Args:
        model: The series' fitted booster for the arm.
        arm: The arm's name.
        run_rows: The run's frame rows for the series, every valid time with every member.
        row: The series' row of `common.series_table()`.
        arrays: The series' arrays on the dense grid.
        grid: The series' PIT grid at the refit.

    Returns:
        One row per valid time: the key columns and the eleven quantile columns.

    Raises:
        ValueError: If a valid time lacks a member.
    """
    prepared = common.with_derived(
        frame=run_rows, latitude=row["latitude"], longitude=row["longitude"]
    )
    prepared = add_lags(frame=prepared, power=arrays.power)
    if arm == "clim_pit":
        prepared = with_pits(frame=prepared, grid=grid)
    prepared = prepared.sort("valid_time", "ensemble_member")
    members = prepared["ensemble_member"].n_unique()
    if prepared.height != prepared.select(*common.KEYS).n_unique() * members:
        msg = "every valid time must hold every member"
        raise ValueError(msg)
    matrix = prepared.select(input_columns(arm=arm)).to_numpy().astype(np.float32)
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
    dict[str, dict[str, labelling.SeriesLabels]],
]:
    """Return the Primary series table, arrays, and labels, loaded once per worker process."""
    if "series" not in _CACHE:
        series = common.series_table().filter(pl.col("substation_type").is_in(common.GROUPS))
        _CACHE["series"] = series
        _CACHE["arrays"] = common.load_series_arrays(series=series)
        oracle = labelling.load_labels(series=series)
        xl = labelling.load_xl_labels(series=series, oracle=oracle)
        _CACHE["labels"] = {site: {"oracle": oracle[site], "xl": xl[site]} for site in oracle}
        _CACHE["rows"] = {}
    return _CACHE["series"], _CACHE["arrays"], _CACHE["labels"]  # ty: ignore[invalid-return-type]


def _training_rows_cached(
    *, row: dict, arrays: common.SeriesArrays, labels: dict[str, labelling.SeriesLabels]
) -> pl.DataFrame:
    """Return a series' training rows, kept for the worker's later jobs."""
    cache: dict[str, pl.DataFrame] = _CACHE["rows"]  # ty: ignore[invalid-assignment]
    if row["site"] not in cache:
        cache[row["site"]] = training_rows(row=row, arrays=arrays, labels=labels)
    return cache[row["site"]]


def _keys() -> tuple[np.ndarray, np.ndarray]:
    """Return the grid's PIT cell keys, computed once per worker process."""
    if "keys" not in _CACHE:
        _CACHE["keys"] = pit.cell_keys()
    return _CACHE["keys"]  # ty: ignore[invalid-return-type]


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
    series, arrays, labels = _context()
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
    grids = {
        site: pit.pit_grid(
            power=arrays[site].power,
            ara=ara_mask(labels=labels[site]),
            refit=refit,
            keys=_keys(),
        )
        for site in {site for site, _arm in todo}
    }
    models: dict[tuple[str, str], xgb.Booster] = {}
    lines = []
    for site, arm in todo:
        started = time.monotonic()
        row = rows_of[site]
        known = known_rows(
            rows=_training_rows_cached(row=row, arrays=arrays[site], labels=labels[site]),
            refit=refit,
        )
        matrix = train_matrix(rows=known, row=row, refit=refit, arm=arm, grid=grids[site])
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
                    arm=arm,
                    run_rows=frame.filter(
                        pl.col("time_series_id") == rows_of[site]["time_series_id"]
                    ),
                    row=rows_of[site],
                    arrays=arrays[site],
                    grid=grids[site],
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
    parser.add_argument("--workers", type=int, default=2, help="processes; each gets 8 // workers")
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
