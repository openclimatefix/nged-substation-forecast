"""Fit the two booster arms of experiment 00096 and hold the shared feature helpers.

One job is one monthly refit (P-roll, experiment 00049) of one arm for a chunk of Primary series. A
refit at time `t` trains an N1 model on the ENS control member's rows whose valid time lies in the
365 days ending 3.5 hours before `t` and whose run began at or before `t`, weighted by a recency
half-life anchored at `t`, for 150 rounds at a learning rate of 0.15 (experiment 00086's recipe),
with 11 quantile levels. The booster is saved to `work/boosters/<arm>/<site>_s0_<refit>.ubj`.
Existing files stay, so a stopped run resumes. Seed 0 only.

Arms (training labels are hindsight labels, which is allowed). Both drop the rows whose valid time
lies inside an event as targets, replace each lag by the lag minus the signed MW at the lag time,
and add the three signed MW values as columns (experiment 00088's `nra_oracle` recipe):

- `nra_real`: the events are the real oracle (experiment 41) events only.
- `nra_synth`: the events are the real oracle events and the synthetic transfers of
  `e00096_synth`. The lags are read from the recorded power plus the injected power, and the signed
  MW of a synthetic transfer is its median injected MW, constant over the transfer.

    uv run python ml-experiments/00096-*/e00096_fit.py [--workers 4] [--smoke]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Final

import e00096_common as common
import e00096_labels as labelling
import e00096_regimes as regimes
import e00096_synth as synth
import numpy as np
import polars as pl
import xgboost as xgb

CORES: Final[int] = 8
NEW_ARMS: Final[tuple[str, ...]] = ("nra_real", "nra_synth")
MW_COLUMNS: Final[tuple[str, ...]] = (
    "mw_origin",
    "mw_same_time_latest_day",
    "mw_same_time_latest_week",
)
SOURCE: Final[str] = "arm"
"""The signed-MW frame columns of an arm are `arm_<MW column>`."""
SITE_CHUNKS: Final[int] = 2
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


def input_columns(*, with_mw: bool) -> list[str]:
    """Return a booster's input columns in matrix order: the plain set, plus the MW columns."""
    base = [*common.feature_columns(), *common.N1_EXTRA_COLUMNS, *common.LAG_COLUMNS]
    return [*base, *MW_COLUMNS] if with_mw else base


def add_lags(
    *, frame: pl.DataFrame, power: np.ndarray, signed_mw: dict[str, np.ndarray]
) -> pl.DataFrame:
    """Add the three origin lags and, for each source, the signed MW at the same lag times.

    Args:
        frame: Rows with `valid_time` and `power_fcst_init_time`.
        power: Power over effective capacity on the dense grid (the injected power for a scenario).
        signed_mw: Each source's signed MW over effective capacity on the dense grid, 0.0 where no
            event is known.

    Returns:
        The frame with `lag_*` and `<source>_mw_*` columns.
    """
    target = np.asarray(common.grid_index(times=frame["valid_time"]))
    origin = np.asarray(common.grid_index(times=frame["power_fcst_init_time"]))
    lags = pl.DataFrame(
        common.origin_lags(power=power, target=target, origin=origin),
        schema=list(common.LAG_COLUMNS),
        orient="row",
    ).cast(pl.Float32)
    out = frame.hstack(lags)
    for source, array in signed_mw.items():
        moved = pl.DataFrame(
            common.origin_lags(power=array, target=target, origin=origin),
            schema=[f"{source}_{name}" for name in MW_COLUMNS],
            orient="row",
        ).cast(pl.Float32)
        out = out.hstack(moved)
    return out


def corrected(*, frame: pl.DataFrame, source: str) -> pl.DataFrame:
    """Return the frame with each lag minus its signed MW, and the MW under the `MW_COLUMNS` names.

    Args:
        frame: A frame from `add_lags` that holds the source's `<source>_mw_*` columns.
        source: The source whose MW is used.

    Returns:
        The frame with corrected lags and the MW columns.
    """
    moved = [f"{source}_{name}" for name in MW_COLUMNS]
    return frame.with_columns(
        *(
            (pl.col(lag) - pl.col(mw)).alias(lag)
            for lag, mw in zip(common.LAG_COLUMNS, moved, strict=True)
        ),
        *(pl.col(mw).alias(name) for mw, name in zip(moved, MW_COLUMNS, strict=True)),
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


def training_rows(
    *,
    row: dict,
    arrays: common.SeriesArrays,
    labels: labelling.SeriesLabels,
    synthetic: synth.SyntheticArrays | None,
) -> pl.DataFrame:
    """Return one series' control-member rows with derived columns, lags, signed MW, event flags.

    The rows run to 2026-06-29. The power of the rows after 2025-06-30 is filled from the dense
    grid. Every row's eligibility for a refit is applied per refit, not here.

    Args:
        row: The series' row of `common.series_table()`.
        arrays: The series' arrays on the dense grid.
        labels: The series' real oracle labels.
        synthetic: The series' synthetic transfers, or None for the real-only arm.

    Returns:
        The rows keyed by series, run, and valid time, with `ensemble_member`, `power` (in MW, the
        recorded power), the weather columns, the derived columns, `lead_hours`, the three lags
        (read from the recorded plus injected power), the three signed MW columns `arm_mw_*`, and
        `in_event` (the valid time lies inside a real or a synthetic transfer).
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
    power, signed, in_event = arrays.power, labels.signed_mw, labels.in_event
    if synthetic is not None:
        power = power + synthetic.delta
        signed = signed + synthetic.signed_mw
        in_event = in_event | synthetic.in_event
    rows = add_lags(frame=rows, power=power, signed_mw={SOURCE: signed})
    valid_index = np.asarray(common.grid_index(times=rows["valid_time"]))
    return rows.with_columns(pl.Series("in_event", in_event[valid_index]))


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


def train_matrix(*, rows: pl.DataFrame, row: dict, refit: datetime) -> xgb.DMatrix:
    """Return the weighted training matrix of one series' rows at `refit`."""
    rows = corrected(frame=rows.filter(~pl.col("in_event")), source=SOURCE)
    age_days = (refit - rows["valid_time"]).dt.total_seconds() / 86_400
    return xgb.DMatrix(
        rows.select(input_columns(with_mw=True)).to_numpy().astype(np.float32),
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


def booster_path(*, arm: str, site: str, refit: datetime, root: Path | None = None) -> Path:
    """Return the file one fitted booster is saved to."""
    if root is None:
        root = common.BOOSTERS_DIR
    return root / arm / f"{site}_s{common.SEED}_{refit:%Y%m%dT%H%M}.ubj"


_CACHE: dict[str, object] = {}


def context() -> tuple[
    pl.DataFrame,
    dict[str, common.SeriesArrays],
    dict[str, dict[str, labelling.SeriesLabels]],
]:
    """Return the Primary series table, arrays, and labels, loaded once per process."""
    if "series" not in _CACHE:
        series = common.series_table().filter(pl.col("substation_type").is_in(common.GROUPS))
        _CACHE["series"] = series
        _CACHE["arrays"] = common.load_series_arrays(series=series)
        oracle = labelling.load_labels(series=series)
        _CACHE["labels"] = {site: {"oracle": oracle[site]} for site in oracle}
        _CACHE["rows"] = {}
    return _CACHE["series"], _CACHE["arrays"], _CACHE["labels"]  # ty: ignore[invalid-return-type]


def _training_rows_cached(
    *,
    row: dict,
    arrays: common.SeriesArrays,
    labels: dict[str, labelling.SeriesLabels],
    arm: str,
) -> pl.DataFrame:
    """Return an arm's training rows for a series, kept for the worker's later jobs."""
    cache: dict[tuple[str, str], pl.DataFrame] = _CACHE["rows"]  # ty: ignore[invalid-assignment]
    key = (row["site"], arm)
    if key not in cache:
        if arm == "nra_synth" and "synthetic" not in _CACHE:
            _CACHE["synthetic"] = synth.load_or_build()[1]
        synthetic = _CACHE["synthetic"][row["site"]] if arm == "nra_synth" else None  # ty: ignore[not-subscriptable]
        cache[key] = training_rows(
            row=row, arrays=arrays, labels=labels["oracle"], synthetic=synthetic
        )
    return cache[key]


def run_job(
    *, refit: datetime, sites: list[str], root: Path, rounds: int, nthread: int
) -> list[str]:
    """Fit one refit of each new arm for each series that lacks its booster; save the boosters.

    Args:
        refit: The refit time.
        sites: The series labels to fit.
        root: The folder holding one subfolder per arm.
        rounds: Boosting rounds.
        nthread: Threads for each fit.

    Returns:
        One progress line per series and arm fitted.
    """
    series, arrays, labels = context()
    rows_of = {row["site"]: row for row in series.iter_rows(named=True)}
    lines = []
    for site in sites:
        for arm in NEW_ARMS:
            target = booster_path(arm=arm, site=site, refit=refit, root=root)
            if target.exists():
                continue
            started = time.monotonic()
            row = rows_of[site]
            known = known_rows(
                rows=_training_rows_cached(
                    row=row, arrays=arrays[site], labels=labels[site], arm=arm
                ),
                refit=refit,
            )
            matrix = train_matrix(rows=known, row=row, refit=refit)
            model = fit(matrix=matrix, rounds=rounds, nthread=nthread)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f"tmp_{target.name}")
            model.save_model(temporary)
            temporary.rename(target)
            lines.append(
                f"{arm} {site} refit {refit:%Y-%m-%d}: {matrix.num_row():,} of {known.height:,} "
                f"rows, fit {time.monotonic() - started:.0f} s"
            )
    return lines


def main() -> None:
    """Fit every new arm's boosters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, help="processes; each gets 8 // workers")
    parser.add_argument("--smoke", action="store_true", help="2 series, 2 refits, 20 rounds")
    args = parser.parse_args()
    series = common.series_table().filter(pl.col("substation_type").is_in(common.GROUPS))
    sites = series["site"].to_list()
    refits = list(regimes.MONTHLY)
    root = common.BOOSTERS_DIR
    rounds = ROUNDS
    workers = args.workers
    if args.smoke:
        sites, refits = sites[:2], refits[:2]
        root = common.BOOSTERS_DIR.with_name("boosters_smoke")
        rounds = 20
        workers = 1
    synth.load_or_build()
    nthread = max(1, CORES // workers)
    started = time.monotonic()
    chunks = [sites[i::SITE_CHUNKS] for i in range(min(SITE_CHUNKS, len(sites)))]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                run_job, refit=refit, sites=chunk, root=root, rounds=rounds, nthread=nthread
            )
            for refit in refits
            for chunk in chunks
        ]
        for done, future in enumerate(futures, start=1):
            for line in future.result():
                print(f"[{done}/{len(futures)}] {line}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
