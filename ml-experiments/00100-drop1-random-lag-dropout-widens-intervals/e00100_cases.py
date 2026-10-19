"""Forecast experiment 00088's injected cases with experiment 00100's boosters, and score them.

For each batch of injected events (series disjoint within 7 days) and each shape, the script
injects the transfers into the dense half-hourly power, rebuilds the lags from the injected power,
and forecasts every (event, series, origin) case of family A (the newest usable lag is inside the
transfer) and family B (the transfer ended before the newest usable lag, within the lag window).
The variants are `plain` (the `plain11` booster on the contaminated lags), `ceiling` (the same
booster on the unmodified lags), and the `mask`, `drop10`, and `drop30` boosters and `plain11` with
every lag inside a mask set to missing. The mask is the injected interval plus the labelled real
events (`inj`, oracle masking) or the causal RO-NC detector's intervals plus the labelled real
events (`caus`).

Each variant's mixture quantiles are scored against the unmodified power over lead bands, for
valid times inside the injected transfer (`inside`) and for all valid times (`all`). One parquet
file per batch and shape holds the cells. The causal detector runs once per (shape, batch, origin)
on residuals up to the cut, the origin minus 3.5 hours.

    uv run python ml-experiments/00100-*/e00100_cases.py [--pilot] [--workers 4]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import e00100_causal as causal
import e00100_common as common
import e00100_fit as fitting
import e00100_grid as grid
import e00100_harness as harness
import e00100_regimes as regimes
import numpy as np
import polars as pl
import xgboost as xgb
from e00100_inject import Event, Injection
from e00100_profile import Shape, inject_shaped

OFFSET: Final[int] = harness.OFFSET
"""Half-hours from the dense grid's start (2019-09-01) to experiment 00082's grid (2022-01-01)."""
SCORED_SHAPES: Final[tuple[Shape, ...]] = ("S1", "S2", "S3", "S4", "S6")
MAX_LEAD_HOURS: Final[float] = 240.0
MEMBERS: Final[tuple[int, ...]] = tuple(range(0, 51, 5))
"""The control member and every fifth member: 11 of the 51, the same in every arm."""
FAMILY_A_DAYS: Final[int] = 7
LAG_WINDOW: Final[int] = 7 * grid.BLOCK
"""A lag is at most 7 days before the newest usable value."""
BANDS: Final[tuple[tuple[str, float, float], ...]] = common.LEAD_BANDS[:4]
OUT_DIR: Final[Path] = common.CASES_DIR
VARIANTS: Final[tuple[tuple[str, str, str | None], ...]] = (
    ("plain", "plain11", None),
    ("ceiling", "plain11", None),
    ("plain_masked_inj", "plain11", "inj"),
    ("mask_inj", "mask", "inj"),
    ("drop10_inj", "drop10", "inj"),
    ("drop30_inj", "drop30", "inj"),
    ("mask_caus", "mask", "caus"),
    ("drop10_caus", "drop10", "caus"),
    ("drop30_caus", "drop30", "caus"),
)
"""Each variant's name, its booster arm, and its mask source (None: no mask)."""
_STATE: dict[str, Any] = {}


def newest_stamp(*, origin: datetime) -> int:
    """Return the experiment 00082 stamp of the newest usable observation for an origin."""
    return grid.stamp_of(when=origin) - common.NEWEST_AGE


def classify(*, event: Event, newest: int) -> tuple[str, float] | None:
    """Return a case's family and its hours since onset (A) or since end (B), or None.

    Args:
        event: The injected event; `end` is the first stamp after it.
        newest: The stamp of the newest usable observation.

    Returns:
        `("A", hours since onset)` when the newest usable value is inside the event and at most
        7 days after the onset; `("B", hours since end)` when the event ended at or before the
        newest value and within the 7-day lag window; otherwise None.
    """
    if event.onset <= newest < event.end and newest - event.onset <= FAMILY_A_DAYS * grid.BLOCK:
        return "A", (newest - event.onset) / 2
    if event.end <= newest and newest - event.end < LAG_WINDOW:
        return "B", (newest - event.end) / 2
    return None


def injected_arrays(
    *, site_row: dict, arrays: common.SeriesArrays, injection: Injection, series_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a series' injected power and the injected signed MW on the dense grid.

    Args:
        site_row: The series' row of `common.series_table()`.
        arrays: The series' clean arrays.
        injection: The batch's injection.
        series_id: The series' time series id.

    Returns:
        The power over capacity with the injected MW added, and the oracle-style signed MW over
        capacity: each event's median injected MW, constant over the event, summed over events.
    """
    capacity = site_row["effective_capacity_mw"]
    power = arrays.power.copy()
    delta = injection.delta[series_id] / capacity
    power[OFFSET : OFFSET + delta.size] += delta.astype(np.float32)
    signed = np.zeros_like(power)
    for event in injection.events:
        if series_id in event.series:
            median = float(np.median(injection.delta[series_id][event.onset : event.end]))
            signed[OFFSET + event.onset : OFFSET + event.end] += np.float32(median / capacity)
    return power, signed


def causal_array(
    *,
    intervals: list[causal.CausalInterval],
    series_id: int,
    capacity: float,
    size: int,
    merge: str = "sum",
) -> np.ndarray:
    """Return the signed MW over capacity that the causal intervals of one series mark.

    Args:
        intervals: The causal detector's intervals.
        series_id: The series to mark.
        capacity: The series' effective capacity in MW.
        size: The length of the dense grid.
        merge: `sum` adds overlapping intervals, as the XL training labels do; `max` takes the
            step of largest size at each half-hour (a post hoc variant).

    Returns:
        The signed MW over capacity on the dense grid.
    """
    out = np.zeros(size, dtype=np.float32)
    mine = [i for i in intervals if i.series == series_id]
    if merge == "max":
        mine.sort(key=lambda i: abs(i.step_mw))
    for interval in mine:
        value = np.float32(interval.step_mw / capacity)
        if merge == "max":
            out[OFFSET + interval.onset : OFFSET + interval.end] = value
        else:
            out[OFFSET + interval.onset : OFFSET + interval.end] += value
    return out


def booster(*, arm: str, site: str, refit: datetime) -> xgb.Booster:
    """Return a fitted booster, loaded once per process."""
    cache: dict = _STATE.setdefault("boosters", {})
    path = fitting.booster_path(arm=arm, site=site, refit=refit)
    if path not in cache:
        model = xgb.Booster()
        model.load_model(path)
        expected = len(fitting.input_columns())
        if model.num_features() != expected:
            msg = f"{path} has {model.num_features()} features, expected {expected}"
            raise ValueError(msg)
        cache[path] = model
    return cache[path]


def run_frame(*, run: datetime, series_id: int, row: dict) -> pl.DataFrame:
    """Return a run's member rows for one series with derived columns, kept for later shapes."""
    cache: dict = _STATE.setdefault("frames", {})
    key = (run, series_id)
    if key not in cache:
        if len(cache) > 400:
            cache.clear()
        frame = (
            pl.read_parquet(common.demand_frame_path(run=run))
            .filter(
                (pl.col("time_series_id") == series_id)
                & pl.col("ensemble_member").is_in(MEMBERS)
                & (pl.col("lead_hours") <= MAX_LEAD_HOURS)
            )
            .sort("valid_time", "ensemble_member")
        )
        cache[key] = common.with_derived(
            frame=frame, latitude=row["latitude"], longitude=row["longitude"]
        )
    return cache[key]


def predict(
    *,
    model: xgb.Booster,
    arm: str,
    base: pl.DataFrame,
    power: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Return the mixture's 11 quantiles per valid time, with the spread factor applied.

    Args:
        model: The booster.
        arm: The booster's arm; its `mask_lags` decides whether lags inside the mask are dropped.
        base: The run's member rows for the series.
        power: The power over capacity the lags are read from.
        mask: The half-hours whose lags are set to missing if the arm masks lags.

    Returns:
        An array of shape (valid times, 11).
    """
    frame = fitting.add_lags(frame=base, power=power, event=mask)
    frame = fitting.for_arm(frame=frame, arm=fitting.ARMS[arm], training=False)
    matrix = frame.select(fitting.input_columns()).to_numpy().astype(np.float32)
    members = len(MEMBERS)
    if frame.height != frame["valid_time"].n_unique() * members:
        msg = "every valid time must hold every member"
        raise ValueError(msg)
    raw = model.predict(xgb.DMatrix(matrix)).reshape(-1, members, len(common.LEVELS))
    pooled = fitting.mixture_quantiles(member_quantiles=raw).astype(np.float64)
    median = pooled[:, [common.MEDIAN]]
    return np.sort(median + common.SPREAD_FACTOR * (pooled - median), axis=1)


def score_cell(
    *,
    quantiles: np.ndarray,
    actual: np.ndarray,
    lead: np.ndarray,
    keep: np.ndarray,
    inside: np.ndarray,
    p99: float,
) -> list[dict]:
    """Return the mean metrics of one forecast in each lead band, for all and inside valid times.

    Args:
        quantiles: Shape (valid times, 11).
        actual: The observed power over capacity per valid time.
        lead: The lead time in hours per valid time.
        keep: The valid times that may be scored.
        inside: The valid times that lie inside an injected transfer of the series.
        p99: The series' `historical_p99`.

    Returns:
        One row per lead band and target kind (`all` valid times, or only those `inside`), with
        the mean case metrics and the mean widths `w90` and `w98` in points of capacity.
    """
    metrics = common.case_metrics(
        quantiles=quantiles[keep],
        actual=actual[keep],
        threshold_p99=np.full(int(keep.sum()), p99),
    )
    metrics["w90"] = (quantiles[keep][:, 9] - quantiles[keep][:, 1]) * 100.0
    metrics["w98"] = (quantiles[keep][:, 10] - quantiles[keep][:, 0]) * 100.0
    rows = []
    for kind, selected in (("all", np.ones(int(keep.sum()), dtype=bool)), ("inside", inside[keep])):
        for name, low, high in BANDS:
            in_band = (lead[keep] >= low) & (lead[keep] < high) & selected
            if not in_band.any():
                continue
            rows.append(
                {
                    "band": name,
                    "target": kind,
                    "n": int(in_band.sum()),
                    **{key: float(values[in_band].mean()) for key, values in metrics.items()},
                }
            )
    return rows


def forecast_cell(
    *,
    site: str,
    origin: datetime,
    power: np.ndarray,
    masks: dict[str, np.ndarray],
    inside_array: np.ndarray,
    variants: tuple[tuple[str, str, str | None], ...],
) -> list[dict]:
    """Return every variant's per-band scores for one series and origin.

    Args:
        site: The series label.
        origin: The forecast origin (`power_fcst_init_time`).
        power: The power over capacity the contaminated variants read their lags from.
        masks: The lag masks by source (`inj`, `caus`).
        inside_array: The half-hours inside an injected transfer of the series.
        variants: The variants to forecast.

    Returns:
        One row per variant, lead band, and target kind.
    """
    series, arrays, labels = fitting.context()
    row = next(r for r in series.iter_rows(named=True) if r["site"] == site)
    run = origin - common.PUBLICATION_DELAY
    refit = regimes.chosen_refit(refits=regimes.MONTHLY, origin=origin)
    base = run_frame(run=run, series_id=int(row["time_series_id"]), row=row)
    valid = base.filter(pl.col("ensemble_member") == MEMBERS[0])["valid_time"]
    lead = base.filter(pl.col("ensemble_member") == MEMBERS[0])["lead_hours"].to_numpy()
    index = np.asarray(common.grid_index(times=valid))
    actual = common.take(values=arrays[site].power, index=index).astype(np.float64)
    blocked = labels[site][1]
    keep = np.isfinite(actual) & ~blocked[index]
    inside = common.take(values=inside_array.astype(np.float32), index=index) > 0.5
    p99 = common.historical_thresholds(power=arrays[site].power)["p99"]
    rows = []
    for name, arm, source in variants:
        clean = name == "ceiling"
        quantiles = predict(
            model=booster(arm=arm, site=site, refit=refit),
            arm=arm,
            base=base,
            power=arrays[site].power if clean else power,
            mask=labels[site][0] if source is None else masks[source],
        )
        bands = score_cell(
            quantiles=quantiles, actual=actual, lead=lead, keep=keep, inside=inside, p99=p99
        )
        rows.extend({"variant": name, **band} for band in bands)
    return rows


def run_batch(job: tuple[int, list[Event], Shape]) -> str:
    """Inject one batch with one shape, forecast its cases, and write the cells.

    Args:
        job: The batch's seed, its events, and the shape.

    Returns:
        A progress line.
    """
    seed, batch, shape = job
    out_dir: Path = OUT_DIR
    variants: tuple = _STATE.get("variants", VARIANTS)
    target = out_dir / f"cells_{shape}_{seed}.parquet"
    if target.exists():
        return f"{target.name} exists"
    started = time.monotonic()
    state = _STATE["harness"]
    series, arrays, labels = fitting.context()
    rows_of = {int(r["time_series_id"]): r for r in series.iter_rows(named=True)}
    panel = state["panel"]
    injection = inject_shaped(
        power=panel["power"], events=batch, seed=seed, shape=shape, shares=state["shares"]
    )
    residual = injection.fast_residual(panel["residual"])
    offset = injection.fast_residual(state["offset_panel"]["residual"])
    powers = {
        sid: injected_arrays(
            site_row=rows_of[sid],
            arrays=arrays[rows_of[sid]["site"]],
            injection=injection,
            series_id=sid,
        )
        for sid in rows_of
    }
    cells: list[dict] = []
    origins = [
        o
        for o in common.origins_of_subset(subset="gap")
        if _STATE.get("origin_ok", lambda _: True)(o)
    ]
    causal_cache: dict[int, list[causal.CausalInterval]] = {}
    for origin in origins:
        newest = newest_stamp(origin=origin)
        todo = []
        for event in batch:
            kind = classify(event=event, newest=newest)
            if kind is None:
                continue
            todo.extend((event, sid, kind) for sid in event.series if sid in rows_of)
        if not todo:
            continue
        cut = newest + 1
        if cut not in causal_cache:
            causal_cache[cut] = causal.causal_intervals(
                state=state,
                residual=residual,
                offset=offset,
                cut=cut,
            )
        intervals = causal_cache[cut]
        for event, sid, (family, hours) in todo:
            row = rows_of[sid]
            power, injected_mw = powers[sid]
            size = power.size
            real = labels[row["site"]][0]
            caus = causal_array(
                intervals=intervals,
                series_id=sid,
                capacity=row["effective_capacity_mw"],
                size=size,
            )
            masks = {"inj": (injected_mw != 0.0) | real, "caus": (caus != 0.0) | real}
            lag_index = np.array([common.grid_index(times=origin) - common.NEWEST_AGE])
            newest_inj = float(injected_mw[lag_index][0])
            newest_caus = float(caus[lag_index][0])
            cells.extend(
                {
                    "shape": shape,
                    "seed": seed,
                    "event_id": event.event_id,
                    "series": sid,
                    "site": row["site"],
                    "origin": origin,
                    "family": family,
                    "hours": hours,
                    "inj_mw_newest": newest_inj,
                    "caus_mw_newest": newest_caus,
                    **cell,
                }
                for cell in forecast_cell(
                    site=row["site"],
                    origin=origin,
                    power=power,
                    masks=masks,
                    inside_array=injected_mw != 0.0,
                    variants=variants,
                )
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    common.atomic_write(
        frame=event_table(batch=batch, injection=injection, shape=shape, seed=seed, state=state),
        target=out_dir / f"events_{shape}_{seed}.parquet",
    )
    common.atomic_write(frame=pl.DataFrame(cells, infer_schema_length=None), target=target)
    return f"{target.name}: {len(cells):,} cells in {time.monotonic() - started:.0f} s"


def event_table(
    *,
    batch: list[Event],
    injection: Injection,
    shape: str,
    seed: int,
    state: dict,
) -> pl.DataFrame:
    """Return one row per injected event: its size in MW, its donor step in noise units, its month.

    Args:
        batch: The batch's events.
        injection: The batch's injection under one shape.
        shape: The shape's name.
        seed: The batch's seed.
        state: `harness.make_state()`.

    Returns:
        The events with `moved_mw` (the mean MW taken from the donor), `donor_step_z` (the donor's
        mean MW in the first 24 hours over its noise scale), and `onset_month` (UTC year and month).
    """
    scale = state["scales"]["X"]
    rows = []
    for event in batch:
        donor = injection.delta[event.donor]
        first_day = slice(event.onset, min(event.end, event.onset + grid.BLOCK))
        rows.append(
            {
                "shape": shape,
                "seed": seed,
                "event_id": event.event_id,
                "donor": event.donor,
                "n_receivers": len(event.receivers),
                "hours": (event.end - event.onset) / 2,
                "size_band": event.size_band,
                "duration_band": event.duration_band,
                "moved_mw": float(-np.mean(donor[event.onset : event.end])),
                "donor_step_z": float(-np.mean(donor[first_day]) / scale[event.donor]),
                "onset_month": grid.stamp_year_months()[event.onset],
            }
        )
    return pl.DataFrame(rows)


def check_alignment(*, state: dict) -> None:
    """Fail unless the panel's power over capacity equals the dense grid wherever both exist."""
    series, arrays, _ = fitting.context()
    for row in series.iter_rows(named=True):
        panel = state["panel"]["power"][int(row["time_series_id"])]
        dense = arrays[row["site"]].power[OFFSET : OFFSET + panel.size]
        both = np.isfinite(panel) & np.isfinite(dense)
        gap = np.abs(panel[both] / row["effective_capacity_mw"] - dense[both]).max()
        if gap > 1e-5:
            msg = f"series {row['site']}: panel and dense power differ by {gap}"
            raise ValueError(msg)


def init_worker() -> None:
    """Load the dense-grid context and the harness state once per worker."""
    fitting.context()
    _STATE["harness"] = harness.make_state()
    check_alignment(state=_STATE["harness"])


def main() -> None:
    """Forecast every batch and shape."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pilot", action="store_true", help="one small batch, one shape")
    args = parser.parse_args()
    state = harness.make_state()
    events = harness.cached_events(power=state["panel"]["power"])
    batches = harness.seeded_batches(events=events)
    jobs = [(seed, batch, shape) for shape in SCORED_SHAPES for seed, batch in batches]
    if args.pilot:
        init_worker()
        _STATE["origin_ok"] = lambda o: o.month == 8  # pilot: the refit whose boosters exist

        def aug_cases(item: tuple[int, list[Event]]) -> int:
            return sum(
                classify(event=e, newest=newest_stamp(origin=o)) is not None
                for e in item[1]
                for o in common.origins_of_subset(subset="gap")
                if o.month == 8
            )

        seed, batch = max(batches, key=aug_cases)
        print(run_batch((seed, batch, "S2")))
        return
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        for done, line in enumerate(pool.map(run_batch, jobs), start=1):
            print(f"[{done}/{len(jobs)}] {line}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
