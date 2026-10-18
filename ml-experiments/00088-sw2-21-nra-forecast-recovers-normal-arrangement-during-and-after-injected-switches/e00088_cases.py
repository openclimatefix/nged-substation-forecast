"""Forecast the injected cases of experiment 00088 with every arm, and write per-case scores.

For each batch of injected events (series disjoint within 7 days) and each shape, the script
injects the transfers into the dense half-hourly power, rebuilds the lags from the injected power,
and forecasts every (event, series, origin) case of family A (the newest usable lag is inside the
transfer) and family B (the transfer ended before the newest usable lag, within the lag window)
with eight variants:

- `plain`: arm 1, the experiment 00086 `plain` booster on the contaminated lags;
- `ceiling`: arm 5, the same booster on the unmodified lags;
- `lagmw_xl_<mw>`, `nra_oracle_<mw>`, `nra_xl_<mw>`: arms 2, 3, and 4, each with the forecast-time
  MW from the injection (`inj`) or from the causal XL detector (`caus`).

Each variant's mixture quantiles are scored against the unmodified power over lead bands. One
parquet file per batch and shape holds the cells. The causal detector runs once per (shape, batch,
origin) on residuals up to the cut, the origin minus 3.5 hours.

    uv run python ml-experiments/00088-*/e00088_cases.py [--pilot] [--workers 4]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Final

import e00088_causal as causal
import e00088_common as common
import e00088_fit as fitting
import e00088_grid as grid
import e00088_harness as harness
import e00088_regimes as regimes
import numpy as np
import polars as pl
import xgboost as xgb
from e00088_inject import Event, Injection
from e00088_profile import inject_shaped

OFFSET: Final[int] = harness.OFFSET
"""Half-hours from the dense grid's start (2019-09-01) to experiment 00082's grid (2022-01-01)."""
SCORED_SHAPES: Final[tuple[str, ...]] = ("S1", "S2", "S3", "S4", "S6")
MAX_LEAD_HOURS: Final[float] = 240.0
MEMBERS: Final[tuple[int, ...]] = tuple(range(0, 51, 5))
"""The control member and every fifth member: 11 of the 51, the same in every arm."""
FAMILY_A_DAYS: Final[int] = 7
LAG_WINDOW: Final[int] = 7 * grid.BLOCK
"""A lag is at most 7 days before the newest usable value."""
BANDS: Final[tuple[tuple[str, float, float], ...]] = common.LEAD_BANDS[:4]
OUT_DIR: Final[Path] = common.CASES_DIR
VARIANTS: Final[tuple[tuple[str, str, str | None], ...]] = (
    ("plain", "plain", None),
    ("ceiling", "plain", None),
    ("lagmw_xl_inj", "lagmw_xl", "inj"),
    ("lagmw_xl_caus", "lagmw_xl", "caus"),
    ("nra_oracle_inj", "nra_oracle", "inj"),
    ("nra_oracle_caus", "nra_oracle", "caus"),
    ("nra_xl_inj", "nra_xl", "inj"),
    ("nra_xl_caus", "nra_xl", "caus"),
)
"""Each variant's name, its booster arm, and its forecast-time MW source (None: no correction)."""
_STATE: dict[str, object] = {}


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
    *, intervals: list[causal.CausalInterval], series_id: int, capacity: float, size: int
) -> np.ndarray:
    """Return the signed MW over capacity that the causal intervals of one series mark."""
    out = np.zeros(size, dtype=np.float32)
    for interval in intervals:
        if interval.series == series_id:
            out[OFFSET + interval.onset : OFFSET + interval.end] += np.float32(
                interval.step_mw / capacity
            )
    return out


def booster(*, arm: str, site: str, refit: datetime) -> xgb.Booster:
    """Return a fitted booster, loaded once per process."""
    cache: dict = _STATE.setdefault("boosters", {})  # ty: ignore[invalid-assignment]
    path = fitting.booster_path(arm=arm, site=site, refit=refit)
    if path not in cache:
        model = xgb.Booster()
        model.load_model(path)
        cache[path] = model
    return cache[path]


def run_frame(*, run: datetime, series_id: int, row: dict) -> pl.DataFrame:
    """Return a run's member rows for one series with derived columns, kept for later shapes."""
    cache: dict = _STATE.setdefault("frames", {})  # ty: ignore[invalid-assignment]
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
    base: pl.DataFrame,
    power: np.ndarray,
    signed_mw: np.ndarray | None,
    with_mw: bool,
    source: str,
) -> np.ndarray:
    """Return the mixture's nine quantiles per valid time, with the spread factor applied.

    Args:
        model: The booster.
        base: The run's member rows for the series.
        power: The power over capacity the lags are read from.
        signed_mw: The signed MW over capacity the lags are corrected by, or None.
        with_mw: Whether the booster takes the MW columns.
        source: The name given to the MW source.

    Returns:
        An array of shape (valid times, 9).
    """
    frame = fitting.add_lags(
        frame=base,
        power=power,
        signed_mw={source: signed_mw if signed_mw is not None else np.zeros_like(power)},
    )
    if with_mw:
        frame = fitting.corrected(frame=frame, source=source)
    matrix = frame.select(fitting.input_columns(with_mw=with_mw)).to_numpy().astype(np.float32)
    members = len(MEMBERS)
    raw = model.predict(xgb.DMatrix(matrix)).reshape(-1, members, len(common.LEVELS))
    pooled = fitting.mixture_quantiles(member_quantiles=raw).astype(np.float64)
    median = pooled[:, [common.MEDIAN]]
    return np.sort(median + common.SPREAD_FACTOR * (pooled - median), axis=1)


def score_cell(
    *, quantiles: np.ndarray, actual: np.ndarray, lead: np.ndarray, keep: np.ndarray, p99: float
) -> list[dict]:
    """Return the mean metrics of one forecast in each lead band, on the kept valid times."""
    metrics = common.case_metrics(
        quantiles=quantiles[keep],
        actual=actual[keep],
        threshold_p99=np.full(int(keep.sum()), p99),
    )
    rows = []
    for name, low, high in BANDS:
        in_band = (lead[keep] >= low) & (lead[keep] < high)
        if not in_band.any():
            continue
        rows.append(
            {
                "band": name,
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
    signed: dict[str, np.ndarray],
    variants: tuple[tuple[str, str, str | None], ...],
    truth: np.ndarray | None = None,
) -> list[dict]:
    """Return every variant's per-band scores for one series and origin.

    Args:
        site: The series label.
        origin: The forecast origin (`power_fcst_init_time`).
        power: The power over capacity the contaminated variants read their lags from.
        signed: The signed MW over capacity by source (`inj`, `caus`).
        variants: The variants to forecast.
        truth: The power over capacity the forecast is scored against, or None for the clean
            observed power.

    Returns:
        One row per variant and lead band.
    """
    series, arrays, labels = fitting.context()
    row = next(r for r in series.iter_rows(named=True) if r["site"] == site)
    run = origin - common.PUBLICATION_DELAY
    refit = regimes.chosen_refit(refits=regimes.MONTHLY, origin=origin)
    base = run_frame(run=run, series_id=int(row["time_series_id"]), row=row)
    valid = base.filter(pl.col("ensemble_member") == MEMBERS[0])["valid_time"]
    lead = base.filter(pl.col("ensemble_member") == MEMBERS[0])["lead_hours"].to_numpy()
    index = np.asarray(common.grid_index(times=valid))
    scored_against = arrays[site].power if truth is None else truth
    actual = common.take(values=scored_against, index=index).astype(np.float64)
    blocked = labels[site]["oracle"].in_event | labels[site]["oracle"].closed
    keep = np.isfinite(actual) & ~blocked[index]
    p99 = common.historical_thresholds(power=arrays[site].power)["p99"]
    rows = []
    for name, arm, mw in variants:
        clean = name == "ceiling"
        quantiles = predict(
            model=booster(arm=arm, site=site, refit=refit),
            base=base,
            power=arrays[site].power if clean else power,
            signed_mw=None if mw is None else signed[mw],
            with_mw=mw is not None,
            source=mw or "none",
        )
        bands = score_cell(quantiles=quantiles, actual=actual, lead=lead, keep=keep, p99=p99)
        rows.extend({"variant": name, **band} for band in bands)
    return rows


def run_batch(job: tuple[int, list[Event], str]) -> str:
    """Inject one batch with one shape, forecast its cases, and write the cells.

    Args:
        job: The batch's seed, its events, and the shape.

    Returns:
        A progress line.
    """
    seed, batch, shape = job
    target = OUT_DIR / f"cells_{shape}_{seed}.parquet"
    if target.exists():
        return f"{target.name} exists"
    started = time.monotonic()
    state = _STATE["harness"]
    series, arrays, _ = fitting.context()
    rows_of = {int(r["time_series_id"]): r for r in series.iter_rows(named=True)}
    panel = state["panel"]  # ty: ignore[invalid-argument-type]
    injection = inject_shaped(
        power=panel["power"], events=batch, seed=seed, shape=shape, shares=state["shares"]
    )  # ty: ignore[invalid-argument-type]
    residual = injection.fast_residual(panel["residual"])
    offset = injection.fast_residual(state["offset_panel"]["residual"])  # ty: ignore[invalid-argument-type]
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
        if _STATE.get("origin_ok", lambda _: True)(o)  # ty: ignore[call-non-callable]
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
                cut=cut,  # ty: ignore[invalid-argument-type]
            )
        intervals = causal_cache[cut]
        for event, sid, (family, hours) in todo:
            row = rows_of[sid]
            power, injected_mw = powers[sid]
            size = power.size
            signed = {
                "inj": injected_mw,
                "caus": causal_array(
                    intervals=intervals,
                    series_id=sid,
                    capacity=row["effective_capacity_mw"],
                    size=size,
                ),
            }
            lag_index = np.array([common.grid_index(times=origin) - common.NEWEST_AGE])
            newest_inj = float(signed["inj"][lag_index][0])
            newest_caus = float(signed["caus"][lag_index][0])
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
                    site=row["site"], origin=origin, power=power, signed=signed, variants=VARIANTS
                )
            )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    common.atomic_write(
        frame=event_table(batch=batch, injection=injection, shape=shape, seed=seed, state=state),
        target=OUT_DIR / f"events_{shape}_{seed}.parquet",
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


def init_worker() -> None:
    """Load the dense-grid context and the harness state once per worker."""
    fitting.context()
    _STATE["harness"] = harness.make_state()


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
