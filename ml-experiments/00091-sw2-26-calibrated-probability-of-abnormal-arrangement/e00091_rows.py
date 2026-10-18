"""Write the lag-window rows of experiment 91: the causal detector's inputs and the truth.

For each batch of injected events (series disjoint within 7 days) and each shape, the script
injects the transfers, runs the causal XL detector at the cut of every daily origin that has an
injected event in its lag window, and writes one row per (series, lag stamp, origin) with the
inputs of the three arms and the truth. A `REAL` job does the same on the real residuals at every
origin, where the truth is the logged closures and the labelled events.

    uv run python ml-experiments/00091-*/e00091_rows.py [--pilot] [--workers 4]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Final, cast

import e00091_candidates as cands
import e00091_common as common
import e00091_grid as grid
import e00091_harness as harness
import e00091_labels as labelling
import numpy as np
import polars as pl
from e00091_inject import Event, Injection
from e00091_profile import Shape, inject_shaped

OFFSET: Final[int] = harness.OFFSET
SCORED_SHAPES: Final[tuple[str, ...]] = ("S1", "S2", "S3", "S4", "S6")
LAG_WINDOW: Final[int] = 7 * grid.BLOCK
"""A lag is at most 7 days before the newest usable value."""
LAG_STEP: Final[int] = 12
"""Every sixth hour of the lag window is a row."""
ROWS_DIR: Final[Path] = common.WORK_DIR / "rows"
_STATE: dict[str, Any] = {}


def newest_stamp(*, origin: datetime) -> int:
    """Return the experiment 82 stamp of the newest usable observation for an origin."""
    return grid.stamp_of(when=origin) - common.NEWEST_AGE


def real_ara_masks() -> dict[int, np.ndarray]:
    """Series-stamp masks of a logged closure or a labelled event, on experiment 82's grid."""
    series = common.series_table().filter(pl.col("substation_type") == "Primary")
    labels = labelling.load_labels(series=series)
    ids = {int(r["time_series_id"]): r["site"] for r in series.iter_rows(named=True)}
    logged = grid.near_log_masks(pad_stamps=0)
    out = {}
    for sid in grid.SERIES_IDS:
        lab = labels[ids[sid]]
        window = slice(OFFSET, OFFSET + grid.GRID_LENGTH)
        out[sid] = lab.in_event[window] | lab.closed[window] | logged[sid]
    return out


def lag_stamps(*, cut: int) -> np.ndarray:
    """The lag stamps of a cut: the newest usable stamp, then every `LAG_STEP` back to 7 days."""
    return np.arange(cut - 1, cut - 1 - LAG_WINDOW, -LAG_STEP)


def event_stats(*, batch: list[Event], injection: Injection, scale: dict[int, float]) -> dict:
    """Per event: the donor's first-day step in noise units, the MW moved, and its hours."""
    out = {}
    for event in batch:
        donor = injection.delta[event.donor]
        first_day = slice(event.onset, min(event.end, event.onset + grid.BLOCK))
        out[event.event_id] = (
            float(-np.mean(donor[first_day]) / scale[event.donor]),
            float(-np.mean(donor[event.onset : event.end])),
            (event.end - event.onset) / 2,
        )
    return out


def injected_truth(
    *, events: list[Event], series: int, lags: np.ndarray, stats: dict, cut: int
) -> tuple[np.ndarray, np.ndarray]:
    """Where a lag stamp lies inside an injected interval on the series, and that event's size.

    Returns:
        The truth flag, and an array of (step in noise units, MW moved, hours, hours since onset
        at the cut) of the covering event with the largest step, NaN where no event covers.
    """
    flag = np.zeros(lags.size, dtype=bool)
    detail = np.full((lags.size, 5), np.nan)
    for event in events:
        if series not in event.series:
            continue
        inside = (lags >= event.onset) & (lags < event.end)
        z, mw, hours = stats[event.event_id]
        better = inside & (np.isnan(detail[:, 0]) | (z > detail[:, 0]))
        detail[better] = (z, mw, hours, (cut - event.onset) / 2, event.onset)
        flag |= inside
    return flag, detail


def build_rows(
    *,
    state: dict,
    residual: dict,
    offset: dict,
    cut: int,
    events: list[Event],
    ara: dict[int, np.ndarray],
    shape: str,
    seed: int,
    real: bool,
    stats: dict,
) -> pl.DataFrame:
    """Return the rows of one cut: the detector inputs, the truth, and the grouping columns."""
    found = cands.causal_candidates(state=state, residual=residual, offset=offset, cut=cut)
    lags = lag_stamps(cut=cut)
    table = cands.lag_rows(found=found, series_ids=grid.SERIES_IDS, lags=lags, cut=cut)
    truth, blocked, details = [], [], []
    for sid in grid.SERIES_IDS:
        known = ara[sid][lags]
        blocked.append(known)
        if real:
            truth.append(known)
            details.append(np.full((lags.size, 5), np.nan))
        else:
            flag, detail = injected_truth(
                events=events, series=sid, lags=lags, stats=stats, cut=cut
            )
            truth.append(flag)
            details.append(detail)
    detail = np.vstack(details)
    active = [e for e in events if e.onset < cut and e.end > cut - LAG_WINDOW]
    reference = max(active, key=lambda e: e.onset) if active else None
    ref_stamp = reference.onset if reference else cut
    months = grid.stamp_months()
    return table.with_columns(
        y=pl.Series(np.concatenate(truth)),
        ev_step_z=pl.Series(detail[:, 0]),
        ev_mw=pl.Series(detail[:, 1]),
        ev_hours=pl.Series(detail[:, 2]),
        ev_age_hours=pl.Series(detail[:, 3]),
        ev_onset=pl.Series(detail[:, 4]),
        blocked=pl.Series(np.concatenate(blocked)),
        cut=pl.lit(cut),
        shape=pl.lit(shape),
        seed=pl.lit(seed),
        ref_month=pl.lit(int(months[min(ref_stamp, cut - 1)])),
        cut_age_hours=pl.lit((cut - reference.onset) / 2 if reference else float("nan")),
    )


def run_job(job: tuple[int, list[Event] | None, str]) -> str:
    """Write the rows of one batch and shape (or of the real residuals)."""
    seed, batch, shape = job
    target = ROWS_DIR / f"rows_{shape}_{seed}.parquet"
    if target.exists():
        return f"{target.name} exists"
    started = time.monotonic()
    state = _STATE["harness"]
    ara = _STATE["ara"]
    panel = state["panel"]
    origins = common.origins_of_subset(subset="gap")
    only: set[int] | None = _STATE.get("only_cuts")
    frames = []
    if batch is None:
        residual, offset, events = panel["residual"], state["offset_panel"]["residual"], []
    else:
        injection: Injection = inject_shaped(
            power=panel["power"],
            events=batch,
            seed=seed,
            shape=cast(Shape, shape),
            shares=state["shares"],
        )
        residual = injection.fast_residual(panel["residual"])
        offset = injection.fast_residual(state["offset_panel"]["residual"])
        events = batch
    stats = (
        {}
        if batch is None
        else event_stats(batch=batch, injection=injection, scale=state["scales"]["X"])
    )
    for origin in origins:
        cut = newest_stamp(origin=origin) + 1
        if only is not None and cut not in only:
            continue
        if batch is not None and not any(
            e.onset < cut and e.end > cut - LAG_WINDOW for e in events
        ):
            continue
        frames.append(
            build_rows(
                state=state,
                residual=residual,
                offset=offset,
                cut=cut,
                events=events,
                ara=ara,
                shape=shape,
                seed=seed,
                real=batch is None,
                stats=stats,
            )
        )
    ROWS_DIR.mkdir(parents=True, exist_ok=True)
    if not frames:
        return f"{target.name}: no cut has an injected event in its lag window"
    frame = pl.concat(frames)
    common.atomic_write(frame=frame, target=target)
    return f"{target.name}: {frame.height:,} rows in {time.monotonic() - started:.0f} s"


def init_worker() -> None:
    """Load the harness state and the real ARA masks once per worker."""
    _STATE["harness"] = harness.make_state()
    _STATE["ara"] = real_ara_masks()


def main() -> None:
    """Write the rows of every batch and shape, then the real rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pilot", action="store_true", help="one batch, one shape, 3 cuts")
    args = parser.parse_args()
    state = harness.make_state()
    events = harness.cached_events(power=state["panel"]["power"])
    batches = harness.seeded_batches(events=events)
    jobs: list[tuple[int, list[Event] | None, str]] = [
        (seed, batch, shape) for shape in SCORED_SHAPES for seed, batch in batches
    ]
    jobs.append((0, None, "REAL"))
    if args.pilot:
        init_worker()
        batch_seed, batch = batches[0]
        origins = common.origins_of_subset(subset="gap")
        _STATE["only_cuts"] = {newest_stamp(origin=o) + 1 for o in origins[100:104]}
        globals()["ROWS_DIR"] = common.WORK_DIR / "rows_pilot"
        for job in ((batch_seed, batch, "S2"), (0, None, "REAL")):
            print(run_job(job))
        return
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        for done, line in enumerate(pool.map(run_job, jobs), start=1):
            print(f"[{done}/{len(jobs)}] {line}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
