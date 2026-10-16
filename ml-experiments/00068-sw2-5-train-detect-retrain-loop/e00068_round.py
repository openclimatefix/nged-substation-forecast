"""One round of the loop: build the residual panel from the refits, detect, score, write the mask.

A round has a name such as `A_r2` (world A, round 2). The steps, in order:

- `build_panel`: the refit's predictions as a panel of measured power, expected power, residual.
- `detect_round`: match the D0M threshold to at most 80 odd-month unexplained alarms, build PE's
  intervals, write the alarm and interval tables and the mask for the next round's refit.
- `score_world_b` and `score_world_a`: found or not, for every injected transfer.
- `score_real_transfers`: the labelled real transfers of world A.
- `error_table`: the refit's absolute error per series and month outside logged switching.
"""

from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00068_common import (
    GRID_LENGTH,
    GRID_START,
    LABELLED_PATH,
    SERIES_IDS,
    WORK_DIR,
    load_panel,
    near_log_masks,
    stamp_months,
    stamp_year_months,
)
from e00068_detect import (
    EXPLAIN_PAD,
    NO_VETO,
    explained_flags,
    real_events,
    score_real,
    tune_d2,
)
from e00068_detectors import alarms, prepare
from e00068_endrule import end_only_intervals
from e00068_fit import FEATURES_PATH, round_dir
from e00068_inject import BLOCK, Event
from e00068_intervals import Interval, build_intervals, relative_residual
from e00068_profile import NON_MATCHING, Shape, fit_profile_shares, inject_shaped
from e00068_score import by_series, match_event

WINDOW: Final[int] = 48
N_WORKERS: Final[int] = 4
_STATE: dict = {}


def panel_path(*, name: str) -> Path:
    """Path of a round's panel parquet."""
    return round_dir(name=name) / "panel.parquet"


def build_panel(*, name: str) -> None:
    """Write the round's panel: `series, stamp, valid_time, obs_mw, expected_mw, res_mw`."""
    capacity = (
        pl.scan_parquet(FEATURES_PATH)
        .group_by("time_series_id")
        .agg(pl.col("effective_capacity_mw").median().alias("capacity"))
        .collect()
    )
    frames = [pl.read_parquet(round_dir(name=name) / f"{s}.parquet") for s in SERIES_IDS]
    delta = (
        pl.read_parquet(WORK_DIR / "world_b_delta.parquet")
        if name.startswith("B")
        else pl.DataFrame(schema={"series": pl.Int64, "stamp": pl.Int64, "delta_mw": pl.Float64})
    )
    panel = (
        pl.concat(frames)
        .join(capacity, on="time_series_id")
        .select(
            series=pl.col("time_series_id").cast(pl.Int64),
            valid_time="valid_time",
            obs_mw=(pl.col("target") * pl.col("capacity")).cast(pl.Float64),
            expected_mw=(pl.col("pred") * pl.col("capacity")).cast(pl.Float64),
        )
        .with_columns(
            stamp=((pl.col("valid_time") - GRID_START).dt.total_seconds() // 1800).cast(pl.Int64)
        )
        .join(delta, on=["series", "stamp"], how="left")
        .with_columns(obs_mw=pl.col("obs_mw") + pl.col("delta_mw").fill_null(0.0))
        .drop("delta_mw")
        .with_columns(res_mw=pl.col("obs_mw") - pl.col("expected_mw"))
        .filter(pl.col("stamp").is_between(0, GRID_LENGTH - 1))
        .sort("series", "stamp")
    )
    panel.write_parquet(panel_path(name=name))


def load_round(*, name: str) -> dict[str, dict[int, np.ndarray]]:
    """The round's panel as arrays per series."""
    return load_panel(path=panel_path(name=name))


def detectors(
    *, residual: dict, expected: dict, scale: dict, threshold: float
) -> tuple[pl.DataFrame, list[Interval]]:
    """D0M's alarms and PE's intervals on one residual panel."""
    relative = {s: relative_residual(residual=residual[s], expected=expected[s]) for s in residual}
    inputs = prepare(residuals=residual, window=WINDOW, scale=scale)
    table = alarms(inputs=inputs, detector="D0", threshold=threshold)
    d0m = build_intervals(alarm_table=table, relative=relative, window=WINDOW, detector="D0")
    return table, end_only_intervals(intervals=d0m, inputs=inputs)


def detect_round(*, name: str, next_name: str | None, reference_threshold: float | None) -> dict:
    """Tune, detect, write the tables and the next round's mask; return the round's statistics."""
    panel = load_round(name=name)
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    plain = prepare(residuals=panel["residual"], window=WINDOW)
    threshold, _ = tune_d2(inputs=plain, veto_z=NO_VETO, logged=logged)
    table, intervals = detectors(
        residual=panel["residual"],
        expected=panel["expected"],
        scale=plain.scale,
        threshold=threshold,
    )
    months = stamp_months()
    stamps = table["stamp"].to_numpy()
    unexplained = ~explained_flags(table=table, logged=logged)
    counts = {
        "odd_unexplained": int((unexplained & (months[stamps] % 2 == 1)).sum()),
        "even_unexplained": int((unexplained & (months[stamps] % 2 == 0)).sum()),
    }
    out = round_dir(name=name)
    table.write_parquet(out / "alarms.parquet")
    frame = pl.DataFrame(
        [
            {"series": iv.series, "onset": iv.onset, "end": iv.end, "alarm_series": iv.alarm_series}
            for iv in intervals
        ],
        schema={
            "series": pl.Int64,
            "onset": pl.Int64,
            "end": pl.Int64,
            "alarm_series": pl.Int64,
        },
    )
    frame.write_parquet(out / "intervals.parquet")
    if next_name is not None:
        nxt = round_dir(name=next_name)
        nxt.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(nxt / "mask_intervals.parquet")
    stats = {
        "round": name,
        "threshold": threshold,
        "median_scale": float(np.median(list(plain.scale.values()))),
        "scale": {str(s): v for s, v in plain.scale.items()},
        "n_alarms": table.height,
        "n_intervals": len(intervals),
        "masked_half_hours": int(sum(iv.end - iv.onset for iv in intervals)),
        **counts,
    }
    if reference_threshold is not None:
        ref_table, _ = detectors(
            residual=panel["residual"],
            expected=panel["expected"],
            scale=plain.scale,
            threshold=reference_threshold,
        )
        ref_stamps = ref_table["stamp"].to_numpy()
        ref_unexplained = ~explained_flags(table=ref_table, logged=logged)
        stats["odd_unexplained_at_round0_threshold"] = int(
            (ref_unexplained & (months[ref_stamps] % 2 == 1)).sum()
        )
    (out / "stats.json").write_text(json.dumps(stats))
    print(
        f"{name}: threshold {threshold}, odd unexplained {counts['odd_unexplained']}, "
        f"even {counts['even_unexplained']}, {len(intervals)} intervals",
        flush=True,
    )
    return stats


def random_mask(*, source_name: str, name: str, seed: int = 7) -> None:
    """Write the control mask: each interval moved to a random start in its series and month."""
    rng = np.random.default_rng(seed)
    year_months = stamp_year_months()
    frame = pl.read_parquet(round_dir(name=source_name) / "mask_intervals.parquet")
    taken = {s: np.zeros(GRID_LENGTH, dtype=bool) for s in SERIES_IDS}
    for row in frame.iter_rows(named=True):
        taken[row["series"]][row["onset"] : row["end"]] = True
    rows = []
    for row in frame.iter_rows(named=True):
        length = row["end"] - row["onset"]
        month = year_months[row["onset"]]
        in_month = np.flatnonzero(year_months == month)
        lo, hi = int(in_month.min()), int(in_month.max()) - length
        placed = (row["onset"], row["end"])
        if hi > lo:
            for _ in range(100):
                start = int(rng.integers(lo, hi))
                if not taken[row["series"]][start : start + length].any():
                    placed = (start, start + length)
                    taken[row["series"]][placed[0] : placed[1]] = True
                    break
        rows.append({"series": row["series"], "onset": placed[0], "end": placed[1]})
    out = round_dir(name=name)
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(out / "mask_intervals.parquet")


def init_worker(state: dict) -> None:
    """Pool initializer."""
    _STATE.update(state)


def _score_a_batch(args: tuple[int, list[Event], Shape]) -> list[dict]:
    """World A, one batch and shape: add the transfers to the residual and match the intervals."""
    seed, batch, shape = args
    panel = _STATE["panel"]
    injection = inject_shaped(
        power=panel["power"], events=batch, seed=seed, shape=shape, shares=_STATE["shares"]
    )
    _, intervals = detectors(
        residual=injection.fast_residual(panel["residual"]),
        expected=panel["expected"],
        scale=_STATE["scale"],
        threshold=_STATE["threshold"],
    )
    return match_rows(
        batch=batch, intervals=intervals, injection_delta=injection.delta, shape=shape
    )


def match_rows(
    *, batch: list[Event], intervals: list[Interval], injection_delta: dict, shape: str
) -> list[dict]:
    """One row per event: found, onset found, MW moved, and the donor step in noise units."""
    grouped = by_series(intervals=intervals)
    scale = _STATE["scale0"]
    rows = []
    for event in batch:
        match = match_event(
            series=event.series, onset=event.onset, end=event.end, intervals=grouped
        )
        span = slice(event.onset, min(event.end, event.onset + BLOCK))
        rows.append(
            {
                "event_id": event.event_id,
                "shape": shape,
                "found": match.found,
                "onset_found": match.onset_found,
                "moved_mw": float(-np.mean(injection_delta[event.donor][event.onset : event.end])),
                "donor_step_z": float(-np.mean(injection_delta[event.donor][span]))
                / scale[event.donor],
                "metered_receivers": len(event.receivers),
                "hours": (event.end - event.onset) / 2,
                "year_month": int(stamp_year_months()[event.onset]),
            }
        )
    return rows


def score_world_a(
    *,
    name: str,
    batches: list[tuple[int, list[Event]]],
    scale0: dict,
    panel: dict | None = None,
    threshold: float | None = None,
) -> pl.DataFrame:
    """Score the world-A transfers under S2, S3, and S4, injected into the residual only.

    `panel` and `threshold` default to the round's own; the round-0 check passes the stored ones.
    """
    panel = panel if panel is not None else load_round(name=name)
    if threshold is None:
        threshold = json.loads((round_dir(name=name) / "stats.json").read_text())["threshold"]
    state = {
        "panel": panel,
        "shares": fit_profile_shares(panel=load_panel()),
        "scale": prepare(residuals=panel["residual"], window=WINDOW).scale,
        "scale0": scale0,
        "threshold": threshold,
    }
    jobs = [(seed, batch, shape) for shape in NON_MATCHING for seed, batch in batches]
    with ProcessPoolExecutor(
        N_WORKERS,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
        initargs=(state,),
    ) as pool:
        rows = [r for part in pool.map(_score_a_batch, jobs) for r in part]
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        world=pl.lit("A_into_residual"), round=pl.lit(name)
    )


def score_world_b(*, name: str, events: list[Event], scale0: dict) -> pl.DataFrame:
    """Score world B: the transfers are in the data, so the round's own residual has them."""
    panel = load_round(name=name)
    _STATE.update({"scale0": scale0})
    intervals = pl.read_parquet(round_dir(name=name) / "intervals.parquet")
    frame = [
        Interval(
            series=r["series"],
            onset=r["onset"],
            end=r["end"],
            known_from=r["end"],
            alarm_series=r["alarm_series"],
            detector="D0",
            role="",
            relative_step=0.0,
        )
        for r in intervals.iter_rows(named=True)
    ]
    table = pl.read_parquet(WORK_DIR / "world_b_events.parquet")
    delta = pl.read_parquet(WORK_DIR / "world_b_delta.parquet")
    arrays = {s: np.zeros(GRID_LENGTH) for s in panel["power"]}
    for s, stamp, value in delta.select("series", "stamp", "delta_mw").iter_rows():
        arrays[s][stamp] = value
    shapes = dict(zip(table["event_id"], table["shape"], strict=True))
    rows = []
    for event in events:
        rows += match_rows(
            batch=[event], intervals=frame, injection_delta=arrays, shape=shapes[event.event_id]
        )
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        world=pl.lit("B_in_training"), round=pl.lit(name)
    )


def score_real_transfers(*, name: str) -> pl.DataFrame:
    """Score the labelled real transfers of world A against the round's intervals."""
    panel = load_round(name=name)
    intervals = pl.read_parquet(round_dir(name=name) / "intervals.parquet")
    found = [
        Interval(
            series=r["series"],
            onset=r["onset"],
            end=r["end"],
            known_from=r["end"],
            alarm_series=r["alarm_series"],
            detector="D0",
            role="",
            relative_step=0.0,
        )
        for r in intervals.iter_rows(named=True)
    ]
    real = real_events(power=panel["power"])
    moved = pl.read_parquet(LABELLED_PATH).select("card_id", "moved_mw_p50")
    scored = score_real(results={name: (pl.DataFrame(), found)}, real=real)
    return scored.join(moved, on="card_id", validate="m:1").with_columns(round=pl.lit(name))


def error_table(*, name: str) -> pl.DataFrame:
    """Absolute error summed per series and calendar month on half-hours outside logged switching.

    Half-hours within 24 hours of a logged closure on the series, or of a labelled real transfer on
    the series, are dropped. The sums and counts are kept so that months can be resampled.
    """
    panel = load_round(name=name)
    near = near_log_masks(pad_stamps=EXPLAIN_PAD)
    for row in real_events(power=panel["power"]).iter_rows(named=True):
        for s in (int(x) for x in row["series"].split(",")):
            if s in near:
                near[s][max(row["onset"] - BLOCK, 0) : row["end"] + BLOCK] = True
    year_months = stamp_year_months()
    capacity = (
        pl.scan_parquet(FEATURES_PATH)
        .group_by("time_series_id")
        .agg(pl.col("effective_capacity_mw").median().alias("capacity"))
        .collect()
    )
    cap = dict(zip(capacity["time_series_id"], capacity["capacity"], strict=True))
    rows = []
    for s in SERIES_IDS:
        error = np.abs(panel["residual"][s]) / cap[s]
        ok = np.isfinite(error) & ~near[s]
        for month in np.unique(year_months[ok]):
            pick = ok & (year_months == month)
            rows.append(
                {
                    "series": s,
                    "year_month": int(month),
                    "abs_error_sum": float(error[pick].sum()),
                    "n": int(pick.sum()),
                }
            )
    return pl.DataFrame(rows).with_columns(round=pl.lit(name))
