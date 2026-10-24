# ruff: noqa: E501, PLR0915
"""Score the knowledge sources and the quality sweep at one substation in one injected world.

Every arm is GAM-K's cached member forecasts plus a Kalman filter correction run on the world's
stream (the real residual stream plus the injected transfers), with experiment 115's K0 spread, and
is scored against the true normal-arrangement series, never against the stream with the transfers
in it. A knowledge source is a list of spans (start, end, when each is known); the MASK design sets
the stream's readings missing inside a known span, and the SUB design subtracts the span's megawatts
(MW). The quality sweep degrades the perfect spans in controlled steps with fixed seeds.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00297_injected.py run [--limit N]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Final

import e00297_fed as fed
import e00297_nra as nra
import numpy as np
import polars as pl
from e00297_common import WORK_DIR, stamp_months

N_WORKERS: Final[int] = 4
WORLDS: Final[Path] = WORK_DIR.parent / "00141-switching-detector-fed-into-gam-k" / "worlds"
CHOSEN: Final[Path] = (
    WORK_DIR.parent / "00141-switching-detector-fed-into-gam-k" / "scores" / "chosen.json"
)
OUT: Final[Path] = WORK_DIR / "injected"
OUT_EXTRA: Final[Path] = WORK_DIR / "injected_extra"
PERIODS: Final[tuple[str, ...]] = ("replicate", "scored")
RECALLS: Final[tuple[float, ...]] = (0.25, 0.5, 0.75)
DELAYS: Final[tuple[int, ...]] = (6, 12, 24, 48)
EXTRA_DELAYS: Final[tuple[int, ...]] = (96, 144, 192)
"""Post hoc: delays of 2, 3, and 4 days (the pre-registered delays all end before the DURING and
AFTER rows begin, because those rows have a cutoff at least 96 half-hours after the onset)."""
EXTRA: Final[bool] = os.environ.get("E208_EXTRA") == "1"
"""Set by the environment variable E208_EXTRA=1 (workers are spawned, so a flag must travel in the environment): score only K0 and the post hoc delays, into `injected_extra`."""
"""Half-hours of delay added to the perfect onset (3, 6, 12, 24 hours)."""
BLURS: Final[tuple[float, ...]] = (0.25, 0.5)
REPAIRS: Final[tuple[float, ...]] = (0.25, 0.5, 0.75, 1.0)
BEFORE_ALARM_STAMPS: Final[int] = nra.ONSET_LAG
"""Phase boundary: a forecast whose data cutoff is within this many half-hours of the true onset
is in BEFORE_ALARM, the time a live alarm normally takes to arrive."""
AFTER_STAMPS: Final[int] = 96


def worlds_list() -> list[int]:
    """The seeds of experiment 141's stored worlds."""
    return sorted(int(p.stem.split("_")[1]) for p in WORLDS.glob("world_*.parquet"))


def load_world(*, seed: int) -> tuple[pl.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """A world's intervals, the MW added to each series by stamp, and its event table."""
    intervals = pl.read_parquet(WORLDS / f"world_{seed}.parquet")
    with np.load(WORLDS / f"world_{seed}.npz") as stored:
        delta = {k: stored[k] for k in stored.files if k.startswith("delta_")}
        events = stored["events"]
    return intervals, delta, events


def site_events(*, delta: np.ndarray, events: np.ndarray) -> list[dict[str, float]]:
    """The events that move MW at this site: id, onset and end stamp, and moved share of load."""
    out = []
    for row in events:
        onset, end = int(row[2]), int(row[3])
        if np.any(np.nan_to_num(delta[onset:end])):
            out.append({"id": int(row[0]), "onset": onset, "end": end, "fraction": float(row[4])})
    return sorted(out, key=lambda e: e["onset"])


def perfect_spans(
    *, events: list[dict[str, float]], points: np.ndarray, data: Any, delay: int = 0
) -> list[nra.Span]:
    """The true spans, known `delay` half-hours after the true onset; the end is known at once."""
    offset = fed.stamp_offset(data)
    spans = []
    for e in events:
        onset, end = int(e["onset"]) + offset, int(e["end"]) + offset
        masked = np.zeros_like(points)
        masked[onset:end] = points[onset:end]
        start = onset + delay
        held = masked.copy()
        held[end:] = masked[min(start, end - 1)]
        spans.append(nra.Span(onset, end, start, max(end, start + 1), masked, held))
    return spans


def with_amount(*, span: nra.Span, factor: float) -> nra.Span:
    """The span with its MW scaled (the blur of the MW)."""
    scale = lambda a: None if a is None else a * factor  # noqa: E731
    return dataclasses.replace(span, amount=scale(span.amount), open_amount=scale(span.open_amount))


def matches(*, span: nra.Span, event: dict[str, float], offset: int) -> bool:
    """Whether a detected span starts at the true onset of an injected event."""
    return abs(span.onset - (int(event["onset"]) + offset)) <= nra.ONSET_TOLERANCE


def knowledge_sources(
    *,
    data: Any,
    events: list[dict[str, float]],
    points: np.ndarray,
    detected: list[nra.Span],
    rng_seed: int,
) -> dict[str, tuple[list[nra.Span], tuple[str, ...]]]:
    """Every knowledge source as spans, with the designs (`MASK`, `SUB`) it is scored under."""
    offset = fed.stamp_offset(data)
    both = ("MASK", "SUB")
    perfect = perfect_spans(events=events, points=points, data=data)
    lagged = nra.true_spans(
        events=[(int(e["onset"]), int(e["end"])) for e in events], delta_points=points, data=data
    )
    found = [[s for s in detected if matches(span=s, event=e, offset=offset)] for e in events]
    out: dict[str, tuple[list[nra.Span], tuple[str, ...]]] = {
        "PERFECT": (perfect, both),
        "PERFECT_LAG": (lagged, both),
        "DET": (detected, both),
        "DET_NOFALSE": ([s for hits in found for s in hits], both),
    }
    for recall in RECALLS:
        rng = np.random.default_rng([rng_seed, int(recall * 100)])
        kept = rng.random(len(events)) < recall
        out[f"RECALL_{recall}"] = ([s for s, k in zip(perfect, kept, strict=True) if k], both)
    for delay in DELAYS:
        out[f"DELAY_{delay}"] = (
            perfect_spans(events=events, points=points, data=data, delay=delay),
            both,
        )
    for blur in BLURS:
        rng = np.random.default_rng([rng_seed, 7, int(blur * 100)])
        signs = rng.choice([-1.0, 1.0], size=len(events))
        out[f"BLUR_{blur}"] = (
            [
                with_amount(span=s, factor=1.0 + blur * g)
                for s, g in zip(perfect, signs, strict=True)
            ],
            ("SUB",),
        )
    for repair in REPAIRS:
        rng = np.random.default_rng([rng_seed, 11, int(repair * 100)])
        spans = list(detected)
        for hits, truth in zip(found, lagged, strict=True):
            if not hits and rng.random() < repair:
                spans.append(truth)
        out[f"REPAIR_{repair}"] = (spans, both)
    return out


def phase_of(
    *, cutoff: np.ndarray, target: np.ndarray, onset: np.ndarray, end: np.ndarray
) -> np.ndarray:
    """The phase of each row, fixed by the true transfer and the forecast cutoff alone.

    BEFORE_ALARM takes precedence: a row whose cutoff is within `BEFORE_ALARM_STAMPS` of the onset
    is in it even when its target time lies after the true end (short transfers).
    """
    return np.where(
        cutoff < onset + BEFORE_ALARM_STAMPS,
        "BEFORE_ALARM",
        np.where(target < end, "DURING", "AFTER"),
    )


def job(args: tuple[str, int]) -> str:
    """Score every arm at one site in one world."""
    site, seed = args
    path = (OUT_EXTRA if EXTRA else OUT) / f"{site}_{seed}.parquet"
    if path.exists():
        return str(path)
    started = time.time()
    data = fed.get_site(site)
    intervals, delta, events_all = load_world(seed=seed)
    stamp_delta = delta[f"delta_{fed.SERIES_OF_SITE[site]}"]
    if not np.any(np.nan_to_num(stamp_delta)):
        return ""
    n = len(data.stream)
    offset = fed.stamp_offset(data)
    points = fed.to_positions(
        array=100.0 * np.nan_to_num(stamp_delta) / fed.CAPACITY_MW[site], data=data
    )
    stream = np.where(np.isfinite(data.stream), data.stream + points, np.nan)
    events = site_events(delta=stamp_delta, events=events_all)
    detected = nra.detected_spans(data=data, site=site, intervals=intervals)
    site_index = sorted(fed.SERIES_OF_SITE).index(site)
    sources = knowledge_sources(
        data=data, events=events, points=points, detected=detected, rng_seed=seed * 10 + site_index
    )
    if EXTRA:
        sources = {
            f"DELAYX_{d}": (
                perfect_spans(events=events, points=points, data=data, delay=d),
                ("MASK",),
            )
            for d in EXTRA_DELAYS
        }
    cutoffs = data.cutoffs(delay_hours=fed.DELAY_HOURS)[data.run_of_row]
    months = stamp_months()
    onset_pos = np.array([int(e["onset"]) + offset for e in events])
    end_pos = np.array([int(e["end"]) + offset for e in events])
    window = np.full(n, -1, dtype=np.int64)
    for k in range(len(events)):
        window[onset_pos[k] : min(end_pos[k] + AFTER_STAMPS, n)] = k
    frames = []
    for period in PERIODS:
        rows = fed.rows_where(data=data, mask_by_position=window >= 0, period=period)
        k = window[data.position[rows]]
        rows, k = rows[k >= 0], k[k >= 0]
        keep = cutoffs[rows] >= onset_pos[k]
        rows, k = rows[keep], k[keep]
        if not len(rows):
            continue
        corrections = {
            "K0|NONE": fed.correction_for(
                data=data, stream=stream, prior=np.zeros(n), multiplier=1.0
            ),
        }
        if not EXTRA:
            corrections["CLEAN|NONE"] = fed.correction_for(
                data=data, stream=data.stream, prior=np.zeros(n), multiplier=1.0
            )
            corrections["GAM_ONLY|NONE"] = np.zeros(len(data.run_of_row))
        for name, (spans, designs) in sources.items():
            for design in designs:
                corrections[f"{name}|{design}"] = nra.subtraction_correction(
                    data=data,
                    stream=stream,
                    spans=spans,
                    cutoffs=cutoffs,
                    rows=rows,
                    mask=design == "MASK",
                )
        first = int(np.flatnonzero(points)[0])
        early = np.flatnonzero(cutoffs < first)
        gap = (
            np.nanmax(np.abs(corrections["CLEAN|NONE"][early] - corrections["K0|NONE"][early]))
            if len(early) and not EXTRA
            else 0.0
        )
        if not gap < 1e-9:
            raise AssertionError(f"CLEAN differs from K0 before the first injected stamp: {gap}")
        scored = nra.score_nra(data=data, site=site, corrections=corrections, rows=rows)
        pos = data.position[rows]
        meta = pl.DataFrame(
            {
                "row": rows,
                "event": np.array([int(e["id"]) for e in events])[k],
                "phase": phase_of(
                    cutoff=cutoffs[rows], target=pos, onset=onset_pos[k], end=end_pos[k]
                ),
                "event_hours": ((end_pos - onset_pos) / 2.0)[k],
                "fraction": np.array([e["fraction"] for e in events])[k],
                "onset_month": months[np.clip(onset_pos[k] - offset, 0, len(months) - 1)],
            }
        )
        frames.append(
            scored.join(meta, on="row").with_columns(
                world=pl.lit(seed),
                period=pl.lit(period),
                arm_name=pl.col("arm").str.split("|").list.get(0),
                design=pl.col("arm").str.split("|").list.get(1),
            )
        )
    if not frames:
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pl.concat(frames).write_parquet(tmp)
    tmp.rename(path)
    print(
        f"{site} {seed}: {len(events)} events, {len(sources)} sources, {time.time() - started:.0f} s",
        flush=True,
    )
    return str(path)


def main() -> None:
    """Run the injected stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["run", "one", "extra"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if (args.stage == "extra") != EXTRA:
        raise SystemExit("run the extra stage with E208_EXTRA=1, and no other stage with it")
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, w) for w in worlds_list() for s in fed.SERIES_OF_SITE]
    if args.stage == "one":
        print(job(jobs[0]))
        return
    if args.limit:
        jobs = jobs[: args.limit]
    with ProcessPoolExecutor(N_WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        done = [p for p in pool.map(job, jobs) if p]
    print(json.dumps({"files": len(done)}))


if __name__ == "__main__":
    main()
