# ruff: noqa: E501
# ruff: noqa: E501, PLR0915
"""Score the knowledge sources and the quality sweep at one substation in one injected world.

Every arm is GAM-K's cached member forecasts plus a Kalman filter correction run on the world's
stream (the real residual stream plus the injected transfers), with experiment 115's K0 spread, and
is scored against the true normal-arrangement series, never against the stream with the transfers
in it. A knowledge source is a list of spans (start, end, when each is known); the MASK design sets
the stream's readings missing inside a known span, and the SUB design subtracts the span's megawatts
(MW). The quality sweep degrades the perfect spans in controlled steps with fixed seeds.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00227_injected.py run [--limit N]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Final

import e00227_fed as fed
import e00227_nra as nra
import e00227_uses as uses
import numpy as np
import polars as pl
from e00227_common import WORK_DIR, stamp_months

N_WORKERS: Final[int] = 6
WORLDS: Final[Path] = WORK_DIR.parent / "00141-switching-detector-fed-into-gam-k" / "worlds"
CHOSEN: Final[Path] = (
    WORK_DIR.parent / "00141-switching-detector-fed-into-gam-k" / "scores" / "chosen.json"
)
SOFT_LARGE: Final[bool] = bool(os.environ.get("E227_SOFT_LARGE"))
"""Post hoc: run only MASK and soft down-weighting with much larger noise factors."""
OUT: Final[Path] = WORK_DIR / ("injected_soft_large" if SOFT_LARGE else "injected")
PERIODS: Final[tuple[str, ...]] = ("replicate", "scored")
BEFORE_ALARM_STAMPS: Final[int] = nra.ONSET_LAG
"""Phase boundary: a forecast whose data cutoff is within this many half-hours of the true onset
is in BEFORE_ALARM, the time a live alarm normally takes to arrive."""
AFTER_STAMPS: Final[int] = 96
NEAR_STAMPS: Final[int] = 288
"""Half-hours after a false alarm's end is known over which its effect on ordinary rows is scored."""


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


def matches(*, span: nra.Span, event: dict[str, float], offset: int) -> bool:
    """Whether a detected span starts at the true onset of an injected event."""
    return abs(span.onset - (int(event["onset"]) + offset)) <= nra.ONSET_TOLERANCE


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
    path = OUT / f"{site}_{seed}.parquet"
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
    perfect = perfect_spans(events=events, points=points, data=data)
    designs = (
        ("MASK", "SOFT_64", "SOFT_1024", "SOFT_16384")
        if SOFT_LARGE
        else ("MASK", "SUB", "SOFT_2", "SOFT_4", "SOFT_16", "HALF")
    )
    sources = {
        "PERFECT": (perfect, designs),
        "DET": (detected, designs),
    }
    cutoffs = data.cutoffs(delay_hours=fed.DELAY_HOURS)[data.run_of_row]
    months = stamp_months()
    onset_pos = np.array([int(e["onset"]) + offset for e in events])
    end_pos = np.array([int(e["end"]) + offset for e in events])
    window = np.full(n, -1, dtype=np.int64)
    for k in range(len(events)):
        window[onset_pos[k] : min(end_pos[k] + AFTER_STAMPS, n)] = k
    blocked = np.zeros(n, dtype=bool)
    for k in range(len(events)):
        blocked[max(onset_pos[k] - 96, 0) : min(end_pos[k] + nra.TAIL, n)] = True
    false_spans = [
        s for s in detected if not any(matches(span=s, event=e, offset=offset) for e in events)
    ]
    near = np.zeros(len(data.run_of_row), dtype=bool)
    for s in false_spans:
        near |= (cutoffs >= s.onset) & (cutoffs < s.known + NEAR_STAMPS)
    frames = []
    for period in PERIODS:
        rows = fed.rows_where(data=data, mask_by_position=window >= 0, period=period)
        k = window[data.position[rows]]
        rows, k = rows[k >= 0], k[k >= 0]
        keep = cutoffs[rows] >= onset_pos[k]
        rows, k = rows[keep], k[keep]
        ordinary = fed.rows_where(data=data, mask_by_position=~blocked, period=period)
        ordinary = ordinary[near[ordinary]]
        rows = np.concatenate([rows, ordinary]).astype(np.int64)
        k = np.concatenate([k, np.full(len(ordinary), -1, dtype=np.int64)])
        if not len(rows):
            continue
        corrections = {
            "K0|NONE": fed.correction_for(
                data=data, stream=stream, prior=np.zeros(n), multiplier=1.0
            ),
        }
        corrections["CLEAN|NONE"] = fed.correction_for(
            data=data, stream=data.stream, prior=np.zeros(n), multiplier=1.0
        )
        corrections["GAM_ONLY|NONE"] = np.zeros(len(data.run_of_row))
        for name, (spans, designs) in sources.items():
            for design in designs:
                if design == "HALF":
                    continue
                corrections[f"{name}|{design}"] = (
                    uses.soft(
                        data=data,
                        stream=stream,
                        spans=spans,
                        cutoffs=cutoffs,
                        rows=rows,
                        factor=float(design.split("_")[1]),
                    )
                    if design.startswith("SOFT")
                    else nra.subtraction_correction(
                        data=data,
                        stream=stream,
                        spans=spans,
                        cutoffs=cutoffs,
                        rows=rows,
                        mask=design == "MASK",
                    )
                )
            corrections[f"{name}|HALF"] = uses.half(
                k0=corrections["K0|NONE"], masked=corrections[f"{name}|MASK"]
            )
        first = int(np.flatnonzero(points)[0])
        early = np.flatnonzero(cutoffs < first)
        gap = (
            np.nanmax(np.abs(corrections["CLEAN|NONE"][early] - corrections["K0|NONE"][early]))
            if len(early)
            else 0.0
        )
        if not gap < 1e-9:
            raise AssertionError(f"CLEAN differs from K0 before the first injected stamp: {gap}")
        scored = nra.score_nra(data=data, site=site, corrections=corrections, rows=rows)
        pos = data.position[rows]
        real_event = k >= 0
        ki = np.where(real_event, k, 0)
        phase = phase_of(cutoff=cutoffs[rows], target=pos, onset=onset_pos[ki], end=end_pos[ki])
        meta = pl.DataFrame(
            {
                "row": rows,
                "event": np.where(real_event, np.array([int(e["id"]) for e in events])[ki], -1),
                "phase": np.where(real_event, phase, "ORDINARY"),
                "event_hours": np.where(real_event, ((end_pos - onset_pos) / 2.0)[ki], np.nan),
                "fraction": np.where(
                    real_event, np.array([e["fraction"] for e in events])[ki], np.nan
                ),
                "onset_month": months[np.clip(onset_pos[ki] - offset, 0, len(months) - 1)],
                "n_false": len(false_spans),
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
    parser.add_argument("stage", choices=["run", "one"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
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
