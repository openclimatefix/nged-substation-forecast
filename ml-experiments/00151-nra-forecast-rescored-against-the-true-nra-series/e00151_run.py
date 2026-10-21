"""Score the NRA arms of experiment 151 at one site in one injected world, or on the real year.

Usage (inside `systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue`):

    uv run --no-sync python e00151_run.py injected [--limit N]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import e00151_fed as fed
import e00151_nra as nra
import numpy as np
import polars as pl
from e00151_common import WORK_DIR

N_WORKERS: Final[int] = 6
WORLDS: Final[Path] = WORK_DIR.parent / "00141-switching-detector-fed-into-gam-k" / "worlds"
CHOSEN: Final[Path] = (
    WORK_DIR.parent / "00141-switching-detector-fed-into-gam-k" / "scores" / "chosen.json"
)
OUT: Final[Path] = WORK_DIR / "scores_v2"
PERIODS: Final[tuple[str, ...]] = ("replicate", "scored")
FALSE_ALARMS: Final[Path] = (
    WORK_DIR.parent / "briefs" / "sw_false_alarms" / "work" / "alarms_flagged.parquet"
)
"""The unexplained RO-NC alarms (no logged closure, no labelled transfer) of the false-alarm
analysis: the real alarms the placebo arm subtracts."""
PLACEBO_WEEKS: Final[tuple[int, ...]] = (-6, -3, 3, 6)


def false_alarm_spans(*, spans: list[nra.Span], series: int, offset: int) -> list[nra.Span]:
    """The detected spans of a series whose onset matches an unexplained real alarm."""
    stamps = (
        pl.read_parquet(FALSE_ALARMS)
        .filter(pl.col("unexplained") & (pl.col("series") == series))["stamp"]
        .to_numpy()
    )
    return [s for s in spans if np.any(np.abs(stamps + offset - s.onset) <= nra.ONSET_TOLERANCE)]


def worlds_list() -> list[int]:
    """The seeds of experiment 141's stored worlds."""
    return sorted(int(p.stem.split("_")[1]) for p in WORLDS.glob("world_*.parquet"))


def load_world(*, seed: int) -> tuple[pl.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """A world's intervals, the MW added to each series by stamp, and its events."""
    intervals = pl.read_parquet(WORLDS / f"world_{seed}.parquet")
    with np.load(WORLDS / f"world_{seed}.npz") as stored:
        delta = {k: stored[k] for k in stored.files if k.startswith("delta_")}
        events = stored["events"]
    return intervals, delta, events


def site_events(*, delta: np.ndarray, events: np.ndarray) -> list[tuple[int, int]]:
    """The (onset, end) stamps of the events that move MW at this site."""
    out = []
    for row in events:
        onset, end = int(row[2]), int(row[3])
        if np.any(np.nan_to_num(delta[onset:end])):
            out.append((onset, end))
    return sorted(out)


def window_tags(
    *, data, events: list[tuple[int, int]], spans: list[nra.Span], n: int
) -> dict[str, np.ndarray]:
    """Per-position tags: the event index, the window kind, and when the alarm is known.

    `inside` is the true transfer, `after48` the 48 hours after its true end, and `after_det`
    the 48 hours after the detected end of a detected transfer. `alarm_known` is the stamp from
    which the transfer's alarm is known (infinity when the detector missed it).
    """
    offset = fed.stamp_offset(data)
    event = np.full(n, -1, dtype=np.int64)
    inside = np.zeros(n, dtype=bool)
    after48 = np.zeros(n, dtype=bool)
    after_det = np.zeros(n, dtype=bool)
    onset_pos = np.array([on + offset for on, _ in events], dtype=np.int64)
    alarm_known = np.full(len(events), np.inf)
    for k, (on, en) in enumerate(events):
        lo, hi = on + offset, en + offset
        inside[lo:hi] = True
        after48[hi : min(hi + nra.TAIL, n)] = True
        event[lo : min(hi + nra.TAIL, n)] = k
        matches = [s for s in spans if abs(s.onset - lo) <= nra.ONSET_TOLERANCE]
        if matches:
            first = min(matches, key=lambda s: s.start)
            alarm_known[k] = first.start
            ends = [s.end for s in matches]
            end_det = max(ends)
            span = slice(end_det, min(end_det + nra.TAIL, n))
            after_det[span] = True
            event[span] = np.where(event[span] < 0, k, event[span])
    return {
        "event": event,
        "inside": inside,
        "after48": after48 & ~inside,
        "after_det": after_det,
        "onset_pos": onset_pos,
        "alarm_known": alarm_known,
    }


def injected_job(args: tuple[str, int]) -> str:
    """Score every arm at one site in one world."""
    site, seed = args
    path = OUT / f"{site}_{seed}.parquet"
    if path.exists():
        return str(path)
    data = fed.get_site(site)
    intervals, delta, events = load_world(seed=seed)
    stamp_delta = delta[f"delta_{fed.SERIES_OF_SITE[site]}"]
    if not np.any(np.nan_to_num(stamp_delta)):
        return ""
    chosen = json.loads(CHOSEN.read_text())
    n = len(data.stream)
    points = fed.to_positions(
        array=100.0 * np.nan_to_num(stamp_delta) / fed.CAPACITY_MW[site], data=data
    )
    stream = np.where(np.isfinite(data.stream), data.stream + points, np.nan)
    evs = site_events(delta=stamp_delta, events=events)
    spans_det = nra.detected_spans(data=data, site=site, intervals=intervals)
    spans_det96 = nra.detected_spans(data=data, site=site, intervals=intervals, end_lag=96)
    spans_true = nra.true_spans(events=evs, delta_points=points, data=data)
    tags = window_tags(data=data, events=evs, spans=spans_det, n=n)
    # the detected intervals feed the two-regime filter as its causal signal
    r2 = intervals.filter(pl.col("setting") == "R0_M48").with_columns(
        setting=pl.lit("R2_M48"),
        known_from=pl.max_horizontal(
            pl.col("known_from"), pl.col("end") + nra.END_LAG, pl.col("onset") + nra.ONSET_LAG + 1
        ),
    )
    signals = fed.build_signals(
        data=data,
        site=site,
        intervals=pl.concat([r2, intervals.filter(pl.col("setting") == "R0_M48")]),
        delta_mw=stamp_delta,
        p_on=chosen["p_on"],
    )
    cutoffs = data.cutoffs(delay_hours=fed.DELAY_HOURS)[data.run_of_row]
    frames = []
    for period in PERIODS:
        mask = tags["inside"] | tags["after48"] | tags["after_det"]
        rows = fed.rows_where(data=data, mask_by_position=mask, period=period)
        if not len(rows):
            continue
        ev = tags["event"][data.position[rows]]
        rows, ev = rows[ev >= 0], ev[ev >= 0]
        c = cutoffs[rows]
        origin = np.where(
            c < tags["onset_pos"][ev],
            "pre_onset",
            np.where(c < tags["alarm_known"][ev], "lag", "post"),
        )
        keep = origin != "pre_onset"
        rows, ev, origin = rows[keep], ev[keep], origin[keep]
        if not len(rows):
            continue
        rng = np.random.default_rng(seed + 31 * sorted(fed.SERIES_OF_SITE).index(site))
        weeks = int(rng.choice(PLACEBO_WEEKS))
        corrections = {
            "K0": fed.correction_for(
                data=data, stream=stream, prior=np.zeros(n), multiplier=1.0
            ),
            "CLEAN": fed.correction_for(
                data=data, stream=data.stream, prior=np.zeros(n), multiplier=1.0
            ),
            "SKF": fed.correction_for(
                data=data, stream=stream, prior=signals.causal, multiplier=chosen["multiplier"]
            ),
            "SORACLE": fed.correction_for(
                data=data, stream=stream, prior=signals.oracle, multiplier=chosen["multiplier"]
            ),
            "SUB_DET": nra.subtraction_correction(
                data=data, stream=stream, spans=spans_det, cutoffs=cutoffs, rows=rows
            ),
            "GAM_ONLY": np.zeros(len(data.run_of_row)),
            "MASK": nra.subtraction_correction(
                data=data, stream=stream, spans=spans_det, cutoffs=cutoffs, rows=rows, mask=True
            ),
            "SUB_DET_LAG96": nra.subtraction_correction(
                data=data, stream=stream, spans=spans_det96, cutoffs=cutoffs, rows=rows
            ),
            "SUB_TRUE_LAG": nra.subtraction_correction(
                data=data, stream=stream, spans=spans_true, cutoffs=cutoffs, rows=rows
            ),
            "SUB_TRUE_CONST": nra.subtraction_correction(
                data=data,
                stream=stream,
                spans=nra.constant_true_spans(spans=spans_true, delta_points=points),
                cutoffs=cutoffs,
                rows=rows,
            ),
            "SUB_FALSE": nra.subtraction_correction(
                data=data,
                stream=stream,
                spans=false_alarm_spans(
                    spans=spans_det,
                    series=fed.SERIES_OF_SITE[site],
                    offset=fed.stamp_offset(data),
                ),
                cutoffs=cutoffs,
                rows=rows,
            ),
            "SUB_PLACEBO": nra.subtraction_correction(
                data=data,
                stream=stream,
                spans=nra.shifted(spans=spans_det, weeks=weeks, n=n),
                cutoffs=cutoffs,
                rows=rows,
            ),
        }
        first = int(np.flatnonzero(points)[0])
        early = np.flatnonzero(cutoffs < first)
        gap = (
            np.nanmax(np.abs(corrections["CLEAN"][early] - corrections["K0"][early]))
            if len(early)
            else 0.0
        )
        if not gap < 1e-9:
            raise AssertionError(f"CLEAN differs from K0 before the first injected stamp: {gap}")
        scored = nra.score_nra(data=data, site=site, corrections=corrections, rows=rows)
        pos = data.position[rows]
        meta = pl.DataFrame(
            {
                "row": rows,
                "origin": origin,
                "event": ev,
                "inside": tags["inside"][pos],
                "after48": tags["after48"][pos],
                "after_det": tags["after_det"][pos],
                "missed": ~np.isfinite(tags["alarm_known"][ev]),
                "event_hours": np.array([(en - on) / 2 for on, en in evs])[ev],
            }
        )
        frames.append(
            scored.join(meta, on="row").with_columns(
                world=pl.lit(seed), period=pl.lit(period), placebo_weeks=pl.lit(weeks)
            )
        )
    if not frames:
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(frames).write_parquet(path)
    return str(path)


def main() -> None:
    """Run the injected stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["injected", "one"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, w) for s in fed.SERIES_OF_SITE for w in worlds_list()]
    if args.stage == "one":
        print(injected_job(jobs[0]))
        return
    if args.limit:
        jobs = jobs[: args.limit]
    with ProcessPoolExecutor(N_WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        done = [p for p in pool.map(injected_job, jobs) if p]
    print(f"{len(done)} files written")


if __name__ == "__main__":
    main()
