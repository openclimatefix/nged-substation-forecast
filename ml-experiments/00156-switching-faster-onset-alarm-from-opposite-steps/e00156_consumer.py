"""Stage 2: the mask forecast driven by RO-NC, the pair alarm, and their union, in the 44 worlds.

Every arm is GAM-K's member forecasts plus a Kalman-filter correction run on a stream. The streams
differ only in which readings are treated as missing: MASK-RONC masks the readings inside RO-NC's
intervals as they are known, MASK-NEW those of the pair alarm, and MASK-UNION those of either.
Scores are against the true normal-running-arrangement series (the stream without the injected MW).

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00156_consumer.py injected [--limit N]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Final

import e00156_alarm as alarm
import e00156_fed as fed
import e00156_nra as nra
import numpy as np
import polars as pl
from e00156_common import WORK_DIR

N_WORKERS: Final[int] = 4
WORLDS: Final[Path] = WORK_DIR.parent / "00141-switching-detector-fed-into-gam-k" / "worlds"
DETECT: Final[Path] = WORK_DIR / "detect"
OUT: Final[Path] = WORK_DIR / "scores"
OUT_POSTHOC: Final[Path] = WORK_DIR / "scores_posthoc"
POSTHOC_WINDOWS: Final[tuple[int, ...]] = (4, 12)
PERIODS: Final[tuple[str, ...]] = ("replicate", "scored")


def headline_window() -> int:
    """The window chosen on odd-month injected transfers, written by the stage 1 report."""
    return int(json.loads((DETECT / "headline.json").read_text())["window"])


def worlds_list() -> list[int]:
    """The seeds of experiment 141's stored worlds."""
    return sorted(int(p.stem.split("_")[1]) for p in WORLDS.glob("world_*.parquet"))


def load_world(*, seed: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """The MW added to each series by stamp, and the events."""
    with np.load(WORLDS / f"world_{seed}.npz") as stored:
        return {k: stored[k] for k in stored.files if k.startswith("delta_")}, stored["events"]


def site_events(*, delta: np.ndarray, events: np.ndarray) -> list[tuple[int, int]]:
    """The (onset, end) stamps of the events that move MW at this site."""
    out = []
    for row in events:
        onset, end = int(row[2]), int(row[3])
        if np.any(np.nan_to_num(delta[onset:end])):
            out.append((onset, end))
    return sorted(out)


def event_tags(
    *, data: Any, events: list[tuple[int, int]], families: dict[str, list], n: int
) -> dict:
    """Per-position event index and window kind, and when each alarm family knows each event."""
    offset = fed.stamp_offset(data)
    event = np.full(n, -1, dtype=np.int64)
    inside = np.zeros(n, dtype=bool)
    after = np.zeros(n, dtype=bool)
    onset_pos = np.array([on + offset for on, _ in events], dtype=np.int64)
    known = {name: np.full(len(events), np.inf) for name in families}
    for k, (on, en) in enumerate(events):
        lo, hi = on + offset, en + offset
        inside[lo:hi] = True
        after[hi : min(hi + nra.TAIL, n)] = True
        event[lo : min(hi + nra.TAIL, n)] = k
        for name, spans in families.items():
            matches = [s.start for s in spans if abs(s.onset - lo) <= nra.ONSET_TOLERANCE]
            if matches:
                known[name][k] = min(matches)
    return {
        "event": event,
        "inside": inside,
        "after": after & ~inside,
        "onset_pos": onset_pos,
        "known": known,
    }


def prefer_new(*, ronc: pl.DataFrame, new: pl.DataFrame) -> pl.DataFrame:
    """Post hoc union: the pair alarm's intervals, plus RO-NC's that start where it raised none.

    An RO-NC interval is dropped when a pair-alarm interval on the same series starts within 24
    hours of it (either side), so the pair alarm's onset and end are used where it fired.
    """
    ronc = alarm.with_ronc_onset_known(intervals=ronc)
    keep = ronc.filter(
        ~pl.struct("series", "onset").map_elements(
            lambda r: (
                new.filter(
                    (pl.col("series") == r["series"]) & ((pl.col("onset") - r["onset"]).abs() <= 48)
                ).height
                > 0
            ),
            return_dtype=pl.Boolean,
        )
    )
    return pl.concat([new, keep], how="diagonal_relaxed")


def injected_job(args: tuple[str, int, bool]) -> str:
    """Score every arm at one site in one world; `posthoc` scores the post hoc arms instead."""
    site, seed, posthoc = args
    path = (OUT_POSTHOC if posthoc else OUT) / f"{site}_{seed}.parquet"
    if path.exists():
        return str(path)
    window = headline_window()
    data = fed.get_site(site)
    delta, events = load_world(seed=seed)
    stamp_delta = delta[f"delta_{fed.SERIES_OF_SITE[site]}"]
    if not np.any(np.nan_to_num(stamp_delta)):
        return ""
    n = len(data.stream)
    points = fed.to_positions(
        array=100.0 * np.nan_to_num(stamp_delta) / fed.CAPACITY_MW[site], data=data
    )
    stream = np.where(np.isfinite(data.stream), data.stream + points, np.nan)
    evs = site_events(delta=stamp_delta, events=events)
    ronc = pl.read_parquet(WORLDS / f"world_{seed}.parquet")
    intervals = {
        "RONC": ronc,
        "NEW": pl.read_parquet(DETECT / f"world_{seed}_W{window}_pair.parquet"),
        "UNION": pl.read_parquet(DETECT / f"world_{seed}_W{window}_union.parquet"),
    }
    if posthoc:
        intervals = {
            "NEW_ALARM": intervals["NEW"].filter(pl.col("role") == "alarm"),
            "UNION2": prefer_new(ronc=ronc, new=intervals["NEW"]),
            "UNION2_MATCHED": prefer_new(
                ronc=ronc.filter(
                    pl.struct("series", "onset").map_elements(
                        lambda r: any(abs(r["onset"] - on) <= nra.ONSET_TOLERANCE for on, _ in evs),
                        return_dtype=pl.Boolean,
                    )
                ),
                new=intervals["NEW"],
            ),
            **{
                f"NEW_W{w}": pl.read_parquet(DETECT / f"world_{seed}_W{w}_pair.parquet")
                for w in POSTHOC_WINDOWS
            },
        }
    spans = {
        name: nra.detected_spans(data=data, site=site, intervals=table)
        for name, table in intervals.items()
    }
    tag_spans = {} if posthoc else spans
    tags = event_tags(data=data, events=evs, families=tag_spans, n=n)
    cutoffs = data.cutoffs(delay_hours=fed.DELAY_HOURS)[data.run_of_row]
    frames = []
    for period in PERIODS:
        rows = fed.rows_where(
            data=data, mask_by_position=tags["inside"] | tags["after"], period=period
        )
        if not len(rows):
            continue
        ev = tags["event"][data.position[rows]]
        rows, ev = rows[ev >= 0], ev[ev >= 0]
        keep = cutoffs[rows] >= tags["onset_pos"][ev]
        rows, ev = rows[keep], ev[keep]
        if not len(rows):
            continue
        corrections = {
            "K0": fed.correction_for(data=data, stream=stream, prior=np.zeros(n), multiplier=1.0),
            "CLEAN": fed.correction_for(
                data=data, stream=data.stream, prior=np.zeros(n), multiplier=1.0
            ),
            "GAM_ONLY": np.zeros(len(data.run_of_row)),
        }
        families = (
            (
                ("MASK_NEW_ALARM", "NEW_ALARM"),
                ("MASK_UNION2", "UNION2"),
                ("MASK_UNION2_MATCHED", "UNION2_MATCHED"),
                *((f"MASK_NEW_W{w}", f"NEW_W{w}") for w in POSTHOC_WINDOWS),
            )
            if posthoc
            else (("MASK_RONC", "RONC"), ("MASK_NEW", "NEW"), ("MASK_UNION", "UNION"))
        )
        for name, family in families:
            corrections[name] = nra.subtraction_correction(
                data=data, stream=stream, spans=spans[family], cutoffs=cutoffs, rows=rows, mask=True
            )
        scored = nra.score_nra(data=data, site=site, corrections=corrections, rows=rows)
        pos = data.position[rows]
        meta = pl.DataFrame(
            {
                "row": rows,
                "event": ev,
                "cutoff_minus_onset": cutoffs[rows] - tags["onset_pos"][ev],
                **(
                    {}
                    if posthoc
                    else {
                        "known_ronc": tags["known"]["RONC"][ev] - cutoffs[rows],
                        "known_new": tags["known"]["NEW"][ev] - cutoffs[rows],
                        "known_union": tags["known"]["UNION"][ev] - cutoffs[rows],
                    }
                ),
                "inside": tags["inside"][pos],
                "event_hours": np.array([(en - on) / 2 for on, en in evs])[ev],
            }
        )
        frames.append(
            scored.join(meta, on="row").with_columns(
                world=pl.lit(seed), period=pl.lit(period), window=pl.lit(window)
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
    parser.add_argument("--posthoc", action="store_true")
    args = parser.parse_args()
    (OUT_POSTHOC if args.posthoc else OUT).mkdir(parents=True, exist_ok=True)
    jobs = [(s, w, args.posthoc) for s in fed.SERIES_OF_SITE for w in worlds_list()]
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
