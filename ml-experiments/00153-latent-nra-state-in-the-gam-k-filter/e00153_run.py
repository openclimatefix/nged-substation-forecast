# ruff: noqa: ANN001, PLR0915
"""Score the NRA arms of experiment 153 at one site in one injected world, or on the real year.

Usage (inside `systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue`):

    uv run --no-sync python e00153_run.py injected [--limit N] | placebo | one
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import e00153_fed as fed
import e00153_nra as nra
import numpy as np
import polars as pl
from e00153_common import WORK_DIR

N_WORKERS: Final[int] = 4
WORLDS: Final[Path] = WORK_DIR.parent / "00141-switching-detector-fed-into-gam-k" / "worlds"
CHOSEN: Final[Path] = (
    WORK_DIR.parent / "00141-switching-detector-fed-into-gam-k" / "scores" / "chosen.json"
)
OUT: Final[Path] = WORK_DIR / "scores_v2"
PLACEBO_OUT: Final[Path] = WORK_DIR / "placebo_v2"
PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
SITES: Final[tuple[str, ...]] = ("D11", "D16")
"""The first test's pair: the two sites with the most injected transfers."""
N_WORLDS: Final[int] = 8
RATIOS: Final[tuple[float, ...]] = (10.0, 100.0, 1000.0, 10000.0)
PLACEBO_WEEKS: Final[tuple[int, ...]] = (-6, -3, 3, 6)


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
    spans_oracle = nra.oracle_spans(events=evs, data=data)
    spans_trueend = nra.true_end_spans(spans=spans_det, events=evs, data=data)
    spans_det96 = nra.detected_spans(data=data, site=site, intervals=intervals, end_lag=96)
    tags = window_tags(data=data, events=evs, spans=spans_det, n=n)
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
    mask = tags["inside"] | tags["after48"] | tags["after_det"]
    all_rows = [fed.rows_where(data=data, mask_by_position=mask, period=p) for p in PERIODS]
    rows = np.unique(np.concatenate(all_rows))
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
        return ""
    corrections = {
        "K0": fed.correction_for(data=data, stream=stream, prior=np.zeros(n), multiplier=1.0),
        "CLEAN": fed.correction_for(
            data=data, stream=data.stream, prior=np.zeros(n), multiplier=1.0
        ),
        "SKF": fed.correction_for(
            data=data, stream=stream, prior=signals.causal, multiplier=chosen["multiplier"]
        ),
    }
    corrections["GAM_ONLY"] = np.zeros(len(cutoffs))
    corrections["MASK"] = nra.mask_correction(
        data=data, stream=stream, spans=spans_det, cutoffs=cutoffs, rows=rows
    )
    corrections["MASK_ORACLE"] = nra.mask_correction(
        data=data, stream=stream, spans=spans_oracle, cutoffs=cutoffs, rows=rows
    )
    for ratio in RATIOS:
        for name, spans, cap in (
            ("NEW_DET", spans_det, nra.CAP),
            ("NEW_DET96", spans_det96, nra.CAP),
            ("NEW_DETNOCAP", spans_det, None),
            ("NEW_TRUEEND", spans_trueend, nra.CAP),
            ("NEW_ORACLE", spans_oracle, nra.CAP),
        ):
            corrections[f"{name}_{int(ratio)}"] = nra.latent_correction(
                data=data,
                stream=stream,
                spans=spans,
                cutoffs=cutoffs,
                rows=rows,
                ratio=ratio,
                cap=cap,
            )
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
    period_of = np.full(len(rows), "", dtype=object)
    for p, pr in zip(PERIODS, all_rows, strict=True):
        period_of[np.isin(rows, pr)] = p  # a row in two periods keeps the later one
    durations = np.array([(en - on) / 48.0 for on, en in evs])
    meta = pl.DataFrame(
        {
            "row": rows,
            "origin": origin,
            "event": ev,
            "inside": tags["inside"][pos],
            "after48": tags["after48"][pos],
            "after_det": tags["after_det"][pos],
            "missed": ~np.isfinite(tags["alarm_known"][ev]),
            "period": period_of.astype(str),
            "duration_days": durations[ev],
        }
    )
    out = scored.join(meta, on="row").with_columns(world=pl.lit(seed))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    out.write_parquet(tmp)
    tmp.rename(path)
    return str(path)


REAL_INTERVALS: Final[Path] = (
    WORK_DIR.parent / "00148-switching-end-rule-lookahead-ro-nc" / "run" / "intervals.parquet"
)


def placebo_job(args: tuple[str, int]) -> str:
    """Score K0, the masks, and the new filter on the real stream with the real alarms.

    No transfer is injected. The intervals are experiment 148's causal alarms with the 6-hour
    lookahead end rule on the recorded power (real transfers, bank holidays, flagged zeros, and
    false alarms alike). The score is against the recorded series, with the logged switching
    periods left out, so an arm that differs from K0 is moving the forecast away from the
    GAM-K level for no known reason.
    """
    site, _ = args
    path = PLACEBO_OUT / f"{site}_real.parquet"
    if path.exists():
        return str(path)
    data = fed.get_site(site)
    intervals = pl.read_parquet(REAL_INTERVALS)
    n = len(data.stream)
    spans = nra.detected_spans(data=data, site=site, intervals=intervals, setting="L6_M48")
    if not spans:
        return ""
    cutoffs = data.cutoffs(delay_hours=fed.DELAY_HOURS)[data.run_of_row]
    mask = np.zeros(n, dtype=bool)
    for s in spans:
        mask[s.start : min(s.known + nra.TAIL, n)] = True
    rows = np.unique(
        np.concatenate(
            [fed.rows_where(data=data, mask_by_position=mask, period=p) for p in PERIODS]
        )
    )
    rows = rows[cutoffs[rows] >= min(s.start for s in spans)]
    corrections = {
        "K0": fed.correction_for(data=data, stream=data.stream, prior=np.zeros(n), multiplier=1.0),
        "MASK": nra.mask_correction(
            data=data, stream=data.stream, spans=spans, cutoffs=cutoffs, rows=rows
        ),
        "GAM_ONLY": np.zeros(len(cutoffs)),
    }
    for ratio in RATIOS:
        corrections[f"NEW_DET_{int(ratio)}"] = nra.latent_correction(
            data=data, stream=data.stream, spans=spans, cutoffs=cutoffs, rows=rows, ratio=ratio
        )
    scored = nra.score_nra(data=data, site=site, corrections=corrections, rows=rows)
    pos = data.position[rows]
    in_open = np.zeros(n, dtype=bool)
    for s in spans:
        in_open[s.onset : s.known] = True
    out = scored.join(pl.DataFrame({"row": rows, "in_alarm": in_open[pos]}), on="row").with_columns(
        world=pl.lit(0)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    out.write_parquet(tmp)
    tmp.rename(path)
    return str(path)


def main() -> None:
    """Run one stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["injected", "placebo", "one"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    worlds = worlds_list()[:N_WORLDS]
    jobs = [(s, w) for s in SITES for w in worlds]
    if args.stage == "one":
        print(injected_job(jobs[0]))
        return
    if args.limit:
        jobs = jobs[: args.limit]
    job = placebo_job if args.stage == "placebo" else injected_job
    with ProcessPoolExecutor(N_WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        done = [p for p in pool.map(job, jobs) if p]
    print(f"{len(done)} files written")


if __name__ == "__main__":
    main()
