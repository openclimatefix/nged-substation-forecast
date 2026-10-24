# ruff: noqa: E501
"""Score the label-free detector masks on experiment 297's injected worlds, against the true NRA.

A world is a Primary's real GAM-K residual stream plus the megawatts of injected transfers (the
worlds of experiment 141). Every arm shares experiment 297's cached ensemble-weather member
forecasts and K0 spread, so arms differ only in the readings the Kalman filter is allowed to see:
`K0` (no mask, arm C), `PERFECT` (the true spans set aside, known at the onset, arm D), `A` (the
12 hour mean normal score detector), and `B` (the 24 hour mean residual detector). Both detectors
read only the world's stream. Their thresholds and the normal-score function are fitted on the
site's real stream (no injection) over all odd-month half-hours, 5 alarms per site-year, and no
neighbour filter applies (the receivers of an injected transfer are unmetered).

    PYTHONPATH=ml-experiments/00297-lsc-need-a-detector-no-detector-arms-against-the-mask:ml-experiments/00309-lsc-two-forecast-gap-choosing-without-truth \
        systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00327_inj.py run
"""

from __future__ import annotations

import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist
from typing import Final

import e00297_fed as fed
import e00297_injected as inj
import e00297_nra as nra
import e00297_score as score
import numpy as np
import polars as pl
from e00297_common import stamp_months
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00327-lsc-label-free-detect-mask-train-pipeline"
OUT: Final[Path] = WORK / "inj"
STEP: Final[int] = 12
"""Half-hours between the detector's evaluation points (6 hours)."""
DELAY: Final[int] = 7
"""Half-hours between an evaluation point and the newest reading it may use."""
WIDTH: Final[dict[str, int]] = {"A": 24, "B": 48}
TARGET_RATE: Final[float] = 5.0
CLOSE_SHARE: Final[float] = 0.5
MAX_OPEN: Final[int] = 56
ONSET_BACK: Final[int] = 47
N_PROBS: Final[int] = 199
Z_LIMIT: Final[float] = 3.5
MATCH_EARLY: Final[int] = 4
N_WORKERS: Final[int] = 4


@lru_cache(maxsize=2)
def band_function(site: str) -> tuple[np.ndarray, np.ndarray]:
    """Normal-score lookup per hour band from the REAL stream's odd-month half-hours, no label filter."""
    data = fed.get_site(site)
    band = data.cache.calendar.band.astype(np.int64)
    ok = np.isfinite(data.stream) & (((data.month_code % 100) % 2) == 1)
    probs = np.arange(1, N_PROBS + 1) / (N_PROBS + 1)
    inverse = np.array([NormalDist().inv_cdf(p) for p in probs])
    table = np.full((int(band.max()) + 1, N_PROBS), np.nan)
    for b in range(table.shape[0]):
        table[b] = np.quantile(data.stream[ok & (band == b)], probs)
    return table, inverse


def base_series(*, site: str, stream: np.ndarray, kind: str) -> np.ndarray:
    """Per-position quantity the detector averages: the normal score (A) or the residual (B)."""
    if kind == "B":
        return stream
    data = fed.get_site(site)
    table, inverse = band_function(site)
    band = data.cache.calendar.band.astype(np.int64)
    out = np.full(len(stream), np.nan)
    for b in range(table.shape[0]):
        idx = (band == b) & np.isfinite(stream)
        out[idx] = np.interp(stream[idx], table[b], inverse)
    return np.clip(out, -Z_LIMIT, Z_LIMIT)


def window_signal(*, x: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluation positions and the mean of x over the `width` positions ending DELAY before each."""
    n = len(x)
    points = np.arange(1488, n, STEP)
    cutoffs = points - DELAY - 1
    valid = np.isfinite(x)
    csum = np.concatenate([[0.0], np.cumsum(np.where(valid, x, 0.0))])
    cnt = np.concatenate([[0], np.cumsum(valid)])
    hi = cutoffs + 1
    lo = np.maximum(hi - width, 0)
    k = cnt[hi] - cnt[lo]
    return points, np.where(k >= width / 2, (csum[hi] - csum[lo]) / np.maximum(k, 1), np.nan)


def machine(*, score_: np.ndarray, threshold: float) -> list[tuple[int, int, int]]:
    """Alarm episodes (open index, first low index, closing index or -1), as in `e00327_detect`."""
    out, i, n = [], 0, len(score_)
    while i < n:
        s = score_[i]
        if np.isfinite(s) and abs(s) > threshold:
            sign, low, first_low, closed, j = np.sign(s), 0, -1, -1, i + 1
            while j < n and j - i <= MAX_OPEN:
                v = score_[j]
                if np.isfinite(v) and sign * v < CLOSE_SHARE * threshold:
                    first_low = j if low == 0 else first_low
                    low += 1
                    if low >= 2:
                        closed = j
                        break
                else:
                    low = 0
                j += 1
            out.append((i, first_low if closed >= 0 else min(i + MAX_OPEN, n - 1), closed))
            i = (closed if closed >= 0 else j) + 1
        else:
            i += 1
    return out


@lru_cache(maxsize=16)
def site_threshold(site: str, kind: str) -> float:
    """Smallest threshold giving `TARGET_RATE` odd-month alarms per year on the site's real stream."""
    data = fed.get_site(site)
    points, sig = window_signal(
        x=base_series(site=site, stream=data.stream, kind=kind), width=WIDTH[kind]
    )
    odd = ((data.month_code[points] % 100) % 2) == 1
    years = float((odd & np.isfinite(sig)).sum()) * STEP / 48.0 / 365.25
    lo, hi = 0.0, float(np.nanmax(np.abs(sig)))
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        rate = sum(1 for e in machine(score_=sig, threshold=mid) if odd[e[0]]) / years
        lo, hi = (mid, hi) if rate > TARGET_RATE else (lo, mid)
    return hi


def detected(*, site: str, stream: np.ndarray, kind: str) -> list[nra.Span]:
    """Mask spans a detector raises on one world's stream."""
    points, sig = window_signal(
        x=base_series(site=site, stream=stream, kind=kind), width=WIDTH[kind]
    )
    thr = site_threshold(site, kind)
    spans = []
    for o, first_low, closed in machine(score_=sig, threshold=thr):
        c_open = int(points[o] - DELAY - 1)
        onset = max(c_open - ONSET_BACK, 0)
        end = int(points[first_low] - DELAY - 1)
        known = int(
            points[closed if closed >= 0 else min(o + MAX_OPEN + 1, len(points) - 1)] - DELAY - 1
        )
        end = max(end, onset + 1)
        spans.append(nra.Span(onset, end, c_open, max(known, end + 1), 0.0))
    return spans


def job(args: tuple[str, int]) -> str:  # noqa: PLR0915
    """Score the arms at one site in one world and write the result atomically."""
    site, seed = args
    path = OUT / f"{site}_{seed}.parquet"
    if path.exists():
        return str(path)
    started = time.time()
    data = fed.get_site(site)
    _, delta, events_all = inj.load_world(seed=seed)
    stamp_delta = delta[f"delta_{fed.SERIES_OF_SITE[site]}"]
    if not np.any(np.nan_to_num(stamp_delta)):
        return ""
    n = len(data.stream)
    offset = fed.stamp_offset(data)
    points = fed.to_positions(
        array=100.0 * np.nan_to_num(stamp_delta) / fed.CAPACITY_MW[site], data=data
    )
    stream = np.where(np.isfinite(data.stream), data.stream + points, np.nan)
    events = inj.site_events(delta=stamp_delta, events=events_all)
    perfect = inj.perfect_spans(events=events, points=points, data=data)
    sources = {
        "PERFECT": perfect,
        "A": detected(site=site, stream=stream, kind="A"),
        "B": detected(site=site, stream=stream, kind="B"),
    }
    cutoffs_run = data.cutoffs(delay_hours=fed.DELAY_HOURS)
    cutoffs = cutoffs_run[data.run_of_row]
    months = stamp_months()
    onset_pos = np.array([int(e["onset"]) + offset for e in events])
    end_pos = np.array([int(e["end"]) + offset for e in events])
    near = np.full(n, -1, dtype=np.int64)
    shift = np.zeros(n, dtype=bool)
    for k in range(len(events)):
        lo, hi = (
            max(onset_pos[k] - score.MARGIN_STAMPS, 0),
            min(end_pos[k] + score.MARGIN_STAMPS, n),
        )
        near[lo:hi] = k
        shift[onset_pos[k] : min(end_pos[k] + score.ORDINARY_TAIL_STAMPS, n)] = True
    edges = score.edges_of(data=data)
    spread = fed.k0_spread(site)
    frames = []
    for period in score.PERIODS:
        near_rows = fed.rows_where(data=data, mask_by_position=near >= 0, period=period)
        quiet = fed.rows_where(data=data, mask_by_position=~shift, period=period)
        quiet = quiet[
            (data.run_of_row[quiet] % score.ORDINARY_RUN_STRIDE == 0)
            & (near[data.position[quiet]] < 0)
        ]
        quiet = quiet[~shift[cutoffs[quiet].clip(0, n - 1)]]
        rows = np.unique(np.concatenate([near_rows, quiet]))
        if not len(rows):
            continue
        pos_rows = data.position[rows]
        corrections = {
            "K0": fed.correction_for(data=data, stream=stream, prior=np.zeros(n), multiplier=1.0)
        }
        for name, spans in sources.items():
            corrections[name] = nra.subtraction_correction(
                data=data, stream=stream, spans=spans, cutoffs=cutoffs, rows=rows, mask=True
            )
        arms = {
            name: (
                fed.quantiles_at(data=data, correction=c, spread=spread, rows=rows),
                data.member_mean[rows] + c[rows],
            )
            for name, c in corrections.items()
        }
        scored = score.score_arms(arms=arms, rows=rows, data=data, delta_points=points, edges=edges)
        k = near[pos_rows]
        has_event = k >= 0
        kk = np.where(has_event, k, 0)
        in_shift = (
            has_event
            & (pos_rows >= onset_pos[kk])
            & (pos_rows < end_pos[kk] + score.ORDINARY_TAIL_STAMPS)
        )
        phase = np.where(
            has_event
            & (cutoffs[rows] >= onset_pos[kk])
            & (pos_rows >= onset_pos[kk])
            & (pos_rows < end_pos[kk] + inj.AFTER_STAMPS),
            inj.phase_of(
                cutoff=cutoffs[rows], target=pos_rows, onset=onset_pos[kk], end=end_pos[kk]
            ),
            np.where(has_event, "MARGIN", "ORDINARY"),
        )
        meta = pl.DataFrame(
            {
                "row": rows,
                "lead_hours": data.lead_hours[rows],
                "event": np.where(has_event, np.array([int(e["id"]) for e in events])[kk], -1),
                "phase": phase,
                "in_shift": in_shift,
                "event_hours": np.where(has_event, ((end_pos - onset_pos) / 2.0)[kk], np.nan),
                "fraction": np.where(
                    has_event, np.array([e["fraction"] for e in events])[kk], np.nan
                ),
                "onset_month": np.where(
                    has_event, months[np.clip(onset_pos[kk] - offset, 0, len(months) - 1)], 0
                ).astype(np.int8),
                "month": score.row_month(data=data, rows=rows).astype(np.int8),
                "actual_nra": data.actual[rows],
                "actual_obs": data.actual[rows] + points[pos_rows],
            }
        )
        keep = [
            c
            for c in scored.columns
            if c in ("arm", "row")
            or c.endswith(("_nra", "_obs"))
            or c in ("q01", "q05", "q50", "q95", "q99")
        ]
        frames.append(
            scored.select(keep)
            .join(meta, on="row")
            .with_columns(
                site=pl.lit(site), world=pl.lit(seed, dtype=pl.Int32), period=pl.lit(period)
            )
        )
    if not frames:
        return ""
    alarm_rows = []
    for name in ("A", "B"):
        for s in sources[name]:
            hit = [
                e["id"]
                for e, o, en in zip(events, onset_pos, end_pos, strict=True)
                if o - MATCH_EARLY <= s.start <= en
            ]
            alarm_rows.append(
                {
                    "arm": name,
                    "site": site,
                    "world": seed,
                    "start": s.start,
                    "onset": s.onset,
                    "end": s.end,
                    "matched": bool(hit),
                }
            )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "alarms").mkdir(exist_ok=True)
    pl.DataFrame(
        alarm_rows
        or {"arm": [], "site": [], "world": [], "start": [], "onset": [], "end": [], "matched": []},
        schema={
            "arm": pl.String,
            "site": pl.String,
            "world": pl.Int64,
            "start": pl.Int64,
            "onset": pl.Int64,
            "end": pl.Int64,
            "matched": pl.Boolean,
        },
    ).write_parquet(OUT / "alarms" / f"{site}_{seed}.parquet")
    pl.DataFrame(
        {
            "event": [int(e["id"]) for e in events],
            "onset": onset_pos,
            "end": end_pos,
            "fraction": [e["fraction"] for e in events],
            "A": [
                any(s.start >= o - MATCH_EARLY and s.start <= en for s in sources["A"])
                for o, en in zip(onset_pos, end_pos, strict=True)
            ],
            "B": [
                any(s.start >= o - MATCH_EARLY and s.start <= en for s in sources["B"])
                for o, en in zip(onset_pos, end_pos, strict=True)
            ],
        }
    ).with_columns(site=pl.lit(site), world=pl.lit(seed, dtype=pl.Int64)).write_parquet(
        OUT / "alarms" / f"events_{site}_{seed}.parquet"
    )
    tmp = path.with_suffix(".tmp")
    pl.concat(frames).with_columns(pl.col(pl.Float64).cast(pl.Float32)).write_parquet(tmp)
    tmp.rename(path)
    print(f"{site} {seed}: {len(events)} events, {time.time() - started:.0f} s", flush=True)
    return str(path)


def main() -> None:
    """Run all worlds and sites in spawned workers."""
    jobs = [(s, w) for w in inj.worlds_list() for s in fed.SERIES_OF_SITE]
    if len(sys.argv) > 2:
        jobs = jobs[: int(sys.argv[2])]
    with ProcessPoolExecutor(N_WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        done = [p for p in pool.map(job, jobs) if p]
    print(len(done), "files")


if __name__ == "__main__":
    main()
