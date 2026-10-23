# ruff: noqa: E501, PLR0915
"""Score the degraded 'true megawatts added back' arms, and store the extra per-row inputs.

Rows, targets, and scoring are experiment 297's pass 1 (`e00297_score.py`). The arm is
experiment 297's PERFECT_ADD: the true megawatts of each known shift are subtracted from the
readings the Kalman filter sees, and added back to the forecast at the target. Each degraded arm
changes one property of the known spans (recall, onset delay, megawatt error, false alarms), with
the other properties perfect. Per row this script also stores what the new heads read: band
position, the perfect-knowledge indicators, and the added-back megawatts. Per world and
substation it stores the events, the RO-NC alarm spans, and the band-position series per run.

    PYTHONPATH=ml-experiments/00297-lsc-need-a-detector-no-detector-arms-against-the-mask \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00312_degraded.py run [--limit N]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from statistics import NormalDist

import e00297_fed as fed
import e00297_injected as inj
import e00297_nra as nra
import e00297_score as score
import numpy as np
import polars as pl
from e00297_common import stamp_months
from e00312_common import (
    BAND_DIR,
    BIASES,
    BLURS,
    DEG_DIR,
    DELAYS,
    EVENTS_DIR,
    EXTRA_DIR,
    FALSE_PER_TRUE,
    N_PROBS,
    PRECISIONS,
    RECALLS,
    TAIL_STAMPS,
    WIDTH,
    Z_LIMIT,
)

N_WORKERS = 4
MARGIN_STAMPS = score.MARGIN_STAMPS
ORDINARY_TAIL_STAMPS = score.ORDINARY_TAIL_STAMPS
ORDINARY_RUN_STRIDE = score.ORDINARY_RUN_STRIDE


@lru_cache(maxsize=2)
def band_function(site: str) -> tuple[np.ndarray, np.ndarray]:
    """The normal-score lookup of band position at a site.

    Returns the empirical quantiles of the real residual stream per hour band (shape (bands,
    N_PROBS)), fitted on odd-month half-hours outside the logged masks, and the matching normal
    quantiles.
    """
    data = fed.get_site(site)
    band = data.cache.calendar.band.astype(np.int64)
    month = data.month_code % 100
    ok = np.isfinite(data.stream) & ((month % 2) == 1) & ~data.features.switching_mask
    probs = np.arange(1, N_PROBS + 1) / (N_PROBS + 1)
    inverse = np.array([NormalDist().inv_cdf(p) for p in probs])
    n_bands = int(band.max()) + 1
    table = np.full((n_bands, N_PROBS), np.nan)
    for b in range(n_bands):
        sample = data.stream[ok & (band == b)]
        if len(sample) >= 50:
            table[b] = np.quantile(sample, probs)
    return table, inverse


def normal_scores(*, site: str, stream: np.ndarray) -> np.ndarray:
    """Normal score (clipped to +/-3.5) of each grid half-hour's stream value in its hour band."""
    data = fed.get_site(site)
    table, inverse = band_function(site)
    band = data.cache.calendar.band.astype(np.int64)
    out = np.full(len(stream), np.nan)
    for b in range(table.shape[0]):
        idx = (band == b) & np.isfinite(stream)
        if idx.any() and np.isfinite(table[b]).all():
            out[idx] = np.interp(stream[idx], table[b], inverse)
    return np.clip(out, -Z_LIMIT, Z_LIMIT)


def window_means(*, x: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    """Mean of x over the WIDTH half-hours ending at each cutoff; NaN under half valid."""
    valid = np.isfinite(x)
    csum = np.concatenate([[0.0], np.cumsum(np.where(valid, x, 0.0))])
    cnt = np.concatenate([[0], np.cumsum(valid)])
    hi = np.clip(cutoffs + 1, 0, len(x))
    lo = np.maximum(hi - WIDTH, 0)
    n = cnt[hi] - cnt[lo]
    return np.where(n >= WIDTH / 2, (csum[hi] - csum[lo]) / np.maximum(n, 1), np.nan)


def false_spans(
    *,
    rng: np.random.Generator,
    n_false: int,
    perfect: list[nra.Span],
    first: int,
    n: int,
) -> list[nra.Span]:
    """False spans: random onset outside every true shift and its tail, true-like length and size.

    The length and the mean absolute megawatts come from randomly chosen true shifts at the site;
    the sign is random. A false span is known at its onset and its end at its end.
    """
    if not perfect or n_false <= 0:
        return []
    lengths = np.array([s.end - s.onset for s in perfect])
    sizes = np.array([float(np.nanmean(np.abs(s.amount[s.onset : s.end]))) for s in perfect])
    busy = np.zeros(n, dtype=bool)
    for s in perfect:
        busy[max(s.onset - 1, 0) : min(s.end + TAIL_STAMPS, n)] = True
    out: list[nra.Span] = []
    tries = 0
    while len(out) < n_false and tries < 50 * n_false:
        tries += 1
        k = int(rng.integers(0, len(perfect)))
        length = int(lengths[k])
        onset = int(rng.integers(first, n - length - 1))
        if busy[onset : onset + length + TAIL_STAMPS].any():
            continue
        amount = float(sizes[int(rng.integers(0, len(perfect)))]) * float(rng.choice([-1.0, 1.0]))
        busy[onset : onset + length + TAIL_STAMPS] = True
        out.append(nra.Span(onset, onset + length, onset, onset + length, amount))
    return out


def count_false(*, precision: float, n_true: int) -> int:
    """Number of false spans giving `precision` against `n_true` true spans."""
    return round(FALSE_PER_TRUE[precision] * n_true)


def build_sources(
    *,
    data: fed.site_module.SiteData,
    events: list[dict[str, float]],
    points: np.ndarray,
    perfect: list[nra.Span],
    rng_seed: int,
    first: int,
    n: int,
) -> dict[str, list[nra.Span]]:
    """The spans of every degraded arm."""
    out: dict[str, list[nra.Span]] = {"ADD_PERFECT": perfect}
    for recall in RECALLS:
        rng = np.random.default_rng([rng_seed, int(recall * 100)])
        kept = rng.random(len(events)) < recall
        out[f"ADD_R{int(recall * 100)}"] = [s for s, k in zip(perfect, kept, strict=True) if k]
    for delay in DELAYS:
        out[f"ADD_D{delay}"] = inj.perfect_spans(
            events=events, points=points, data=data, delay=delay
        )
    for blur in BLURS:
        rng = np.random.default_rng([rng_seed, 7, int(blur * 100)])
        signs = rng.choice([-1.0, 1.0], size=len(events))
        out[f"ADD_B{int(blur * 100)}"] = [
            inj.with_amount(span=s, factor=1.0 + blur * g)
            for s, g in zip(perfect, signs, strict=True)
        ]
    for name, factor in BIASES:
        out[f"ADD_{name}"] = [inj.with_amount(span=s, factor=factor) for s in perfect]
    for precision in PRECISIONS:
        rng = np.random.default_rng([rng_seed, 13, int(precision * 100)])
        out[f"ADD_P{int(precision * 100)}"] = list(perfect) + false_spans(
            rng=rng,
            n_false=count_false(precision=precision, n_true=len(perfect)),
            perfect=perfect,
            first=first,
            n=n,
        )
    # post hoc mix: recall 0.5, delay 12, blur 25%, precision 0.75
    rng = np.random.default_rng([rng_seed, 17])
    delayed = inj.perfect_spans(events=events, points=points, data=data, delay=12)
    kept = rng.random(len(events)) < 0.5
    signs = rng.choice([-1.0, 1.0], size=len(events))
    spans = [
        inj.with_amount(span=s, factor=1.0 + 0.25 * g)
        for s, k, g in zip(delayed, kept, signs, strict=True)
        if k
    ]
    out["ADD_MIX"] = spans + false_spans(
        rng=rng,
        n_false=count_false(precision=0.75, n_true=len(spans)),
        perfect=perfect,
        first=first,
        n=n,
    )
    return out


def added_back(
    *, spans: list[nra.Span], rows: np.ndarray, pos_rows: np.ndarray, cutoffs: np.ndarray
) -> np.ndarray:
    """The points added back at each row's target: spans known at its cutoff and covering its target."""
    added = np.zeros(len(rows))
    for s in spans:
        sel = (pos_rows >= s.onset) & (pos_rows < s.end) & (cutoffs[rows] >= s.start)
        amount = s.amount[pos_rows[sel]] if isinstance(s.amount, np.ndarray) else s.amount
        added[sel] = amount
    return added


def indicators(
    *, perfect: list[nra.Span], rows: np.ndarray, pos_rows: np.ndarray, cutoffs: np.ndarray
) -> dict[str, np.ndarray]:
    """The perfect-knowledge indicators of each row.

    `ind_active`: a shift is known to be on at the origin. `ind_recent`: a shift ended within 72
    hours before the origin. `ind_target`: the target lies inside a true shift known at the origin.
    """
    cut = cutoffs[rows]
    active = np.zeros(len(rows))
    recent = np.zeros(len(rows))
    target = np.zeros(len(rows))
    for s in perfect:
        active[(cut >= s.start) & (cut < s.known)] = 1.0
        recent[(cut >= s.known) & (cut < s.known + TAIL_STAMPS)] = 1.0
        target[(cut >= s.start) & (pos_rows >= s.onset) & (pos_rows < s.end)] = 1.0
    return {"ind_active": active, "ind_recent": recent, "ind_target": target}


def job(args: tuple[str, int]) -> str:
    """Score the degraded arms at one site in one world; write scores, extras, events, band."""
    site, seed = args
    path = DEG_DIR / f"{site}_{seed}.parquet"
    if path.exists():
        return str(path)
    started = time.time()
    data = fed.get_site(site)
    intervals, delta, events_all = inj.load_world(seed=seed)
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
    detected = nra.detected_spans(data=data, site=site, intervals=intervals)
    cutoffs_run = data.cutoffs(delay_hours=fed.DELAY_HOURS)
    cutoffs = cutoffs_run[data.run_of_row]
    months = stamp_months()
    onset_pos = np.array([int(e["onset"]) + offset for e in events])
    end_pos = np.array([int(e["end"]) + offset for e in events])
    near = np.full(n, -1, dtype=np.int64)
    shift = np.zeros(n, dtype=bool)
    for k in range(len(events)):
        lo, hi = max(onset_pos[k] - MARGIN_STAMPS, 0), min(end_pos[k] + MARGIN_STAMPS, n)
        near[lo:hi] = k
        shift[onset_pos[k] : min(end_pos[k] + ORDINARY_TAIL_STAMPS, n)] = True
    edges = score.edges_of(data=data)
    spread = fed.k0_spread(site)
    site_index = sorted(fed.SERIES_OF_SITE).index(site)
    first = int(data.position[data.period_rows(period="replicate")].min())
    sources = build_sources(
        data=data,
        events=events,
        points=points,
        perfect=perfect,
        rng_seed=seed * 10 + site_index,
        first=first,
        n=n,
    )
    pit_grid = normal_scores(site=site, stream=stream)
    pit_run = window_means(x=pit_grid, cutoffs=cutoffs_run)
    ext_run = window_means(x=np.abs(pit_grid), cutoffs=cutoffs_run)
    score_frames, extra_frames = [], []
    for period in score.PERIODS:
        near_rows = fed.rows_where(data=data, mask_by_position=near >= 0, period=period)
        quiet = fed.rows_where(data=data, mask_by_position=~shift, period=period)
        quiet = quiet[
            (data.run_of_row[quiet] % ORDINARY_RUN_STRIDE == 0) & (near[data.position[quiet]] < 0)
        ]
        quiet = quiet[~shift[cutoffs[quiet].clip(0, n - 1)]]
        rows = np.unique(np.concatenate([near_rows, quiet]))
        if not len(rows):
            continue
        pos_rows = data.position[rows]
        arms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, spans in sources.items():
            correction = nra.subtraction_correction(
                data=data, stream=stream, spans=spans, cutoffs=cutoffs, rows=rows, mask=False
            )
            back = added_back(spans=spans, rows=rows, pos_rows=pos_rows, cutoffs=cutoffs)
            q = fed.quantiles_at(data=data, correction=correction, spread=spread, rows=rows)
            arms[name] = (q + back[:, None], data.member_mean[rows] + correction[rows] + back)
        scored = score.score_arms(arms=arms, rows=rows, data=data, delta_points=points, edges=edges)
        keep = [
            c
            for c in scored.columns
            if c in ("arm", "row") or c.startswith(("qs_obs", "tw95_obs", "q0", "q5", "q9", "ae_"))
        ]
        score_frames.append(
            scored.select(keep).with_columns(
                site=pl.lit(site), world=pl.lit(seed, dtype=pl.Int32), period=pl.lit(period)
            )
        )
        run_index = data.run_of_row[rows]
        extras = {
            "row": rows,
            "added_mw": added_back(spans=perfect, rows=rows, pos_rows=pos_rows, cutoffs=cutoffs),
            "pit24": pit_run[run_index],
            "ext24": ext_run[run_index],
            **indicators(perfect=perfect, rows=rows, pos_rows=pos_rows, cutoffs=cutoffs),
        }
        extra_frames.append(
            pl.DataFrame(extras).with_columns(
                site=pl.lit(site), world=pl.lit(seed, dtype=pl.Int32), period=pl.lit(period)
            )
        )
    if not score_frames:
        return ""
    for d in (DEG_DIR, EXTRA_DIR, BAND_DIR, EVENTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    band_frame = pl.DataFrame(
        {
            "run_index": np.arange(len(cutoffs_run)),
            "cutoff": cutoffs_run,
            "month": months[np.clip(cutoffs_run - offset, 0, len(months) - 1)].astype(np.int8),
            "pit24": pit_run,
            "ext24": ext_run,
            "usable": np.array([p in ("replicate", "scored") for p in data.run_period]),
        }
    ).with_columns(site=pl.lit(site), world=pl.lit(seed, dtype=pl.Int32))
    event_frame = pl.DataFrame(
        {
            "id": [int(e["id"]) for e in events],
            "onset": onset_pos,
            "end": end_pos,
            "fraction": [e["fraction"] for e in events],
            "onset_month": [
                int(months[np.clip(o - offset, 0, len(months) - 1)]) for o in onset_pos
            ],
            "ronc_matches": [
                sum(inj.matches(span=s, event=e, offset=offset) for s in detected) for e in events
            ],
            "ronc_delay": [
                min(
                    (
                        s.start - (int(e["onset"]) + offset)
                        for s in detected
                        if inj.matches(span=s, event=e, offset=offset)
                    ),
                    default=-1,
                )
                for e in events
            ],
        }
    ).with_columns(site=pl.lit(site), world=pl.lit(seed, dtype=pl.Int32))
    detected_frame = pl.DataFrame(
        {
            "onset": [s.onset for s in detected],
            "start": [s.start for s in detected],
            "end": [s.end for s in detected],
            "month": [int(months[np.clip(s.onset - offset, 0, len(months) - 1)]) for s in detected],
            "matched": [
                any(inj.matches(span=s, event=e, offset=offset) for e in events) for s in detected
            ],
        },
        schema={
            "onset": pl.Int64,
            "start": pl.Int64,
            "end": pl.Int64,
            "month": pl.Int64,
            "matched": pl.Boolean,
        },
    ).with_columns(site=pl.lit(site), world=pl.lit(seed, dtype=pl.Int32))
    for frame, directory, name in (
        (pl.concat(extra_frames), EXTRA_DIR, f"{site}_{seed}.parquet"),
        (band_frame, BAND_DIR, f"{site}_{seed}.parquet"),
        (event_frame, EVENTS_DIR, f"events_{site}_{seed}.parquet"),
        (detected_frame, EVENTS_DIR, f"ronc_{site}_{seed}.parquet"),
        (pl.concat(score_frames), DEG_DIR, f"{site}_{seed}.parquet"),
    ):
        tmp = directory / (name + ".tmp")
        frame.with_columns(
            pl.col(pl.Float64).cast(pl.Float32)
            if directory in (DEG_DIR, EXTRA_DIR)
            else pl.col(pl.Float64)
        ).write_parquet(tmp)
        tmp.rename(directory / name)
    print(
        f"{site} {seed}: {len(events)} events, {len(sources)} arms, {time.time() - started:.0f} s",
        flush=True,
    )
    return str(path)


def main() -> None:
    """Run the stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["run", "one"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--site", default="D5")
    args = parser.parse_args()
    jobs = [(s, w) for w in inj.worlds_list() for s in fed.SERIES_OF_SITE]
    if args.stage == "one":
        print(job((args.site, inj.worlds_list()[0])))
        return
    if args.limit:
        jobs = jobs[: args.limit]
    with ProcessPoolExecutor(N_WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        done = [p for p in pool.map(job, jobs) if p]
    print(json.dumps({"files": len(done)}))


if __name__ == "__main__":
    main()
