# ruff: noqa: E501, PLR0915
"""Pass 1: score K0, GAM_ONLY, PERFECT, PERFECT_ADD, DET, and DET_ADD, and store each row's features.

Every arm is GAM-K's cached member forecasts plus a Kalman filter correction run on the world's
stream (the real residual stream plus the injected megawatts), as in experiment 283, whose rows,
targets, and metrics this script keeps. GAM_ONLY has no correction and its own spread, fitted on
the residuals of the other month parity. DET sets aside the readings inside the RO-NC alarm
intervals that experiment 141 stored in the world. The `_ADD` arms subtract the known megawatts
from the stream and add them back to the forecast at the target time (post hoc, lab conditions).

Besides the scores, a per-row `features_<site>_<world>.parquet` stores what pass 2 fits on: the
K0 centre and correction, the correction at lead 0, its scaled level u, the as-of residual ratio,
and the calendar fields.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00297_score.py run [--limit N]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Final

import e00297_fed as fed
import e00297_injected as inj
import e00297_nra as nra
import numpy as np
import polars as pl
from e00297_common import WORK_DIR, stamp_months
from studies.dp_ens import LeadHourSpread

N_WORKERS: Final[int] = 4
OUT: Final[Path] = WORK_DIR / "scores"
FEATURES: Final[Path] = WORK_DIR / "features"
PERIODS: Final[tuple[str, ...]] = ("replicate", "scored")
MARGIN_STAMPS: Final[int] = 336
"""Seven days either side of a shift."""
ORDINARY_TAIL_STAMPS: Final[int] = 144
ORDINARY_RUN_STRIDE: Final[int] = 4
ASOF_WINDOW: Final[int] = 336
"""Seven days of half-hours."""
ASOF_MIN_COUNT: Final[int] = 48
ADD_ARMS: Final[tuple[str, ...]] = ("PERFECT_ADD", "DET_ADD")
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])


def qscore(*, actual: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Twice the mean pinball loss over the 11 levels, per row, in points."""
    error = actual[:, None] - q
    return 2.0 * np.mean(
        np.maximum(LEVELS[None, :] * error, (LEVELS[None, :] - 1.0) * error), axis=1
    )


def edges_of(*, data: fed.site_module.SiteData) -> dict[str, float]:
    """The 95th and 99th percentile of power (points) strictly before the first replicate row."""
    first = int(data.position[data.period_rows(period="replicate")].min())
    grid_actual = np.full(len(data.cache.end_times), np.nan)
    grid_actual[data.position] = data.actual
    history = grid_actual[:first]
    history = history[np.isfinite(history)]
    return {"95": float(np.quantile(history, 0.95)), "99": float(np.quantile(history, 0.99))}


def row_month(*, data: fed.site_module.SiteData, rows: np.ndarray) -> np.ndarray:
    """The calendar month (1 to 12) of each row's target."""
    return (data.month_code[data.position[rows]] % 100).astype(np.int64)


@lru_cache(maxsize=2)
def gam_only_spreads(site: str) -> dict[int, LeadHourSpread]:
    """GAM_ONLY's spread for each month parity of the scored rows.

    Key `p` holds the spread fitted on the residuals (actual minus the member mean) of rows whose
    target month has parity `1 - p`, so rows of parity `p` are scored out of sample.
    """
    data = fed.get_site(site)
    rows = np.flatnonzero(data.scorable)
    month = row_month(data=data, rows=rows)
    residual = data.actual[rows] - data.member_mean[rows]
    out = {}
    for parity in (0, 1):
        chosen = (month % 2) == (1 - parity)
        sel = rows[chosen]
        out[parity] = LeadHourSpread.fit(
            residuals=residual[chosen],
            band=data.hour_band[sel],
            quarter=data.quarter[sel],
            lead_hours=data.lead_hours[sel],
        )
    return out


def gam_only_quantiles(
    *, data: fed.site_module.SiteData, site: str, rows: np.ndarray
) -> np.ndarray:
    """GAM_ONLY's 11 quantiles at `rows`, each from the spread fitted on the other month parity."""
    spreads = gam_only_spreads(site)
    month = row_month(data=data, rows=rows)
    out = np.empty((len(rows), 11))
    zero = np.zeros(data.n_rows)
    for parity in (0, 1):
        chosen = np.flatnonzero((month % 2) == parity)
        if len(chosen):
            out[chosen] = fed.quantiles_at(
                data=data, correction=zero, spread=spreads[parity], rows=rows[chosen]
            )
    return out


@lru_cache(maxsize=2)
def site_scales(site: str) -> dict[str, float]:
    """The scale of K0's correction at lead 0 and of the stream, both on the real record.

    `c0_scale` is the robust standard deviation of the correction at the first row of each run
    (1.4826 times the median absolute value), so u = c0 / c0_scale is the filter level in units of
    its own history. `stream_rms` is the root mean square of the real stream in odd months.
    """
    data = fed.get_site(site)
    n = len(data.stream)
    correction = fed.correction_for(
        data=data, stream=data.stream, prior=np.zeros(n), multiplier=1.0
    )
    c0 = correction[data.offsets[:-1]]
    c0 = c0[np.isfinite(c0)]
    month = (data.month_code % 100).astype(np.int64)
    odd = data.stream[(month % 2) == 1]
    odd = odd[np.isfinite(odd)]
    return {
        "c0_scale": float(1.4826 * np.median(np.abs(c0))),
        "stream_rms": float(np.sqrt(np.mean(odd**2))),
    }


def asof_ratio_per_run(
    *, data: fed.site_module.SiteData, stream: np.ndarray, cutoffs_run: np.ndarray, rms: float
) -> np.ndarray:
    """Ratio of the last week's stream RMS known at each run's cutoff to the usual RMS, clipped 1 to 3."""
    finite = np.isfinite(stream)
    squares = np.cumsum(np.where(finite, stream**2, 0.0))
    counts = np.cumsum(finite.astype(np.int64))
    squares = np.concatenate([[0.0], squares])
    counts = np.concatenate([[0], counts])
    hi = np.clip(cutoffs_run + 1, 0, len(stream))
    lo = np.clip(hi - ASOF_WINDOW, 0, len(stream))
    count = counts[hi] - counts[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.sqrt((squares[hi] - squares[lo]) / np.maximum(count, 1)) / rms
    return np.where(count >= ASOF_MIN_COUNT, np.clip(ratio, 1.0, 3.0), 1.0)


def score_arms(
    *,
    arms: dict[str, tuple[np.ndarray, np.ndarray]],
    rows: np.ndarray,
    data: fed.site_module.SiteData,
    delta_points: np.ndarray,
    edges: dict[str, float],
) -> pl.DataFrame:
    """One row per (arm, row): scores against both targets, plus q01, q05, q50, q95, and q99.

    Args:
        arms: For each arm, its quantiles of shape (rows, 11) and its mean forecast.
        rows: The scored rows.
        data: The site data.
        delta_points: The injected points per grid position.
        edges: The 95th and 99th percentile of earlier power.

    Returns:
        The long score table.
    """
    actual_nra = data.actual[rows]
    actual_obs = actual_nra + delta_points[data.position[rows]]
    frames = []
    for arm, (q, mean_forecast) in arms.items():
        columns: dict[str, object] = {
            "arm": arm,
            "row": rows,
            "q99": q[:, 10],
            "q95": q[:, 9],
            "q50": q[:, 5],
            "q05": q[:, 1],
            "q01": q[:, 0],
        }
        for target, actual in (("nra", actual_nra), ("obs", actual_obs)):
            columns[f"qs_{target}"] = qscore(actual=actual, q=q)
            columns[f"ae_median_{target}"] = np.abs(q[:, 5] - actual)
            columns[f"ae_mean_{target}"] = np.abs(mean_forecast - actual)
            for label, edge in edges.items():
                columns[f"tw{label}_{target}"] = qscore(
                    actual=np.maximum(actual, edge), q=np.maximum(q, edge)
                )
        frames.append(pl.DataFrame(columns))
    return pl.concat(frames)


def job(args: tuple[str, int]) -> str:
    """Score the arms at one site in one world and write the result atomically."""
    site, seed = args
    path = OUT / f"{site}_{seed}.parquet"
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
    edges = edges_of(data=data)
    scales = site_scales(site)
    spread = fed.k0_spread(site)
    score_frames, feature_frames = [], []
    for period in PERIODS:
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
        k0 = fed.correction_for(data=data, stream=stream, prior=np.zeros(n), multiplier=1.0)
        corrections = {"K0": k0}
        back: dict[str, np.ndarray] = {}
        for name, spans in (("PERFECT", perfect), ("DET", detected)):
            corrections[name] = nra.subtraction_correction(
                data=data, stream=stream, spans=spans, cutoffs=cutoffs, rows=rows, mask=True
            )
            corrections[f"{name}_ADD"] = nra.subtraction_correction(
                data=data, stream=stream, spans=spans, cutoffs=cutoffs, rows=rows, mask=False
            )
            added = np.zeros(len(rows))
            for s in spans:
                sel = (pos_rows >= s.onset) & (pos_rows < s.end) & (cutoffs[rows] >= s.start)
                amount = s.amount[pos_rows[sel]] if isinstance(s.amount, np.ndarray) else s.amount
                added[sel] = amount
            back[f"{name}_ADD"] = added
        arms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, correction in corrections.items():
            q = fed.quantiles_at(data=data, correction=correction, spread=spread, rows=rows)
            shift_back = back.get(name, np.zeros(len(rows)))
            arms[name] = (
                q + shift_back[:, None],
                data.member_mean[rows] + correction[rows] + shift_back,
            )
        arms["GAM_ONLY"] = (
            gam_only_quantiles(data=data, site=site, rows=rows),
            data.member_mean[rows],
        )
        arms["GAM_ONLY_K0SPREAD"] = (
            fed.quantiles_at(data=data, correction=np.zeros(data.n_rows), spread=spread, rows=rows),
            data.member_mean[rows],
        )
        scored = score_arms(arms=arms, rows=rows, data=data, delta_points=points, edges=edges)
        k = near[pos_rows]
        has_event = k >= 0
        kk = np.where(has_event, k, 0)
        in_shift = (
            has_event
            & (pos_rows >= onset_pos[kk])
            & (pos_rows < end_pos[kk] + ORDINARY_TAIL_STAMPS)
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
        origin = data.run_origins[data.run_of_row[rows]]
        run_index = data.run_of_row[rows]
        c0_run = k0[data.offsets[:-1]]
        ratio_run = asof_ratio_per_run(
            data=data, stream=stream, cutoffs_run=cutoffs_run, rms=scales["stream_rms"]
        )
        meta = pl.DataFrame(
            {
                "row": rows,
                "position": pos_rows,
                "run": data.runs[run_index],
                "power_fcst_init_time": origin.astype("datetime64[us]"),
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
                "actual_nra": data.actual[rows],
                "actual_obs": data.actual[rows] + points[pos_rows],
            }
        )
        feature = pl.DataFrame(
            {
                "row": rows,
                "month": row_month(data=data, rows=rows).astype(np.int8),
                "hour_band": data.hour_band[rows],
                "quarter": data.quarter[rows],
                "gam_mean": data.member_mean[rows],
                "k0_mean": data.member_mean[rows] + k0[rows],
                "k0_correction": k0[rows],
                "c0": c0_run[run_index],
                "u": c0_run[run_index] / scales["c0_scale"],
                "asof_ratio": ratio_run[run_index],
                "run_index": run_index,
            }
        )
        score_frames.append(
            scored.join(meta, on="row").with_columns(
                site=pl.lit(site), world=pl.lit(seed, dtype=pl.Int32), period=pl.lit(period)
            )
        )
        feature_frames.append(
            feature.join(
                meta.select("row", "phase", "in_shift", "lead_hours"), on="row"
            ).with_columns(
                site=pl.lit(site), world=pl.lit(seed, dtype=pl.Int32), period=pl.lit(period)
            )
        )
    if not score_frames:
        return ""
    OUT.mkdir(parents=True, exist_ok=True)
    FEATURES.mkdir(parents=True, exist_ok=True)
    for frames, target in (
        (feature_frames, FEATURES / f"{site}_{seed}.parquet"),
        (score_frames, path),
    ):
        tmp = target.with_suffix(".tmp")
        pl.concat(frames).with_columns(pl.col(pl.Float64).cast(pl.Float32)).write_parquet(tmp)
        tmp.rename(target)
    print(f"{site} {seed}: {len(events)} events, {time.time() - started:.0f} s", flush=True)
    return str(path)


def site_constants() -> None:
    """Write each site's edges, effective capacity, earlier maximum flow, and scales."""
    out = {}
    for site in fed.SERIES_OF_SITE:
        data = fed.get_site(site)
        first = int(data.position[data.period_rows(period="replicate")].min())
        grid_actual = np.full(len(data.cache.end_times), np.nan)
        grid_actual[data.position] = data.actual
        history = grid_actual[:first]
        out[site] = {
            "edges": edges_of(data=data),
            "capacity_mw": fed.CAPACITY_MW[site],
            "max_before_points": float(np.nanmax(history)),
            "series": fed.SERIES_OF_SITE[site],
            **site_scales(site),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "site_constants.json").write_text(json.dumps(out, indent=2))


def main() -> None:
    """Run pass 1."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["run", "one", "constants"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "constants":
        site_constants()
        return
    jobs = [(s, w) for w in inj.worlds_list() for s in fed.SERIES_OF_SITE]
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
