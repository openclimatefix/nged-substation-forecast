# ruff: noqa: PLR0915
"""Score K0, PERFECT and PERFECT_LAG at one substation in one injected world, on both targets.

Every arm is GAM-K's cached member forecasts plus a Kalman filter correction run on the world's
stream (the real residual stream plus the injected megawatts), with experiment 115's K0 spread, as
in experiment 208. K0 sets nothing aside. PERFECT sets the readings inside each true span aside
(the MASK design) from the true onset, with the end known at once. PERFECT_LAG does the same after
a live service's delays (onset 96 half-hours, end 60 half-hours after the end).

Each row is scored against two targets: the true normal-arrangement (NRA) series, which is the real
series, and observed power, which is the real series plus the injected megawatts. Rows are the
rows whose target lies within 7 days either side of an injected shift, plus every fourth daily
run's rows that lie outside every shift and the 144 half-hours after it (the ordinary rows).

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00296_score.py run [--limit N]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import e00296_fed as fed
import e00296_injected as inj
import e00296_nra as nra
import numpy as np
import polars as pl
from e00296_common import WORK_DIR, stamp_months

N_WORKERS: Final[int] = 4
OUT: Final[Path] = WORK_DIR / "scores"
PERIODS: Final[tuple[str, ...]] = ("replicate", "scored")
MARGIN_STAMPS: Final[int] = 336
"""Seven days either side of a shift."""
ORDINARY_TAIL_STAMPS: Final[int] = 144
"""The 144 half-hours after a shift's end that are not ordinary rows."""
ORDINARY_RUN_STRIDE: Final[int] = 4
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
WORLDS: Final[Path] = WORK_DIR / "worlds"
LD_MIN_HOURS: Final[float] = 48.0
LD_MIN_DEPTH: Final[float] = 0.10
"""The long-and-deep class: shifts of at least 2 days and at least 10% of load."""


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


def score_arms(
    *,
    data: fed.site_module.SiteData,
    site: str,
    corrections: dict[str, np.ndarray],
    rows: np.ndarray,
    delta_points: np.ndarray,
    edges: dict[str, float],
    add: dict[str, np.ndarray] | None = None,
) -> pl.DataFrame:
    """One row per (arm, row): scores against both targets, plus the 0.95 quantile."""
    spread = fed.k0_spread(site)
    actual_nra = data.actual[rows]
    actual_obs = actual_nra + delta_points[data.position[rows]]
    frames = []
    for arm, correction in corrections.items():
        q = fed.quantiles_at(data=data, correction=correction, spread=spread, rows=rows)
        shift_back = add[arm] if add is not None else np.zeros(len(rows))
        q = q + shift_back[:, None]
        mean_forecast = data.member_mean[rows] + correction[rows] + shift_back
        columns: dict[str, object] = {
            "arm": arm,
            "row": rows,
            "q95": q[:, 9],
            "q50": q[:, 5],
            "q05": q[:, 1],
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


def worlds_list() -> list[tuple[str, int]]:
    """The (site, world) pairs of this experiment's injected worlds."""
    table = pl.read_parquet(WORLDS / "events.parquet")
    return sorted(table.select("site", "world").unique().iter_rows())


def is_long_deep(*, events: pl.DataFrame) -> np.ndarray:
    """True for shifts of at least 2 days and at least 10% of load (the long-and-deep class)."""
    return ((events["hours"] >= LD_MIN_HOURS) & (events["depth"] >= LD_MIN_DEPTH - 1e-9)).to_numpy()


def job(args: tuple[str, int]) -> str:
    """Score the arms at one site in one world and write the result atomically."""
    site, seed = args
    path = OUT / f"{site}_{seed}.parquet"
    if path.exists():
        return str(path)
    started = time.time()
    data = fed.get_site(site)
    table = (
        pl.read_parquet(WORLDS / "events.parquet")
        .filter((pl.col("site") == site) & (pl.col("world") == seed))
        .sort("onset")
    )
    with np.load(WORLDS / f"{site}_{seed}.npz") as stored:
        stamp_delta = stored["delta"]
    n = len(data.stream)
    offset = fed.stamp_offset(data)
    points = fed.to_positions(
        array=100.0 * np.nan_to_num(stamp_delta) / fed.CAPACITY_MW[site], data=data
    )
    stream = np.where(np.isfinite(data.stream), data.stream + points, np.nan)
    events = [
        {
            "id": int(r["event"]),
            "onset": int(r["onset"]),
            "end": int(r["end"]),
            "fraction": r["depth"],
        }
        for r in table.iter_rows(named=True)
    ]
    long_deep = is_long_deep(events=table)
    perfect = inj.perfect_spans(events=events, points=points, data=data)
    lagged = nra.true_spans(
        events=[(e["onset"], e["end"]) for e in events], delta_points=points, data=data
    )
    pick = lambda spans, keep: [s for s, k in zip(spans, keep, strict=True) if k]  # noqa: E731
    sources = {
        "PERFECT": perfect,
        "PERFECT_LAG": lagged,
        "PERFECT_LD": pick(perfect, long_deep),
        "PERFECT_SMALL": pick(perfect, ~long_deep),
    }
    cutoffs = data.cutoffs(delay_hours=fed.DELAY_HOURS)[data.run_of_row]
    months = stamp_months()
    onset_pos = np.array([e["onset"] + offset for e in events])
    end_pos = np.array([e["end"] + offset for e in events])
    near = np.full(n, -1, dtype=np.int64)
    shift = np.zeros(n, dtype=bool)
    for k in range(len(events)):
        lo, hi = max(onset_pos[k] - MARGIN_STAMPS, 0), min(end_pos[k] + MARGIN_STAMPS, n)
        near[lo:hi] = k
        shift[onset_pos[k] : min(end_pos[k] + ORDINARY_TAIL_STAMPS, n)] = True
    edges = edges_of(data=data)
    frames = []
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
        corrections = {
            "K0": fed.correction_for(data=data, stream=stream, prior=np.zeros(n), multiplier=1.0)
        }
        for name, spans in sources.items():
            corrections[name] = nra.subtraction_correction(
                data=data, stream=stream, spans=spans, cutoffs=cutoffs, rows=rows, mask=True
            )
        scored = score_arms(
            data=data,
            site=site,
            corrections=corrections,
            rows=rows,
            delta_points=points,
            edges=edges,
        )
        corrections_add: dict[str, np.ndarray] = {}
        add: dict[str, np.ndarray] = {}
        pos_rows = data.position[rows]
        for name, spans in sources.items():
            corrections_add[f"{name}_ADD"] = nra.subtraction_correction(
                data=data, stream=stream, spans=spans, cutoffs=cutoffs, rows=rows, mask=False
            )
            back = np.zeros(len(rows))
            for span in spans:
                sel = (
                    (pos_rows >= span.onset) & (pos_rows < span.end) & (cutoffs[rows] >= span.start)
                )
                back[sel] = points[pos_rows[sel]]
            add[f"{name}_ADD"] = back
        scored_add = score_arms(
            data=data,
            site=site,
            corrections=corrections_add,
            rows=rows,
            delta_points=points,
            edges=edges,
            add=add,
        )
        scored = pl.concat([scored, scored_add])
        pos = pos_rows
        k = near[pos]
        has_event = k >= 0
        kk = np.where(has_event, k, 0)
        in_shift = has_event & (pos >= onset_pos[kk]) & (pos < end_pos[kk] + ORDINARY_TAIL_STAMPS)
        phase = np.where(
            has_event
            & (cutoffs[rows] >= onset_pos[kk])
            & (pos >= onset_pos[kk])
            & (pos < end_pos[kk] + inj.AFTER_STAMPS),
            inj.phase_of(cutoff=cutoffs[rows], target=pos, onset=onset_pos[kk], end=end_pos[kk]),
            np.where(has_event, "MARGIN", "ORDINARY"),
        )
        origin = data.run_origins[data.run_of_row[rows]]
        event_rows = {
            c: table[c].to_numpy()[kk] for c in ("hours", "depth", "sign", "shape", "duration_bin")
        }
        meta = pl.DataFrame(
            {
                "row": rows,
                "position": pos,
                "run": data.runs[data.run_of_row[rows]],
                "power_fcst_init_time": origin.astype("datetime64[us]"),
                "lead_hours": data.lead_hours[rows],
                "event": np.where(has_event, np.array([e["id"] for e in events])[kk], -1),
                "phase": phase,
                "in_shift": in_shift,
                "event_hours": np.where(has_event, event_rows["hours"], np.nan),
                "fraction": np.where(has_event, event_rows["depth"], np.nan),
                "sign": np.where(has_event, event_rows["sign"], np.nan),
                "shape": np.where(has_event, event_rows["shape"], ""),
                "duration_bin": np.where(has_event, event_rows["duration_bin"], ""),
                "onset_month": np.where(
                    has_event, months[np.clip(onset_pos[kk] - offset, 0, len(months) - 1)], 0
                ).astype(np.int8),
                "actual_nra": data.actual[rows],
                "actual_obs": data.actual[rows] + points[pos],
            }
        )
        frames.append(
            scored.join(meta, on="row").with_columns(
                site=pl.lit(site),
                world=pl.lit(seed, dtype=pl.Int32),
                period=pl.lit(period),
            )
        )
    if not frames:
        return ""
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pl.concat(frames).with_columns(pl.col(pl.Float64).cast(pl.Float32)).write_parquet(tmp)
    tmp.rename(path)
    print(f"{site} {seed}: {len(events)} events, {time.time() - started:.0f} s", flush=True)
    return str(path)


def site_constants() -> None:
    """Write each site's edges, effective capacity and earlier maximum flow, for the report."""
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
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "site_constants.json").write_text(json.dumps(out, indent=2))


def main() -> None:
    """Run the scoring stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["run", "one", "constants"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "constants":
        site_constants()
        return
    jobs = worlds_list()
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
