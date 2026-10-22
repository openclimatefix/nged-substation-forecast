"""Score the unmasked stack and the hard masks at one substation on every cached run's real rows.

Experiment 166 scored the scored year only. This script scores every row with a reading of every
cached run (forecast origins 2024-04-10 to 2026-06-29), so the no-harm contrast rests on 27 months.
The arms are K0 (no mask), MASK_GATED (pair alarms with a balance score of at least
`ev.BALANCE_GATE`), and, for reference, MASK_NEW and MASK_UNION. Each arm is GAM-K's cached member
forecasts plus a correction from the Kalman filter run on the real residual stream with readings
set missing inside the known alarm intervals, with experiment 115's K0 spread.

Writes `rows_<site>.parquet` (one row per arm and scored row), `flags_<site>.parquet` (per row:
inside a logged transfer and its 72 hours after, and the alarm active at the forecast origin), and
`masked_<site>.parquet` (per gated span: its start, known, and logged-transfer overlap).

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00171_run.py D5
"""

from __future__ import annotations

import sys
import time

import e00171_events as ev
import e00171_fed as fed
import e00171_nra as nra
import numpy as np
import polars as pl
from e00171_common import WORK_DIR
from e00171_g_arms import quantile_score

ARMS = ("K0", "MASK_GATED", "MASK_NEW", "MASK_UNION")
TABLES = {"MASK_GATED": "GATED", "MASK_NEW": "NEW", "MASK_UNION": "UNION"}


def main(site: str) -> None:
    """Compute and store one substation's per-row scores."""
    started = time.time()
    data = fed.get_site(site)
    offset = fed.stamp_offset(data)
    n = len(data.stream)
    tables = ev.alarm_tables()
    series = fed.SERIES_OF_SITE[site]
    logged = ev.logged_transfers().filter(pl.col("series").list.contains(series))
    blocked = np.zeros(n, dtype=bool)
    for r in logged.iter_rows(named=True):
        lo, hi = r["onset"] + offset, r["end"] + offset
        blocked[max(lo, 0) : min(hi + nra.TAIL, n)] = True
    rows = np.flatnonzero(np.isfinite(data.actual))
    cutoffs = data.cutoffs(delay_hours=fed.DELAY_HOURS)[data.run_of_row]
    spans = {
        arm: nra.detected_spans(data=data, site=site, intervals=tables[key])
        for arm, key in TABLES.items()
    }
    corrections = {
        "K0": fed.correction_for(data=data, stream=data.stream, prior=np.zeros(n), multiplier=1.0)
    }
    for arm, arm_spans in spans.items():
        corrections[arm] = nra.subtraction_correction(
            data=data, stream=data.stream, spans=arm_spans, cutoffs=cutoffs, rows=rows, mask=True
        )
        print(f"{site} {arm} done at {time.time() - started:.0f} s", flush=True)
    spread = fed.k0_spread(site)
    frames = []
    for arm in ARMS:
        q = fed.quantiles_at(data=data, correction=corrections[arm], spread=spread, rows=rows)
        frames.append(
            pl.DataFrame(
                {
                    "arm": arm,
                    "site": site,
                    "row": rows,
                    "power_fcst_init_time": data.run_origins[data.run_of_row[rows]],
                    "valid_time": data.cache.end_times[data.position[rows]],
                    "lead_hours": data.lead_hours[rows],
                    "run_period": data.run_period[data.run_of_row[rows]],
                    "actual": data.actual[rows],
                    "qs": quantile_score(actual=data.actual[rows], quantiles=q),
                    "q50": q[:, 5],
                    "mean_forecast": data.member_mean[rows] + corrections[arm][rows],
                }
            )
        )
    pl.concat(frames).write_parquet(WORK_DIR / f"rows_{site}.parquet")
    flags = {
        "site": site,
        "row": rows,
        "in_logged": blocked[data.position[rows]],
    }
    cut = cutoffs[rows]
    for arm, arm_spans in spans.items():
        active = np.zeros(len(rows), dtype=bool)
        for s in arm_spans:
            active |= (cut >= s.start) & (cut < s.known)
        flags[f"active_{arm}"] = active
    pl.DataFrame(flags).write_parquet(WORK_DIR / f"flags_{site}.parquet")
    masked = []
    masked.extend(
        {
            "arm": arm,
            "site": site,
            "start": s.start,
            "known": s.known,
            "start_position": min(max(s.start, 0), n - 1),
            "n_half_hours": s.known - s.start,
            "n_outside_logged": int((~blocked[max(s.start, 0) : min(s.known, n)]).sum()),
        }
        for arm, arm_spans in spans.items()
        for s in arm_spans
    )
    table = pl.DataFrame(
        masked,
        schema={
            "arm": pl.String,
            "site": pl.String,
            "start": pl.Int64,
            "known": pl.Int64,
            "start_position": pl.Int64,
            "n_half_hours": pl.Int64,
            "n_outside_logged": pl.Int64,
        },
    )
    start_time = pl.Series(data.cache.end_times[table["start_position"].to_numpy()])
    table.with_columns(start_time=start_time).write_parquet(WORK_DIR / f"masked_{site}.parquet")
    print(f"{site}: {len(rows)} rows, {time.time() - started:.0f} s")


if __name__ == "__main__":
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    main(sys.argv[1])
