"""Score the stack's arms at one substation on the real scored year and the logged-transfer row sets.

Writes `rows_<site>.parquet` (one row per arm and scored row), `events_<site>.parquet`, and
`rowsets_<site>.parquet` (which rows belong to which event and row set) under the experiment's
private folder. Every arm is GAM-K's cached member forecasts plus a correction from the Kalman filter
run on the real residual stream with readings set missing inside the known alarm intervals, with
experiment 115's K0 spread, so a contrast between arms is a difference in location only.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00163_run.py D5
"""

from __future__ import annotations

import sys
import time
from typing import Final

import e00163_events as ev
import e00163_fed as fed
import e00163_nra as nra
import numpy as np
import polars as pl
from e00163_common import WORK_DIR
from e00163_g_arms import quantile_score

ARMS: Final[tuple[str, ...]] = (
    "K0",
    "GAM_ONLY",
    "MASK_NEW",
    "MASK_RONC",
    "MASK_UNION",
    "ORACLE",
)


def main(site: str) -> None:  # noqa: PLR0915
    """Compute and store one substation's per-row scores."""
    started = time.time()
    data = fed.get_site(site)
    offset = fed.stamp_offset(data)
    n = len(data.stream)
    tables = ev.alarm_tables()
    events = ev.site_events(site=site, tables=tables, offset=offset)
    sets = ev.row_sets(data=data, events=events)
    event_rows = np.unique(np.concatenate(list(sets.values()))) if sets else np.array([], int)
    series = fed.SERIES_OF_SITE[site]
    logged = ev.logged_transfers().filter(pl.col("series").list.contains(series))
    blocked = np.zeros(n, dtype=bool)
    for r in logged.iter_rows(named=True):
        lo, hi = r["onset"] + offset, r["end"] + offset
        blocked[max(lo, 0) : min(hi + nra.TAIL, n)] = True
    scored = data.period_rows(period="scored") & np.isfinite(data.actual)
    quiet = np.flatnonzero(scored & ~blocked[data.position] & ~data.features.switching_mask[data.position])
    rows = np.unique(np.concatenate([quiet, event_rows]))
    cutoffs = data.cutoffs(delay_hours=fed.DELAY_HOURS)[data.run_of_row]
    spans = {
        "MASK_NEW": nra.detected_spans(data=data, site=site, intervals=tables["NEW"]),
        "MASK_RONC": nra.detected_spans(data=data, site=site, intervals=tables["RONC"]),
        "MASK_UNION": nra.detected_spans(data=data, site=site, intervals=tables["UNION"]),
    }
    corrections = {
        "K0": fed.correction_for(
            data=data, stream=data.stream, prior=np.zeros(n), multiplier=1.0
        ),
        "GAM_ONLY": np.zeros(len(data.run_of_row)),
    }
    for arm, arm_spans in spans.items():
        corrections[arm] = nra.subtraction_correction(
            data=data, stream=data.stream, spans=arm_spans, cutoffs=cutoffs, rows=rows, mask=True
        )
        print(f"{site} {arm} done at {time.time() - started:.0f} s", flush=True)
    corrections["ORACLE"] = nra.subtraction_correction(
        data=data,
        stream=data.stream,
        spans=ev.oracle_spans(events=events),
        cutoffs=cutoffs,
        rows=event_rows,
        mask=True,
    )
    spread = fed.k0_spread(site)
    origin = data.run_origins[data.run_of_row[rows]]
    valid = data.cache.end_times[data.position[rows]]
    frames = []
    for arm in ARMS:
        use = event_rows if arm == "ORACLE" else rows
        q = fed.quantiles_at(data=data, correction=corrections[arm], spread=spread, rows=use)
        frames.append(
            pl.DataFrame(
                {
                    "arm": arm,
                    "site": site,
                    "row": use,
                    "power_fcst_init_time": data.run_origins[data.run_of_row[use]],
                    "valid_time": data.cache.end_times[data.position[use]],
                    "lead_hours": data.lead_hours[use],
                    "actual": data.actual[use],
                    "qs": quantile_score(actual=data.actual[use], quantiles=q),
                    "q01": q[:, 0],
                    "q05": q[:, 1],
                    "q50": q[:, 5],
                    "q95": q[:, 9],
                    "q99": q[:, 10],
                    "mean_forecast": data.member_mean[use] + corrections[arm][use],
                }
            )
        )
    del origin, valid
    pl.concat(frames).write_parquet(WORK_DIR / f"rows_{site}.parquet")
    events.write_parquet(WORK_DIR / f"events_{site}.parquet")
    pl.DataFrame(
        [
            {"set": key.split("|")[0], "card": key.split("|")[1], "site": site, "row": int(r)}
            for key, value in sets.items()
            for r in value
        ],
        schema={"set": pl.String, "card": pl.String, "site": pl.String, "row": pl.Int64},
    ).write_parquet(WORK_DIR / f"rowsets_{site}.parquet")
    quiet_frame = pl.DataFrame({"site": site, "row": quiet})
    quiet_frame.write_parquet(WORK_DIR / f"quiet_{site}.parquet")
    print(f"{site}: {len(rows)} rows, {len(events)} events, {time.time() - started:.0f} s")


if __name__ == "__main__":
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    main(sys.argv[1])
