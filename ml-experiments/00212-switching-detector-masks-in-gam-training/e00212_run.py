"""Score the three training-mask arms of experiment 212 at one substation.

Every arm is GAM-K's member forecasts from the arm's own monthly fits plus the K0 Kalman correction
on the residual stream the arm's fits give, with experiment 115's K0 spread, so contrasts between
arms are differences in location only. Arms: A (no training mask), B (hand labels masked, experiment
111's fits), and C0, C1, C2 (the learned-score alarms of experiment 195's three folds masked).

Rows: experiment 203's ordinary rows (a reading, outside every logged transfer and its 72 hours
after), over every cached run. Each row carries flags for the CLEAN set (scored-year runs), and for
POST-OLD and POST-NEW (the run's latest monthly fit has a logged transfer in its 24-month training
window that ended before the cut, at least or fewer than 192 half-hours before it).

Writes `meta_<site>`, `scores_<site>` parquet files under the experiment's private folder.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00212_run.py D5
"""

from __future__ import annotations

import sys
import time

import e00212_events as ev
import e00212_fed as fed
import e00212_fits as fits_module
import e00212_nra as nra
import numpy as np
import polars as pl
from e00212_common import WORK_DIR
from e00212_g_arms import quantile_score
from studies.dp_features import naive_utc

ARMS = fits_module.ARMS
NEW_STAMPS = 192
"""A logged transfer that ended fewer than this many half-hours (4 days) before the cut is not yet
known to the re-ranker."""
WINDOW_HALF_HOURS = 48 * 730
"""24 months of half-hours, the fit's training window."""


def post_flags(*, data, events: pl.DataFrame, offset: int, cuts: list) -> pl.DataFrame:  # noqa: ANN001
    """Per cached run: POST-NEW, POST-OLD, and the cluster (card of the latest transfer in the window)."""
    cut_positions = np.array(
        [int(np.searchsorted(data.cache.end_times, naive_utc(moment=c), side="right")) for c in cuts]
    )
    cut_times = np.array([naive_utc(moment=c) for c in cuts], dtype="datetime64[us]")
    origins = data.run_origins
    ends = events["end"].to_numpy() + offset
    cards = events["card"].to_list()
    rows = []
    for origin in origins:
        idx = int(np.searchsorted(cut_times + np.timedelta64(210, "m"), origin, side="right")) - 1
        if idx < 0:
            rows.append((False, False, None))
            continue
        cut = cut_positions[idx]
        gap = cut - ends
        in_window = (gap >= 0) & (gap < WINDOW_HALF_HOURS)
        new = bool((in_window & (gap < NEW_STAMPS)).any())
        old = bool((in_window & (gap >= NEW_STAMPS)).any()) and not new
        cluster = None
        if old:
            latest = int(np.argmax(np.where(in_window & (gap >= NEW_STAMPS), ends, -1)))
            cluster = cards[latest]
        rows.append((new, old, cluster))
    return pl.DataFrame(rows, schema={"post_new": pl.Boolean, "post_old": pl.Boolean, "cluster": pl.String}, orient="row")


def main(site: str) -> None:
    """Compute and store one substation's per-row scores for every arm."""
    started = time.time()
    series = fed.SERIES_OF_SITE[site]
    base = fed.get_site(site, "B")
    offset = fed.stamp_offset(base)
    n = len(base.stream)
    logged = ev.logged_transfers().filter(pl.col("series").list.contains(series))
    blocked = np.zeros(n, dtype=bool)
    for r in logged.iter_rows(named=True):
        blocked[max(r["onset"] + offset, 0) : min(r["end"] + offset + nra.TAIL, n)] = True
    rows = np.flatnonzero(np.isfinite(base.actual) & ~blocked[base.position])
    spread = fed.k0_spread(site)
    cuts = [f.cut for f in __import__("e00212_g_members").load_fits(site=site)]
    flags = post_flags(data=base, events=logged, offset=offset, cuts=cuts)
    run_flags = flags[base.run_of_row[rows]]
    scores = []
    for arm in ARMS:
        data = fed.get_site(site, arm)
        if not (np.array_equal(data.position, base.position) and np.array_equal(data.run_of_row, base.run_of_row)):
            msg = f"{site} {arm}: rows differ from arm B"
            raise ValueError(msg)
        correction = fed.correction_for(
            data=data, stream=data.stream, prior=np.zeros(n), multiplier=1.0
        )
        q = fed.quantiles_at(data=data, correction=correction, spread=spread, rows=rows)
        scores.append(
            pl.DataFrame(
                {
                    "arm": arm,
                    "row": rows,
                    "qs": quantile_score(actual=data.actual[rows], quantiles=q).astype(np.float32),
                    "q01": q[:, 0].astype(np.float32),
                    "q05": q[:, 1].astype(np.float32),
                    "q50": q[:, 5].astype(np.float32),
                    "q95": q[:, 9].astype(np.float32),
                    "q99": q[:, 10].astype(np.float32),
                    "mean_forecast": (data.member_mean[rows] + correction[rows]).astype(np.float32),
                }
            )
        )
        print(f"{site} {arm} done at {time.time() - started:.0f} s", flush=True)
    pl.concat(scores).write_parquet(WORK_DIR / f"scores_{site}.parquet")
    meta = pl.DataFrame(
        {
            "site": site,
            "row": rows,
            "power_fcst_init_time": base.run_origins[base.run_of_row[rows]],
            "valid_time": base.cache.end_times[base.position[rows]],
            "lead_hours": base.lead_hours[rows],
            "run_period": base.run_period[base.run_of_row[rows]],
            "actual": base.actual[rows],
        }
    ).hstack(run_flags)
    meta.write_parquet(WORK_DIR / f"meta_{site}.parquet")
    print(f"{site}: {len(rows)} rows, {time.time() - started:.0f} s")


if __name__ == "__main__":
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    main(sys.argv[1])
