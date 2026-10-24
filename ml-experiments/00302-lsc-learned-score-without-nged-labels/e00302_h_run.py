"""Replay this experiment's learned-score alarms through the masked forecast at one substation.

Every arm is GAM-K's cached member forecasts plus a correction from the Kalman filter run on the
real residual stream with readings set missing inside the known alarm intervals, with experiment
115's K0 spread, so a contrast between arms is a difference in location only. Arms: K0 (no mask),
PAIR_NEW and PAIR_GATED (experiments 163 and 171, recomputed as positive controls), and, for each
of experiment 195's three folds, LEARNED_R, LEARNED_F, RONC_195, and LEARNED_R96 (post hoc).

Rows: experiment 171's ordinary rows (a reading, outside every logged transfer and its 72 hours
after, every cached run) and experiment 163's AFTER rows (the 72 hours of target time after each
logged transfer ends, forecast origin at or after the logged onset, every cached run, with the
run's period kept so the report can restrict to 163's scored and replicate months).

Writes `meta_<site>`, `after_<site>`, `scores_<site>`, `tails_<site>`, and `spans_<site>`
parquet files under the experiment's private folder.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00302_h_run.py D5
"""

from __future__ import annotations

import sys
import time

import e00302_h_alarms as al
import e00302_h_events as ev
import e00302_h_fed as fed
import e00302_h_nra as nra
import numpy as np
import polars as pl
from e00302_h_common import WORK_DIR
from e00302_h_g_arms import quantile_score

AFTER_STAMPS = ev.AFTER_STAMPS


def main(site: str, arms: tuple[str, ...]) -> None:
    """Compute one substation's ordinary-row scores for K0 and each arm's three folds."""
    started = time.time()
    data = fed.get_site(site)
    offset = fed.stamp_offset(data)
    n = len(data.stream)
    series = fed.SERIES_OF_SITE[site]
    logged = ev.logged_transfers().filter(pl.col("series").list.contains(series))
    blocked = np.zeros(n, dtype=bool)
    for r in logged.iter_rows(named=True):
        blocked[max(r["onset"] + offset, 0) : min(r["end"] + offset + nra.TAIL, n)] = True
    rows = np.flatnonzero(np.isfinite(data.actual) & ~blocked[data.position]).astype(np.int64)
    cutoffs = data.cutoffs(delay_hours=fed.DELAY_HOURS)[data.run_of_row]
    spread = fed.k0_spread(site)
    corrections = {
        ("K0", -1): fed.correction_for(
            data=data, stream=data.stream, prior=np.zeros(n), multiplier=1.0
        )
    }
    for arm in arms:
        for fold, table in al.tables(arm=arm).items():
            spans = nra.detected_spans(data=data, site=site, intervals=table)
            corrections[(arm, fold)] = nra.subtraction_correction(
                data=data, stream=data.stream, spans=spans, cutoffs=cutoffs, rows=rows, mask=True
            )
            print(f"{site} {(arm, fold)} done at {time.time() - started:.0f} s", flush=True)
    scores = []
    for (arm, fold), correction in corrections.items():
        q = fed.quantiles_at(data=data, correction=correction, spread=spread, rows=rows)
        scores.append(
            pl.DataFrame(
                {
                    "arm": arm,
                    "fold": fold,
                    "row": rows,
                    "qs": quantile_score(actual=data.actual[rows], quantiles=q).astype(np.float32),
                    "q50": q[:, 5].astype(np.float32),
                    "mean_forecast": (data.member_mean[rows] + correction[rows]).astype(np.float32),
                }
            )
        )
    pl.concat(scores).write_parquet(WORK_DIR / f"h_scores_{site}.parquet")
    pl.DataFrame(
        {
            "site": site,
            "row": rows,
            "valid_time": data.cache.end_times[data.position[rows]],
            "power_fcst_init_time": data.run_origins[data.run_of_row[rows]],
            "lead_hours": data.lead_hours[rows],
        }
    ).write_parquet(WORK_DIR / f"h_meta_{site}.parquet")
    print(f"{site}: {len(rows)} ordinary rows, {len(arms)} arms, {time.time() - started:.0f} s")


if __name__ == "__main__":
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    main(sys.argv[1], tuple(sys.argv[2:]) or al.ARMS)
