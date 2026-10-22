"""Replay experiment 195's learned-score alarms through the masked forecast at one substation.

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
        env OMP_NUM_THREADS=1 uv run --no-sync python e00203_run.py D5
"""

from __future__ import annotations

import sys
import time

import e00203_alarms as al
import e00203_events as ev
import e00203_fed as fed
import e00203_nra as nra
import numpy as np
import polars as pl
from e00203_common import WORK_DIR
from e00203_g_arms import quantile_score

AFTER_STAMPS = ev.AFTER_STAMPS


def after_rows(*, data, events: pl.DataFrame) -> pl.DataFrame:  # noqa: ANN001
    """Rows (card, row) of the 72 hours of target time after each event's end, every cached run."""
    origin_pos = np.searchsorted(data.cache.end_times, data.run_origins)[data.run_of_row]
    ok = np.isfinite(data.actual)
    out = []
    for r in events.iter_rows(named=True):
        chosen = np.flatnonzero(
            ok
            & (origin_pos >= r["onset"])
            & (data.position >= r["end"])
            & (data.position < r["end"] + AFTER_STAMPS)
        )
        out.append(
            pl.DataFrame(
                {
                    "card": [r["card"]] * len(chosen),
                    "site": [r["site"]] * len(chosen),
                    "row": chosen,
                },
                schema={"card": pl.String, "site": pl.String, "row": pl.Int64},
            )
        )
    return (
        pl.concat(out)
        if out
        else pl.DataFrame(schema={"card": pl.String, "site": pl.String, "row": pl.Int64})
    )


def main(site: str) -> None:
    """Compute and store one substation's per-row scores for every arm."""
    started = time.time()
    data = fed.get_site(site)
    offset = fed.stamp_offset(data)
    n = len(data.stream)
    series = fed.SERIES_OF_SITE[site]
    pair_tables = ev.alarm_tables()
    logged = ev.logged_transfers().filter(pl.col("series").list.contains(series))
    blocked = np.zeros(n, dtype=bool)
    for r in logged.iter_rows(named=True):
        blocked[max(r["onset"] + offset, 0) : min(r["end"] + offset + nra.TAIL, n)] = True
    events = ev.site_events(site=site, tables=pair_tables, offset=offset)
    after = after_rows(data=data, events=events)
    ordinary = np.flatnonzero(np.isfinite(data.actual) & ~blocked[data.position])
    rows = np.unique(np.concatenate([ordinary, after["row"].to_numpy()]).astype(np.int64))
    cutoffs = data.cutoffs(delay_hours=fed.DELAY_HOURS)[data.run_of_row]
    spread = fed.k0_spread(site)
    folds_of_card = al.card_folds()

    # (arm, fold) -> spans; the arms without a fold use fold -1.
    spans: dict[tuple[str, int], list] = {
        ("PAIR_NEW", -1): nra.detected_spans(data=data, site=site, intervals=pair_tables["NEW"]),
        ("PAIR_GATED", -1): nra.detected_spans(
            data=data, site=site, intervals=pair_tables["GATED"]
        ),
    }
    for arm in al.ARMS:
        for fold, table in al.learned_tables(arm=arm).items():
            spans[(arm, fold)] = nra.detected_spans(data=data, site=site, intervals=table)
    corrections = {
        ("K0", -1): fed.correction_for(
            data=data, stream=data.stream, prior=np.zeros(n), multiplier=1.0
        )
    }
    for key, arm_spans in spans.items():
        corrections[key] = nra.subtraction_correction(
            data=data, stream=data.stream, spans=arm_spans, cutoffs=cutoffs, rows=rows, mask=True
        )
        print(f"{site} {key} done at {time.time() - started:.0f} s", flush=True)

    after_rows_ = np.unique(after["row"].to_numpy())
    scores, tails = [], []
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
        is_after = np.isin(rows, after_rows_)
        tails.append(
            pl.DataFrame(
                {
                    "arm": arm,
                    "fold": fold,
                    "row": rows[is_after],
                    "q01": q[is_after, 0],
                    "q05": q[is_after, 1],
                    "q95": q[is_after, 9],
                    "q99": q[is_after, 10],
                }
            )
        )
    pl.concat(scores).write_parquet(WORK_DIR / f"scores_{site}.parquet")
    pl.concat(tails).write_parquet(WORK_DIR / f"tails_{site}.parquet")
    pl.DataFrame(
        {
            "site": site,
            "row": rows,
            "power_fcst_init_time": data.run_origins[data.run_of_row[rows]],
            "valid_time": data.cache.end_times[data.position[rows]],
            "lead_hours": data.lead_hours[rows],
            "run_period": data.run_period[data.run_of_row[rows]],
            "actual": data.actual[rows],
            "ordinary": np.isin(rows, ordinary),
        }
    ).write_parquet(WORK_DIR / f"meta_{site}.parquet")
    fold_of_card = pl.DataFrame(
        {"card": list(folds_of_card), "fold": list(folds_of_card.values())},
        schema={"card": pl.String, "fold": pl.Int64},
    )
    after.join(fold_of_card, on="card", how="left").with_columns(
        pl.col("fold").fill_null(-1)
    ).write_parquet(WORK_DIR / f"after_{site}.parquet")
    masked = [
        {
            "site": site,
            "arm": arm,
            "fold": fold,
            "start": s.start,
            "known": s.known,
            "n_half_hours": s.known - s.start,
            "n_outside_logged": int((~blocked[max(s.start, 0) : min(s.known, n)]).sum()),
        }
        for (arm, fold), arm_spans in spans.items()
        for s in arm_spans
    ]
    pl.DataFrame(
        masked,
        schema={
            "site": pl.String,
            "arm": pl.String,
            "fold": pl.Int64,
            "start": pl.Int64,
            "known": pl.Int64,
            "n_half_hours": pl.Int64,
            "n_outside_logged": pl.Int64,
        },
    ).write_parquet(WORK_DIR / f"spans_{site}.parquet")
    print(f"{site}: {len(rows)} rows, {len(events)} events, {time.time() - started:.0f} s")


if __name__ == "__main__":
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    main(sys.argv[1])
