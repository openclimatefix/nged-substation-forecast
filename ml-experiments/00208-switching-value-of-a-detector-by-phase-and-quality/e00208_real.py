"""Score the knowledge sources on the real record's logged transfers, and the false-alarm harm.

Per substation, with GAM-K's cached member forecasts and the Kalman filter correction run on the
real stream (experiments 163, 171, and 203), arms are:

- K0 (no mask); LOGGED (NGED's switching table: logged onset, end known at once);
- RONC (the stored refit-offset neighbour-count detector, experiment 147's causal tables);
- PAIR_NEW and PAIR_GATED (experiments 156 and 166);
- LEARNED_R, LEARNED_F, and RONC_195 (experiment 195's alarms), one set of spans per fold.

Rows of a logged transfer are the forecasts whose data cutoff is at or after the logged onset and
whose target time lies from the onset to 144 half-hours after the logged end, each tagged with its
phase and with whether it is also in experiment 163's AFTER set (forecast origin at or after the
onset, target within 144 half-hours after the end). The precision sweep adds false alarms to the
logged spans and scores the ordinary rows near them.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00208_real.py D5
"""

from __future__ import annotations

import sys
import time
from typing import Final

import e00208_alarms as al
import e00208_events as ev
import e00208_fed as fed
import e00208_nra as nra
import numpy as np
import polars as pl
from e00208_common import WORK_DIR
from e00208_g_arms import quantile_score
from e00208_injected import phase_of

SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16", "D9")
PRECISIONS: Final[tuple[float, ...]] = (0.75, 0.5)
N_DRAWS: Final[int] = 5
NEAR_STAMPS: Final[int] = 288
"""Half-hours after a false alarm's end is known over which its filter effect is scored (6 days)."""
WINDOW_STAMPS: Final[int] = 144
"""72 hours of target time after the logged end are kept; the AFTER phase is the first 96."""
AFTER_STAMPS: Final[int] = 96
ORDINARY_ROWS_ALL_SITES: Final[int] = 1_579_711
"""Experiment 171's ordinary rows at the six substations (experiment 203 printed the same count)."""
OUT = WORK_DIR / "real"


def logged_pair_count() -> int:
    """Logged transfers touching each decision substation, counted once per substation."""
    logged = ev.logged_transfers()
    return int(
        sum(
            logged.filter(pl.col("series").list.contains(fed.SERIES_OF_SITE[s])).height
            for s in SITES
        )
    )


def onset_month(*, data: object, position: int) -> int:
    """The calendar month (1 to 12) of a grid position."""
    times = data.cache.end_times  # type: ignore[attr-defined]
    return int(str(times[min(max(position, 0), len(times) - 1)].astype("datetime64[M]"))[5:7])


def main(site: str) -> None:
    """Compute and store one substation's event-row scores and false-alarm harm."""
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    data = fed.get_site(site)
    offset = fed.stamp_offset(data)
    n = len(data.stream)
    series = fed.SERIES_OF_SITE[site]
    tables = ev.alarm_tables()
    logged = ev.logged_transfers().filter(pl.col("series").list.contains(series))
    blocked = np.zeros(n, dtype=bool)
    for r in logged.iter_rows(named=True):
        blocked[max(r["onset"] + offset, 0) : min(r["end"] + offset + nra.TAIL, n)] = True
    events = ev.site_events(site=site, tables=tables, offset=offset)
    cutoffs = data.cutoffs(delay_hours=fed.DELAY_HOURS)[data.run_of_row]
    origin_pos = np.searchsorted(data.cache.end_times, data.run_origins)[data.run_of_row]
    eligible = np.isin(data.run_period, ["scored", "replicate"])[data.run_of_row]
    ok = np.isfinite(data.actual)

    # ---- rows of each logged transfer, with phase ----
    pieces = []
    for r in events.iter_rows(named=True):
        chosen = np.flatnonzero(
            ok
            & eligible
            & ((cutoffs >= r["onset"]) | (origin_pos >= r["onset"]))
            & (data.position >= r["onset"])
            & (data.position < r["end"] + WINDOW_STAMPS)
        )
        phase = phase_of(
            cutoff=cutoffs[chosen],
            target=data.position[chosen],
            onset=np.full(len(chosen), r["onset"]),
            end=np.full(len(chosen), r["end"]),
        )
        # targets from 96 to 144 half-hours after the end are in experiment 163's AFTER set only
        phase = np.where(data.position[chosen] >= r["end"] + AFTER_STAMPS, "TAIL", phase)
        in_163 = (
            (origin_pos[chosen] >= r["onset"])
            & (data.position[chosen] >= r["end"])
            & (data.position[chosen] < r["end"] + WINDOW_STAMPS)
        )
        cutoff_ok = cutoffs[chosen] >= r["onset"]
        pieces.append(
            pl.DataFrame(
                {
                    "card": [r["card"]] * len(chosen),
                    "row": chosen,
                    "phase": phase,
                    "in_163_after": in_163,
                    "cutoff_ok": cutoff_ok,
                    "hours": [(r["end"] - r["onset"]) / 2.0] * len(chosen),
                    "onset_month": [onset_month(data=data, position=r["onset"])] * len(chosen),
                },
                schema={
                    "card": pl.String,
                    "row": pl.Int64,
                    "phase": pl.String,
                    "in_163_after": pl.Boolean,
                    "cutoff_ok": pl.Boolean,
                    "hours": pl.Float64,
                    "onset_month": pl.Int64,
                },
            )
        )
    event_rows = pl.concat(pieces) if pieces else pl.DataFrame()
    rows = np.unique(event_rows["row"].to_numpy()).astype(np.int64) if len(event_rows) else np.array([], dtype=np.int64)
    spread = fed.k0_spread(site)

    # ---- spans of every knowledge source ----
    spans: dict[tuple[str, int], list[nra.Span]] = {
        ("LOGGED", -1): ev.oracle_spans(events=events),
        ("RONC", -1): nra.detected_spans(data=data, site=site, intervals=tables["RONC"]),
        ("PAIR_NEW", -1): nra.detected_spans(data=data, site=site, intervals=tables["NEW"]),
        ("PAIR_GATED", -1): nra.detected_spans(data=data, site=site, intervals=tables["GATED"]),
    }
    for arm in ("LEARNED_R", "LEARNED_F", "RONC_195"):
        for fold, table in al.learned_tables(arm=arm).items():
            spans[(arm, fold)] = nra.detected_spans(data=data, site=site, intervals=table)
    base = fed.correction_for(data=data, stream=data.stream, prior=np.zeros(n), multiplier=1.0)
    frames = []
    if len(rows):
        corrections = {("K0", -1): base}
        for key, arm_spans in spans.items():
            corrections[key] = nra.subtraction_correction(
                data=data, stream=data.stream, spans=arm_spans, cutoffs=cutoffs, rows=rows, mask=True
            )
        for (arm, fold), correction in corrections.items():
            q = fed.quantiles_at(data=data, correction=correction, spread=spread, rows=rows)
            frames.append(
                pl.DataFrame(
                    {
                        "arm": arm,
                        "fold": fold,
                        "row": rows,
                        "qs": quantile_score(actual=data.actual[rows], quantiles=q),
                        "q01": q[:, 0],
                        "q05": q[:, 1],
                        "q50": q[:, 5],
                        "q95": q[:, 9],
                        "q99": q[:, 10],
                        "mean_forecast": data.member_mean[rows] + correction[rows],
                        "actual": data.actual[rows],
                        "lead_hours": data.lead_hours[rows],
                        "target": data.position[rows],
                    }
                )
            )
        pl.concat(frames).write_parquet(OUT / f"event_scores_{site}.parquet")
        event_rows.with_columns(site=pl.lit(site)).write_parquet(OUT / f"event_rows_{site}.parquet")
    print(f"{site}: events scored at {time.time() - started:.0f} s", flush=True)

    # ---- false-alarm harm on ordinary rows ----
    ordinary = np.flatnonzero(ok & ~blocked[data.position])
    month_of_row = (
        data.cache.end_times[data.position[ordinary]].astype("datetime64[M]").astype(int) + 1
    )
    pl.DataFrame(
        {"site": site, "month": month_of_row, "row": ordinary}, schema_overrides={"row": pl.Int64}
    ).group_by("site", "month").len().write_parquet(OUT / f"ordinary_count_{site}.parquet")
    durations = np.array(
        [r["end"] - r["onset"] for r in ev.logged_transfers().iter_rows(named=True)]
    )
    lo, hi = int(data.position[ordinary].min()) + 200, int(data.position[ordinary].max()) - 600
    share = len(ordinary) / ORDINARY_ROWS_ALL_SITES
    total_events = logged_pair_count()
    harm = []
    live_logged = spans[("LOGGED", -1)]
    for precision in PRECISIONS:
        n_false_total = total_events * (1.0 - precision) / precision
        for draw in range(N_DRAWS):
            rng = np.random.default_rng([208, SITES.index(site), int(precision * 100), draw])
            expected = n_false_total * share
            count = int(np.floor(expected) + (rng.random() < expected - np.floor(expected)))
            false_spans = []
            while len(false_spans) < count:
                onset = int(rng.integers(lo, hi))
                length = int(rng.choice(durations))
                if blocked[max(onset - 96, 0) : min(onset + length + nra.TAIL, n)].any():
                    continue
                false_spans.append(
                    nra.Span(
                        onset, onset + length, onset + nra.ONSET_LAG,
                        onset + length + nra.END_LAG, 0.0,
                    )
                )
            if not false_spans:
                continue
            near = np.zeros(len(data.run_of_row), dtype=bool)
            for s in false_spans:
                near |= (cutoffs >= s.onset) & (cutoffs < s.known + NEAR_STAMPS)
            near_rows = np.flatnonzero(near & ok & ~blocked[data.position]).astype(np.int64)
            if not len(near_rows):
                continue
            correction = nra.subtraction_correction(
                data=data,
                stream=data.stream,
                spans=live_logged + false_spans,
                cutoffs=cutoffs,
                rows=near_rows,
                mask=True,
            )
            q_arm = fed.quantiles_at(data=data, correction=correction, spread=spread, rows=near_rows)
            alone = nra.subtraction_correction(
                data=data, stream=data.stream, spans=live_logged, cutoffs=cutoffs,
                rows=near_rows, mask=True,
            )
            q_logged = fed.quantiles_at(data=data, correction=alone, spread=spread, rows=near_rows)
            diff = quantile_score(actual=data.actual[near_rows], quantiles=q_arm) - quantile_score(
                actual=data.actual[near_rows], quantiles=q_logged
            )
            harm.append(
                pl.DataFrame(
                    {
                        "site": site,
                        "precision": precision,
                        "draw": draw,
                        "month": data.cache.end_times[data.position[near_rows]]
                        .astype("datetime64[M]")
                        .astype(int)
                        + 1,
                        "diff": diff,
                        "n_false": len(false_spans),
                    }
                )
            )
    if harm:
        pl.concat(harm).write_parquet(OUT / f"false_alarm_{site}.parquet")
    print(f"{site}: done in {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
