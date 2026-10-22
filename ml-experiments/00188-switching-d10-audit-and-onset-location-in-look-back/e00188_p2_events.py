"""Part 2, stage 1: the alarms of every injected world and of the real stream, with located onsets.

For each alarm family (the pair alarm and RO-NC) this writes `alarms_<tag>.parquet` (one row per
distinct alarm, with the stamp it is known, its first usable origin, and its located onset) and
`events_<tag>.parquet` (one row per transfer at a decision substation that an alarm finds, with the
true onset). `<tag>` is `world_<seed>` for an injected world or `real` for the logged transfers.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00188_p2_events.py injected|real
"""

from __future__ import annotations

import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import e00188_alarm as alarm
import e00188_detect_run as detect_run
import e00188_events as ev
import e00188_fed as fed
import e00188_locate as loc
import numpy as np
import polars as pl
from e00188_common import WORK_DIR
from e00188_d_inject import inject_power

N_WORKERS: Final[int] = 4
DETECT: Final[Path] = detect_run.OUT
WORLDS: Final[Path] = detect_run.WORLDS_141
SITE_SERIES: Final[dict[str, int]] = fed.SERIES_OF_SITE
OUT: Final[Path] = WORK_DIR / "p2"


def window() -> int:
    """The pair alarm's headline window of experiment 156."""
    return int(json.loads((DETECT / "headline.json").read_text())["window"])


def match_rows(
    *, intervals: pl.DataFrame, series: int, lo: int, hi: int
) -> pl.DataFrame:
    """Interval rows on `series` whose step stamp lies in [lo, hi], earliest known first."""
    return intervals.filter(
        (pl.col("series") == series) & (pl.col("onset") >= lo) & (pl.col("onset") <= hi)
    ).sort("onset_known")


def event_rows(
    *,
    transfers: list[dict],
    tables: dict[str, pl.DataFrame],
    alarms: dict[str, pl.DataFrame],
    tag: str,
    real: bool,
) -> pl.DataFrame:
    """One row per (family, decision substation, transfer) found by an alarm of the family."""
    rows = []
    for family, table in tables.items():
        for t in transfers:
            for site, series in SITE_SERIES.items():
                if series not in t["series"]:
                    continue
                lo = t["onset"] - loc.MATCH
                hi = t["end"] - 1 if real else t["onset"] + loc.MATCH
                hits = match_rows(intervals=table, series=series, lo=lo, hi=hi)
                if hits.is_empty():
                    continue
                h = hits.row(0, named=True)
                key = h["alarm_series"] if "alarm_series" in h else series
                found = alarms[family].filter(
                    (pl.col("alarm_series") == key) & (pl.col("onset") == h["onset"])
                )
                if found.is_empty():
                    continue
                a = found.row(0, named=True)
                rows.append(
                    {
                        "tag": tag,
                        "family": family,
                        "site": site,
                        "event": t["event"],
                        "true_onset": t["onset"],
                        "true_end": t["end"],
                        "alarm_series": a["alarm_series"],
                        "onset": a["onset"],
                        "known": a["known"],
                        "end": a["end"],
                        "first_origin": a["first_origin"],
                        "located": a["located"],
                    }
                )
    return pl.DataFrame(rows)


def world_job(args: tuple[int, list]) -> str:
    """Alarms and events of one injected world."""
    seed, batch = args
    path = OUT / f"events_world_{seed}.parquet"
    if path.exists():
        return str(path)
    state = detect_run.state()
    injection = inject_power(power=state["panel"]["power"], events=batch, seed=seed)
    residual = injection.fast_residual(state["panel"]["residual"])
    w = window()
    ronc = alarm.with_ronc_onset_known(
        intervals=pl.read_parquet(WORLDS / f"world_{seed}.parquet").filter(
            pl.col("setting") == "R0_M48"
        )
    )
    pair = pl.read_parquet(DETECT / f"world_{seed}_W{w}_pair.parquet")
    tables = {"PAIR": pair, "RONC": ronc}
    alarms = {k: loc.alarm_rows(table=t, residual=residual, family=k) for k, t in tables.items()}
    transfers = [
        {"event": e.event_id, "onset": e.onset, "end": e.end, "series": list(e.series)}
        for e in batch
    ]
    events = event_rows(transfers=transfers, tables=tables, alarms=alarms, tag=f"world_{seed}", real=False)
    OUT.mkdir(parents=True, exist_ok=True)
    pl.concat(list(alarms.values())).write_parquet(OUT / f"alarms_world_{seed}.parquet")
    events.write_parquet(path)
    return str(path)


def real_job() -> str:
    """Alarms and events of the real stream and the logged transfers."""
    state = detect_run.state()
    residual = state["panel"]["residual"]
    tables = ev.alarm_tables()
    tables = {"PAIR": tables["NEW"], "RONC": tables["RONC"].filter(pl.col("setting") == "R0_M48")}
    alarms = {k: loc.alarm_rows(table=t, residual=residual, family=k) for k, t in tables.items()}
    logged = ev.logged_transfers()
    transfers = [
        {"event": r["card"], "onset": r["onset"], "end": r["end"], "series": r["series"]}
        for r in logged.iter_rows(named=True)
    ]
    events = event_rows(transfers=transfers, tables=tables, alarms=alarms, tag="real", real=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pl.concat(list(alarms.values())).write_parquet(OUT / "alarms_real.parquet")
    events.write_parquet(OUT / "events_real.parquet")
    return str(OUT / "events_real.parquet")


def main(mode: str) -> None:
    """Run the injected worlds in a pool, or the real stream."""
    if mode == "real":
        print(real_job())
        return
    jobs = detect_run.world_events()
    print(f"{len(jobs)} worlds", flush=True)
    with ProcessPoolExecutor(N_WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        done = list(pool.map(world_job, jobs))
    print(f"{len(done)} worlds written")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main(sys.argv[1])
