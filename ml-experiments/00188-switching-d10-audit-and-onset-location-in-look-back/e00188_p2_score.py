"""Part 2, stage 2: score the mask forecast with the mask starting at four different stamps.

Every arm is GAM-K's cached member forecasts plus a correction from the Kalman filter run on the
stream with readings set missing inside the known alarm intervals. The arms differ only in the
first stamp of an interval that is masked:

- `MASK_ALARM`: the stamp the alarm is known (readings before it, back to the step, stay in).
- `MASK_STEP`: the detector's own step stamp (experiment 171's arm).
- `MASK_LOCATED`: the onset located in the look-back at the first 6-hourly origin that sees the alarm.
- `MASK_TRUE`: the true onset (injected onset, or logged onset) for alarms that find a transfer.

`K0` is no mask, and `CLEAN` (injected worlds only) is the stream without the injected megawatts.
The scored rows are every lead of the daily run whose newest usable stamp lies within 24 hours
after the alarm's first usable origin.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00188_p2_score.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import e00188_alarm as alarm
import e00188_events as ev
import e00188_fed as fed
import e00188_locate as loc
import e00188_nra as nra
import numpy as np
import polars as pl
from e00188_common import WORK_DIR

N_WORKERS: Final[int] = 4
P2: Final[Path] = WORK_DIR / "p2"
OUT: Final[Path] = WORK_DIR / "p2_scores"
WORLDS: Final[Path] = WORK_DIR.parent / "00141-switching-detector-fed-into-gam-k" / "worlds"
DETECT: Final[Path] = WORK_DIR.parent / "00156-switching-faster-onset-alarm-from-opposite-steps" / "detect"
RUN_WINDOW: Final[int] = 48
"""Half-hours (24 hours) after the first usable origin within which a run's newest stamp may lie."""
ARMS: Final[tuple[str, ...]] = ("MASK_ALARM", "MASK_STEP", "MASK_LOCATED", "MASK_TRUE")


def pair_window() -> int:
    """The pair alarm's headline window of experiment 156."""
    return int(json.loads((DETECT / "headline.json").read_text())["window"])


def interval_tables(*, tag: str) -> dict[str, pl.DataFrame]:
    """The PAIR and RONC interval tables of a world or of the real stream."""
    if tag == "real":
        tables = ev.alarm_tables()
        return {"PAIR": tables["NEW"], "RONC": tables["RONC"].filter(pl.col("setting") == "R0_M48")}
    seed = int(tag.split("_")[1])
    return {
        "PAIR": pl.read_parquet(DETECT / f"world_{seed}_W{pair_window()}_pair.parquet"),
        "RONC": alarm.with_ronc_onset_known(
            intervals=pl.read_parquet(WORLDS / f"world_{seed}.parquet").filter(
                pl.col("setting") == "R0_M48"
            )
        ),
    }


def family_spans(
    *, data, site: str, table: pl.DataFrame, alarms: pl.DataFrame, true_onsets: dict
) -> dict[str, list[nra.Span]]:
    """One span list per arm for a family at a site: the arms differ in the span's onset."""
    offset = fed.stamp_offset(data)
    series = fed.SERIES_OF_SITE[site]
    key = "alarm_series" if "alarm_series" in table.columns else None
    rows = table.filter(pl.col("series") == series)
    out: dict[str, list[nra.Span]] = {a: [] for a in ARMS}
    for r in rows.iter_rows(named=True):
        a_series = r[key] if key else series
        found = alarms.filter((pl.col("alarm_series") == a_series) & (pl.col("onset") == r["onset"]))
        if found.is_empty():
            continue
        a = found.row(0, named=True)
        step = r["step_mw"]
        points = 100.0 * step / fed.CAPACITY_MW[site] if np.isfinite(step) else 0.0
        end = r["end"] + offset
        start = a["known"] + offset
        known = max(r["known_from"] + offset, end + nra.END_LAG, start + 1)
        onsets = {
            "MASK_ALARM": a["known"],
            "MASK_STEP": a["onset"],
            "MASK_LOCATED": a["located"],
            "MASK_TRUE": true_onsets.get((a_series, a["onset"]), a["onset"]),
        }
        for arm, onset in onsets.items():
            out[arm].append(nra.Span(onset + offset, end, start, known, points))
    return out


def job(args: tuple[str, str]) -> str:
    """Score every arm at one site in one world, or in the real stream."""
    site, tag = args
    path = OUT / f"{site}_{tag}.parquet"
    if path.exists():
        return str(path)
    events_all = pl.read_parquet(P2 / f"events_{tag}.parquet") if (P2 / f"events_{tag}.parquet").stat().st_size else None
    if events_all is None or events_all.is_empty() or "site" not in events_all.columns:
        return ""
    events = events_all.filter(pl.col("site") == site)
    if events.is_empty():
        return ""
    real = tag == "real"
    data = fed.get_site(site)
    offset = fed.stamp_offset(data)
    n = len(data.stream)
    cutoffs_run = data.cutoffs(delay_hours=fed.DELAY_HOURS)
    usable = cutoffs_run >= 0
    assert np.all((cutoffs_run[usable] - offset) % loc.HALF == loc.ORIGIN_PHASE), "origin phase"
    cutoffs = cutoffs_run[data.run_of_row]
    stream = data.stream
    if not real:
        seed = int(tag.split("_")[1])
        with np.load(WORLDS / f"world_{seed}.npz") as stored:
            delta = stored[f"delta_{fed.SERIES_OF_SITE[site]}"]
        stream = np.where(
            np.isfinite(data.stream),
            data.stream
            + fed.to_positions(array=100.0 * np.nan_to_num(delta) / fed.CAPACITY_MW[site], data=data),
            np.nan,
        )
    ok = np.isfinite(data.actual)
    if not real:
        ok &= ~data.features.switching_mask[data.position]
        ok &= data.period_rows(period="replicate") | data.period_rows(period="scored")
    tables = interval_tables(tag=tag)
    all_alarms = pl.read_parquet(P2 / f"alarms_{tag}.parquet")
    frames = []
    for family, table in tables.items():
        evs = events.filter(pl.col("family") == family)
        if evs.is_empty():
            continue
        alarms = all_alarms.filter(pl.col("family") == family)
        true_onsets = {(r["alarm_series"], r["onset"]): r["true_onset"] for r in evs.iter_rows(named=True)}
        spans = family_spans(data=data, site=site, table=table, alarms=alarms, true_onsets=true_onsets)
        pairs = []
        for r in evs.iter_rows(named=True):
            c1 = r["first_origin"] + offset
            runs = np.flatnonzero((cutoffs_run >= c1) & (cutoffs_run < c1 + RUN_WINDOW))
            for run in runs:
                rows = np.flatnonzero((data.run_of_row == run) & ok)
                pairs.extend((int(x), r["event"]) for x in rows)
        if not pairs:
            continue
        rows = np.unique([p[0] for p in pairs])
        corrections = {
            "K0": fed.correction_for(data=data, stream=stream, prior=np.zeros(n), multiplier=1.0)
        }
        if not real:
            corrections["CLEAN"] = fed.correction_for(
                data=data, stream=data.stream, prior=np.zeros(n), multiplier=1.0
            )
        for arm in ARMS:
            corrections[arm] = nra.subtraction_correction(
                data=data, stream=stream, spans=spans[arm], cutoffs=cutoffs, rows=rows, mask=True
            )
        scored = nra.score_nra(data=data, site=site, corrections=corrections, rows=rows)
        mapping = pl.DataFrame(pairs, schema={"row": pl.Int64, "event": pl.String}, orient="row")
        meta = evs.select("event", "true_onset", "onset", "known", "located", "first_origin")
        joined = scored.join(mapping, on="row").join(meta, on="event")
        frames.append(
            joined.with_columns(
                family=pl.lit(family),
                tag=pl.lit(tag),
                cutoff=pl.Series(cutoffs[joined["row"].to_numpy()] - offset),
            )
        )
    if not frames:
        return ""
    OUT.mkdir(parents=True, exist_ok=True)
    pl.concat(frames).write_parquet(path)
    return str(path)


def main() -> None:
    """Score every site in every world and in the real stream."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tags = ["real"] + sorted(p.stem.removeprefix("events_") for p in P2.glob("events_world_*.parquet"))
    jobs = [(s, t) for t in tags for s in fed.SERIES_OF_SITE]
    if args.only:
        jobs = [j for j in jobs if args.only in j[1]]
    if args.limit:
        jobs = jobs[: args.limit]
    with ProcessPoolExecutor(N_WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        done = [p for p in pool.map(job, jobs) if p]
    print(f"{len(done)} files written")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
