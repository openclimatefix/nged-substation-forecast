# ruff: noqa: E501
"""Member-level solar responses past 168 hours, from experiment 184's member code.

For every run of the three periods at the six Primaries, this script passes the 51 ensemble
members through experiment 184's T-all solar term with the forecast window set to 480 half-hours
(experiment 317's setting), and stores each member's response (`delta_m`) at the rows of
experiment 317's convex chain whose lead is above 168 hours. `--gate` instead recomputes rows of
168 hours or less for two runs and compares them with experiment 184's stored `delta_m`.

    PYTHONPATH=ml-experiments/00184-dp-ens-daylight-calibration \
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00319-*/e00319_members.py [--workers 6] [--gate]
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import e00184_common as c184  # ty: ignore[unresolved-import]
import e00184_members as m184  # ty: ignore[unresolved-import]
import e00184_tall as tall  # ty: ignore[unresolved-import]
import e00319_common as common
import numpy as np
import polars as pl
import studies.dp_ens_weather as weather
import torch  # ty: ignore[unresolved-import]


def positions(*, cache, full: dict, rows: pl.DataFrame) -> np.ndarray:  # noqa: ANN001
    """Return each row's index into a run's window."""
    start = int(full["window_start"][0])
    position = (
        np.searchsorted(cache.end_times, rows["valid_time"].dt.replace_time_zone(None).to_numpy())
        - start
    )
    if position.min() < 0 or position.max() >= len(full["base"]):
        msg = "a valid time lies outside the run's window"
        raise ValueError(msg)
    return position


def job(arguments: tuple[str, str, bool]) -> str:
    """Compute one site and period's member responses (or run the gate on two runs)."""
    site, period, gate = arguments
    torch.set_num_threads(1)
    weather.HORIZON_HALF_HOURS = common.HORIZON_HALF_HOURS
    started = time.monotonic()
    cache = c184.load_cache()
    setup = m184.load_context(site=site, cache=cache)
    fits = tall.load_fits(cadence="monthly", site=site)
    chain = pl.read_parquet(common.CHAIN_DIR / f"{site}.parquet").filter(pl.col("period") == period)
    if gate:
        chain = chain.filter(pl.col("lead_hours") <= common.STORED_HORIZON_HOURS)
        keep_runs = chain["run"].unique().sort().to_list()[:2]
        chain = chain.filter(pl.col("run").is_in(keep_runs))
    else:
        chain = chain.filter(pl.col("lead_hours") > common.STORED_HORIZON_HOURS)
    pieces = []
    for run_index, rows in chain.group_by("run", maintain_order=True):
        full = m184.compute_run(cache=cache, setup=setup, fits=fits, index=int(run_index[0]))
        position = positions(cache=cache, full=full, rows=rows)
        pieces.append(
            {
                "run": rows["run"].to_numpy(),
                "valid_time": rows["valid_time"].dt.replace_time_zone(None).to_numpy(),
                "delta_m": full["delta_m"][position].astype(np.float32),
                "n_members": full["n_members"][position],
            }
        )
    out = {name: np.concatenate([p[name] for p in pieces]) for name in pieces[0]}
    if gate:
        stored = np.load(common.E184_ARMS.parent / "members" / f"{site}_{period}.npz")
        key = {
            (int(r), v): i
            for i, (r, v) in enumerate(zip(stored["run"], stored["valid_time"], strict=True))
        }
        index = np.array(
            [key[(int(r), v)] for r, v in zip(out["run"], out["valid_time"], strict=True)]
        )
        diff = np.nanmax(np.abs(out["delta_m"] - stored["delta_m"][index]))
        return f"GATE {site} {period}: {len(index)} rows, max |delta_m diff| {diff:.2e}, {time.monotonic() - started:.0f} s"
    target = common.MEMBERS_DIR / f"{site}_{period}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp.npz")
    np.savez(temporary, **out)  # ty: ignore[invalid-argument-type]
    temporary.rename(target)
    return f"{site} {period}: {len(out['run']):,} rows, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Compute every site and period in parallel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--sites", nargs="+", default=list(common.SCORED_SITES))
    args = parser.parse_args()
    periods = ("scored",) if args.gate else common.PERIODS
    jobs = [(s, p, args.gate) for p in periods for s in args.sites]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
