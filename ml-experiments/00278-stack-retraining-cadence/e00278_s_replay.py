# ruff: noqa: E501
"""Replay the live weather layer's two stored columns with factors refitted monthly or quarterly.

Copied from experiment 202. For each scored month, the offset factor and spread factor per lead band
are fitted on the runs of the trailing 365 days whose last valid time (origin plus 168 hours) is at
or before the first origin of the month the factors were fitted at. The monthly arm fits at the
row's own month; the quarterly arm fits at the first month of the row's quarter (July, October,
January, April) and applies those factors to all three months. Writes `work/arms/scored_<cadence>.npy`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00278-*/e00278_s_replay.py
"""

import multiprocessing as mp
from collections.abc import Callable
from datetime import timedelta

import e00278_s_arms as arms
import e00278_s_common as common
import e00278_s_data as data
import numpy as np
import polars as pl

HORIZON = timedelta(hours=data.HORIZON_HOURS)


def load_all() -> tuple[pl.DataFrame, np.ndarray, int]:
    """Return every stored period's rows (tuning, replicate, scored) and member responses.

    Returns:
        The concatenated frame, the (rows, 51) responses, and the row index where the scored rows
        start.
    """
    frames, deltas = [], []
    for period in ("tuning", "replicate", "scored"):
        frame, delta = arms.labelled(period=period)
        frames.append(frame.with_columns(period=pl.lit(period)))
        deltas.append(delta)
    frame = pl.concat(frames, how="vertical")
    n_scored = frames[-1].height
    return frame, np.concatenate(deltas), frame.height - n_scored


def run_origin_array(*, frame: pl.DataFrame) -> np.ndarray:
    """Return each row's run origin (00 UTC run date plus the publication delay) as datetime64."""
    dates = common.run_dates()
    origins = np.array(
        [np.datetime64(common.origin_of(run=d).replace(tzinfo=None), "us") for d in dates]
    )
    return origins[frame["run"].to_numpy()]


def stats_float32(*, delta: np.ndarray) -> dict[str, np.ndarray]:
    """Return the two stored columns (`mu`, `sigma`) rounded to float32, plus the unused p10 and p90."""
    stats = arms.summaries(delta=delta)
    return {k: v.astype(np.float32).astype(np.float64) for k, v in stats.items()}


def maker(stats: dict[str, np.ndarray], rows: np.ndarray) -> Callable[[float], np.ndarray]:
    """Return `make_points(spread)` for the Gaussian arm over the chosen rows."""
    sub = {k: v[rows] for k, v in stats.items()}
    return lambda s: np.nan_to_num(arms.gaussian_points(stats=sub, spread=s))


def fit_on(
    *, frame: pl.DataFrame, stats: dict[str, np.ndarray], rows: np.ndarray, spread: bool
) -> list[tuple[float, float]]:
    """Fit `(f, s)` per lead band on the chosen rows."""
    return arms.fit_factors(frame=frame[rows], make_points=maker(stats, rows), fit_spread=spread)


STATE: dict = {}
"""Rows, responses, and origins shared with the spawned workers (set in `main`)."""
N_WORKERS = 6


def anchor_month(*, label: str, cadence: str) -> str:
    """Return the month (`YYYY-MM`) whose start the factors for `label` are fitted at."""
    year, month = (int(part) for part in label.split("-"))
    if cadence == "quarterly":
        month = 3 * ((month - 1) // 3) + 1
    return f"{year}-{month:02d}"


def replay_month(m: str) -> tuple[str, np.ndarray, dict]:
    """Fit and apply the monthly and quarterly factors for one scored month."""
    frame, origin = STATE["frame"], STATE["origin"]
    scored, month = STATE["scored"], STATE["month"]
    rows_m = np.flatnonzero(scored & (month == m))
    sub = frame[rows_m]
    horizon = np.timedelta64(int(HORIZON.total_seconds()), "s")
    quantiles: dict = {}
    log: dict = {}
    fitted: dict[str, list] = {}
    for cadence in ("monthly", "quarterly"):
        anchor = anchor_month(label=m, cadence=cadence)
        if anchor == m and "monthly" in fitted:
            fitted[cadence] = fitted["monthly"]
            log[cadence] = log["monthly"]
            quantiles[cadence] = quantiles["monthly"]
            continue
        first = origin[np.flatnonzero(scored & (month == anchor))].min()
        completed = (origin + horizon <= first) & (
            origin >= first - np.timedelta64(common.WINDOW_DAYS, "D")
        )
        fit_rows = np.flatnonzero(completed)
        assert origin[fit_rows].max() + horizon <= first
        assert first <= origin[rows_m].min()
        factors = fit_on(frame=frame, stats=STATE["stats32"], rows=fit_rows, spread=True)
        fitted[cadence] = factors
        log[cadence] = {"anchor": anchor, "n_fit_rows": len(fit_rows), "factors": factors}
        quantiles[cadence] = arms.apply_arm(
            frame=sub, make_points=maker(STATE["stats32"], rows_m), factors=factors
        )
    print(m, "done", flush=True)
    return m, rows_m, {"q": quantiles, "log": log}


def init_state() -> None:
    """Load the rows and responses in the calling process (workers are spawned, not forked)."""
    frame, delta, n_fit_end = load_all()
    STATE.update(
        frame=frame,
        delta=delta,
        origin=run_origin_array(frame=frame),
        scored=np.arange(frame.height) >= n_fit_end,
        month=frame["month"].to_numpy(),
        stats32=stats_float32(delta=delta),
        stats64=arms.summaries(delta=delta),
    )


def main() -> None:
    """Replay every scored month at both cadences and write the arrays."""
    init_state()
    frame, scored = STATE["frame"], STATE["scored"]
    months = sorted(set(STATE["month"][scored]))
    for key in ("delta", "stats32", "stats64"):
        del STATE[key]
    with mp.get_context("spawn").Pool(N_WORKERS, initializer=init_state) as pool:
        results = pool.map(replay_month, months, chunksize=1)
    scored_index = np.flatnonzero(scored)
    out = {
        n: frame.select(list(data.QUANTILES)).to_numpy().astype(np.float64)[scored]
        for n in ("monthly", "quarterly")
    }
    factors_log = {}
    for m, rows_m, res in results:
        pos = np.searchsorted(scored_index, rows_m)
        for name, q in res["q"].items():
            out[name][pos] = q
        factors_log[m] = res["log"]
    arm_dir = common.WORK_DIR / "arms"
    arm_dir.mkdir(parents=True, exist_ok=True)
    for name, q in out.items():
        np.save(arm_dir / f"scored_{name}.npy", np.sort(q, axis=1).astype(np.float32))
    frame[scored_index].write_parquet(arm_dir / "scored_frame.parquet")
    common.write_json_atomic(path=common.WORK_DIR / "solar_factors.json", content=factors_log)


if __name__ == "__main__":
    main()
