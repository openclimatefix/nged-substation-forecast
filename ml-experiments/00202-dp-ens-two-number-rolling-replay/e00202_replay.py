# ruff: noqa: E501
"""Replay the live weather layer's two stored columns, with factors refitted monthly.

For each scored month, the offset factor `f` and spread factor `s` per lead band are fitted on the
runs of the trailing 365 days whose last valid time (origin plus 168 hours) is at or before the
month's first origin. The two columns (mean and standard deviation of the 51 members' solar
responses) are rounded to float32, as the weather layer would store them. Arms:

| Arm | Meaning |
|---|---|
| `r2` | rolling replay, float32 columns, factors from completed history |
| `r2f64` | the same with float64 columns (gate G2) |
| `r2o` | factors fitted on the scored month itself (in-sample ceiling, outside the verdict) |
| `rr` | placebo: every response zero, rolling factors |

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00202-*/e00202_replay.py
"""

import json
import multiprocessing as mp
from collections.abc import Callable
from datetime import timedelta

import e00202_arms as arms
import e00202_common as common
import e00202_data as data
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
N_WORKERS = 3


def replay_month(m: str) -> tuple[str, np.ndarray, dict]:
    """Fit and apply every arm for one scored month; return its row positions' quantiles."""
    frame, delta, origin = STATE["frame"], STATE["delta"], STATE["origin"]
    scored, month = STATE["scored"], STATE["month"]
    rows_m = np.flatnonzero(scored & (month == m))
    first = origin[rows_m].min()
    horizon = np.timedelta64(int(HORIZON.total_seconds()), "s")
    completed = (origin + horizon <= first) & (
        origin >= first - np.timedelta64(common.WINDOW_DAYS, "D")
    )
    fit_rows = np.flatnonzero(completed)
    assert origin[fit_rows].max() + horizon <= first
    sub = frame[rows_m]
    spec = {
        "r2": (STATE["stats32"], fit_rows),
        "r2f64": (STATE["stats64"], fit_rows),
        "r2o": (STATE["stats32"], rows_m),
    }
    quantiles, log = {}, {"n_fit_rows": len(fit_rows), "first_origin": str(first)}
    for name, (stats, fr) in spec.items():
        factors = fit_on(frame=frame, stats=stats, rows=fr, spread=True)
        log[name] = factors
        quantiles[name] = arms.apply_arm(
            frame=sub, make_points=maker(stats, rows_m), factors=factors
        )
    fit_zero = np.zeros_like(delta[fit_rows])
    factors = arms.fit_factors(
        frame=frame[fit_rows], make_points=lambda s: fit_zero, fit_spread=False
    )
    log["rr"] = factors
    zero = np.zeros_like(delta[rows_m])
    quantiles["rr"] = arms.apply_arm(frame=sub, make_points=lambda s: zero, factors=factors)
    print(m, "done, fit rows", len(fit_rows), flush=True)
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


def gate_g1() -> float:
    """Return the largest quantile difference between 193's stored Q2 and this code path.

    The code path is given 193's fixed factors and float64 columns.
    """
    scored, stats64 = STATE["scored"], STATE["stats64"]
    fixed = json.loads(common.E193_FACTORS.read_text())["q2"]
    g1 = arms.apply_arm(
        frame=STATE["frame"][np.flatnonzero(scored)],
        make_points=lambda s: np.nan_to_num(
            arms.gaussian_points(stats={k: v[scored] for k, v in stats64.items()}, spread=s)
        ),
        factors=[tuple(x) for x in fixed],
    )
    stored_q2 = np.load(common.E193_ARM_DIR / "scored_q2.npy").astype(np.float64)
    return float(np.abs(np.sort(g1, axis=1) - stored_q2).max())


def main() -> None:
    """Replay every scored month and write the arms and gate files."""
    init_state()
    frame, origin, scored = STATE["frame"], STATE["origin"], STATE["scored"]
    dates = common.run_dates()
    inits = np.array([np.datetime64(d.replace(tzinfo=None), "us") for d in dates])[
        frame["run"].to_numpy()
    ]
    gates = {"availability_ok": bool(np.all(inits + np.timedelta64(9, "h") <= origin))}
    gates["g1_max_abs_diff_vs_193_q2"] = gate_g1()
    months = sorted(set(STATE["month"][scored]))
    for key in ("delta", "stats32", "stats64"):
        del STATE[key]
    with mp.get_context("spawn").Pool(N_WORKERS, initializer=init_state) as pool:
        results = pool.map(replay_month, months, chunksize=1)
    scored_index = np.flatnonzero(scored)
    out = {
        n: frame.select(list(data.QUANTILES)).to_numpy().astype(np.float64)[scored]
        for n in ("r2", "r2f64", "r2o", "rr")
    }
    factors_log = {}
    for m, rows_m, res in results:
        pos = np.searchsorted(scored_index, rows_m)
        for name, q in res["q"].items():
            out[name][pos] = q
        factors_log[m] = res["log"]
    scored_frame = frame[scored_index]
    arm_dir = common.WORK_DIR / "arms"
    arm_dir.mkdir(parents=True, exist_ok=True)
    for name, q in out.items():
        np.save(arm_dir / f"scored_{name}.npy", np.sort(q, axis=1).astype(np.float32))
    scored_frame.write_parquet(arm_dir / "scored_frame.parquet")
    common.write_json_atomic(path=common.WORK_DIR / "factors.json", content=factors_log)
    common.write_json_atomic(path=common.WORK_DIR / "gates_pre.json", content=gates)
    print(gates)


if __name__ == "__main__":
    main()
