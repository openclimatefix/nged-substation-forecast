r"""Tune, score, and check the joint edge-flow estimator (EF) against experiment 65's stored D0M.

Stages, each writing to the private output folder:

- `check`: a planted flow and slack are recovered, and the injection reproduces experiment 65's
  moved MW.
- `tune`: for each cell of the grid of fused ratio and slack multiplier, match `lambda` to 80
  unexplained odd-month alarms on the real residual, then score 300 odd-month transfers.
- `score`: the 620 even-month headline transfers under S2, S3, and S4 at the chosen penalties.
- `real`: the real-residual alarms and the labelled real transfers at the chosen penalties.

    uv run --with cvxpy python ml-experiments/00067-*/e00067_run.py check tune score real
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00067_common import (
    GRID_LENGTH,
    LABELLED_PATH,
    ML_EXPERIMENTS_DATA_DIR,
    TUNING_MONTHS,
    WORK_DIR,
    load_panel,
    near_log_masks,
    stamp_months,
)
from e00067_detect import (
    EXPLAIN_PAD,
    SEED_43,
    SEED_NEW,
    new_events,
    original_events,
    real_events,
    score_real,
    seeded_batches,
)
from e00067_detectors import refit_stamps
from e00067_edgeflow import (
    BIN,
    HUBER_THRESHOLD,
    WINDOW_BINS,
    EdgeFlowProblem,
    Penalties,
    build_network,
    find_runs,
    first_bin_around,
    fit_window,
    hourly_panel,
    noise_scales,
    window_starts,
)
from e00067_inject import BLOCK, HEADLINE_BAND, Event
from e00067_intervals import Interval
from e00067_profile import NON_MATCHING, Shape, fit_profile_shares, inject_shaped
from e00067_score import by_series, match_event

EXPERIMENT_58_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00058-sw2-8b-donor-step-recut-and-stronger-veto"
)
EXPERIMENT_65_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00065-sw2-14-combined-single-series-and-neighbour-score"
)
TARGET: Final[int] = 80
"""D0M's odd-month unexplained-alarm count, which EF is held to."""
FUSED_RATIOS: Final[tuple[float, ...]] = (1.0, 3.0, 10.0)
SLACK_MULTIPLIERS: Final[tuple[float, ...]] = (1.5, 3.0)
LAMBDA_RANGE: Final[tuple[float, float]] = (0.3, 30.0)
BISECTION_STEPS: Final[int] = 7
TUNING_EVENTS: Final[int] = 300
HOLIDAY_FIRST_ID: Final[int] = 3000
REFIT_PAD: Final[int] = BLOCK
"""A run that starts within 24 hours of a monthly refit of the residual model is dropped."""
N_WORKERS: Final[int] = 4
_STATE: dict = {}
_RUN: dict = {"huber": HUBER_THRESHOLD}
"""The Huber threshold of this run; `main` sets it from `--huber`."""


def init_worker(state: dict) -> None:
    """Pool initializer: share the panel, hourly noise scales, and profile shares."""
    network = build_network()
    z = hourly_panel(residual=state["panel"]["residual"], network=network)
    state["network"], state["sigma"] = network, state["sigma"]
    state["problem"] = EdgeFlowProblem(network=network, sigma=state["sigma"], huber=state["huber"])
    state["z_plain"] = z / state["sigma"][:, None]
    _STATE.update(state)


def refit_mask() -> np.ndarray:
    """True within `REFIT_PAD` stamps of a monthly refit of the residual model."""
    mask = np.zeros(GRID_LENGTH, dtype=bool)
    for stamp in refit_stamps():
        mask[max(stamp - REFIT_PAD + 1, 0) : stamp + REFIT_PAD] = True
    return mask


def usable(*, intervals: list[Interval]) -> list[Interval]:
    """Keep the intervals whose onset is not within 24 hours of a refit."""
    near = refit_mask()
    return [iv for iv in intervals if not near[iv.onset]]


def z_of(*, residual: dict[int, np.ndarray]) -> np.ndarray:
    """Normalised hourly residual of every series, shape (series, bins)."""
    return hourly_panel(residual=residual, network=_STATE["network"]) / _STATE["sigma"][:, None]


def real_window_job(args: tuple[int, int, int, Penalties]) -> list[Interval]:
    """Fit one window of the real residual; return the intervals that start in its kept range."""
    first, keep_lo, keep_hi, penalties = args
    intervals = fit_window(
        problem=_STATE["problem"], z=_STATE["z_plain"], first_bin=first, penalties=penalties
    )
    return [iv for iv in intervals if BIN * keep_lo <= iv.onset < BIN * keep_hi]


def event_job(args: tuple[int, list[Event], Shape, list[int], Penalties]) -> list[dict]:
    """Inject one batch with one shape and fit the window around each chosen transfer."""
    seed, batch, shape, chosen, penalties = args
    panel = _STATE["panel"]
    injection = inject_shaped(
        power=panel["power"], events=batch, seed=seed, shape=shape, shares=_STATE["shares"]
    )
    z = z_of(residual=injection.fast_residual(panel["residual"]))
    rows = []
    for event in batch:
        if event.event_id not in chosen:
            continue
        first = first_bin_around(onset_stamp=event.onset, total_bins=z.shape[1])
        intervals = usable(
            intervals=fit_window(
                problem=_STATE["problem"], z=z, first_bin=first, penalties=penalties
            )
        )
        match = match_event(
            series=event.series,
            onset=event.onset,
            end=event.end,
            intervals=by_series(intervals=intervals),
        )
        rows.append(
            {
                "event_id": event.event_id,
                "setting": "EF_PE",
                "shape": shape,
                "found": match.found,
                "onset_found": match.onset_found,
                "moved_mw": float(-np.mean(injection.delta[event.donor][event.onset : event.end])),
                "inaccurate": _STATE["problem"].last_status != "optimal",
            }
        )
    return rows


def run_pool(*, function: Callable, jobs: list, state: dict) -> list:
    """Run `function` over `jobs` in worker processes that share `state`."""
    with ProcessPoolExecutor(
        N_WORKERS,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
        initargs=(state,),
    ) as pool:
        return list(pool.map(function, jobs))


def make_state() -> dict:
    """The panel, hourly noise scales, and profile shares that every worker needs."""
    panel = load_panel()
    network = build_network()
    sigma = noise_scales(panel=hourly_panel(residual=panel["residual"], network=network))
    return {
        "panel": panel,
        "sigma": sigma,
        "shares": fit_profile_shares(panel=panel),
        "huber": _RUN["huber"],
    }


def event_groups(*, panel: dict) -> list[tuple[int, list[Event]]]:
    """Experiment 58's original, extra, and holiday transfers in its batches and with its seeds."""
    original = original_events(power=panel["power"])
    extra, holiday = new_events(power=panel["power"])
    return seeded_batches(events=original, seed=SEED_43) + seeded_batches(
        events=extra + holiday, seed=SEED_NEW
    )


def chosen_ids(*, even: bool) -> list[int]:
    """Headline-band, non-holiday transfers that start in even (or odd) months, by event id."""
    events = pl.read_parquet(EXPERIMENT_58_DIR / "events.parquet")
    month = stamp_months()[events["onset"].to_numpy()]
    selected = events.filter(
        (pl.col("size_band") == HEADLINE_BAND[0])
        & (pl.col("duration_band") == HEADLINE_BAND[1])
        & (pl.col("set") != "holiday")
        & pl.Series((month % 2 == 0) if even else (month % 2 == 1))
    ).sort("event_id")
    return selected["event_id"].to_list()


def event_jobs(
    *, groups: list, wanted: dict[int, list[Shape]], penalties: Penalties
) -> list[tuple]:
    """One job per batch and shape, holding the wanted transfers of that batch and shape."""
    jobs = []
    for seed, batch in groups:
        for shape in NON_MATCHING:
            chosen = [e.event_id for e in batch if shape in wanted.get(e.event_id, [])]
            if chosen:
                jobs.append((seed, batch, shape, chosen, penalties))
    return jobs


def alarm_table(*, intervals: list[Interval]) -> pl.DataFrame:
    """Two alarms per series and run, at its onset and at its end, as D0M raises one per step.

    D0M alarms on the onset step and on the return step of a block of false load, so a run counts
    twice here too. Each alarm is explained when a logged closure touches its series within 24
    hours of its own stamp.
    """
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    months = stamp_months()
    runs = sorted({(iv.series, iv.onset, iv.end) for iv in intervals})
    rows = [
        {
            "series": series,
            "stamp": stamp,
            "kind": kind,
            "month": int(months[stamp]),
            "explained": bool(logged[series][stamp]),
        }
        for series, onset, end in runs
        for kind, stamp in (("onset", onset), ("end", min(end, GRID_LENGTH - 1)))
    ]
    return pl.DataFrame(
        rows,
        schema={
            "series": pl.Int64,
            "stamp": pl.Int64,
            "kind": pl.String,
            "month": pl.Int64,
            "explained": pl.Boolean,
        },
    )


def real_intervals(*, state: dict, penalties: Penalties) -> list[Interval]:
    """Fit every window of the real residual and return the usable runs."""
    total_bins = GRID_LENGTH // BIN
    jobs = [(a, b, c, penalties) for a, b, c in window_starts(total_bins=total_bins)]
    pieces = run_pool(function=real_window_job, jobs=jobs, state=state)
    return usable(intervals=[iv for part in pieces for iv in part])


def unexplained(*, table: pl.DataFrame) -> tuple[int, int]:
    """Unexplained alarm counts in odd and even months."""
    bad = table.filter(~pl.col("explained"))
    odd = bad.filter(pl.col("month").is_in(list(TUNING_MONTHS))).height
    return odd, bad.height - odd


def match_lambda(*, state: dict, fused_ratio: float, slack_multiplier: float) -> tuple[float, int]:
    """The smallest `lambda`, on a log scale, with at most `TARGET` odd-month unexplained alarms."""
    stretch = _RUN["huber"] / HUBER_THRESHOLD
    low, high = np.log(LAMBDA_RANGE[0] * stretch), np.log(LAMBDA_RANGE[1] * stretch)
    best = (float(np.exp(high)), -1)
    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (low + high)
        penalties = Penalties.from_knobs(
            level=float(np.exp(mid)), fused_ratio=fused_ratio, slack_multiplier=slack_multiplier
        )
        odd, _ = unexplained(
            table=alarm_table(intervals=real_intervals(state=state, penalties=penalties))
        )
        print(f"    lambda {np.exp(mid):.3f}: {odd} odd-month unexplained alarms", flush=True)
        if odd <= TARGET:
            high, best = mid, (float(np.exp(mid)), odd)
        else:
            low = mid
    return best


def stored_weights() -> dict[int, float]:
    """Each transfer's weight: its mean moved MW under S1, from experiment 65's stored results."""
    stored = pl.read_parquet(EXPERIMENT_65_DIR / "event_results.parquet")
    one = stored.filter(pl.col("shape") == "S1", pl.col("setting") == "D0M_TR")
    return dict(zip(one["event_id"].to_list(), one["moved_mw"].to_list(), strict=True))


def weighted_recall(*, rows: pl.DataFrame, weights: dict[int, float]) -> float:
    """MW-weighted recall of a table of per-transfer rows."""
    w = np.array([weights[i] for i in rows["event_id"].to_list()])
    return float((w * rows["found"].to_numpy()).sum() / w.sum())


def tune(*, out: Path) -> None:
    """Match `lambda` in every grid cell on the real residual, then pick the cell by recall."""
    state = make_state()
    groups = event_groups(panel=state["panel"])
    ids = chosen_ids(even=False)
    chosen = ids[:: max(len(ids) // TUNING_EVENTS, 1)][:TUNING_EVENTS]
    wanted = {i: [NON_MATCHING[i % 3]] for i in chosen}
    weights = stored_weights()
    cells = []
    wide = _RUN["huber"] != HUBER_THRESHOLD
    for ratio in FUSED_RATIOS[1:] if wide else FUSED_RATIOS:
        for multiplier in SLACK_MULTIPLIERS[:1] if wide else SLACK_MULTIPLIERS:
            print(f"cell fused ratio {ratio}, slack multiplier {multiplier}", flush=True)
            level, count = match_lambda(state=state, fused_ratio=ratio, slack_multiplier=multiplier)
            if count < 0:
                print("  no lambda in range reached the alarm target; cell dropped", flush=True)
                cells.append({"fused_ratio": ratio, "slack_multiplier": multiplier, "recall": -1})
                continue
            penalties = Penalties.from_knobs(
                level=level, fused_ratio=ratio, slack_multiplier=multiplier
            )
            rows = pl.DataFrame(
                [
                    r
                    for part in run_pool(
                        function=event_job,
                        jobs=event_jobs(groups=groups, wanted=wanted, penalties=penalties),
                        state=state,
                    )
                    for r in part
                ]
            )
            recall = weighted_recall(rows=rows, weights=weights)
            print(f"  lambda {level:.3f}, alarms {count}, recall {recall:.3f}", flush=True)
            cells.append(
                {
                    "fused_ratio": ratio,
                    "slack_multiplier": multiplier,
                    "level": level,
                    "odd_unexplained": count,
                    "recall": recall,
                    "n_transfers": rows.height,
                }
            )
            (out / "tuning_partial.json").write_text(json.dumps(cells))
    if all(c["recall"] < 0 for c in cells):
        raise SystemExit("STOP: no cell reached the alarm target; widen LAMBDA_RANGE")
    best = max(cells, key=lambda c: (c["recall"], c.get("level", 0.0)))
    print(f"chosen: {best}")
    (out / "tuning.json").write_text(
        json.dumps({"cells": cells, "chosen": best, "huber": _RUN["huber"]})
    )


def chosen_penalties(*, out: Path) -> Penalties:
    """The penalties that `tune` chose."""
    tuning = json.loads((out / "tuning.json").read_text())
    if tuning["huber"] != _RUN["huber"]:
        raise SystemExit(f"STOP: tuned at Huber threshold {tuning['huber']}, not {_RUN['huber']}")
    best = tuning["chosen"]
    return Penalties.from_knobs(
        level=best["level"],
        fused_ratio=best["fused_ratio"],
        slack_multiplier=best["slack_multiplier"],
    )


def score(*, out: Path) -> None:
    """Score the 620 even-month headline transfers under S2, S3, and S4."""
    state = make_state()
    groups = event_groups(panel=state["panel"])
    wanted = {i: list(NON_MATCHING) for i in chosen_ids(even=True)}
    jobs = event_jobs(groups=groups, wanted=wanted, penalties=chosen_penalties(out=out))
    rows = [r for part in run_pool(function=event_job, jobs=jobs, state=state) for r in part]
    scored = pl.DataFrame(rows)
    scored.write_parquet(out / "event_results.parquet")
    print(f"{scored.height} rows; {int(scored['inaccurate'].sum())} fits were optimal_inaccurate")


def real(*, out: Path) -> None:
    """Fit the real residual and score the labelled real transfers."""
    state = make_state()
    penalties = chosen_penalties(out=out)
    intervals = real_intervals(state=state, penalties=penalties)
    table = alarm_table(intervals=intervals)
    table.write_parquet(out / "alarms.parquet")
    odd, even = unexplained(table=table)
    print(f"unexplained alarms: odd months {odd}, even months {even}")
    if odd > TARGET:
        raise SystemExit(f"STOP: {odd} odd-month unexplained alarms, above the target {TARGET}")
    events = real_events(power=state["panel"]["power"])
    moved = pl.read_parquet(LABELLED_PATH).select("card_id", "moved_mw_p50")
    scored = score_real(results={"EF_PE": (pl.DataFrame(), intervals)}, real=events).join(
        moved, on="card_id", validate="m:1"
    )
    scored.with_columns(month=pl.Series(stamp_months()[scored["onset"].to_numpy()])).write_parquet(
        out / "real_results.parquet"
    )


def check(*, out: Path) -> None:
    """A planted flow and slack are recovered, and the injection matches experiment 65's."""
    state = make_state()
    init_worker(state)
    network, sigma = _STATE["network"], _STATE["sigma"]
    z = np.zeros((len(network.series), WINDOW_BINS))
    # With a cheap slack (every node has an MVA neighbour that a flow could use instead):
    # A 5-noise-unit flow from series 8 to 10 for bins 300 to 500, and a slack on series 12.
    edge = network.edges.index((8, 10))
    scale = _STATE["problem"].edge_sigma[edge] / sigma[network.series.index(10)]
    z[network.series.index(10), 300:500] += 5.0 * scale
    z[network.series.index(8), 300:500] -= (
        5.0 * _STATE["problem"].edge_sigma[edge] / sigma[network.series.index(8)]
    )
    z[network.series.index(12), 600:800] += 5.0
    flow, slack = _STATE["problem"].solve(
        z=z, penalties=Penalties.from_knobs(level=0.5, fused_ratio=3.0, slack_multiplier=0.5)
    )
    runs = find_runs(values=flow[edge])
    slack_runs = find_runs(values=slack[network.series.index(12)])
    print(f"planted flow 300-500 recovered as {runs}; planted slack 600-800 as {slack_runs}")
    if not (
        len(runs) == 1
        and abs(runs[0][0] - 300) <= 3
        and abs(runs[0][1] - 500) <= 3
        and len(slack_runs) == 1
        and abs(slack_runs[0][0] - 600) <= 3
        and abs(slack_runs[0][1] - 800) <= 3
    ):
        raise SystemExit("STOP: the fit does not recover a planted flow and slack")
    stored = pl.read_parquet(EXPERIMENT_65_DIR / "event_results.parquet").filter(
        pl.col("setting") == "D0M_PE"
    )
    seed, batch = event_groups(panel=state["panel"])[0]
    for shape in NON_MATCHING:
        injection = inject_shaped(
            power=state["panel"]["power"],
            events=batch,
            seed=seed,
            shape=shape,
            shares=state["shares"],
        )
        for event in batch[:20]:
            moved = float(-np.mean(injection.delta[event.donor][event.onset : event.end]))
            old = stored.filter(pl.col("event_id") == event.event_id, pl.col("shape") == shape)[
                "moved_mw"
            ].item()
            if abs(moved - old) > 1e-6 * max(abs(old), 1.0):
                raise SystemExit(
                    f"STOP: transfer {event.event_id} {shape} moves {moved}, not {old}"
                )
    print("injection reproduces experiment 65's moved MW")


def main() -> None:
    """Run the requested stages."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stages", nargs="+", choices=("check", "tune", "score", "real"))
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    parser.add_argument("--huber", type=float, default=HUBER_THRESHOLD)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    _RUN["huber"] = args.huber
    stages = {"check": check, "tune": tune, "score": score, "real": real}
    for stage in args.stages:
        stages[stage](out=args.out)


if __name__ == "__main__":
    main()
