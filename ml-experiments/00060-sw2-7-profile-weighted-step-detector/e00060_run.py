r"""Match D3's threshold on odd months, then score D0M and D3 on injected and real transfers.

The injected transfers use five shapes; the real ones are the labelled transfers.

Stages, in order. Each writes files to the private output folder.

- `tune`: D3's matched threshold (and D0M's, which must come back as 4.0).
- `harness-check`: with shape S1, D0M must reproduce experiment 58's outcome for every transfer.
- `score`: the outcome of every injected transfer for both detectors and every shape.
- `real`: the labelled real transfers' outcomes and the real alarm counts.

    uv run python ml-experiments/00060-*/e00060_run.py tune harness-check score real
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00060_common import LABELLED_PATH, ML_EXPERIMENTS_DATA_DIR, WORK_DIR, near_log_masks
from e00060_detect import (
    EXPLAIN_PAD,
    NO_VETO,
    SEED_43,
    SEED_NEW,
    explained_flags,
    load,
    new_events,
    original_events,
    real_alarm_rows,
    real_events,
    score_real,
    seeded_batches,
    tune_d2,
)
from e00060_detectors import alarms, prepare
from e00060_inject import BLOCK, Event
from e00060_intervals import build_intervals, relative_residual
from e00060_profile import (
    SHAPES,
    Shape,
    fit_profile_shares,
    inject_shaped,
    prepare_profile,
    shares_json,
)
from e00060_score import by_series, match_event

WINDOW: Final[int] = 48
D0M_THRESHOLD: Final[float] = 4.0
EXPERIMENT_58_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00058-sw2-8b-donor-step-recut-and-stronger-veto"
)
N_WORKERS: Final[int] = 4
SETTINGS: Final[tuple[str, ...]] = ("D0M_48", "D3_48")
_STATE: dict = {}


def build_detectors(*, residual: dict, expected: dict, scales: dict, thresholds: dict) -> dict:
    """Alarms and plain intervals of D0M and D3 on one residual panel."""
    relative = {s: relative_residual(residual=residual[s], expected=expected[s]) for s in residual}
    inputs = {
        "D0M_48": prepare(residuals=residual, window=WINDOW, scale=scales["plain"]),
        "D3_48": prepare_profile(
            residuals=residual, expected=expected, window=WINDOW, scale=scales["profile"]
        ),
    }
    out = {}
    for setting, one in inputs.items():
        table = alarms(inputs=one, detector="D0", threshold=thresholds[setting])
        out[setting] = (
            table,
            build_intervals(alarm_table=table, relative=relative, window=WINDOW, detector="D0"),
        )
    return out


def init_worker(state: dict) -> None:
    """Pool initializer: share the panel, scales, and thresholds."""
    _STATE.update(state)


def score_batch(args: tuple[int, list[Event], Shape]) -> list[dict]:
    """Inject one batch with one shape, detect with both detectors, return one row per event."""
    seed, batch, shape = args
    panel = _STATE["panel"]
    injection = inject_shaped(
        power=panel["power"], events=batch, seed=seed, shape=shape, shares=_STATE["shares"]
    )
    results = build_detectors(
        residual=injection.fast_residual(panel["residual"]),
        expected=panel["expected"],
        scales=_STATE["scales"],
        thresholds=_STATE["thresholds"],
    )
    plain_scale = _STATE["scales"]["plain"]
    rows = []
    for setting, (_, intervals) in results.items():
        grouped = by_series(intervals=intervals)
        for event in batch:
            match = match_event(
                series=event.series, onset=event.onset, end=event.end, intervals=grouped
            )
            span = slice(event.onset, min(event.end, event.onset + BLOCK))
            donor_mw = float(-np.mean(injection.delta[event.donor][span]))
            rows.append(
                {
                    "event_id": event.event_id,
                    "setting": setting,
                    "shape": shape,
                    "found": match.found,
                    "onset_found": match.onset_found,
                    "moved_mw": float(
                        -np.mean(injection.delta[event.donor][event.onset : event.end])
                    ),
                    "donor_step_z": donor_mw / plain_scale[event.donor],
                    "largest_step_z": max(
                        abs(float(np.mean(injection.delta[s][span]))) / plain_scale[s]
                        for s in event.series
                    ),
                }
            )
    return rows


def run_shapes(*, shapes: tuple[Shape, ...], groups: list, state: dict) -> pl.DataFrame:
    """Score every batch under every shape in worker processes."""
    jobs = [(seed, batch, shape) for shape in shapes for seed, batch in groups]
    with ProcessPoolExecutor(
        N_WORKERS,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
        initargs=(state,),
    ) as pool:
        rows = [r for part in pool.map(score_batch, jobs) for r in part]
    return pl.DataFrame(rows, infer_schema_length=None)


def both_scales(*, panel: dict) -> dict:
    """The uninjected odd-month noise scales of the plain and profile-weighted steps."""
    return {
        "plain": prepare(residuals=panel["residual"], window=WINDOW).scale,
        "profile": prepare_profile(
            residuals=panel["residual"], expected=panel["expected"], window=WINDOW
        ).scale,
    }


def read_thresholds(*, out: Path) -> dict[str, float]:
    """The thresholds `tune` wrote."""
    return json.loads((out / "thresholds.json").read_text())


def tune(*, out: Path) -> None:
    """Match D3's threshold to D0M's odd-month unexplained-alarm count (80), and confirm D0M's."""
    panel, _ = load()
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    plain = prepare(residuals=panel["residual"], window=WINDOW)
    profile = prepare_profile(
        residuals=panel["residual"], expected=panel["expected"], window=WINDOW
    )
    d0m, plain_curve = tune_d2(inputs=plain, veto_z=NO_VETO, logged=logged)
    d3, profile_curve = tune_d2(inputs=profile, veto_z=NO_VETO, logged=logged)
    print(f"matched thresholds: D0M {d0m}, D3 {d3}")
    if d0m != D0M_THRESHOLD:
        raise SystemExit("STOP: D0M's matched threshold is not experiment 58's 4.0")
    (out / "thresholds.json").write_text(json.dumps({"D0M_48": d0m, "D3_48": d3}))
    pl.DataFrame(
        [{"detector": "D0M", "threshold": t, "odd_unexplained": c} for t, c in plain_curve]
        + [{"detector": "D3", "threshold": t, "odd_unexplained": c} for t, c in profile_curve]
    ).write_parquet(out / "curves.parquet")


def event_groups(*, panel: dict) -> tuple[list, list[tuple[int, list[Event]]]]:
    """Experiment 58's original, extra, and holiday events in its batches, so S1 reproduces it."""
    original = original_events(power=panel["power"])
    extra, holiday = new_events(power=panel["power"])
    groups = seeded_batches(events=original, seed=SEED_43) + seeded_batches(
        events=extra + holiday, seed=SEED_NEW
    )
    return original + extra + holiday, groups


def make_state(*, out: Path) -> tuple[dict, list]:
    """Panel, scales, thresholds, and fitted shares for the workers, with the event batches."""
    panel, _ = load()
    _, groups = event_groups(panel=panel)
    shares = fit_profile_shares(panel=panel)
    return {
        "panel": panel,
        "scales": both_scales(panel=panel),
        "thresholds": read_thresholds(out=out),
        "shares": shares,
    }, groups


def harness_check(*, out: Path) -> None:
    """With shape S1, D0M must give experiment 58's outcome for every one of the 1,900 transfers."""
    state, groups = make_state(out=out)
    scored = run_shapes(shapes=("S1",), groups=groups, state=state).filter(
        pl.col("setting") == "D0M_48"
    )
    stored = pl.read_parquet(EXPERIMENT_58_DIR / "event_results_1.parquet").filter(
        pl.col("setting") == "D0M_48"
    )
    joined = scored.join(stored, on="event_id", suffix="_58", validate="1:1")
    different = joined.filter(pl.col("found") != pl.col("found_58")).height
    print(f"{joined.height} transfers, {different} differ from experiment 58's D0M")
    if joined.height != 1900 or different:
        raise SystemExit("STOP: the harness does not reproduce experiment 58's D0M")
    print("harness reproduced")


def score(*, out: Path) -> None:
    """Score every transfer under all four shapes with both detectors."""
    state, groups = make_state(out=out)
    shares = state["shares"]
    print(f"{len(shares)} fitted profile shares: {shares_json(shares=shares)}", flush=True)
    (out / "profile_shares.json").write_text(shares_json(shares=shares))
    scored = run_shapes(shapes=SHAPES, groups=groups, state=state)
    scored.write_parquet(out / "event_results.parquet")
    print(f"{scored.height} event rows", flush=True)


def real(*, out: Path) -> None:
    """Score the labelled real transfers and count the real alarms of both detectors."""
    panel, _ = load()
    results = build_detectors(
        residual=panel["residual"],
        expected=panel["expected"],
        scales=both_scales(panel=panel),
        thresholds=read_thresholds(out=out),
    )
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    real_alarm_rows(results=results, logged=logged).write_parquet(out / "alarms.parquet")
    events = real_events(power=panel["power"])
    moved = pl.read_parquet(LABELLED_PATH).select("card_id", "moved_mw_p50")
    score_real(results=results, real=events).join(
        moved, on="card_id", validate="m:1"
    ).write_parquet(out / "real_results.parquet")
    print({s: explained_flags(table=t, logged=logged).sum() for s, (t, _) in results.items()})


def main() -> None:
    """Run the requested stages."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stages", nargs="+", choices=("tune", "harness-check", "score", "real"))
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stages = {"tune": tune, "harness-check": harness_check, "score": score, "real": real}
    for stage in args.stages:
        stages[stage](out=args.out)


if __name__ == "__main__":
    main()
