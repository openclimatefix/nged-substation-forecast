r"""Calibrate the six detectors on odd months, then score injected and labelled real transfers.

Stages, in order. Each writes files to the private output folder.

- `harness-check`: the 4-hour and 24-hour single windows at experiment 43's thresholds must
  reproduce its alarm counts and the outcome of each of its injected transfers. Stop if not.
- `tune`: the matched threshold of each single window and of the maximum and sum detectors, and the
  alarm counts of every detector on the real residual.
- `score`: the outcome of every injected transfer, in each of three shapes, for every detector.
- `real`: the outcome of the labelled real transfers for every detector.

    uv run python ml-experiments/00059-*/e00059_run.py harness-check tune score real
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import polars as pl
from e00059_common import EXPERIMENT_43_DIR, WORK_DIR, near_log_masks, stamp_months
from e00059_detect import (
    WINDOW_24_HOURS,
    Thresholds,
    detect_all,
    load,
    rows_for_events,
)
from e00059_events import (
    EXPLAIN_PAD,
    SEED_43,
    SEED_NEW,
    alarm_masks,
    new_events,
    original_events,
    real_events,
    seeded_batches,
)
from e00059_inject import (
    HEADLINE_BAND,
    SHAPES,
    Event,
    ShapeType,
    events_frame,
    inject_power,
)
from e00059_score import match_event
from e00059_windows import (
    TARGET_COUNT,
    calibrate_combined,
    calibrate_singles,
    calibrated_steps,
    odd_unexplained,
    single_inputs,
)

N_WORKERS: Final[int] = 4
HARNESS_ALARM_COUNTS: Final[dict[int, int]] = {8: 432, 48: 297}
HARNESS_THRESHOLDS: Final[dict[int, float]] = {8: 6.0, 24: 4.0, 48: 4.0, 96: 4.0}
"""Experiment 43's thresholds for 4 and 24 hours. The 12-hour and 48-hour entries and
`HARNESS_COMBINED` are arbitrary, and only let `detect_all` run; the check reads W8 and W48."""
HARNESS_COMBINED: Final[float] = 1.5
HARNESS_FOUND_D0_48: Final[int] = 47
"""Of the 160 headline-band transfers in experiment 43's draw, D0 at 24 hours finds 47."""


def read_thresholds(*, out: Path) -> Thresholds:
    """The matched thresholds `tune` wrote."""
    raw = json.loads((out / "thresholds.json").read_text())
    return Thresholds(
        single={int(w): t for w, t in raw["single"].items()}, combined=raw["combined"]
    )


def harness_check(*, out: Path) -> None:
    """Reproduce experiment 43's D0 alarm counts and event outcomes."""
    panel, scales = load()
    thresholds = Thresholds(
        single=HARNESS_THRESHOLDS, combined={"MAX": HARNESS_COMBINED, "SUM": HARNESS_COMBINED}
    )
    detections = detect_all(
        residual=panel["residual"], expected=panel["expected"], scales=scales, thresholds=thresholds
    )
    for window, expected in HARNESS_ALARM_COUNTS.items():
        name = {8: "W8", 48: "W48"}[window]
        count = detections[name].table.height
        print(f"{name}: {count} alarms (experiment 43: {expected})")
        if count != expected:
            raise SystemExit(f"STOP: {name} alarm count differs from experiment 43")
    events = original_events(power=panel["power"])
    rows = score_groups(
        panel=panel,
        scales=scales,
        thresholds=thresholds,
        groups=seeded_batches(events=events, seed=SEED_43),
        shapes=("donor_profile",),
    )
    stored = pl.read_parquet(EXPERIMENT_43_DIR / "event_results.parquet").filter(
        pl.col("variant") == "plain", pl.col("setting") == "alarms_D0_48_4.0"
    )
    ours = rows.filter(pl.col("detector") == "W48", pl.col("variant") == "plain").select(
        "event_id", "found", "onset_found"
    )
    joined = ours.join(stored, on="event_id", suffix="_43", validate="1:1")
    different = joined.filter(
        (pl.col("found") != pl.col("found_43"))
        | (pl.col("onset_found") != pl.col("onset_found_43"))
    )
    headline = events_frame(events=events).filter(
        pl.col("size_band") == HEADLINE_BAND[0], pl.col("duration_band") == HEADLINE_BAND[1]
    )["event_id"]
    found = joined.filter(pl.col("event_id").is_in(headline.implode()))["found"].sum()
    print(f"{joined.height} event rows, {different.height} differ; D0 24 h found {found} of 160")
    if different.height or found != HARNESS_FOUND_D0_48:
        raise SystemExit("STOP: the harness does not reproduce experiment 43")
    print("harness reproduced")


def tune(*, out: Path) -> None:
    """Match every threshold on odd months, then count each detector's alarms on the real data."""
    panel, scales = load()
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    inputs = single_inputs(residuals=panel["residual"], scales=scales)
    single = calibrate_singles(inputs=inputs, logged=logged)
    print("single windows", single, flush=True)
    steps = calibrated_steps(inputs=inputs, thresholds=single)
    combined = calibrate_combined(steps=steps, logged=logged)
    print("combined", combined, flush=True)
    thresholds = Thresholds(single=single, combined=combined)
    (out / "thresholds.json").write_text(json.dumps({"single": single, "combined": combined}))
    detections = detect_all(
        residual=panel["residual"], expected=panel["expected"], scales=scales, thresholds=thresholds
    )
    months = stamp_months()
    pieces = []
    for name, detection in detections.items():
        table = detection.table
        stamps = table["stamp"].to_numpy()
        explained = [bool(logged[s][t]) for s, t in table.select("series", "stamp").iter_rows()]
        pieces.append(
            table.select("series", "stamp", "window").with_columns(
                detector=pl.lit(name),
                month=pl.Series(months[stamps], dtype=pl.Int64),
                explained=pl.Series(explained, dtype=pl.Boolean),
            )
        )
        achieved = odd_unexplained(table=table, logged=logged)
        print(name, table.height, "alarms; odd unexplained", achieved, flush=True)
        if achieved > TARGET_COUNT:
            raise SystemExit(f"STOP: {name} odd-month unexplained count {achieved} is over target")
    pl.concat(pieces).write_parquet(out / "real_alarms.parquet")


_STATE: dict = {}


def init_worker(state: dict) -> None:
    """Pool initializer: share the panel, scales, and thresholds."""
    _STATE.update(state)


def score_batch(args: tuple[int, list[Event], ShapeType]) -> list[dict]:
    """Inject one batch in one shape with the fast residual, detect, and return one row each.

    The `plain` variant builds intervals as experiment 43 does. The `fixed` variant is the post hoc
    arm that tracks the transfer's end more patiently.
    """
    seed, batch, shape = args
    panel = _STATE["panel"]
    injection = inject_power(power=panel["power"], events=batch, seed=seed, shape=shape)
    rows: list[dict] = []
    for variant in ("plain", "fixed"):
        detections = detect_all(
            residual=injection.fast_residual(panel["residual"]),
            expected=panel["expected"],
            scales=_STATE["scales"],
            thresholds=_STATE["thresholds"],
            fixed=variant == "fixed",
        )
        rows += rows_for_events(
            batch=batch,
            injection=injection,
            detections=detections,
            scale_24h=_STATE["scales"][WINDOW_24_HOURS],
            shape=shape,
            variant=variant,
        )
    return rows


def score_groups(
    *,
    panel: dict,
    scales: dict[int, dict[int, float]],
    thresholds: Thresholds,
    groups: list[tuple[int, list[Event]]],
    shapes: tuple[ShapeType, ...],
) -> pl.DataFrame:
    """Score batches of events, each with its own injection seed, in spawned workers."""
    state = {"panel": panel, "scales": scales, "thresholds": thresholds}
    tasks = [(seed, batch, shape) for shape in shapes for seed, batch in groups]
    with ProcessPoolExecutor(
        N_WORKERS,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
        initargs=(state,),
    ) as pool:
        rows = [r for part in pool.map(score_batch, tasks) for r in part]
    return pl.DataFrame(rows, infer_schema_length=None)


def score(*, out: Path) -> None:
    """Score every injected transfer in every shape for every detector."""
    panel, scales = load()
    thresholds = read_thresholds(out=out)
    original = original_events(power=panel["power"])
    extra, holiday = new_events(power=panel["power"])
    full = alarm_masks()
    pieces = []
    for name, events in (("original", original), ("extra", extra), ("holiday", holiday)):
        table = events_frame(events=events).with_columns(set=pl.lit(name))
        clash = [
            any(full[s][max(e.onset - 96, 0) : e.end + 96].any() for s in e.series) for e in events
        ]
        pieces.append(table.with_columns(alarm_quiet=pl.Series([not c for c in clash])))
    pl.concat(pieces).write_parquet(out / "events.parquet")
    groups = seeded_batches(events=original, seed=SEED_43) + seeded_batches(
        events=extra + holiday, seed=SEED_NEW
    )
    rows = score_groups(
        panel=panel, scales=scales, thresholds=thresholds, groups=groups, shapes=SHAPES
    )
    rows.write_parquet(out / "event_results.parquet")
    print(f"{rows.height} event rows", flush=True)


def real(*, out: Path) -> None:
    """Score the labelled real transfers for every detector on the real residual."""
    panel, scales = load()
    thresholds = read_thresholds(out=out)
    detections = detect_all(
        residual=panel["residual"], expected=panel["expected"], scales=scales, thresholds=thresholds
    )
    events = real_events(power=panel["power"])
    rows = []
    for name, detection in detections.items():
        grouped = detection.intervals
        for row in events.iter_rows(named=True):
            series = tuple(int(s) for s in row["series"].split(","))
            match = match_event(
                series=series, onset=row["onset"], end=row["end"], intervals=grouped
            )
            rows.append(
                {
                    **row,
                    "detector": name,
                    "onset_found": match.onset_found,
                    "found": match.found,
                    "end_error": match.end_error,
                }
            )
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(out / "real_results.parquet")
    print(f"{len(rows)} real rows", flush=True)


def main() -> None:
    """Run the requested stages."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stages", nargs="+", choices=("harness-check", "tune", "score", "real"))
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stages = {"harness-check": harness_check, "tune": tune, "score": score, "real": real}
    for stage in args.stages:
        stages[stage](out=args.out)


if __name__ == "__main__":
    main()
