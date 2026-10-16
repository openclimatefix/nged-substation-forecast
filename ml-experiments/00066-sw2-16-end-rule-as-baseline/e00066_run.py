r"""Match the 4-hour threshold on odd months, then score D0M and the end rule at three caps.

The injected transfers use five shapes; the real ones are the labelled transfers.

Stages, in order. Each writes files to the private output folder.

- `tune`: the thresholds of the three alarm settings (24 h at 4.0, 4 h at 6.0, 4 h matched to 80).
- `harness-check`: with shape S1, D0M and the 14-day end rule at 24 hours must reproduce
  experiments 58 and 61 for every transfer of experiment 58's injection set.
- `score`: the outcome of every injected transfer for every arm and every shape, including the
  long set.
- `real`: the labelled real transfers' outcomes and the real alarm counts.

    uv run python ml-experiments/00066-*/e00066_run.py tune harness-check score real
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
from e00066_common import LABELLED_PATH, ML_EXPERIMENTS_DATA_DIR, WORK_DIR, near_log_masks
from e00066_detect import (
    EXPLAIN_PAD,
    NO_VETO,
    SEED_43,
    SEED_NEW,
    explained_flags,
    load,
    log_masks,
    new_events,
    original_events,
    real_alarm_rows,
    real_events,
    score_real,
    seeded_batches,
    tune_d2,
)
from e00066_detectors import alarms, prepare
from e00066_endrule import CAPS_DAYS, end_rule_intervals, first_return_intervals
from e00066_inject import BLOCK, Event, events_frame
from e00066_intervals import build_intervals, relative_residual
from e00066_long import SEED_LONG, draw_long_events
from e00066_profile import SHAPES, Shape, fit_profile_shares, inject_shaped, shares_json
from e00066_score import by_series, match_event

EXPERIMENT_58_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00058-sw2-8b-donor-step-recut-and-stronger-veto"
)
EXPERIMENT_61_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00061-sw2-6-paired-step-intervals"
N_WORKERS: Final[int] = 4
ALARM_SETTINGS: Final[dict[str, int]] = {"48": 48, "8": 8, "8m": 8}
"""Alarm setting tag to window in half-hours: 24 h at 4.0, 4 h at 6.0, 4 h matched to 80 alarms."""
TARGETS: Final[dict[str, int]] = {"48": 80, "8": 120, "8m": 80}
"""Odd-month unexplained-alarm counts the thresholds are matched to."""
EXPECTED_THRESHOLDS: Final[dict[str, float]] = {"48": 4.0, "8": 6.0}
"""Experiments 43 and 58 matched these; `tune` stops if the search disagrees."""
SETTINGS: Final[tuple[str, ...]] = tuple(
    name
    for tag in ALARM_SETTINGS
    for name in (
        f"D0M_{tag}",
        *(f"PE{c}_{tag}" for c in CAPS_DAYS),
        *(f"PF{c}_{tag}" for c in CAPS_DAYS if tag == "48"),
    )
)
"""PF is the post hoc first-return end rule, 24-hour window only."""
_STATE: dict = {}


def build_detectors(*, residual: dict, expected: dict, scales: dict, thresholds: dict) -> dict:
    """Alarms and intervals of D0M and the three end rules at each alarm setting."""
    relative = {s: relative_residual(residual=residual[s], expected=expected[s]) for s in residual}
    inputs = {w: prepare(residuals=residual, window=w, scale=scales[w]) for w in (8, 48)}
    out: dict = {}
    for tag, window in ALARM_SETTINGS.items():
        table = alarms(inputs=inputs[window], detector="D0", threshold=thresholds[tag])
        tracked = build_intervals(
            alarm_table=table, relative=relative, window=window, detector="D0"
        )
        out[f"D0M_{tag}"] = (table, tracked)
        for cap in CAPS_DAYS:
            out[f"PE{cap}_{tag}"] = (
                table,
                end_rule_intervals(intervals=tracked, inputs=inputs[window], cap_days=cap),
            )
            if tag == "48":
                out[f"PF{cap}_{tag}"] = (
                    table,
                    first_return_intervals(intervals=tracked, inputs=inputs[window], cap_days=cap),
                )
    return out


def profile_correlation(*, moved: np.ndarray, expected: np.ndarray) -> float:
    """Correlation of a transfer's moved MW with the donor's expected power; NaN if constant."""
    ok = np.isfinite(expected) & np.isfinite(moved)
    if ok.sum() < 3 or moved[ok].std() < 1e-12 or expected[ok].std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(moved[ok], expected[ok])[0, 1])


def init_worker(state: dict) -> None:
    """Pool initializer: share the panel, scales, and thresholds."""
    _STATE.update(state)


def score_batch(args: tuple[int, list[Event], Shape]) -> list[dict]:
    """Inject one batch with one shape, detect with every arm, return one row per event."""
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
    plain_scale = _STATE["scales"][48]
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
                    "end_error": match.end_error,
                    "corr_expected": profile_correlation(
                        moved=-injection.delta[event.donor][event.onset : event.end],
                        expected=panel["expected"][event.donor][event.onset : event.end],
                    ),
                    "moved_mw": float(
                        -np.mean(injection.delta[event.donor][event.onset : event.end])
                    ),
                    "donor_step_z": donor_mw / plain_scale[event.donor],
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


def read_thresholds(*, out: Path) -> dict[str, float]:
    """The thresholds `tune` wrote."""
    return json.loads((out / "thresholds.json").read_text())


def scales_of(*, panel: dict) -> dict[int, dict[int, float]]:
    """The odd-month noise scale of every series, for both windows."""
    return {w: prepare(residuals=panel["residual"], window=w).scale for w in (8, 48)}


def tune(*, out: Path) -> None:
    """Match the three alarm settings' thresholds to their odd-month unexplained-alarm counts."""
    panel, _ = load()
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    thresholds: dict[str, float] = {}
    curves = []
    for tag, window in ALARM_SETTINGS.items():
        plain = prepare(residuals=panel["residual"], window=window)
        threshold, curve = tune_d2(inputs=plain, veto_z=NO_VETO, logged=logged, target=TARGETS[tag])
        print(f"alarm setting {tag}: threshold {threshold}")
        if tag in EXPECTED_THRESHOLDS and threshold != EXPECTED_THRESHOLDS[tag]:
            raise SystemExit(
                f"STOP: setting {tag} matched {threshold}, not {EXPECTED_THRESHOLDS[tag]}"
            )
        thresholds[tag] = threshold
        curves += [{"setting": tag, "threshold": t, "odd_unexplained": c} for t, c in curve]
    (out / "thresholds.json").write_text(json.dumps(thresholds))
    pl.DataFrame(curves).write_parquet(out / "curves.parquet")


def base_groups(*, panel: dict) -> tuple[list, list[tuple[int, list[Event]]]]:
    """Experiment 58's original, extra, and holiday events in its batches."""
    original = original_events(power=panel["power"])
    extra, holiday = new_events(power=panel["power"])
    groups = seeded_batches(events=original, seed=SEED_43) + seeded_batches(
        events=extra + holiday, seed=SEED_NEW
    )
    return original + extra + holiday, groups


def long_groups(*, panel: dict) -> tuple[list, list[tuple[int, list[Event]]]]:
    """The long set in its batches."""
    events = draw_long_events(quiet=log_masks(), power=panel["power"])
    return events, seeded_batches(events=events, seed=SEED_LONG + 1000)


def make_state(*, out: Path, panel: dict) -> dict:
    """Panel, scales, thresholds, and fitted shares for the workers."""
    return {
        "panel": panel,
        "scales": scales_of(panel=panel),
        "thresholds": read_thresholds(out=out),
        "shares": fit_profile_shares(panel=panel),
    }


def harness_check(*, out: Path) -> None:
    """With S1, D0M and PE14 (24 h) must match experiments 58 and 61 for 1,900 transfers."""
    panel, _ = load()
    state = make_state(out=out, panel=panel)
    _, groups = base_groups(panel=panel)
    scored = run_shapes(shapes=("S1",), groups=groups, state=state)
    for mine, reference, path, their in (
        ("D0M_48", "D0M_48", EXPERIMENT_58_DIR / "event_results_1.parquet", "D0M_48"),
        ("PE14_48", "PE_48", EXPERIMENT_61_DIR / "event_results.parquet", "PE_48"),
    ):
        stored = pl.read_parquet(path).filter(pl.col("setting") == their)
        if "shape" in stored.columns:
            stored = stored.filter(pl.col("shape") == "S1")
        joined = scored.filter(pl.col("setting") == mine).join(
            stored, on="event_id", suffix="_ref", validate="1:1"
        )
        different = joined.filter(pl.col("found") != pl.col("found_ref")).height
        print(f"{mine} vs {reference}: {joined.height} transfers, {different} differ")
        if joined.height != 1900 or different:
            raise SystemExit(f"STOP: the harness does not reproduce {reference}")
    print("harness reproduced")


def score(*, out: Path) -> None:
    """Score every transfer, including the long set, under all five shapes with every arm."""
    panel, _ = load()
    state = make_state(out=out, panel=panel)
    shares = state["shares"]
    (out / "profile_shares.json").write_text(shares_json(shares=shares))
    base_events, groups = base_groups(panel=panel)
    long_events, long_batches = long_groups(panel=panel)
    events = pl.concat(
        [
            events_frame(events=base_events).join(
                pl.read_parquet(EXPERIMENT_58_DIR / "events.parquet").select("event_id", "set"),
                on="event_id",
                validate="1:1",
            ),
            events_frame(events=long_events).with_columns(set=pl.lit("long")),
        ]
    )
    events.write_parquet(out / "events.parquet")
    print(f"{len(base_events)} base and {len(long_events)} long events", flush=True)
    scored = run_shapes(shapes=SHAPES, groups=groups + long_batches, state=state)
    scored.write_parquet(out / "event_results.parquet")
    print(f"{scored.height} event rows", flush=True)


def real(*, out: Path) -> None:
    """Score the labelled real transfers and count the real alarms of every arm."""
    panel, _ = load()
    results = build_detectors(
        residual=panel["residual"],
        expected=panel["expected"],
        scales=scales_of(panel=panel),
        thresholds=read_thresholds(out=out),
    )
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    real_alarm_rows(results=results, logged=logged).write_parquet(out / "alarms.parquet")
    events = real_events(power=panel["power"])
    scored = score_real(results=results, real=events)
    moved = pl.read_parquet(LABELLED_PATH).select("card_id", "moved_mw_p50")
    scored.join(moved, on="card_id", validate="m:1").write_parquet(out / "real_results.parquet")
    print({s: int(explained_flags(table=t, logged=logged).sum()) for s, (t, _) in results.items()})


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
