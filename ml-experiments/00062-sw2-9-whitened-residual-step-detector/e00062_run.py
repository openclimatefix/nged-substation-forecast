r"""Match W0's thresholds on odd months, then score D0M and W0 on injected and real transfers.

The injected transfers use five shapes; the real ones are the labelled transfers.

Stages, in order. Each writes files to the private output folder.

- `tune`: W0's matched thresholds at 24 and 4 hours (D0M's must come back as 4.0 and 6.0).
- `harness-check`: with shape S1, D0M must reproduce experiment 58's outcome for every transfer.
- `score`: the outcome of every injected transfer for both detectors and every shape.
- `real`: the labelled real transfers' outcomes and the real alarm counts.

    uv run python ml-experiments/00060-*/e00062_run.py tune harness-check score real
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
from e00062_common import LABELLED_PATH, ML_EXPERIMENTS_DATA_DIR, WORK_DIR, near_log_masks
from e00062_detect import (
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
from e00062_detectors import alarms, prepare
from e00062_inject import BLOCK, Event
from e00062_intervals import build_intervals, relative_residual
from e00062_profile import SHAPES, Shape, fit_profile_shares, inject_shaped, shares_json
from e00062_score import by_series, match_event
from e00062_whiten import fit_coefficients, prepare_whitened

WINDOWS: Final[tuple[int, ...]] = (48, 8)
"""The 24-hour window (headline) and the 4-hour window, in half-hours."""
D0M_THRESHOLDS: Final[dict[int, float]] = {48: 4.0, 8: 6.0}
EXPERIMENT_58_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00058-sw2-8b-donor-step-recut-and-stronger-veto"
)
N_WORKERS: Final[int] = 4
SETTINGS: Final[tuple[str, ...]] = ("D0M_48", "W0_48", "D0M_8", "W0_8")
_STATE: dict = {}


def build_detectors(
    *, residual: dict, expected: dict, scales: dict, thresholds: dict, coefficients: dict
) -> dict:
    """Alarms and plain intervals of D0M and W0 at both windows on one residual panel."""
    relative = {s: relative_residual(residual=residual[s], expected=expected[s]) for s in residual}
    out = {}
    for window in WINDOWS:
        inputs = {
            f"D0M_{window}": prepare(
                residuals=residual, window=window, scale=scales["plain"][window]
            ),
            f"W0_{window}": prepare_whitened(
                residuals=residual,
                coefficients=coefficients,
                window=window,
                scale=scales["white"][window],
            ),
        }
        for setting, one in inputs.items():
            table = alarms(inputs=one, detector="D0", threshold=thresholds[setting])
            out[setting] = (
                table,
                build_intervals(alarm_table=table, relative=relative, window=window, detector="D0"),
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
        coefficients=_STATE["coefficients"],
    )
    plain_scale = _STATE["scales"]["plain"][48]
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
                    "corr_expected": profile_correlation(
                        moved=-injection.delta[event.donor][event.onset : event.end],
                        expected=panel["expected"][event.donor][event.onset : event.end],
                    ),
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


def both_scales(*, panel: dict, coefficients: dict) -> dict:
    """The uninjected odd-month noise scales of the plain and whitened steps, by window."""
    return {
        "plain": {w: prepare(residuals=panel["residual"], window=w).scale for w in WINDOWS},
        "white": {
            w: prepare_whitened(
                residuals=panel["residual"], coefficients=coefficients, window=w
            ).scale
            for w in WINDOWS
        },
    }


def read_thresholds(*, out: Path) -> dict[str, float]:
    """The thresholds `tune` wrote."""
    return json.loads((out / "thresholds.json").read_text())


def tune(*, out: Path) -> None:
    """Match W0's thresholds to D0M's odd-month unexplained-alarm counts, and confirm D0M's."""
    panel, _ = load()
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    coefficients = fit_coefficients(residuals=panel["residual"])
    thresholds: dict[str, float] = {}
    curves: list[dict] = []
    for window in WINDOWS:
        plain = prepare(residuals=panel["residual"], window=window)
        white = prepare_whitened(
            residuals=panel["residual"], coefficients=coefficients, window=window
        )
        d0m, plain_curve = tune_d2(inputs=plain, veto_z=NO_VETO, logged=logged)
        w0, white_curve = tune_d2(inputs=white, veto_z=NO_VETO, logged=logged)
        print(f"window {window}: matched thresholds D0M {d0m}, W0 {w0}")
        if d0m != D0M_THRESHOLDS[window]:
            raise SystemExit(f"STOP: D0M's matched threshold at window {window} is not the old one")
        thresholds |= {f"D0M_{window}": d0m, f"W0_{window}": w0}
        curves += [
            {"detector": "D0M", "window": window, "threshold": t, "odd_unexplained": c}
            for t, c in plain_curve
        ] + [
            {"detector": "W0", "window": window, "threshold": t, "odd_unexplained": c}
            for t, c in white_curve
        ]
    (out / "thresholds.json").write_text(json.dumps(thresholds))
    pl.DataFrame(curves).write_parquet(out / "curves.parquet")
    pl.DataFrame(
        [
            {"series": s, "slot": h, "coefficient": float(c[h])}
            for s, c in coefficients.items()
            for h in range(len(c))
        ]
    ).write_parquet(out / "coefficients.parquet")


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
    coefficients = fit_coefficients(residuals=panel["residual"])
    return {
        "panel": panel,
        "scales": both_scales(panel=panel, coefficients=coefficients),
        "thresholds": read_thresholds(out=out),
        "shares": shares,
        "coefficients": coefficients,
    }, groups


def harness_check(*, out: Path) -> None:
    """With shape S1, D0M must give experiment 58's outcome for every one of the 1,900 transfers."""
    state, groups = make_state(out=out)
    scored = run_shapes(shapes=("S1",), groups=groups, state=state)
    stored = pl.read_parquet(EXPERIMENT_58_DIR / "event_results_1.parquet")
    for setting in ("D0M_48", "D0M_8"):
        joined = scored.filter(pl.col("setting") == setting).join(
            stored.filter(pl.col("setting") == setting),
            on="event_id",
            suffix="_58",
            validate="1:1",
        )
        different = joined.filter(pl.col("found") != pl.col("found_58")).height
        print(f"{setting}: {joined.height} transfers, {different} differ from experiment 58's")
        if joined.height != 1900 or different:
            raise SystemExit(f"STOP: the harness does not reproduce experiment 58's {setting}")
    print("harness reproduced")


def score(*, out: Path) -> None:
    """Score every transfer under all five shapes with both detectors."""
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
    coefficients = fit_coefficients(residuals=panel["residual"])
    results = build_detectors(
        residual=panel["residual"],
        expected=panel["expected"],
        scales=both_scales(panel=panel, coefficients=coefficients),
        thresholds=read_thresholds(out=out),
        coefficients=coefficients,
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
