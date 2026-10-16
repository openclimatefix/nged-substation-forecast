r"""Match PL's and PF's thresholds on odd months, then score D0M, PE, PL and PF.

The injected transfers use five shapes; the real ones are the labelled transfers.

Stages, in order. Each writes files to the private output folder.

- `tune`: PL's and PF's matched thresholds (and D0M's, which must come back as 4.0).
- `harness-check`: with shape S1, D0M must reproduce experiment 58's outcome for every transfer.
- `score`: the outcome of every injected transfer for every arm and every shape.
- `real`: the labelled real transfers' outcomes and the real alarm counts.

    uv run python ml-experiments/00061-*/e00066_run.py tune harness-check score real
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
    new_events,
    odd_unexplained,
    original_events,
    real_alarm_rows,
    real_events,
    score_real,
    seeded_batches,
    tune_d2,
)
from e00066_detectors import DetectorInputs, alarms, prepare
from e00066_inject import BLOCK, Event
from e00066_intervals import build_intervals, relative_residual
from e00066_paired import (
    alarm_table,
    cut,
    end_only_intervals,
    flat_prior,
    labelled_duration_prior,
    paired_alarms,
    paired_intervals,
)
from e00066_profile import SHAPES, Shape, fit_profile_shares, inject_shaped, shares_json
from e00066_score import by_series, match_event

WINDOW: Final[int] = 48
D0M_THRESHOLD: Final[float] = 4.0
EXPERIMENT_58_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00058-sw2-8b-donor-step-recut-and-stronger-veto"
)
N_WORKERS: Final[int] = 4
SETTINGS: Final[tuple[str, ...]] = ("D0M_48", "PE_48", "PL_48", "PF_48")
_STATE: dict = {}


def build_detectors(
    *, residual: dict, expected: dict, scale: dict, thresholds: dict, priors: dict
) -> dict:
    """Alarms and intervals of D0M, PE, PL and PF on one residual panel."""
    relative = {s: relative_residual(residual=residual[s], expected=expected[s]) for s in residual}
    inputs = prepare(residuals=residual, window=WINDOW, scale=scale)
    table = alarms(inputs=inputs, detector="D0", threshold=thresholds["D0M_48"])
    d0m = build_intervals(alarm_table=table, relative=relative, window=WINDOW, detector="D0")
    out = {
        "D0M_48": (table, d0m),
        "PE_48": (table, end_only_intervals(intervals=d0m, inputs=inputs)),
    }
    for name in ("PL", "PF"):
        kept = cut(
            scored=paired_alarms(inputs=inputs, prior=priors[name]),
            threshold=thresholds[f"{name}_48"],
        )
        out[f"{name}_48"] = (
            alarm_table(kept=kept, shift=inputs.shift),
            paired_intervals(kept=kept, residual=residual, expected=expected, window=WINDOW),
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
        scale=_STATE["scale"],
        thresholds=_STATE["thresholds"],
        priors=_STATE["priors"],
    )
    plain_scale = _STATE["scale"]
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


def priors() -> dict:
    """The log duration priors of PL (labelled durations) and PF (flat)."""
    return {"PL": labelled_duration_prior(), "PF": flat_prior()}


def read_thresholds(*, out: Path) -> dict[str, float]:
    """The thresholds `tune` wrote."""
    return json.loads((out / "thresholds.json").read_text())


def tune_paired(
    *, inputs: DetectorInputs, prior: np.ndarray, logged: dict
) -> tuple[float, list[tuple[float, int]]]:
    """Return the lowest threshold on a 0.05 grid whose odd-month unexplained count is at most 80.

    The scan moves down from 8 in steps of 0.25 to the first count above the target, then up in
    steps of 0.05 to the first count at or under it. The second value holds every pair evaluated.
    """
    scored = paired_alarms(inputs=inputs, prior=prior)
    curve: dict[float, int] = {}

    def count(threshold: float) -> int:
        key = round(threshold, 2)
        if key not in curve:
            kept = cut(scored=scored, threshold=key)
            curve[key] = odd_unexplained(
                table=alarm_table(kept=kept, shift=inputs.shift), logged=logged
            )
        return curve[key]

    threshold = 8.0
    while threshold > 0.5 and count(threshold) <= TARGET:
        threshold = round(threshold - 0.25, 2)
    while count(threshold) > TARGET:
        threshold = round(threshold + 0.05, 2)
    print(f"  odd-month unexplained alarms at {threshold}: {count(threshold)} (target {TARGET})")
    return threshold, sorted(curve.items())


TARGET: Final[int] = 80
"""D0M's odd-month unexplained-alarm count, which PL and PF are matched to."""


def tune(*, out: Path) -> None:
    """Confirm D0M's threshold (4.0, 80 alarms), and match PL's and PF's to its count."""
    panel, _ = load()
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    plain = prepare(residuals=panel["residual"], window=WINDOW)
    d0m, d0m_curve = tune_d2(inputs=plain, veto_z=NO_VETO, logged=logged)
    print(f"matched thresholds: D0M {d0m}")
    if d0m != D0M_THRESHOLD:
        raise SystemExit("STOP: D0M's matched threshold is not experiment 58's 4.0")
    thresholds = {"D0M_48": d0m}
    curves = [{"detector": "D0M", "threshold": t, "odd_unexplained": c} for t, c in d0m_curve]
    for name, prior in priors().items():
        threshold, curve = tune_paired(inputs=plain, prior=prior, logged=logged)
        print(f"{name}: threshold {threshold}")
        thresholds[f"{name}_48"] = threshold
        curves += [{"detector": name, "threshold": t, "odd_unexplained": c} for t, c in curve]
    (out / "thresholds.json").write_text(json.dumps(thresholds))
    pl.DataFrame(curves).write_parquet(out / "curves.parquet")


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
        "scale": prepare(residuals=panel["residual"], window=WINDOW).scale,
        "priors": priors(),
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
    """Score every transfer under all five shapes with every arm."""
    state, groups = make_state(out=out)
    shares = state["shares"]
    print(f"{len(shares)} fitted profile shares: {shares_json(shares=shares)}", flush=True)
    (out / "profile_shares.json").write_text(shares_json(shares=shares))
    scored = run_shapes(shapes=SHAPES, groups=groups, state=state)
    scored.write_parquet(out / "event_results.parquet")
    print(f"{scored.height} event rows", flush=True)


def real(*, out: Path) -> None:
    """Score the labelled real transfers and count the real alarms of every arm.

    PL's scores on the real transfers use a cross-fitted prior: the transfers are split into two
    halves by the parity of their rank in card id, and each half is scored with a prior fitted
    without that half's cards.
    """
    panel, _ = load()
    kwargs = {
        "residual": panel["residual"],
        "expected": panel["expected"],
        "scale": prepare(residuals=panel["residual"], window=WINDOW).scale,
        "thresholds": read_thresholds(out=out),
    }
    results = build_detectors(priors=priors(), **kwargs)
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    real_alarm_rows(results=results, logged=logged).write_parquet(out / "alarms.parquet")
    events = real_events(power=panel["power"]).sort("card_id").with_row_index("rank")
    moved = pl.read_parquet(LABELLED_PATH).select("card_id", "moved_mw_p50")
    scored = score_real(results=results, real=events.drop("rank")).filter(
        pl.col("setting") != "PL_48"
    )
    pieces = [scored]
    for half in (0, 1):
        held = events.filter(pl.col("rank") % 2 == half)
        held_out = frozenset(held["card_id"].to_list())
        cross = build_detectors(
            priors={"PL": labelled_duration_prior(exclude_cards=held_out), "PF": flat_prior()},
            **kwargs,
        )
        pieces.append(score_real(results={"PL_48": cross["PL_48"]}, real=held.drop("rank")))
    pl.concat(pieces).join(moved, on="card_id", validate="m:1").write_parquet(
        out / "real_results.parquet"
    )
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
