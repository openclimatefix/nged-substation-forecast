r"""Score four end rules for RO-NC arm B on a fresh draw and the real transfers.

Arm B's alarms come from experiment 138's stored logistic score and threshold, so nothing is
refitted. R0 is the offline end rule. R1, R2, and R3 call the end with no later data (see
`e00162_h_endrule.py`).

- `score`: the outcome of every fresh injected transfer for every end rule under all five shapes.
- `real`: the labelled real transfers' outcomes for every end rule.

    E162_WORKERS=6 OMP_NUM_THREADS=1 uv run --no-sync python \
        ml-experiments/00147-*/e00162_h_run.py score real
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pickle
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00162_h_arms import ARMS, PERSIST_GAP, SLOW_WINDOW, arm_candidates, arm_inputs
from e00162_h_common import EXPERIMENT_138_RUN_DIR, LABELLED_PATH, WORK_DIR
from e00162_h_detect import log_masks, quiet_masks, real_events, seeded_batches
from e00162_h_detectors import DetectorInputs
from e00162_h_endrule import MERGE_STAMPS, RULES, filter_alarms, live_end_intervals
from e00162_h_inject import BLOCK, Event, Injection, draw_events, events_frame
from e00162_h_intervals import build_intervals
from e00162_h_long import draw_long_events
from e00162_h_profile import SHAPES, Shape, fit_profile_shares, inject_shaped, shares_json
from e00162_h_score import by_series, end_tolerance
from e00162_h_variants import BLANK_SERIES, load_both, scales_for, tracked_relative

N_WORKERS: Final[int] = int(os.environ.get("E162_WORKERS", "6"))
WINDOW: Final[int] = 48
CAP_DAYS: Final[int] = 14
TARGET: Final[int] = 80
ONSET_TOLERANCE: Final[int] = 4
"""Stamps (2 hours) within which an alarm counts as at the start of a transfer."""
SEED_FRESH_BASE: Final[int] = 8200
"""Seed of the fresh headline-band draw; batch `i` is injected with `SEED_FRESH_BASE + 1000 + i`."""
SEED_FRESH_LONG: Final[int] = 8300
FRESH_BASE_EVENTS: Final[int] = 1160
FRESH_BASE_FIRST_ID: Final[int] = 40_000
FRESH_LONG_FIRST_ID: Final[int] = 45_000
_STATE: dict = {}


def init_worker(state: dict) -> None:
    """Pool initializer: share the panels, scales, thresholds, and fitted scores."""
    _STATE.update(state)


def run_pool(*, function: Callable, jobs: list, state: dict) -> list:
    """Run `function` over `jobs` in worker processes that share `state`."""
    with ProcessPoolExecutor(
        N_WORKERS,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
        initargs=(state,),
    ) as pool:
        return list(pool.map(function, jobs))


def all_scales(*, panel: dict, offset_panel: dict) -> dict[str, dict[int, float]]:
    """The odd-month noise unit of each series for every step statistic the arms read.

    `PF14` is the plain 24-hour step, kept only to express donor steps in the units of
    experiments 82 and 130.
    """
    kwargs = {"residual": panel["residual"], "offset": offset_panel["residual"]}
    return {
        "PF14": scales_for(variant="PF14", **kwargs),
        "X": scales_for(variant="X", **kwargs),
        "persist": scales_for(variant="X", gap=PERSIST_GAP, **kwargs),
        "slow": scales_for(variant="X", window=SLOW_WINDOW, **kwargs),
    }


def build_detectors(
    *,
    panel: dict,
    offset_panel: dict,
    residual: dict,
    offset: dict,
    scales: dict,
    thresholds: dict,
    models: dict,
) -> tuple[dict, DetectorInputs]:
    """Alarm table and intervals of every end rule for arm B, and the 24-hour step statistic."""
    relative = tracked_relative(
        variant="X", panel=panel, offset_panel=offset_panel, residual=residual, offset=offset
    )
    spec = ARMS["B"]
    inputs = arm_inputs(spec=spec, residual=residual, offset=offset, scales=scales)
    cand = arm_candidates(spec=spec, inputs=inputs)
    keep = cand.filter(
        pl.Series(models["B"].score(table=cand) > thresholds["B"]) if cand.height else []
    )
    table = keep.select(
        "series",
        "stamp",
        "step_mw",
        z=pl.struct("series", "stamp").map_elements(
            lambda r, s=inputs.fast: float(
                np.nan_to_num(s.shift[r["series"]][r["stamp"]] / s.scale[r["series"]])
            ),
            return_dtype=pl.Float64,
        ),
        partners=pl.lit(""),
    )
    out = {}
    tracked = build_intervals(alarm_table=table, relative=relative, window=WINDOW, detector="D0")
    for rule in RULES:
        live = live_end_intervals(
            intervals=tracked, inputs=inputs.fast, rule=rule, cap_days=CAP_DAYS
        )
        out[rule] = (table, live)
        for tolerance in MERGE_STAMPS:
            merged = filter_alarms(table=table, live=live, tolerance=tolerance)
            rebuilt = build_intervals(
                alarm_table=merged, relative=relative, window=WINDOW, detector="D0"
            )
            out[f"{rule}_M{tolerance}"] = (
                merged,
                live_end_intervals(
                    intervals=rebuilt, inputs=inputs.fast, rule=rule, cap_days=CAP_DAYS
                ),
            )
    return out, inputs.fast


PARTNER_HOURS: Final[int] = 3
"""A partner's reversal counts when it lies within this many hours of the stated end."""
PARTNER_SIZE: Final[tuple[float, float]] = (0.5, 2.0)
"""A partner's return step must be this share of its own onset step, at least and at most."""


def partner_reversal(
    *, fast: DetectorInputs, series: tuple[int, ...], matched: dict
) -> tuple[float, float]:
    """Whether a partner series' step reverses at the stated end, and by how much of its onset step.

    The partners are the other series of the transfer. A partner's onset step is its largest step
    within 1 stamp of the matched onset, in the direction opposite to the matched series' step. The
    reversal is its largest step the other way within `PARTNER_HOURS` of the stated end. Returns
    the ratio of the reversal to the onset step (NaN when there is no partner or no onset step),
    and 1.0 or 0.0 for whether that ratio lies inside `PARTNER_SIZE`.
    """
    own = matched["matched_series"]
    if own < 0 or own not in fast.shift:
        return float("nan"), float("nan")
    onset, end = matched["matched_onset"], matched["matched_end"]
    sign = np.sign(fast.shift[own][onset])
    best = float("nan")
    for partner in series:
        if partner == own or partner not in fast.shift:
            continue
        z = fast.shift[partner] / fast.scale[partner]
        around = z[max(onset - 1, 0) : onset + 2]
        if not np.isfinite(around).any():
            continue
        onset_step = float(np.nanmax(-sign * around))
        if onset_step < 1.5:
            continue
        span = PARTNER_HOURS * 2
        back = z[max(end - span, 0) : end + span + 1]
        if not np.isfinite(back).any():
            continue
        ratio = float(np.nanmax(sign * back)) / onset_step
        best = ratio if np.isnan(best) else max(best, ratio)
    if np.isnan(best):
        return best, float("nan")
    return best, float(PARTNER_SIZE[0] <= best <= PARTNER_SIZE[1])


def match_end(*, series: tuple[int, ...], onset: int, end: int, intervals: dict[int, list]) -> dict:
    """Match a transfer to the interval whose end is closest to the true end.

    Returns whether the onset was found, whether the end is within the tolerance (`found`),
    whether the end is called too early or too late by more than the tolerance, the error in
    stamps (positive when late), and the lag from the true end to the call, in stamps.
    """
    tolerance = end_tolerance(onset=onset, end=end)
    best = None
    for s in series:
        for interval in intervals.get(s, []):
            if abs(interval.onset - onset) > ONSET_TOLERANCE:
                continue
            if best is None or abs(interval.end - end) < abs(best.end - end):
                best = interval
    if best is None:
        return {
            "onset_found": False,
            "found": False,
            "premature": False,
            "late": False,
            "end_error": float("nan"),
            "lag": float("nan"),
            "matched_series": -1,
            "matched_onset": -1,
            "matched_end": -1,
        }
    error = best.end - end
    return {
        "onset_found": True,
        "found": abs(error) <= tolerance,
        "premature": error < -tolerance,
        "late": error > tolerance,
        "end_error": float(error),
        "lag": float(best.known_from - end),
        "matched_series": best.series,
        "matched_onset": best.onset,
        "matched_end": best.end,
    }


def injected(*, seed: int, batch: list[Event], shape: Shape) -> tuple[Injection, dict, dict]:
    """The injection of one batch and shape, and the injected residuals of both models.

    The convex model is frozen, so the residual is the stored residual plus the injected delta.
    """
    panel, offset_panel = _STATE["panel"], _STATE["offset_panel"]
    injection = inject_shaped(
        power=panel["power"], events=batch, seed=seed, shape=shape, shares=_STATE["shares"]
    )
    return (
        injection,
        injection.fast_residual(panel["residual"]),
        injection.fast_residual(offset_panel["residual"]),
    )


def score_batch(args: tuple[int, list[Event], Shape]) -> list[dict]:
    """Inject one batch with one shape, detect, and return one row per event and end rule."""
    seed, batch, shape = args
    injection, residual, offset = injected(seed=seed, batch=batch, shape=shape)
    results, fast = build_detectors(
        panel=_STATE["panel"],
        offset_panel=_STATE["offset_panel"],
        residual=residual,
        offset=offset,
        scales=_STATE["scales"],
        thresholds=_STATE["thresholds"],
        models=_STATE["models"],
    )
    plain_scale = _STATE["scales"]["PF14"]
    rows = []
    for setting, (table, intervals) in results.items():
        grouped = by_series(intervals=intervals)
        steps = {(r["series"], r["stamp"]): r["step_mw"] for r in table.iter_rows(named=True)}
        alarm_intervals = [iv for iv in intervals if iv.role == "alarm"]
        for event in batch:
            if event.donor == BLANK_SERIES:
                continue
            span = slice(event.onset, min(event.end, event.onset + BLOCK))
            moved = float(-np.mean(injection.delta[event.donor][event.onset : event.end]))
            match = match_end(
                series=event.series, onset=event.onset, end=event.end, intervals=grouped
            )
            n_intervals = max(
                sum(
                    1
                    for iv in alarm_intervals
                    if iv.series == series
                    and event.onset - ONSET_TOLERANCE <= iv.onset <= event.end + ONSET_TOLERANCE
                )
                for series in event.series
            )
            step = steps.get((match["matched_series"], match["matched_onset"]))
            ratio, reversed_ok = partner_reversal(fast=fast, series=event.series, matched=match)
            rows.append(
                {
                    "kind": "event",
                    "event_id": event.event_id,
                    "setting": setting,
                    "shape": shape,
                    **match,
                    "n_intervals": n_intervals,
                    "partner_ratio": ratio,
                    "partner_reversal": reversed_ok,
                    "detected_ratio": abs(step) / moved
                    if step is not None and match["matched_series"] == event.donor and moved > 0
                    else float("nan"),
                    "moved_mw": moved,
                    "donor_step_z": float(-np.mean(injection.delta[event.donor][span]))
                    / plain_scale[event.donor],
                }
            )
        matched = sum(
            any(iv.series in e.series and abs(iv.onset - e.onset) <= ONSET_TOLERANCE for e in batch)
            for iv in alarm_intervals
        )
        rows.append(
            {
                "kind": "batch",
                "event_id": -1,
                "setting": setting,
                "shape": shape,
                "n_intervals": len(alarm_intervals),
                "n_matched": matched,
            }
        )
    return rows


def read_json(*, out: Path, name: str) -> dict:
    """A JSON file `train` wrote."""
    return json.loads((out / name).read_text())


class _Unpickler(pickle.Unpickler):
    """Load experiment 138's fitted scores, whose classes live in a module of the same code."""

    def find_class(self, module: str, name: str) -> object:
        """Map experiment 138's module names to this folder's copy."""
        return super().find_class(module.replace("e00138_", "e00162_h_"), name)


def make_state() -> dict:
    """Panels, scales, experiment 138's arm B score and threshold, and fitted shares."""
    panel, offset_panel = load_both()
    with (EXPERIMENT_138_RUN_DIR / "models.pkl").open("rb") as handle:
        models = _Unpickler(handle).load()
    return {
        "panel": panel,
        "offset_panel": offset_panel,
        "scales": all_scales(panel=panel, offset_panel=offset_panel),
        "shares": fit_profile_shares(panel=load_both(source="A16")[0]),
        "thresholds": json.loads((EXPERIMENT_138_RUN_DIR / "thresholds.json").read_text()),
        "models": models,
    }


def draw_fresh(*, panel: dict) -> tuple[list[Event], list[Event]]:
    """A headline-band draw and a long draw."""
    base = draw_events(
        quiet=quiet_masks(),
        power=panel["power"],
        seed=SEED_FRESH_BASE,
        headline_only=FRESH_BASE_EVENTS,
        first_id=FRESH_BASE_FIRST_ID,
    )
    long = draw_long_events(
        quiet=log_masks(), power=panel["power"], seed=SEED_FRESH_LONG, first_id=FRESH_LONG_FIRST_ID
    )
    return base, long


def score(*, out: Path) -> None:
    """Score every fresh transfer under all five shapes with every arm."""
    state = make_state()
    panel = state["panel"]
    (out / "profile_shares.json").write_text(shares_json(shares=state["shares"]))
    base, long = draw_fresh(panel=panel)
    events = pl.concat(
        [
            events_frame(events=base).with_columns(population=pl.lit("base")),
            events_frame(events=long).with_columns(population=pl.lit("long")),
        ]
    )
    events.write_parquet(out / "events.parquet")
    print(f"{len(base)} base and {len(long)} long events", flush=True)
    groups = seeded_batches(events=base, seed=SEED_FRESH_BASE + 1000) + seeded_batches(
        events=long, seed=SEED_FRESH_LONG + 1000
    )
    jobs = [(seed, batch, shape) for shape in SHAPES for seed, batch in groups]
    rows = [r for part in run_pool(function=score_batch, jobs=jobs, state=state) for r in part]
    scored = pl.DataFrame(rows, infer_schema_length=None)
    scored.write_parquet(out / "event_results.parquet")
    print(f"{scored.height} event rows", flush=True)


def real(*, out: Path) -> None:
    """Score the labelled real transfers with every end rule and merge setting."""
    panel, offset_panel = load_both()
    state = make_state()
    results, fast = build_detectors(
        panel=panel,
        offset_panel=offset_panel,
        residual=panel["residual"],
        offset=offset_panel["residual"],
        scales=state["scales"],
        thresholds=state["thresholds"],
        models=state["models"],
    )
    moved = pl.read_parquet(LABELLED_PATH).select("card_id", "moved_mw_p50")
    labelled = real_events(power=panel["power"])
    rows = []
    for setting, (_, intervals) in results.items():
        grouped = by_series(intervals=intervals)
        for row in labelled.iter_rows(named=True):
            series = tuple(int(s) for s in row["series"].split(","))
            n_intervals = max(
                sum(
                    1
                    for iv in intervals
                    if iv.role == "alarm"
                    and iv.series == one
                    and row["onset"] - ONSET_TOLERANCE <= iv.onset <= row["end"] + ONSET_TOLERANCE
                )
                for one in series
            )
            match = match_end(series=series, onset=row["onset"], end=row["end"], intervals=grouped)
            ratio, reversed_ok = partner_reversal(fast=fast, series=series, matched=match)
            rows.append(
                {
                    **row,
                    "setting": setting,
                    **match,
                    "n_intervals": n_intervals,
                    "partner_ratio": ratio,
                    "partner_reversal": reversed_ok,
                }
            )
    scored = pl.DataFrame(rows, infer_schema_length=None).join(moved, on="card_id", validate="m:1")
    scored.write_parquet(out / "real_results.parquet")
    print(f"{scored.height} real rows", flush=True)


def export(*, out: Path) -> None:
    """Write every setting's intervals on the uninjected panel in the layout of experiment 82.

    The layout is `xl_intervals.parquet`'s (series, role, alarm_series, onset, end, step_mw), plus
    `known_from` (the stamp at which the end is called), `setting`, `causal_live` (False for the
    hindsight rule R0), and `merged` (True when overlapping alarms were merged).
    """
    panel, offset_panel = load_both()
    state = make_state()
    results, _ = build_detectors(
        panel=panel,
        offset_panel=offset_panel,
        residual=panel["residual"],
        offset=offset_panel["residual"],
        scales=state["scales"],
        thresholds=state["thresholds"],
        models=state["models"],
    )
    rows = []
    for setting, (table, intervals) in results.items():
        steps = {(r["series"], r["stamp"]): r["step_mw"] for r in table.iter_rows(named=True)}
        rows.extend(
            {
                "series": iv.series,
                "role": iv.role,
                "alarm_series": iv.alarm_series,
                "onset": iv.onset,
                "end": iv.end,
                "step_mw": steps.get((iv.alarm_series, iv.onset), float("nan")),
                "known_from": iv.known_from,
                "setting": setting,
                "causal_live": not setting.startswith("R0"),
                "merged": "_M" in setting,
            }
            for iv in intervals
        )
    table = pl.DataFrame(rows)
    table.write_parquet(out / "intervals.parquet")
    print(f"{table.height} interval rows; series {sorted(table['series'].unique().to_list())}")


def main() -> None:
    """Run the requested stages."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stages", nargs="+", choices=("score", "real", "export"))
    parser.add_argument("--out", type=Path, default=WORK_DIR / "run")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stages = {"score": score, "real": real, "export": export}
    for stage in args.stages:
        stages[stage](out=args.out)


if __name__ == "__main__":
    main()
