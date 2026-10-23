r"""Score the pair rules for the end of a load shift on experiment 148's injected draw.

Arm B's alarms come from experiment 138's stored logistic score and threshold, so the detector is
not refitted. R0 is the lookahead rule of experiment 148 at 24 hours on each side alone, P1 to P3
are the pair rules, and OFF is the offline rule (see `e00253_endrule.py`).

- `train`: fit the small neural scorer of rule P3 on a separate draw that starts in odd months.
- `score`: the outcome of every injected transfer for every rule, with equal and with unequal
  (10%) step sizes.
- `real`: the labelled real transfers' outcomes for every rule.

    E253_WORKERS=4 OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=16G \
        -p OOMPolicy=continue uv run --no-sync python \
        ml-experiments/00253-*/e00253_run.py train score real
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pickle
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00253_arms import ARMS, PERSIST_GAP, SLOW_WINDOW, arm_candidates, arm_inputs
from e00253_common import (
    EXPERIMENT_138_RUN_DIR,
    GRID_LENGTH,
    WORK_DIR,
    stamp_months,
)
from e00253_detect import quiet_masks, real_events, seeded_batches
from e00253_detectors import DetectorInputs
from e00253_endrule import FEATURE_NAMES, RULES, Candidates, live_end_intervals, set_scorer
from e00253_inject import BLOCK, Event, Injection, draw_events, events_frame
from e00253_intervals import build_intervals
from e00253_profile import Shape, fit_profile_shares, inject_shaped
from e00253_score import by_series, end_tolerance
from e00253_variants import BLANK_SERIES, load_both, scales_for, tracked_relative
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_WORKERS: Final[int] = int(os.environ.get("E253_WORKERS", "4"))
WINDOW: Final[int] = 48
CAP_DAYS: Final[int] = 14
TARGET: Final[int] = 80
ONSET_TOLERANCE: Final[int] = 4
"""Stamps (2 hours) within which an alarm counts as at the start of a transfer."""
SEED_FRESH_BASE: Final[int] = 8200
"""Seed of the fresh headline-band draw; batch `i` is injected with `SEED_FRESH_BASE + 1000 + i`."""
SCORED_SHAPES: Final[tuple[Shape, ...]] = ("S2", "S3", "S4")
SEED_TRAIN: Final[int] = 8400
TRAIN_EVENTS: Final[int] = 600
TRAIN_FIRST_ID: Final[int] = 60_000
SEED_UNEQUAL: Final[int] = 253
FRESH_BASE_EVENTS: Final[int] = 1160
FRESH_BASE_FIRST_ID: Final[int] = 40_000
_STATE: dict = {}


def init_worker(state: dict) -> None:
    """Pool initializer: share the panels, scales, thresholds, and fitted scores."""
    _STATE.update(state)
    if "scorer" in state:
        set_scorer(scorer=state["scorer"])


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


def detect(
    *,
    panel: dict,
    offset_panel: dict,
    residual: dict,
    offset: dict,
    scales: dict,
    thresholds: dict,
    models: dict,
) -> tuple[pl.DataFrame, list, DetectorInputs]:
    """Arm B's alarm table, its tracked intervals, and the 24-hour step statistic."""
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
    tracked = build_intervals(alarm_table=table, relative=relative, window=WINDOW, detector="D0")
    return table, tracked, inputs.fast


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
    table, tracked, fast = detect(
        panel=panel,
        offset_panel=offset_panel,
        residual=residual,
        offset=offset,
        scales=scales,
        thresholds=thresholds,
        models=models,
    )
    out = {
        rule: (
            table,
            live_end_intervals(intervals=tracked, inputs=fast, rule=rule, cap_days=CAP_DAYS),
        )
        for rule in RULES
    }
    return out, fast


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


def injected(
    *, seed: int, batch: list[Event], shape: Shape, variant: str = "clean"
) -> tuple[Injection, dict, dict]:
    """The injection of one batch and shape, and the injected residuals of both models.

    The convex model is frozen, so the residual is the stored residual plus the injected delta.
    Variant `C` adds an unrelated step (see `add_unrelated_step`).
    """
    panel, offset_panel = _STATE["panel"], _STATE["offset_panel"]
    injection = inject_shaped(
        power=panel["power"], events=batch, seed=seed, shape=shape, shares=_STATE["shares"]
    )
    if variant == "C":
        add_unrelated_step(injection=injection, batch=batch, seed=seed)
    return (
        injection,
        injection.fast_residual(panel["residual"]),
        injection.fast_residual(offset_panel["residual"]),
    )


def add_unrelated_step(*, injection: Injection, batch: list[Event], seed: int) -> None:
    """Add a 3-day step to one side of each pair transfer, 2 to 20 hours after the true end.

    The step restores the side's level and is 1.0 to 1.5 times the side's own step, so a rule
    that reads the side alone may take it for the end. The other side shows nothing.
    """
    rng = np.random.default_rng((seed, 7))
    for event in batch:
        if not event.receivers or event.donor == BLANK_SERIES:
            continue
        side = int(rng.choice(event.series))
        start = event.end + int(rng.integers(4, 41))
        level = float(np.mean(injection.delta[side][event.onset : event.end]))
        stop = min(start + 3 * BLOCK, GRID_LENGTH)
        injection.delta[side][start:stop] += -level * rng.uniform(1.0, 1.5)


def side_matches(*, event: Event, intervals: dict[int, list], tolerance: float) -> dict[int, dict]:
    """For each side of a transfer, the interval starting nearest the true start, and its end."""
    out: dict[int, dict] = {}
    for side in event.series:
        near = [
            iv for iv in intervals.get(side, []) if abs(iv.onset - event.onset) <= ONSET_TOLERANCE
        ]
        if not near:
            out[side] = {"onset_found": False, "found": False, "premature": False, "end": -1}
            continue
        best = min(near, key=lambda iv: abs(iv.onset - event.onset))
        error = best.end - event.end
        out[side] = {
            "onset_found": True,
            "found": abs(error) <= tolerance,
            "premature": error < -tolerance,
            "end": best.end,
            "known_from": best.known_from,
        }
    return out


def disagreement_hours(*, sides: dict[int, dict]) -> float:
    """The gap in hours between the earliest and latest end called by the sides that alarmed."""
    ends = [m["end"] for m in sides.values() if m["onset_found"]]
    return (max(ends) - min(ends)) / 2 if len(ends) >= 2 else float("nan")


def score_batch(args: tuple[int, list[Event], Shape, str]) -> list[dict]:
    """Inject one batch with one shape and variant, detect, and return one row per side and rule."""
    seed, batch, shape, variant = args
    injection, residual, offset = injected(seed=seed, batch=batch, shape=shape, variant=variant)
    results, _ = build_detectors(
        panel=_STATE["panel"],
        offset_panel=_STATE["offset_panel"],
        residual=residual,
        offset=offset,
        scales=_STATE["scales"],
        thresholds=_STATE["thresholds"],
        models=_STATE["models"],
    )
    rows = []
    for setting, (_, intervals) in results.items():
        grouped = by_series(intervals=intervals)
        for event in batch:
            if event.donor == BLANK_SERIES:
                continue
            moved = float(-np.mean(injection.delta[event.donor][event.onset : event.end]))
            tolerance = end_tolerance(onset=event.onset, end=event.end)
            sides = side_matches(event=event, intervals=grouped, tolerance=tolerance)
            best = match_end(
                series=event.series, onset=event.onset, end=event.end, intervals=grouped
            )
            common = {
                "event_id": event.event_id,
                "setting": setting,
                "shape": shape,
                "variant": variant,
                "moved_mw": moved,
            }
            rows.append(
                {
                    **common,
                    "kind": "event",
                    "found": best["found"],
                    "onset_found": best["onset_found"],
                    "disagreement_hours": disagreement_hours(sides=sides),
                }
            )
            rows.extend(
                {
                    **common,
                    "kind": "side",
                    "side": side,
                    "is_donor": side == event.donor,
                    "onset_found": match["onset_found"],
                    "found": match["found"],
                    "premature": match["premature"],
                    "end_error_hours": (match["end"] - event.end) / 2
                    if match["onset_found"]
                    else float("nan"),
                    "lag_hours": (match["known_from"] - event.end) / 2
                    if match["onset_found"]
                    else float("nan"),
                }
                for side, match in sides.items()
            )
    return rows


def train_batch(args: tuple[int, list[Event], Shape]) -> tuple[np.ndarray, np.ndarray]:
    """Features and labels of every balanced candidate end of every paired alarm in one batch."""
    seed, batch, shape = args
    _, residual, offset = injected(seed=seed, batch=batch, shape=shape)
    _, tracked, fast = detect(
        panel=_STATE["panel"],
        offset_panel=_STATE["offset_panel"],
        residual=residual,
        offset=offset,
        scales=_STATE["scales"],
        thresholds=_STATE["thresholds"],
        models=_STATE["models"],
    )
    features, labels = [], []
    for iv in tracked:
        for event in batch:
            if iv.series not in event.series or abs(iv.onset - event.onset) > ONSET_TOLERANCE:
                continue
            cand = Candidates(inputs=fast, iv=iv, cap_days=CAP_DAYS)
            if not cand.paired:
                continue
            matrix = cand.features()
            keep = np.flatnonzero(cand.qualifies() & cand.balanced() & np.isfinite(matrix).all(1))
            tolerance = end_tolerance(onset=event.onset, end=event.end)
            features.append(matrix[keep])
            labels.append(np.abs(cand.base + keep - event.end) <= tolerance)
    if not features:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=bool)
    return np.vstack(features), np.concatenate(labels)


class _Unpickler(pickle.Unpickler):
    """Load experiment 138's fitted scores, whose classes live in a module of the same code."""

    def find_class(self, module: str, name: str) -> object:
        """Map experiment 138's module names to this folder's copy."""
        return super().find_class(module.replace("e00138", "e00253"), name)


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


def draw_base(*, panel: dict) -> list[Event]:
    """Experiment 148's headline-band draw (same seed)."""
    return draw_events(
        quiet=quiet_masks(),
        power=panel["power"],
        seed=SEED_FRESH_BASE,
        headline_only=FRESH_BASE_EVENTS,
        first_id=FRESH_BASE_FIRST_ID,
    )


def train(*, out: Path) -> None:
    """Fit rule P3's scorer on a draw that starts in odd months."""
    state = make_state()
    panel = state["panel"]
    months = stamp_months()
    events = draw_events(
        quiet=quiet_masks(),
        power=panel["power"],
        seed=SEED_TRAIN,
        headline_only=TRAIN_EVENTS,
        first_id=TRAIN_FIRST_ID,
        onset_ok=lambda onset: bool(months[onset] % 2 == 1),
    )
    groups = seeded_batches(events=events, seed=SEED_TRAIN + 1000)
    jobs = [(seed, batch, shape) for shape in SCORED_SHAPES for seed, batch in groups]
    parts = run_pool(function=train_batch, jobs=jobs, state=state)
    features = np.vstack([f for f, _ in parts])
    labels = np.concatenate([y for _, y in parts])
    scorer = make_pipeline(
        StandardScaler(), MLPClassifier(hidden_layer_sizes=(8,), max_iter=500, random_state=253)
    ).fit(features, labels)
    with (out / "scorer.pkl").open("wb") as handle:
        pickle.dump(scorer, handle)
    print(
        f"{len(events)} training events, {len(labels)} candidates, "
        f"{labels.mean():.3f} positive, training accuracy {scorer.score(features, labels):.3f}",
        flush=True,
    )


def with_unequal_steps(*, events: list[Event]) -> list[Event]:
    """Multiply each transfer's receiver shares by 0.9 or 1.1, chosen by seed."""
    rng = np.random.default_rng(SEED_UNEQUAL)
    return [
        replace(event, weights=tuple(w * f for w in event.weights))
        for event in events
        for f in [float(rng.choice([0.9, 1.1]))]
    ]


def score(*, out: Path) -> None:
    """Score every injected transfer under shapes S2 to S4 with every rule and variant."""
    state = make_state()
    with (out / "scorer.pkl").open("rb") as handle:
        state["scorer"] = pickle.load(handle)
    base = draw_base(panel=state["panel"])
    events_frame(events=base).write_parquet(out / "events.parquet")
    print(f"{len(base)} events", flush=True)
    unequal = with_unequal_steps(events=base)
    jobs = []
    for variant, events in (("clean", base), ("C", base), ("U10", unequal)):
        by_id = {e.event_id: e for e in events}
        for seed, batch in seeded_batches(events=base, seed=SEED_FRESH_BASE + 1000):
            variant_batch = [by_id[e.event_id] for e in batch]
            jobs.extend((seed, variant_batch, shape, variant) for shape in SCORED_SHAPES)
    rows = [r for part in run_pool(function=score_batch, jobs=jobs, state=state) for r in part]
    scored = pl.DataFrame(rows, infer_schema_length=None)
    scored.write_parquet(out / "results.parquet")
    print(f"{scored.height} rows", flush=True)


def real(*, out: Path) -> None:
    """Score the labelled real transfers with every rule, side by side."""
    panel, offset_panel = load_both()
    state = make_state()
    with (out / "scorer.pkl").open("rb") as handle:
        set_scorer(scorer=pickle.load(handle))
    results, _ = build_detectors(
        panel=panel,
        offset_panel=offset_panel,
        residual=panel["residual"],
        offset=offset_panel["residual"],
        scales=state["scales"],
        thresholds=state["thresholds"],
        models=state["models"],
    )
    labelled = real_events(power=panel["power"])
    rows = []
    for setting, (_, intervals) in results.items():
        grouped = by_series(intervals=intervals)
        for row in labelled.iter_rows(named=True):
            series = tuple(int(s) for s in row["series"].split(",") if int(s) in panel["power"])
            if len(series) < 2:
                continue
            event = Event(0, series[0], series[1:], (), row["onset"], row["end"], "", "", 0.0)
            sides = side_matches(
                event=event,
                intervals=grouped,
                tolerance=end_tolerance(onset=row["onset"], end=row["end"]),
            )
            for side, match in sides.items():
                rows.append(
                    {
                        "card_id": row["card_id"],
                        "onset": row["onset"],
                        "end": row["end"],
                        "setting": setting,
                        "side": side,
                        "hours": row["hours"],
                        "onset_found": match["onset_found"],
                        "found": match["found"],
                        "end_error_hours": (match["end"] - row["end"]) / 2
                        if match["onset_found"]
                        else float("nan"),
                        "disagreement_hours": disagreement_hours(sides=sides),
                    }
                )
    scored = pl.DataFrame(rows, infer_schema_length=None)
    scored.write_parquet(out / "real_results.parquet")
    print(f"{scored.height} real rows", flush=True)


def main() -> None:
    """Run the requested stages."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stages", nargs="+", choices=("train", "score", "real"))
    parser.add_argument("--out", type=Path, default=WORK_DIR / "run")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stages = {"train": train, "score": score, "real": real}
    for stage in args.stages:
        stages[stage](out=args.out)


if __name__ == "__main__":
    main()
