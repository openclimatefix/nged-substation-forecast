r"""Train the logistic scores, then score the four arms on a fresh draw and the real transfers.

Arm B is experiment 130's RO-NC on the frozen convex residual. W1, W2, and W3 look for weaker and
slower onsets (see `e00138_arms.py`). Every arm fits its own logistic score on injections from odd
months and sets its own threshold at 80 unexplained odd-month alarms.

Stages, in order. Each writes files to the private output folder.

- `train`: each arm's logistic score and threshold.
- `score`: the outcome of every fresh injected transfer for every arm under all five shapes.
- `real`: the labelled real transfers' outcomes and the real alarm counts.

    E138_WORKERS=6 OMP_NUM_THREADS=1 uv run --no-sync python \
        ml-experiments/00138-*/e00138_run.py train score real
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
from e00138_arms import ARMS, PERSIST_GAP, SLOW_WINDOW, arm_candidates, arm_inputs
from e00138_combined import fit_logistic, positive_flags, threshold_for_count
from e00138_common import (
    LABELLED_PATH,
    TUNING_MONTHS,
    WORK_DIR,
    near_log_masks,
    stamp_months,
)
from e00138_detect import (
    EXPLAIN_PAD,
    SEED_43,
    SEED_NEW,
    explained_flags,
    log_masks,
    new_events,
    original_events,
    quiet_masks,
    real_alarm_rows,
    real_events,
    score_real,
    seeded_batches,
)
from e00138_endrule import first_return_intervals
from e00138_inject import BLOCK, HEADLINE_BAND, Event, Injection, draw_events, events_frame
from e00138_intervals import build_intervals
from e00138_long import draw_long_events
from e00138_profile import SHAPES, Shape, fit_profile_shares, inject_shaped, shares_json
from e00138_score import by_series, match_event
from e00138_variants import BLANK_SERIES, load_both, scales_for, tracked_relative

N_WORKERS: Final[int] = int(os.environ.get("E138_WORKERS", "6"))
WINDOW: Final[int] = 48
CAP_DAYS: Final[int] = 14
TARGET: Final[int] = 80
HOLIDAY_FIRST_ID: Final[int] = 3000
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
) -> dict:
    """Alarm tables and first-return intervals of every arm on one residual panel."""
    out = {}
    relative = tracked_relative(
        variant="X", panel=panel, offset_panel=offset_panel, residual=residual, offset=offset
    )
    for arm, spec in ARMS.items():
        inputs = arm_inputs(spec=spec, residual=residual, offset=offset, scales=scales)
        cand = arm_candidates(spec=spec, inputs=inputs)
        keep = cand.filter(
            pl.Series(models[arm].score(table=cand) > thresholds[arm]) if cand.height else []
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
        tracked = build_intervals(
            alarm_table=table, relative=relative, window=WINDOW, detector="D0"
        )
        out[arm] = (
            table,
            first_return_intervals(intervals=tracked, inputs=inputs.fast, cap_days=CAP_DAYS),
        )
    return out


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


def is_training_event(*, event: Event, months: np.ndarray) -> bool:
    """A headline-band transfer that starts in an odd month, outside the holiday set."""
    return (
        event.donor != BLANK_SERIES
        and event.event_id < HOLIDAY_FIRST_ID
        and (event.size_band, event.duration_band) == HEADLINE_BAND
        and int(months[event.onset]) in TUNING_MONTHS
    )


def positives_batch(args: tuple[int, list[Event], Shape]) -> pl.DataFrame:
    """Candidates within 2 hours of the onset of a training transfer, for every arm."""
    seed, batch, shape = args
    training = [e for e in batch if is_training_event(event=e, months=stamp_months())]
    if not training:
        return pl.DataFrame()
    _, residual, offset = injected(seed=seed, batch=batch, shape=shape)
    parts = []
    for arm, spec in ARMS.items():
        cand = arm_candidates(
            spec=spec,
            inputs=arm_inputs(spec=spec, residual=residual, offset=offset, scales=_STATE["scales"]),
        )
        parts.append(
            cand.filter(pl.Series(positive_flags(table=cand, events=training))).with_columns(
                shape=pl.lit(shape), arm=pl.lit(arm)
            )
        )
    return pl.concat(parts, how="diagonal")


def score_batch(args: tuple[int, list[Event], Shape]) -> list[dict]:
    """Inject one batch with one shape, detect with every arm, return one row per event and arm."""
    seed, batch, shape = args
    injection, residual, offset = injected(seed=seed, batch=batch, shape=shape)
    results = build_detectors(
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
    for arm, (_, intervals) in results.items():
        grouped = by_series(intervals=intervals)
        for event in batch:
            if event.donor == BLANK_SERIES:
                continue
            match = match_event(
                series=event.series, onset=event.onset, end=event.end, intervals=grouped
            )
            span = slice(event.onset, min(event.end, event.onset + BLOCK))
            rows.append(
                {
                    "event_id": event.event_id,
                    "setting": arm,
                    "shape": shape,
                    "found": match.found,
                    "onset_found": match.onset_found,
                    "end_error": match.end_error,
                    "moved_mw": float(
                        -np.mean(injection.delta[event.donor][event.onset : event.end])
                    ),
                    "donor_step_z": float(-np.mean(injection.delta[event.donor][span]))
                    / plain_scale[event.donor],
                }
            )
    return rows


def read_json(*, out: Path, name: str) -> dict:
    """A JSON file `train` wrote."""
    return json.loads((out / name).read_text())


def make_state(*, out: Path, trained: bool = True) -> dict:
    """Panels, scales, thresholds, fitted scores, and fitted shares for the workers."""
    panel, offset_panel = load_both()
    return {
        "panel": panel,
        "offset_panel": offset_panel,
        "scales": all_scales(panel=panel, offset_panel=offset_panel),
        "shares": fit_profile_shares(panel=load_both(source="A16")[0]),
        "thresholds": read_json(out=out, name="thresholds.json") if trained else {},
        "models": pickle.loads((out / "models.pkl").read_bytes()) if trained else {},
    }


def training_groups(*, panel: dict) -> list[tuple[int, list[Event]]]:
    """Experiment 58's original, extra, and holiday events in its batches."""
    original = original_events(power=panel["power"])
    extra, holiday = new_events(power=panel["power"])
    return seeded_batches(events=original, seed=SEED_43) + seeded_batches(
        events=extra + holiday, seed=SEED_NEW
    )


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


def train(*, out: Path) -> None:
    """Fit each arm's logistic score and set its threshold at 80 unexplained odd-month alarms."""
    state = make_state(out=out, trained=False)
    panel, offset_panel = state["panel"], state["offset_panel"]
    jobs = [
        (seed, batch, shape) for shape in SHAPES for seed, batch in training_groups(panel=panel)
    ]
    positive = pl.concat(
        [p for p in run_pool(function=positives_batch, jobs=jobs, state=state) if p.height],
        how="diagonal",
    )
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    models, thresholds = {}, {}
    for arm, spec in ARMS.items():
        cand = arm_candidates(
            spec=spec,
            inputs=arm_inputs(
                spec=spec,
                residual=panel["residual"],
                offset=offset_panel["residual"],
                scales=state["scales"],
            ),
        )
        odd = np.isin(stamp_months()[cand["stamp"].to_numpy()], list(TUNING_MONTHS))
        negative = cand.filter(pl.Series(odd & ~explained_flags(table=cand, logged=logged)))
        arm_positive = positive.filter(pl.col("arm") == arm)
        print(f"{arm}: {arm_positive.height} positive and {negative.height} negative candidates")
        model = fit_logistic(positive=arm_positive, negative=negative, features=spec.features)
        scores = model.score(table=negative)
        thresholds[arm] = threshold_for_count(scores=scores, target=TARGET)
        coef = dict(zip(model.features, np.round(model.coef, 3).tolist(), strict=True))
        print(f"{arm}: threshold {thresholds[arm]:.4f}, kept {(scores > thresholds[arm]).sum()}")
        print(f"  standardised coefficients {coef}")
        models[arm] = model
        arm_positive.write_parquet(out / f"positives_{arm}.parquet")
        negative.write_parquet(out / f"negatives_{arm}.parquet")
    (out / "models.pkl").write_bytes(pickle.dumps(models))
    (out / "thresholds.json").write_text(json.dumps(thresholds))


def score(*, out: Path) -> None:
    """Score every fresh transfer under all five shapes with every arm."""
    state = make_state(out=out)
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
    """Score the labelled real transfers and count the real alarms of every arm."""
    panel, offset_panel = load_both()
    results = build_detectors(
        panel=panel,
        offset_panel=offset_panel,
        residual=panel["residual"],
        offset=offset_panel["residual"],
        scales=all_scales(panel=panel, offset_panel=offset_panel),
        thresholds=read_json(out=out, name="thresholds.json"),
        models=pickle.loads((out / "models.pkl").read_bytes()),
    )
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    real_alarm_rows(results=results, logged=logged).write_parquet(out / "alarms.parquet")
    moved = pl.read_parquet(LABELLED_PATH).select("card_id", "moved_mw_p50")
    scored = score_real(results=results, real=real_events(power=panel["power"])).join(
        moved, on="card_id", validate="m:1"
    )
    scored.write_parquet(out / "real_results.parquet")
    print({s: int(explained_flags(table=t, logged=logged).sum()) for s, (t, _) in results.items()})


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
