r"""Train the two logistic scores, then score them and PF14 on fresh draws and the real transfers.

Stages, in order. Each writes files to the private output folder.

- `tune`: D0M's threshold (must come back as 4.0).
- `train`: LOGIT and LOGIT_SAME from odd-month injections of experiment 58's draw, and each one's
  threshold at 80 unexplained odd-month alarms on the real residual.
- `harness-check`: PF14 must reproduce experiment 69's `found` flags on a sample of its fresh
  draw, and the combined arms must differ from PF14 only through their alarms.
- `score`: the outcome of every fresh injected transfer for every arm under all five shapes.
- `real`: the labelled real transfers' outcomes and the real alarm counts.

    uv run python ml-experiments/00070-*/e00070_run.py tune train harness-check score real
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import pickle
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00070_combined import (
    FEATURES,
    FEATURES_SAME,
    candidates,
    fit_logistic,
    positive_flags,
    threshold_for_count,
)
from e00070_common import (
    LABELLED_PATH,
    ML_EXPERIMENTS_DATA_DIR,
    TUNING_MONTHS,
    WORK_DIR,
    near_log_masks,
    stamp_months,
)
from e00070_detect import (
    EXPLAIN_PAD,
    NO_VETO,
    SEED_43,
    SEED_NEW,
    explained_flags,
    load,
    log_masks,
    new_events,
    original_events,
    quiet_masks,
    real_alarm_rows,
    real_events,
    score_real,
    seeded_batches,
    tune_d2,
)
from e00070_detectors import alarms, prepare
from e00070_endrule import first_return_intervals
from e00070_inject import BLOCK, HEADLINE_BAND, Event, draw_events, events_frame
from e00070_intervals import build_intervals, relative_residual
from e00070_long import draw_long_events
from e00070_profile import SHAPES, Shape, fit_profile_shares, inject_shaped, shares_json
from e00070_score import by_series, match_event

EXPERIMENT_69_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00069-sw2-17-missed-onsets-of-the-end-rule"
)
N_WORKERS: Final[int] = 4
WINDOW: Final[int] = 48
CAP_DAYS: Final[int] = 14
TARGET: Final[int] = 80
"""D0M's odd-month unexplained-alarm count, which every arm is held to."""
D0M_THRESHOLD: Final[float] = 4.0
HOLIDAY_FIRST_ID: Final[int] = 3000
SEED_FRESH_BASE: Final[int] = 7100
"""Seed of the fresh headline-band draw; batch `i` is injected with `SEED_FRESH_BASE + 1000 + i`."""
SEED_FRESH_LONG: Final[int] = 8100
"""Seed of the fresh long draw; batch `i` is injected with `SEED_FRESH_LONG + 1000 + i`."""
FRESH_BASE_EVENTS: Final[int] = 1160
FRESH_BASE_FIRST_ID: Final[int] = 20_000
FRESH_LONG_FIRST_ID: Final[int] = 25_000
EXPERIMENT_69_BASE_SEED: Final[int] = 6900
EXPERIMENT_69_BASE_FIRST_ID: Final[int] = 10_000
CHECK_ROWS: Final[int] = 400
SCORES: Final[tuple[str, ...]] = ("LOGIT", "LOGIT_SAME")
ARMS: Final[tuple[str, ...]] = ("PF14", *SCORES)
_STATE: dict = {}


def init_worker(state: dict) -> None:
    """Pool initializer: share the panel, scale, thresholds, and fitted scores."""
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


def build_detectors(
    *, residual: dict, expected: dict, scale: dict, thresholds: dict, models: dict
) -> dict:
    """Alarm tables and PF14 intervals of PF14, LOGIT, and LOGIT_SAME on one residual panel."""
    relative = {s: relative_residual(residual=residual[s], expected=expected[s]) for s in residual}
    inputs = prepare(residuals=residual, window=WINDOW, scale=scale)
    tables = {"PF14": alarms(inputs=inputs, detector="D0", threshold=thresholds["D0M"])}
    cand = candidates(inputs=inputs)
    for name in SCORES:
        keep = cand.filter(
            pl.Series(models[name].score(table=cand) > thresholds[name]) if cand.height else []
        )
        tables[name] = keep.select(
            "series",
            "stamp",
            "step_mw",
            z=pl.struct("series", "stamp").map_elements(
                lambda r: float(inputs.shift[r["series"]][r["stamp"]] / inputs.scale[r["series"]]),
                return_dtype=pl.Float64,
            ),
            partners=pl.lit(""),
        )
    out = {}
    for arm, table in tables.items():
        tracked = build_intervals(
            alarm_table=table.with_columns(partners=pl.lit("")) if arm == "PF14" else table,
            relative=relative,
            window=WINDOW,
            detector="D0",
        )
        out[arm] = (
            table,
            first_return_intervals(intervals=tracked, inputs=inputs, cap_days=CAP_DAYS),
        )
    return out


def injected_inputs(*, seed: int, batch: list[Event], shape: Shape) -> tuple[dict, object]:
    """The injected residual of one batch and shape, and the injection."""
    panel = _STATE["panel"]
    injection = inject_shaped(
        power=panel["power"], events=batch, seed=seed, shape=shape, shares=_STATE["shares"]
    )
    return injection.fast_residual(panel["residual"]), injection


def is_training_event(*, event: Event, months: np.ndarray) -> bool:
    """A headline-band transfer that starts in an odd month, outside the holiday set."""
    return (
        event.event_id < HOLIDAY_FIRST_ID
        and (event.size_band, event.duration_band) == HEADLINE_BAND
        and int(months[event.onset]) in TUNING_MONTHS
    )


def positives_batch(args: tuple[int, list[Event], Shape]) -> pl.DataFrame:
    """The candidates within 2 hours of the onset of a training transfer in one injected batch."""
    seed, batch, shape = args
    training = [e for e in batch if is_training_event(event=e, months=stamp_months())]
    if not training:
        return pl.DataFrame()
    residual, _ = injected_inputs(seed=seed, batch=batch, shape=shape)
    cand = candidates(inputs=prepare(residuals=residual, window=WINDOW, scale=_STATE["scale"]))
    return cand.filter(pl.Series(positive_flags(table=cand, events=training))).with_columns(
        shape=pl.lit(shape)
    )


def score_batch(args: tuple[int, list[Event], Shape]) -> list[dict]:
    """Inject one batch with one shape, detect with every arm, return one row per event."""
    seed, batch, shape = args
    panel = _STATE["panel"]
    residual, injection = injected_inputs(seed=seed, batch=batch, shape=shape)
    results = build_detectors(
        residual=residual,
        expected=panel["expected"],
        scale=_STATE["scale"],
        thresholds=_STATE["thresholds"],
        models=_STATE["models"],
    )
    rows = []
    for arm, (_, intervals) in results.items():
        grouped = by_series(intervals=intervals)
        for event in batch:
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
                    / _STATE["scale"][event.donor],
                }
            )
    return rows


def read_json(*, out: Path, name: str) -> dict:
    """A JSON file `tune` or `train` wrote."""
    return json.loads((out / name).read_text())


def make_state(*, out: Path, trained: bool = True) -> dict:
    """Panel, scales, thresholds, and fitted scores for the workers."""
    panel, _ = load()
    return {
        "panel": panel,
        "scale": prepare(residuals=panel["residual"], window=WINDOW).scale,
        "shares": fit_profile_shares(panel=panel),
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


def fresh_events(
    *, panel: dict, base_seed: int, base_first_id: int, long_seed: int, long_first_id: int
) -> tuple[list[Event], list[Event]]:
    """A headline-band draw and a long draw."""
    base = draw_events(
        quiet=quiet_masks(),
        power=panel["power"],
        seed=base_seed,
        headline_only=FRESH_BASE_EVENTS,
        first_id=base_first_id,
    )
    long = draw_long_events(
        quiet=log_masks(), power=panel["power"], seed=long_seed, first_id=long_first_id
    )
    return base, long


def tune(*, out: Path) -> None:
    """Check that D0M's matched threshold is 4.0."""
    panel, _ = load()
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    plain = prepare(residuals=panel["residual"], window=WINDOW)
    d0m, _ = tune_d2(inputs=plain, veto_z=NO_VETO, logged=logged, target=TARGET)
    print(f"D0M threshold {d0m}")
    if d0m != D0M_THRESHOLD:
        raise SystemExit(f"STOP: D0M matched {d0m}, not {D0M_THRESHOLD}")
    (out / "thresholds.json").write_text(json.dumps({"D0M": d0m}))


def train(*, out: Path) -> None:
    """Fit LOGIT and LOGIT_SAME and set each one's threshold on the real odd months."""
    state = make_state(out=out, trained=False)
    panel = state["panel"]
    jobs = [
        (seed, batch, shape) for shape in SHAPES for seed, batch in training_groups(panel=panel)
    ]
    positive = pl.concat(
        [p for p in run_pool(function=positives_batch, jobs=jobs, state=state) if p.height]
    )
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    cand = candidates(
        inputs=prepare(residuals=panel["residual"], window=WINDOW, scale=state["scale"])
    )
    odd = np.isin(stamp_months()[cand["stamp"].to_numpy()], list(TUNING_MONTHS))
    negative = cand.filter(pl.Series(odd & ~explained_flags(table=cand, logged=logged)))
    print(f"{positive.height} positive and {negative.height} negative candidates")
    models = {
        "LOGIT": fit_logistic(positive=positive, negative=negative, features=FEATURES),
        "LOGIT_SAME": fit_logistic(positive=positive, negative=negative, features=FEATURES_SAME),
    }
    thresholds = read_json(out=out, name="thresholds.json")
    for name, model in models.items():
        scores = model.score(table=negative)
        thresholds[name] = threshold_for_count(scores=scores, target=TARGET)
        coef = dict(zip(model.features, np.round(model.coef, 3).tolist(), strict=True))
        print(f"{name}: threshold {thresholds[name]:.4f}, kept {(scores > thresholds[name]).sum()}")
        print(f"  standardised coefficients {coef}")
    (out / "models.pkl").write_bytes(pickle.dumps(models))
    (out / "thresholds.json").write_text(json.dumps(thresholds))
    positive.write_parquet(out / "positives.parquet")
    negative.write_parquet(out / "negatives.parquet")


def harness_check(*, out: Path) -> None:
    """PF14 must reproduce experiment 69's flags on a sample of its fresh headline-band draw."""
    state = make_state(out=out)
    panel = state["panel"]
    base = draw_events(
        quiet=quiet_masks(),
        power=panel["power"],
        seed=EXPERIMENT_69_BASE_SEED,
        headline_only=FRESH_BASE_EVENTS,
        first_id=EXPERIMENT_69_BASE_FIRST_ID,
    )
    kept: list[tuple[int, list[Event]]] = []
    total = 0
    for seed, batch in seeded_batches(events=base, seed=EXPERIMENT_69_BASE_SEED + 1000):
        if total >= CHECK_ROWS:
            break
        kept.append((seed, batch))
        total += len(batch)
    jobs = [(seed, batch, shape) for shape in ("S2", "S4") for seed, batch in kept]
    scored = pl.DataFrame(
        [r for part in run_pool(function=score_batch, jobs=jobs, state=state) for r in part],
        infer_schema_length=None,
    ).filter(pl.col("setting") == "PF14")
    stored = pl.read_parquet(EXPERIMENT_69_DIR / "event_results.parquet").filter(
        pl.col("setting") == "PF14_48"
    )
    joined = scored.join(stored, on=["event_id", "shape"], suffix="_69", validate="1:1")
    different = joined.filter(
        (pl.col("found") != pl.col("found_69"))
        | (pl.col("onset_found") != pl.col("onset_found_69"))
    ).height
    print(f"{joined.height} rows, {different} differ from experiment 69's PF14")
    if joined.height != scored.height or different:
        raise SystemExit("STOP: the harness does not reproduce experiment 69's PF14")
    print("harness reproduced")


def score(*, out: Path) -> None:
    """Score every fresh transfer under all five shapes with every arm."""
    state = make_state(out=out)
    panel = state["panel"]
    (out / "profile_shares.json").write_text(shares_json(shares=state["shares"]))
    base, long = fresh_events(
        panel=panel,
        base_seed=SEED_FRESH_BASE,
        base_first_id=FRESH_BASE_FIRST_ID,
        long_seed=SEED_FRESH_LONG,
        long_first_id=FRESH_LONG_FIRST_ID,
    )
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
    panel, _ = load()
    results = build_detectors(
        residual=panel["residual"],
        expected=panel["expected"],
        scale=prepare(residuals=panel["residual"], window=WINDOW).scale,
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
    parser.add_argument(
        "stages", nargs="+", choices=("tune", "train", "harness-check", "score", "real")
    )
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stages = {
        "tune": tune,
        "train": train,
        "harness-check": harness_check,
        "score": score,
        "real": real,
    }
    for stage in args.stages:
        stages[stage](out=args.out)


if __name__ == "__main__":
    main()
