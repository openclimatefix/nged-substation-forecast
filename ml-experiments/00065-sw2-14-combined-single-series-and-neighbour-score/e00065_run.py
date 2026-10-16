r"""Match every arm's alarm count on odd months, train the two combined scores, and score them.

Stages, in order. Each writes files to the private output folder.

- `tune`: the matched thresholds of D0M, D1, D2 (veto 2.0), and D2 at veto 1.0 (post hoc).
- `train`: LR, LOGIT, and LOGIT_SAME (post hoc) from odd-month injections, and their thresholds.
- `harness-check`: with shape S1, D0M must reproduce experiment 58's outcome for every transfer.
- `score`: the outcome of every injected transfer for every arm, end rule, and shape.
- `real`: the labelled real transfers' outcomes and the real alarm counts.

    uv run python ml-experiments/00065-*/e00065_run.py tune train harness-check score real
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
from e00065_combined import (
    FEATURES,
    FEATURES_SAME,
    SCORES,
    candidates,
    fit_likelihood_ratio,
    fit_logistic,
    positive_flags,
    threshold_for_count,
)
from e00065_common import (
    LABELLED_PATH,
    ML_EXPERIMENTS_DATA_DIR,
    TUNING_MONTHS,
    WORK_DIR,
    near_log_masks,
    stamp_months,
)
from e00065_detect import (
    EXPLAIN_PAD,
    NO_VETO,
    SEED_43,
    SEED_NEW,
    explained_flags,
    kept_alarms,
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
from e00065_detectors import DetectorInputs, alarms, prepare
from e00065_ends import end_only_intervals
from e00065_inject import BLOCK, HEADLINE_BAND, Event, Injection
from e00065_intervals import build_intervals, relative_residual
from e00065_profile import SHAPES, Shape, fit_profile_shares, inject_shaped, shares_json
from e00065_score import by_series, match_event

WINDOW: Final[int] = 48
TARGET: Final[int] = 80
"""D0M's odd-month unexplained-alarm count, which every arm is held to."""
D0M_THRESHOLD: Final[float] = 4.0
D2_VETO: Final[float] = 2.0
D2_VETO_POST_HOC: Final[float] = 1.0
HOLIDAY_FIRST_ID: Final[int] = 3000
EXPERIMENT_58_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00058-sw2-8b-donor-step-recut-and-stronger-veto"
)
N_WORKERS: Final[int] = 4
ARMS: Final[tuple[str, ...]] = ("D0M", "D1", "D2", "D2V1", *SCORES)
RULES: Final[tuple[str, ...]] = ("TR", "PE")
"""The tracked end of experiment 43 and the strongest-opposite-step end of experiment 61."""
_STATE: dict = {}


def setting(*, arm: str, rule: str) -> str:
    """The name of one arm under one end rule."""
    return f"{arm}_{rule}"


def tune_down(*, count: Callable[[float], int]) -> tuple[float, list[tuple[float, int]]]:
    """The lowest threshold on a 0.05 grid whose odd-month unexplained count is at most `TARGET`.

    The scan moves down from 8 in steps of 0.25 to the first count above the target, then up in
    steps of 0.05 to the first count at or under it,
    then down while the count stays at or under it. The second value holds every pair evaluated.
    """
    curve: dict[float, int] = {}

    def cached(threshold: float) -> int:
        key = round(threshold, 2)
        if key not in curve:
            curve[key] = count(key)
        return curve[key]

    threshold = 8.0
    while threshold > 1.0 and cached(threshold) <= TARGET:
        threshold = round(threshold - 0.25, 2)
    while cached(threshold) > TARGET:
        threshold = round(threshold + 0.05, 2)
    while threshold - 0.05 >= 1.0 and cached(threshold - 0.05) <= TARGET:
        threshold = round(threshold - 0.05, 2)
    print(f"  odd-month unexplained alarms at {threshold}: {cached(threshold)} (target {TARGET})")
    return threshold, sorted(curve.items())


def score_tables(
    *, inputs: DetectorInputs, models: dict, thresholds: dict
) -> dict[str, pl.DataFrame]:
    """The alarm table of each combined score: the candidates whose score exceeds its threshold."""
    cand = candidates(inputs=inputs)
    out = {}
    for name in SCORES:
        keep = cand.filter(
            pl.Series(models[name].score(table=cand) > thresholds[name]) if cand.height else []
        )
        out[name] = keep.select(
            "series",
            "stamp",
            "step_mw",
            z=pl.struct("series", "stamp").map_elements(
                lambda r: float(inputs.shift[r["series"]][r["stamp"]] / inputs.scale[r["series"]]),
                return_dtype=pl.Float64,
            ),
            partners=pl.lit(""),
        )
    return out


def build_detectors(
    *, residual: dict, expected: dict, scale: dict, thresholds: dict, models: dict
) -> dict:
    """Alarm tables and intervals of every arm under both end rules on one residual panel."""
    relative = {s: relative_residual(residual=residual[s], expected=expected[s]) for s in residual}
    inputs = prepare(residuals=residual, window=WINDOW, scale=scale)
    tables = {
        "D0M": alarms(inputs=inputs, detector="D0", threshold=thresholds["D0M"]).with_columns(
            partners=pl.lit("")
        ),
        "D1": alarms(inputs=inputs, detector="D1", threshold=thresholds["D1"]),
    }
    for arm, veto in (("D2", D2_VETO), ("D2V1", D2_VETO_POST_HOC)):
        kept = kept_alarms(inputs=inputs, threshold=thresholds[arm], veto_z=veto)
        tables[arm] = kept.filter(pl.col("tier") != "vetoed").with_columns(partners=pl.lit(""))
    tables |= score_tables(inputs=inputs, models=models, thresholds=thresholds)
    out = {}
    for arm, table in tables.items():
        tracked = build_intervals(
            alarm_table=table,
            relative=relative,
            window=WINDOW,
            detector="D1" if arm == "D1" else "D0",
        )
        out[setting(arm=arm, rule="TR")] = (table, tracked)
        out[setting(arm=arm, rule="PE")] = (
            table,
            end_only_intervals(intervals=tracked, inputs=inputs),
        )
    return out


def init_worker(state: dict) -> None:
    """Pool initializer: share the panel, scales, thresholds, and fitted scores."""
    _STATE.update(state)


def injected_inputs(*, seed: int, batch: list[Event], shape: Shape) -> tuple[dict, Injection]:
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
    plain_scale = _STATE["scale"]
    rows = []
    for name, (_, intervals) in results.items():
        grouped = by_series(intervals=intervals)
        for event in batch:
            match = match_event(
                series=event.series, onset=event.onset, end=event.end, intervals=grouped
            )
            span = slice(event.onset, min(event.end, event.onset + BLOCK))
            rows.append(
                {
                    "event_id": event.event_id,
                    "setting": name,
                    "shape": shape,
                    "found": match.found,
                    "onset_found": match.onset_found,
                    "moved_mw": float(
                        -np.mean(injection.delta[event.donor][event.onset : event.end])
                    ),
                    "donor_step_z": float(-np.mean(injection.delta[event.donor][span]))
                    / plain_scale[event.donor],
                    "largest_step_z": max(
                        abs(float(np.mean(injection.delta[s][span]))) / plain_scale[s]
                        for s in event.series
                    ),
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


def event_groups(*, panel: dict) -> list[tuple[int, list[Event]]]:
    """Experiment 58's original, extra, and holiday events in its batches, so S1 reproduces it."""
    original = original_events(power=panel["power"])
    extra, holiday = new_events(power=panel["power"])
    return seeded_batches(events=original, seed=SEED_43) + seeded_batches(
        events=extra + holiday, seed=SEED_NEW
    )


def read_json(*, out: Path, name: str) -> dict:
    """A JSON file `tune` or `train` wrote."""
    return json.loads((out / name).read_text())


def make_state(*, out: Path, trained: bool = True) -> tuple[dict, list]:
    """Panel, scales, thresholds, and fitted scores for the workers, with the event batches."""
    panel, _ = load()
    state = {
        "panel": panel,
        "scale": prepare(residuals=panel["residual"], window=WINDOW).scale,
        "shares": fit_profile_shares(panel=panel),
        "thresholds": read_json(out=out, name="thresholds.json"),
        "models": pickle.loads((out / "models.pkl").read_bytes()) if trained else {},
    }
    return state, event_groups(panel=panel)


def tune(*, out: Path) -> None:
    """Match D0M, D1, D2, and D2 at veto 1.0 to `TARGET` unexplained odd-month alarms."""
    panel, _ = load()
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    plain = prepare(residuals=panel["residual"], window=WINDOW)
    d0m, _ = tune_d2(inputs=plain, veto_z=NO_VETO, logged=logged)
    if d0m != D0M_THRESHOLD:
        raise SystemExit("STOP: D0M's matched threshold is not experiment 58's 4.0")
    thresholds = {"D0M": d0m}
    curves = []
    d1, curve = tune_down(
        count=lambda t: odd_unexplained(
            table=alarms(inputs=plain, detector="D1", threshold=t), logged=logged
        )
    )
    thresholds["D1"] = d1
    curves += [{"detector": "D1", "threshold": t, "odd_unexplained": c} for t, c in curve]
    for arm, veto in (("D2", D2_VETO), ("D2V1", D2_VETO_POST_HOC)):
        thresholds[arm], curve = tune_d2(inputs=plain, veto_z=veto, logged=logged)
        curves += [{"detector": arm, "threshold": t, "odd_unexplained": c} for t, c in curve]
    print(f"matched thresholds: {thresholds}")
    (out / "thresholds.json").write_text(json.dumps(thresholds))
    pl.DataFrame(curves).write_parquet(out / "curves.parquet")


def train(*, out: Path) -> None:
    """Fit LR, LOGIT, and LOGIT_SAME and set each one's threshold on the real odd months."""
    state, groups = make_state(out=out, trained=False)
    jobs = [(seed, batch, shape) for shape in SHAPES for seed, batch in groups]
    positive = pl.concat(
        [p for p in run_pool(function=positives_batch, jobs=jobs, state=state) if p.height]
    )
    panel = state["panel"]
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    cand = candidates(
        inputs=prepare(residuals=panel["residual"], window=WINDOW, scale=state["scale"])
    )
    odd = np.isin(stamp_months()[cand["stamp"].to_numpy()], list(TUNING_MONTHS))
    unexplained = ~explained_flags(table=cand, logged=logged)
    negative = cand.filter(pl.Series(odd & unexplained))
    print(f"{positive.height} positive and {negative.height} negative candidates")
    models = {
        "LR": fit_likelihood_ratio(positive=positive, negative=negative),
        "LOGIT": fit_logistic(positive=positive, negative=negative, features=FEATURES),
        "LOGIT_SAME": fit_logistic(positive=positive, negative=negative, features=FEATURES_SAME),
    }
    thresholds = read_json(out=out, name="thresholds.json")
    for name, model in models.items():
        thresholds[name] = threshold_for_count(scores=model.score(table=negative), target=TARGET)
        kept = (model.score(table=negative) > thresholds[name]).sum()
        print(f"{name}: threshold {thresholds[name]:.4f}, odd-month unexplained alarms {kept}")
    (out / "models.pkl").write_bytes(pickle.dumps(models))
    (out / "thresholds.json").write_text(json.dumps(thresholds))
    positive.write_parquet(out / "positives.parquet")
    negative.write_parquet(out / "negatives.parquet")


def harness_check(*, out: Path) -> None:
    """With shape S1, D0M must give experiment 58's outcome for every one of the 1,900 transfers."""
    state, groups = make_state(out=out)
    jobs = [(seed, batch, "S1") for seed, batch in groups]
    scored = pl.DataFrame(
        [r for part in run_pool(function=score_batch, jobs=jobs, state=state) for r in part],
        infer_schema_length=None,
    ).filter(pl.col("setting") == "D0M_TR")
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
    """Score every transfer under all five shapes with every arm and end rule."""
    state, groups = make_state(out=out)
    (out / "profile_shares.json").write_text(shares_json(shares=state["shares"]))
    jobs = [(seed, batch, shape) for shape in SHAPES for seed, batch in groups]
    rows = [r for part in run_pool(function=score_batch, jobs=jobs, state=state) for r in part]
    scored = pl.DataFrame(rows, infer_schema_length=None)
    scored.write_parquet(out / "event_results.parquet")
    print(f"{scored.height} event rows")


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
    events = real_events(power=panel["power"])
    moved = pl.read_parquet(LABELLED_PATH).select("card_id", "moved_mw_p50")
    scored = score_real(results=results, real=events).join(moved, on="card_id", validate="m:1")
    scored.with_columns(month=pl.Series(stamp_months()[scored["onset"].to_numpy()])).write_parquet(
        out / "real_results.parquet"
    )
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
