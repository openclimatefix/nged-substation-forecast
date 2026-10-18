r"""Train the logistic scores, then score four detectors on a fresh draw and the real transfers.

PF14 is the benchmark (experiment 69). X takes its step statistic from two residual models refitted
half a month apart (experiment 71). LOGIT_SAME scores PF14's candidates with a logistic regression
that includes the same-sign neighbour count (experiment 70). XL fits and applies the same logistic
score to the candidates of X.

Stages, in order. Each writes files to the private output folder.

- `tune`: the thresholds of PF14 (must come back as 4.0) and X.
- `train`: LOGIT_SAME on PF14's candidates and XL on X's candidates, from odd-month injections of
  experiment 58's draw, and each one's threshold at 80 unexplained odd-month alarms.
- `harness-check`: on experiment 71's draw, PF14 and X must reproduce experiment 71's `found` flags
  and LOGIT_SAME experiment 70's.
- `score`: the outcome of every fresh injected transfer for every arm under all five shapes.
- `real`: the labelled real transfers' outcomes and the real alarm counts.

    uv run python ml-experiments/00082-*/e00098_run.py tune train harness-check score real
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
from e00098_combined import (
    FEATURES_SAME,
    candidates,
    fit_logistic,
    positive_flags,
    threshold_for_count,
)
from e00098_common import (
    ARM,
    LABELLED_PATH,
    ML_EXPERIMENTS_DATA_DIR,
    TUNING_MONTHS,
    WORK_DIR,
    near_log_masks,
    stamp_months,
)
from e00098_detect import (
    EXPLAIN_PAD,
    NO_VETO,
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
    tune_d2,
)
from e00098_detectors import alarms
from e00098_endrule import first_return_intervals
from e00098_inject import BLOCK, HEADLINE_BAND, Event, Injection, draw_events, events_frame
from e00098_intervals import build_intervals
from e00098_long import draw_long_events
from e00098_profile import SHAPES, Shape, fit_profile_shares, inject_shaped, shares_json
from e00098_score import by_series, match_event
from e00098_variants import (
    Variant,
    check_variants,
    load_both,
    scales_for,
    tracked_relative,
    variant_inputs,
)

EXPERIMENT_69_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00069-sw2-17-missed-onsets-of-the-end-rule"
)
EXPERIMENT_70_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00070-sw2-19-same-sign-neighbour-count"
EXPERIMENT_71_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00071-sw2-18-refit-offset-and-gap-bridging"
)
N_WORKERS: Final[int] = 6
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
SEED_71_BASE: Final[int] = 7100
SEED_71_LONG: Final[int] = 8100
FIRST_ID_71_BASE: Final[int] = 20_000
CHECK_ROWS: Final[int] = 400
CHECK_SHAPES: Final[tuple[Shape, ...]] = ("S1", "S4")
BASIS: Final[dict[str, Variant]] = {
    "PF14": "PF14",
    "X": "X",
    "LOGIT_SAME": "PF14",
    "XL": "X",
}
"""Each arm's step statistic: the stored residual's (PF14) or the refit-offset splice's (X)."""
SCORED: Final[tuple[str, ...]] = ("LOGIT_SAME", "XL")
ARMS: Final[tuple[str, ...]] = tuple(BASIS)
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


def all_scales(*, panel: dict, offset_panel: dict) -> dict[Variant, dict[int, float]]:
    """The odd-month noise scale of every series for both step statistics."""
    return {
        v: scales_for(variant=v, residual=panel["residual"], offset=offset_panel["residual"])
        for v in ("PF14", "X")
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
    for arm, variant in BASIS.items():
        inputs = variant_inputs(
            variant=variant, residual=residual, offset=offset, scale=scales[variant]
        )
        relative = tracked_relative(
            variant=variant,
            panel=panel,
            offset_panel=offset_panel,
            residual=residual,
            offset=offset,
        )
        if arm in SCORED:
            cand = candidates(inputs=inputs)
            keep = cand.filter(
                pl.Series(models[arm].score(table=cand) > thresholds[arm]) if cand.height else []
            )
            table = keep.select(
                "series",
                "stamp",
                "step_mw",
                z=pl.struct("series", "stamp").map_elements(
                    lambda r, s=inputs: float(
                        s.shift[r["series"]][r["stamp"]] / s.scale[r["series"]]
                    ),
                    return_dtype=pl.Float64,
                ),
                partners=pl.lit(""),
            )
        else:
            table = alarms(inputs=inputs, detector="D0", threshold=thresholds[arm])
        tracked = build_intervals(
            alarm_table=table, relative=relative, window=WINDOW, detector="D0"
        )
        out[arm] = (
            table,
            first_return_intervals(intervals=tracked, inputs=inputs, cap_days=CAP_DAYS),
        )
    return out


def injected(*, seed: int, batch: list[Event], shape: Shape) -> tuple[Injection, dict, dict]:
    """The injection of one batch and shape, and the injected residuals of both models."""
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
        event.event_id < HOLIDAY_FIRST_ID
        and (event.size_band, event.duration_band) == HEADLINE_BAND
        and int(months[event.onset]) in TUNING_MONTHS
    )


def positives_batch(args: tuple[int, list[Event], Shape]) -> pl.DataFrame:
    """Candidates within 2 hours of the onset of a training transfer, for both step statistics."""
    seed, batch, shape = args
    training = [e for e in batch if is_training_event(event=e, months=stamp_months())]
    if not training:
        return pl.DataFrame()
    _, residual, offset = injected(seed=seed, batch=batch, shape=shape)
    parts = []
    for variant in ("PF14", "X"):
        inputs = variant_inputs(
            variant=variant, residual=residual, offset=offset, scale=_STATE["scales"][variant]
        )
        cand = candidates(inputs=inputs)
        parts.append(
            cand.filter(pl.Series(positive_flags(table=cand, events=training))).with_columns(
                shape=pl.lit(shape), variant=pl.lit(variant)
            )
        )
    return pl.concat(parts)


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
    """A JSON file `tune` or `train` wrote."""
    return json.loads((out / name).read_text())


def make_state(*, out: Path, trained: bool = True) -> dict:
    """Panels, scales, thresholds, fitted scores, and fitted shares for the workers."""
    panel, offset_panel = load_both()
    return {
        "panel": panel,
        "offset_panel": offset_panel,
        "scales": all_scales(panel=panel, offset_panel=offset_panel),
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


def draw_fresh(
    *, panel: dict, seed_base: int, seed_long: int, base_first_id: int, long_first_id: int
) -> tuple[list[Event], list[Event]]:
    """A headline-band draw and a long draw."""
    base = draw_events(
        quiet=quiet_masks(),
        power=panel["power"],
        seed=seed_base,
        headline_only=FRESH_BASE_EVENTS,
        first_id=base_first_id,
    )
    long = draw_long_events(
        quiet=log_masks(), power=panel["power"], seed=seed_long, first_id=long_first_id
    )
    return base, long


def tune(*, out: Path) -> None:
    """Match PF14 and X to the largest count not above 80 odd-month unexplained alarms."""
    panel, offset_panel = load_both()
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    scales = all_scales(panel=panel, offset_panel=offset_panel)
    thresholds: dict[str, float] = {}
    for variant in ("PF14", "X"):
        inputs = variant_inputs(
            variant=variant,
            residual=panel["residual"],
            offset=offset_panel["residual"],
            scale=scales[variant],
        )
        threshold, curve = tune_d2(inputs=inputs, veto_z=NO_VETO, logged=logged, target=TARGET)
        print(f"{variant}: threshold {threshold}, count {dict(curve)[round(threshold, 2)]}")
        thresholds[variant] = threshold
    if ARM == "h05_stored" and thresholds["PF14"] != 4.0:
        raise SystemExit(f"STOP: PF14 matched {thresholds['PF14']}, not 4.0")
    (out / "thresholds.json").write_text(json.dumps(thresholds))


def train(*, out: Path) -> None:
    """Fit LOGIT_SAME on PF14's candidates and XL on X's, and set each one's threshold."""
    state = make_state(out=out, trained=False)
    state["thresholds"] = read_json(out=out, name="thresholds.json")
    panel, offset_panel = state["panel"], state["offset_panel"]
    jobs = [
        (seed, batch, shape) for shape in SHAPES for seed, batch in training_groups(panel=panel)
    ]
    positive = pl.concat(
        [p for p in run_pool(function=positives_batch, jobs=jobs, state=state) if p.height]
    )
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    models, thresholds = {}, read_json(out=out, name="thresholds.json")
    for arm in SCORED:
        variant = BASIS[arm]
        cand = candidates(
            inputs=variant_inputs(
                variant=variant,
                residual=panel["residual"],
                offset=offset_panel["residual"],
                scale=state["scales"][variant],
            )
        )
        odd = np.isin(stamp_months()[cand["stamp"].to_numpy()], list(TUNING_MONTHS))
        negative = cand.filter(pl.Series(odd & ~explained_flags(table=cand, logged=logged)))
        arm_positive = positive.filter(pl.col("variant") == variant)
        print(f"{arm}: {arm_positive.height} positive and {negative.height} negative candidates")
        model = fit_logistic(positive=arm_positive, negative=negative, features=FEATURES_SAME)
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


def harness_check(*, out: Path) -> None:
    """PF14, X, and LOGIT_SAME must reproduce experiments 71 and 70 on experiment 71's draw."""
    check_variants()
    state = make_state(out=out)
    panel = state["panel"]
    base = draw_events(
        quiet=quiet_masks(),
        power=panel["power"],
        seed=SEED_71_BASE,
        headline_only=FRESH_BASE_EVENTS,
        first_id=FIRST_ID_71_BASE,
    )
    kept: list[tuple[int, list[Event]]] = []
    total = 0
    for seed, batch in seeded_batches(events=base, seed=SEED_71_BASE + 1000):
        if total >= CHECK_ROWS:
            break
        kept.append((seed, batch))
        total += len(batch)
    jobs = [(seed, batch, shape) for shape in CHECK_SHAPES for seed, batch in kept]
    scored = pl.DataFrame(
        [r for part in run_pool(function=score_batch, jobs=jobs, state=state) for r in part],
        infer_schema_length=None,
    )
    stored_71 = pl.read_parquet(EXPERIMENT_71_DIR / "event_results.parquet").with_columns(
        setting=pl.col("setting").str.replace("_48", "")
    )
    stored_70 = pl.read_parquet(EXPERIMENT_70_DIR / "event_results.parquet")
    failed = False
    for arm, stored in (("PF14", stored_71), ("X", stored_71), ("LOGIT_SAME", stored_70)):
        mine = scored.filter(pl.col("setting") == arm)
        joined = mine.join(
            stored.filter(pl.col("setting") == arm),
            on=["event_id", "shape"],
            suffix="_old",
            validate="1:1",
        )
        different = joined.filter(
            (pl.col("found") != pl.col("found_old"))
            | (pl.col("onset_found") != pl.col("onset_found_old"))
        ).height
        print(f"{arm}: {joined.height} rows, {different} differ from the earlier experiment")
        failed |= joined.height != mine.height or bool(different)
    if failed:
        raise SystemExit("STOP: the harness does not reproduce the earlier experiments")
    print("harness reproduced")


def score(*, out: Path) -> None:
    """Score every fresh transfer under all five shapes with every arm."""
    state = make_state(out=out)
    panel = state["panel"]
    (out / "profile_shares.json").write_text(shares_json(shares=state["shares"]))
    base, long = draw_fresh(
        panel=panel,
        seed_base=SEED_FRESH_BASE,
        seed_long=SEED_FRESH_LONG,
        base_first_id=FRESH_BASE_FIRST_ID,
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
