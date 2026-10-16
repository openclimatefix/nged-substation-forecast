r"""Fresh draws, the paired low-threshold onset, and the classification of unalarmed real transfers.

Stages, in order. Each writes files to the private output folder.

- `tune`: D0M's threshold (must come back as 4.0) and the strong threshold of the two PP variants,
  each matched on odd months to 80 unexplained alarms.
- `harness-check`: D0M, PE14, and PF14 must reproduce experiment 66's `found` flags on its
  transfers of experiments 43 and 58, under the shapes in `CHECK_SHAPES`.
- `score`: the outcome of every fresh injected transfer for every arm under all five shapes.
- `real`: the labelled real transfers' outcomes and the real alarm counts.
- `classify`: the class of each labelled real transfer over 1 day with no alarm at its start.

    uv run python ml-experiments/00069-*/e00069_run.py tune harness-check score real classify
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00069_classify import classify
from e00069_common import (
    GRID_LENGTH,
    LABELLED_PATH,
    ML_EXPERIMENTS_DATA_DIR,
    WORK_DIR,
    near_log_masks,
)
from e00069_detect import (
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
from e00069_detectors import DetectorInputs, alarms, prepare
from e00069_endrule import end_rule_intervals, first_return_intervals
from e00069_inject import BLOCK, Event, draw_events, events_frame
from e00069_intervals import build_intervals, relative_residual
from e00069_long import draw_long_events
from e00069_paired import paired_alarms, tune_pp
from e00069_profile import SHAPES, Shape, fit_profile_shares, inject_shaped, shares_json
from e00069_score import by_series, match_event

EXPERIMENT_66_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00066-sw2-16-end-rule-as-baseline"
N_WORKERS: Final[int] = 4
WINDOW: Final[int] = 48
CAP_DAYS: Final[int] = 14
BASE_TARGET: Final[int] = 80
SEED_FRESH_BASE: Final[int] = 6900
"""Seed of the fresh headline-band draw; batch `i` is injected with `SEED_FRESH_BASE + 1000 + i`."""
SEED_FRESH_LONG: Final[int] = 7900
"""Seed of the fresh long draw; batch `i` is injected with `SEED_FRESH_LONG + 1000 + i`."""
FRESH_BASE_EVENTS: Final[int] = 1160
FRESH_BASE_FIRST_ID: Final[int] = 10_000
FRESH_LONG_FIRST_ID: Final[int] = 15_000
CHECK_SHAPES: Final[tuple[Shape, ...]] = ("S1", "S4")
CHECK_ROWS: Final[int] = 400
"""The harness check re-runs the first events of experiment 66's draw, to bound the run time."""
PP_TAGS: Final[dict[float, str]] = {0.5: "50", 0.75: "75"}
ARMS: Final[tuple[str, ...]] = (
    "D0M_48",
    "PE14_48",
    "PF14_48",
    *(f"PP{end}14_{tag}" for tag in PP_TAGS.values() for end in ("E", "F")),
    "SE14_50",
    "SF14_50",
)
"""D0M, the two end rules, each PP variant (weak share 50% or 75%) with each end rule, and the
strong alarms of the primary PP variant alone (S) with each end rule, which separates the effect
of the raised strong threshold from the effect of the weak alarms."""
_STATE: dict = {}


def build_detectors(*, residual: dict, expected: dict, scales: dict, thresholds: dict) -> dict:
    """Alarms and intervals of every arm in `ARMS`."""
    relative = {s: relative_residual(residual=residual[s], expected=expected[s]) for s in residual}
    inputs = prepare(residuals=residual, window=WINDOW, scale=scales[WINDOW])
    out: dict = {}

    def add(*, table: pl.DataFrame, names: tuple[str, str, str] | tuple[str, str]) -> None:
        tracked = build_intervals(
            alarm_table=table, relative=relative, window=WINDOW, detector="D0"
        )
        ends = (
            tracked,
            end_rule_intervals(intervals=tracked, inputs=inputs, cap_days=CAP_DAYS),
            first_return_intervals(intervals=tracked, inputs=inputs, cap_days=CAP_DAYS),
        )
        for name, intervals in zip(names, ends[-len(names) :], strict=True):
            out[name] = (table, intervals)

    add(
        table=alarms(inputs=inputs, detector="D0", threshold=thresholds["base"]),
        names=("D0M_48", "PE14_48", "PF14_48"),
    )
    for share, tag in PP_TAGS.items():
        table = paired_alarms(inputs=inputs, threshold=thresholds[f"pp{tag}"], weak_share=share)
        add(table=table, names=(f"PPE14_{tag}", f"PPF14_{tag}"))
        if share == 0.5:
            add(table=table.filter(~pl.col("weak")), names=("SE14_50", "SF14_50"))
    return out


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
    plain_scale = _STATE["scales"][WINDOW]
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
    """The odd-month noise scale of every series, for the 24-hour window."""
    return {WINDOW: prepare(residuals=panel["residual"], window=WINDOW).scale}


def tune(*, out: Path) -> None:
    """Match D0M to 80 odd-month unexplained alarms, then each PP variant's strong threshold."""
    panel, _ = load()
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    plain = prepare(residuals=panel["residual"], window=WINDOW)
    base, _ = tune_d2(inputs=plain, veto_z=NO_VETO, logged=logged, target=BASE_TARGET)
    print(f"D0M threshold {base}")
    if base != 4.0:
        raise SystemExit(f"STOP: D0M matched {base}, not 4.0")
    thresholds: dict[str, float] = {"base": base}
    curves = []
    for share, tag in PP_TAGS.items():
        threshold, curve = tune_pp(inputs=plain, weak_share=share, logged=logged)
        print(f"PP weak share {share}: strong threshold {threshold}")
        thresholds[f"pp{tag}"] = threshold
        rises = sum(b[1] > a[1] for a, b in itertools.pairwise(curve))
        print(
            f"  count rises {rises} times along the grid; at the chosen threshold: "
            f"{[c for c in curve if c[0] == threshold]}"
        )
        curves += [
            {"weak_share": share, "threshold": t, "odd_unexplained": c, "odd_weak": w}
            for t, c, w in curve
        ]
    (out / "thresholds.json").write_text(json.dumps(thresholds))
    pl.DataFrame(curves).write_parquet(out / "curves.parquet")


def make_state(*, out: Path, panel: dict) -> dict:
    """Panel, scales, thresholds, and fitted shares for the workers."""
    return {
        "panel": panel,
        "scales": scales_of(panel=panel),
        "thresholds": read_thresholds(out=out),
        "shares": fit_profile_shares(panel=panel),
    }


def check_groups(*, panel: dict) -> list[tuple[int, list[Event]]]:
    """The first `CHECK_ROWS` events of experiments 43 and 58 in experiment 66's batches."""
    original = original_events(power=panel["power"])
    extra, holiday = new_events(power=panel["power"])
    groups = seeded_batches(events=original, seed=SEED_43) + seeded_batches(
        events=extra + holiday, seed=SEED_NEW
    )
    kept: list[tuple[int, list[Event]]] = []
    total = 0
    for seed, batch in groups:
        if total >= CHECK_ROWS:
            break
        kept.append((seed, batch))
        total += len(batch)
    return kept


def harness_check(*, out: Path) -> None:
    """D0M, PE14, and PF14 must reproduce experiment 66's flags on a sample of its transfers."""
    panel, _ = load()
    state = make_state(out=out, panel=panel)
    groups = check_groups(panel=panel)
    scored = run_shapes(shapes=CHECK_SHAPES, groups=groups, state=state)
    stored = pl.read_parquet(EXPERIMENT_66_DIR / "event_results.parquet")
    for setting in ("D0M_48", "PE14_48", "PF14_48"):
        mine = scored.filter(pl.col("setting") == setting)
        joined = mine.join(
            stored.filter(pl.col("setting") == setting),
            on=["event_id", "shape"],
            suffix="_66",
            validate="1:1",
        )
        different = joined.filter(
            (pl.col("found") != pl.col("found_66"))
            | (pl.col("onset_found") != pl.col("onset_found_66"))
        ).height
        print(f"{setting}: {joined.height} rows, {different} differ from experiment 66")
        if joined.height != mine.height or different:
            raise SystemExit(f"STOP: the harness does not reproduce experiment 66 at {setting}")
    check_pairing()
    plain = prepare(residuals=panel["residual"], window=WINDOW, scale=state["scales"][WINDOW])
    strong = alarms(inputs=plain, detector="D0", threshold=4.0)
    disabled = paired_alarms(inputs=plain, threshold=4.0, weak_share=1.0).drop("weak")
    if not strong.equals(disabled):
        raise SystemExit("STOP: PP with the weak alarms disabled differs from D0M")
    print("harness reproduced; PP with weak alarms disabled equals D0M")


def check_pairing() -> None:
    """Hand-built steps: a weak onset is accepted only for a matching step back within limits.

    The strong threshold is 4.0, so the weak threshold is 2.0, and the onset step is 2.5.
    """
    scale = {1: 1.0}
    for gap, size, expected in (
        (48, -2.5, True),  # exactly 24 hours later, ratio 1.0
        (47, -2.5, False),  # one stamp too early
        (14 * 48, -2.5, True),  # exactly the cap
        (14 * 48 + 1, -2.5, False),  # one stamp past the cap
        (100, -2.01, True),  # just over the weak threshold
        (100, -1.99, False),  # just under the weak threshold
        (100, -5.0, True),  # ratio 2.0
        (100, -5.1, False),  # ratio over 2.0
        (100, 2.5, False),  # same sign
    ):
        shift = np.zeros(GRID_LENGTH)
        shift[3000], shift[3000 + gap] = 2.5, size
        inputs = DetectorInputs(window=48, shift={1: shift}, scale=scale)
        table = paired_alarms(inputs=inputs, threshold=4.0, weak_share=0.5)
        got = table.filter(pl.col("weak"), pl.col("stamp") == 3000).height == 1
        if got != expected:
            raise SystemExit(f"STOP: pairing gap {gap}, size {size}: got {got}, wanted {expected}")
    print("pairing rule checked on hand-built steps")


def fresh_events(*, panel: dict) -> tuple[list[Event], list[Event]]:
    """The fresh headline-band draw and the fresh long draw."""
    base = draw_events(
        quiet=quiet_masks(),
        power=panel["power"],
        seed=SEED_FRESH_BASE,
        headline_only=FRESH_BASE_EVENTS,
        first_id=FRESH_BASE_FIRST_ID,
    )
    long = draw_long_events(
        quiet=log_masks(),
        power=panel["power"],
        seed=SEED_FRESH_LONG,
        first_id=FRESH_LONG_FIRST_ID,
    )
    return base, long


def score(*, out: Path) -> None:
    """Score every fresh transfer under all five shapes with every arm."""
    panel, _ = load()
    state = make_state(out=out, panel=panel)
    (out / "profile_shares.json").write_text(shares_json(shares=state["shares"]))
    base, long = fresh_events(panel=panel)
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
    scored = run_shapes(shapes=SHAPES, groups=groups, state=state)
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


def classify_real(*, out: Path) -> None:
    """Classify the real transfers over 1 day that D0M leaves without an alarm at the start."""
    panel, _ = load()
    thresholds = read_thresholds(out=out)
    scale = scales_of(panel=panel)[WINDOW]
    inputs = prepare(residuals=panel["residual"], window=WINDOW, scale=scale)
    table = alarms(inputs=inputs, detector="D0", threshold=thresholds["base"])
    logged = log_masks()
    transfers = real_events(power=panel["power"]).filter(pl.col("hours") > 24)
    scored = pl.read_parquet(out / "real_results.parquet").filter(
        pl.col("setting") == "D0M_48", pl.col("hours") > 24
    )
    unalarmed = scored.filter(~pl.col("onset_found")).select("card_id")
    print(f"{transfers.height} transfers over 1 day, {unalarmed.height} with no alarm at the start")
    if (transfers.height, unalarmed.height) != (35, 24):
        raise SystemExit("STOP: the plan says 35 transfers, 24 without an alarm at the start")
    rows = [
        classify(row=row, inputs=inputs, alarm_table=table, logged=logged)
        for row in transfers.join(unalarmed, on="card_id").iter_rows(named=True)
    ]
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(out / "classes.parquet")


def main() -> None:
    """Run the requested stages."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stages", nargs="+", choices=("tune", "harness-check", "score", "real", "classify")
    )
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stages = {
        "tune": tune,
        "harness-check": harness_check,
        "score": score,
        "real": real,
        "classify": classify_real,
    }
    for stage in args.stages:
        stages[stage](out=args.out)


if __name__ == "__main__":
    main()
