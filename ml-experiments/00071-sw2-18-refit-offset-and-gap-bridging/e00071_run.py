r"""Fresh draws and the five arms: PF14, X (refit offset), G (gap bridging), XG, and the control L.

Stages, in order. Each writes files to the private output folder. Run `e00071_fit_offset.py fit`
first.

- `tune`: each arm's threshold, the largest count not above 80 odd-month unexplained alarms on the
  0.05 grid. PF14 must come back at 4.0.
- `harness-check`: the PF14 arm must reproduce experiment 69's `found` flags on its own draw, and
  the hand-built checks of the variants must pass.
- `score`: the outcome of every injected transfer for every arm under all five shapes, on the
  fresh headline-band draw, the fresh long draw, and the draw whose onsets lie in refit zones.
- `real`: the labelled real transfers' outcomes and the real alarm counts.
- `diagnose`: the splice's agreement between the two models, and the gaps of the panel.
- `classify`: the class of each labelled real transfer over 1 day with no PF14 alarm at its start.

    uv run python ml-experiments/00071-*/e00071_run.py tune harness-check score real \
        diagnose classify
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
from e00071_classify import classify
from e00071_common import (
    GRID_LENGTH,
    LABELLED_PATH,
    ML_EXPERIMENTS_DATA_DIR,
    SERIES_IDS,
    WORK_DIR,
    near_log_masks,
)
from e00071_detect import (
    EXPLAIN_PAD,
    NO_VETO,
    explained_flags,
    log_masks,
    quiet_masks,
    real_alarm_rows,
    real_events,
    score_real,
    seeded_batches,
    tune_d2,
)
from e00071_detectors import alarms, prepare, refit_stamps
from e00071_endrule import first_return_intervals
from e00071_inject import BLOCK, Event, draw_events, events_frame
from e00071_intervals import build_intervals
from e00071_long import draw_long_events
from e00071_profile import SHAPES, Shape, fit_profile_shares, inject_shaped, shares_json
from e00071_score import by_series, end_tolerance, match_event
from e00071_variants import (
    VARIANTS,
    Variant,
    check_variants,
    load_both,
    refit_zone,
    scales_for,
    second_refit_stamps,
    shifts,
    tracked_relative,
    variant_inputs,
)

EXPERIMENT_69_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00069-sw2-17-missed-onsets-of-the-end-rule"
)
N_WORKERS: Final[int] = 4
WINDOW: Final[int] = 48
CAP_DAYS: Final[int] = 14
TARGET: Final[int] = 80
SEED_FRESH_BASE: Final[int] = 7100
"""Seed of the fresh headline-band draw; batch `i` is injected with `SEED_FRESH_BASE + 1000 + i`."""
SEED_FRESH_LONG: Final[int] = 8100
SEED_ZONE: Final[int] = 9100
SEED_69_BASE: Final[int] = 6900
SEED_69_LONG: Final[int] = 7900
FRESH_BASE_EVENTS: Final[int] = 1160
ZONE_EVENTS: Final[int] = 300
FRESH_BASE_FIRST_ID: Final[int] = 20_000
FRESH_LONG_FIRST_ID: Final[int] = 25_000
ZONE_FIRST_ID: Final[int] = 30_000
CHECK_SHAPES: Final[tuple[Shape, ...]] = ("S1", "S4")
CHECK_ROWS: Final[int] = 400
ARMS: Final[tuple[str, ...]] = tuple(f"{v}_48" for v in VARIANTS)
"""PF14 (the benchmark), X, G, XG, and L, all with the first-return end rule, cap 14 days."""
_STATE: dict = {}


def build_detectors(
    *, panel: dict, offset_panel: dict, residual: dict, offset: dict, scales: dict, thresholds: dict
) -> dict:
    """Alarms and first-return intervals of every arm in `ARMS`."""
    out: dict = {}
    for variant in VARIANTS:
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
        table = alarms(inputs=inputs, detector="D0", threshold=thresholds[variant])
        tracked = build_intervals(
            alarm_table=table, relative=relative, window=WINDOW, detector="D0"
        )
        out[f"{variant}_48"] = (
            table,
            first_return_intervals(intervals=tracked, inputs=inputs, cap_days=CAP_DAYS),
        )
    return out


def init_worker(state: dict) -> None:
    """Pool initializer: share the panels, scales, thresholds, and fitted shares."""
    _STATE.update(state)


def score_batch(args: tuple[int, list[Event], Shape]) -> list[dict]:
    """Inject one batch with one shape, detect with every arm, return one row per event and arm."""
    seed, batch, shape = args
    panel, offset_panel = _STATE["panel"], _STATE["offset_panel"]
    injection = inject_shaped(
        power=panel["power"], events=batch, seed=seed, shape=shape, shares=_STATE["shares"]
    )
    results = build_detectors(
        panel=panel,
        offset_panel=offset_panel,
        residual=injection.fast_residual(panel["residual"]),
        offset=injection.fast_residual(offset_panel["residual"]),
        scales=_STATE["scales"],
        thresholds=_STATE["thresholds"],
    )
    plain_scale = _STATE["scales"]["PF14"]
    rows = []
    for setting, (_, intervals) in results.items():
        grouped = by_series(intervals=intervals)
        for event in batch:
            match = match_event(
                series=event.series, onset=event.onset, end=event.end, intervals=grouped
            )
            scaled = match_event(
                series=event.series,
                onset=event.onset,
                end=event.end,
                intervals=grouped,
                onset_tolerance=end_tolerance(onset=event.onset, end=event.end),
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
                    "found_scaled": scaled.found,
                    "onset_found_scaled": scaled.onset_found,
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


def all_scales(*, panel: dict, offset_panel: dict) -> dict[Variant, dict[int, float]]:
    """The odd-month noise scale of every series for every arm, from the uninjected residuals."""
    return {
        v: scales_for(variant=v, residual=panel["residual"], offset=offset_panel["residual"])
        for v in VARIANTS
    }


def tune(*, out: Path) -> None:
    """Match every arm to the largest count not above 80 odd-month unexplained alarms."""
    panel, offset_panel = load_both()
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    scales = all_scales(panel=panel, offset_panel=offset_panel)
    thresholds: dict[str, float] = {}
    rows = []
    for variant in VARIANTS:
        inputs = variant_inputs(
            variant=variant,
            residual=panel["residual"],
            offset=offset_panel["residual"],
            scale=scales[variant],
        )
        threshold, curve = tune_d2(inputs=inputs, veto_z=NO_VETO, logged=logged, target=TARGET)
        print(f"{variant}: threshold {threshold}, count {dict(curve)[round(threshold, 2)]}")
        thresholds[variant] = threshold
        rows += [{"variant": variant, "threshold": t, "odd_unexplained": c} for t, c in curve]
    if thresholds["PF14"] != 4.0:
        raise SystemExit(f"STOP: PF14 matched {thresholds['PF14']}, not 4.0")
    (out / "thresholds.json").write_text(json.dumps(thresholds))
    pl.DataFrame(rows).write_parquet(out / "curves.parquet")


def make_state(*, out: Path, panel: dict, offset_panel: dict) -> dict:
    """Panels, scales, thresholds, and fitted shares for the workers."""
    return {
        "panel": panel,
        "offset_panel": offset_panel,
        "scales": all_scales(panel=panel, offset_panel=offset_panel),
        "thresholds": read_thresholds(out=out),
        "shares": fit_profile_shares(panel=panel),
    }


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


def draw_zone(*, panel: dict) -> list[Event]:
    """Headline-band transfers whose onsets lie within 24 hours of a refit of the first model."""
    zone = refit_zone()
    return draw_events(
        quiet=quiet_masks(),
        power=panel["power"],
        seed=SEED_ZONE,
        headline_only=ZONE_EVENTS,
        first_id=ZONE_FIRST_ID,
        onset_ok=lambda onset: bool(zone[onset]),
    )


def harness_check(*, out: Path) -> None:
    """The PF14 arm must reproduce experiment 69's flags on a sample of its own draw."""
    check_variants()
    panel, offset_panel = load_both()
    state = make_state(out=out, panel=panel, offset_panel=offset_panel)
    base, long = draw_fresh(
        panel=panel,
        seed_base=SEED_69_BASE,
        seed_long=SEED_69_LONG,
        base_first_id=10_000,
        long_first_id=15_000,
    )
    groups = seeded_batches(events=base, seed=SEED_69_BASE + 1000) + seeded_batches(
        events=long, seed=SEED_69_LONG + 1000
    )
    kept, total = [], 0
    for group in groups:
        if total >= CHECK_ROWS:
            break
        kept.append(group)
        total += len(group[1])
    scored = run_shapes(shapes=CHECK_SHAPES, groups=kept, state=state)
    stored = pl.read_parquet(EXPERIMENT_69_DIR / "event_results.parquet")
    mine = scored.filter(pl.col("setting") == "PF14_48")
    joined = mine.join(
        stored.filter(pl.col("setting") == "PF14_48"),
        on=["event_id", "shape"],
        suffix="_69",
        validate="1:1",
    )
    different = joined.filter(
        (pl.col("found") != pl.col("found_69"))
        | (pl.col("onset_found") != pl.col("onset_found_69"))
    ).height
    print(f"PF14_48: {joined.height} rows, {different} differ from experiment 69")
    if joined.height != mine.height or different:
        raise SystemExit("STOP: the harness does not reproduce experiment 69's PF14")
    print("harness reproduced")


def score(*, out: Path) -> None:
    """Score every injected transfer under all five shapes with every arm."""
    panel, offset_panel = load_both()
    state = make_state(out=out, panel=panel, offset_panel=offset_panel)
    (out / "profile_shares.json").write_text(shares_json(shares=state["shares"]))
    base, long = draw_fresh(
        panel=panel,
        seed_base=SEED_FRESH_BASE,
        seed_long=SEED_FRESH_LONG,
        base_first_id=FRESH_BASE_FIRST_ID,
        long_first_id=FRESH_LONG_FIRST_ID,
    )
    zone = draw_zone(panel=panel)
    events = pl.concat(
        [
            events_frame(events=part).with_columns(population=pl.lit(name))
            for name, part in (("base", base), ("long", long), ("zone", zone))
        ]
    )
    events.write_parquet(out / "events.parquet")
    print(f"{len(base)} base, {len(long)} long, and {len(zone)} zone events", flush=True)
    groups = (
        seeded_batches(events=base, seed=SEED_FRESH_BASE + 1000)
        + seeded_batches(events=long, seed=SEED_FRESH_LONG + 1000)
        + seeded_batches(events=zone, seed=SEED_ZONE + 1000)
    )
    scored = run_shapes(shapes=SHAPES, groups=groups, state=state)
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
        thresholds=read_thresholds(out=out),
    )
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    real_alarm_rows(results=results, logged=logged).write_parquet(out / "alarms.parquet")
    events = real_events(power=panel["power"])
    scored = score_real(results=results, real=events)
    moved = pl.read_parquet(LABELLED_PATH).select("card_id", "moved_mw_p50")
    scored.join(moved, on="card_id", validate="m:1").write_parquet(out / "real_results.parquet")
    print({s: int(explained_flags(table=t, logged=logged).sum()) for s, (t, _) in results.items()})


def diagnose(*, out: Path) -> None:
    """Agreement of the two models, the splice stamps, and the panel's gaps."""
    panel, offset_panel = load_both()
    scales = all_scales(panel=panel, offset_panel=offset_panel)["PF14"]
    first = shifts(variant="PF14", residual=panel["residual"], offset={})
    second = shifts(variant="PF14", residual=offset_panel["residual"], offset={})
    zone = refit_zone()
    second_zone = np.zeros(GRID_LENGTH, dtype=bool)
    for stamp in second_refit_stamps():
        second_zone[max(stamp - WINDOW + 1, 0) : stamp + WINDOW] = True
    clean = ~zone & ~second_zone
    rows = []
    for series in SERIES_IDS:
        if series not in second:
            continue
        both = clean & np.isfinite(first[series]) & np.isfinite(second[series])
        gap = np.abs(first[series][both] - second[series][both]) / scales[series]
        rows.append(
            {
                "series": series,
                "stamps": int(both.sum()),
                "median_abs_difference_z": float(np.median(gap)),
                "p99_abs_difference_z": float(np.quantile(gap, 0.99)),
            }
        )
    pl.DataFrame(rows).write_parquet(out / "splice_agreement.parquet")
    offsets = np.arange(-3 * 48, 3 * 48 + 1, 12)
    profile = []
    for series in SERIES_IDS:
        if series not in offset_panel["expected"]:
            continue
        difference = np.abs(panel["expected"][series] - offset_panel["expected"][series])
        scale_mw = np.nanmedian(np.abs(panel["power"][series]))
        for refit_name, stamps in (("1st", refit_stamps()), ("16th", second_refit_stamps())):
            for offset in offsets:
                at = stamps + offset
                at = at[(at >= 0) & (at < GRID_LENGTH)]
                values = difference[at]
                if np.isfinite(values).any():
                    profile.append(
                        {
                            "series": series,
                            "refit": refit_name,
                            "offset_half_hours": int(offset),
                            "mean_abs_difference_share": float(np.nanmean(values) / scale_mw),
                        }
                    )
    pl.DataFrame(profile).write_parquet(out / "refit_profile.parquet")
    gaps = []
    for series in SERIES_IDS:
        values = panel["residual"][series]
        present = np.flatnonzero(np.isfinite(values))
        inner = ~np.isfinite(values[present[0] : present[-1] + 1])
        edges = np.diff(np.concatenate([[0], inner.astype(int), [0]]))
        for start, stop in zip(
            np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True
        ):
            gaps.append(
                {"series": series, "start": int(present[0] + start), "length": int(stop - start)}
            )
    pl.DataFrame(gaps).write_parquet(out / "gaps.parquet")
    print(f"{len(rows)} series compared, {len(gaps)} inner gaps")


def classify_real(*, out: Path) -> None:
    """Classify the real transfers over 1 day that have no PF14 alarm at their start."""
    panel, _ = load_both()
    thresholds = read_thresholds(out=out)
    inputs = prepare(residuals=panel["residual"], window=WINDOW)
    table = alarms(inputs=inputs, detector="D0", threshold=thresholds["PF14"])
    logged = log_masks()
    transfers = real_events(power=panel["power"]).filter(pl.col("hours") > 24)
    scored = pl.read_parquet(out / "real_results.parquet").filter(
        pl.col("setting") == "PF14_48", pl.col("hours") > 24
    )
    unalarmed = scored.filter(~pl.col("onset_found")).select("card_id")
    print(
        f"{transfers.height} transfers over 1 day, "
        f"{unalarmed.height} with no PF14 alarm at the start"
    )
    rows = [
        classify(row=row, inputs=inputs, alarm_table=table, logged=logged)
        for row in transfers.join(unalarmed, on="card_id").iter_rows(named=True)
    ]
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(out / "classes.parquet")


def main() -> None:
    """Run the requested stages."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stages",
        nargs="+",
        choices=("tune", "harness-check", "score", "real", "diagnose", "classify"),
    )
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stages = {
        "tune": tune,
        "harness-check": harness_check,
        "score": score,
        "real": real,
        "diagnose": diagnose,
        "classify": classify_real,
    }
    for stage in args.stages:
        stages[stage](out=args.out)


if __name__ == "__main__":
    main()
