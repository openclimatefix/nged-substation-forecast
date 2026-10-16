"""Run the detectors on the real residual and on injected transfers; write the scoreboard tables.

uv run python ml-experiments/00043-sw2-1-detector-scoreboard/e00043_run.py
"""

from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Final

import numpy as np
import polars as pl
from e00043_common import (
    LABELLED_META_PATH,
    LABELLED_PATH,
    OPERATING_POINTS,
    WORK_DIR,
    load_panel,
    near_log_masks,
    stamp_of,
)
from e00043_detectors import alarms, prepare
from e00043_inject import (
    QUIET_STAMPS,
    Event,
    batches,
    draw_events,
    events_frame,
    inject_power,
)
from e00043_intervals import build_intervals, relative_residual
from e00043_score import by_series, match_event

SEED: Final[int] = 43
N_WORKERS: Final[int] = 4
VARIANTS: Final[tuple[tuple[str, bool], ...]] = (("plain", False), ("fixed", True))
"""The interval builder of experiment 38 as published, and its post hoc fixed variant."""
EXPLAIN_PAD: Final[int] = 48
"""An alarm within 24 hours of a logged closure on its series is explained by that closure."""

_STATE: dict[str, Any] = {}


def _detect(
    *,
    residual: dict[int, np.ndarray],
    expected: dict[int, np.ndarray],
    scales: dict[int, dict[int, float]],
) -> dict[tuple[str, str], tuple[pl.DataFrame, list]]:
    """Alarms and intervals for each operating point and interval variant."""
    relative = {s: relative_residual(residual=residual[s], expected=expected[s]) for s in residual}
    out: dict[tuple[str, str], tuple[pl.DataFrame, list]] = {}
    inputs = {w: prepare(residuals=residual, window=w, scale=scales.get(w)) for w in (8, 48)}
    for name, (detector, window, threshold) in OPERATING_POINTS.items():
        table = alarms(inputs=inputs[window], detector=detector, threshold=threshold)
        for variant, fixed in VARIANTS:
            intervals = build_intervals(
                alarm_table=table,
                relative=relative,
                window=window,
                detector=detector,
                fixed=fixed,
            )
            out[(name, variant)] = (table, intervals)
    return out


def _init(state: dict[str, Any]) -> None:
    _STATE.update(state)


def _score_batch(args: tuple[int, list[Event]]) -> list[dict]:
    """Inject one batch, run every detector, and return one row per event, detector, variant."""
    index, batch = args
    panel = _STATE["panel"]
    injection = inject_power(power=panel["power"], events=batch, seed=SEED + index)
    residual = injection.fast_residual(panel["residual"])
    results = _detect(residual=residual, expected=panel["expected"], scales=_STATE["scales"])
    rows: list[dict] = []
    for (name, variant), (_, intervals) in results.items():
        grouped = by_series(intervals=intervals)
        for event in batch:
            match = match_event(
                series=event.series, onset=event.onset, end=event.end, intervals=grouped
            )
            rows.append(
                {
                    "event_id": event.event_id,
                    "setting": name,
                    "variant": variant,
                    "onset_found": match.onset_found,
                    "found": match.found,
                    "end_error": match.end_error,
                    "known_delay": match.known_delay,
                }
            )
    return rows


def _scales(*, residual: dict[int, np.ndarray]) -> dict[int, dict[int, float]]:
    return {w: prepare(residuals=residual, window=w).scale for w in (8, 48)}


def _baseline(*, panel: dict, scales: dict) -> dict:
    results = _detect(residual=panel["residual"], expected=panel["expected"], scales=scales)
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    rows = []
    for (name, variant), (table, intervals) in results.items():
        if variant != "plain":
            continue
        matched = [bool(logged[s][t]) for s, t in table.select("series", "stamp").iter_rows()]
        rows.append(
            {
                "setting": name,
                "alarms": table.height,
                "logged_matched": int(sum(matched)),
                "unexplained": table.height - int(sum(matched)),
                "intervals": len(intervals),
            }
        )
        table.with_columns(logged_matched=pl.Series(matched, dtype=pl.Boolean)).write_parquet(
            WORK_DIR / f"baseline_alarms_{name}.parquet"
        )
    return {"summary": pl.DataFrame(rows), "results": results}


def _real_events() -> pl.DataFrame:
    """The labelled real transfers (not hidden cards): stamps, series, and conditions."""
    frame = (
        pl.read_parquet(LABELLED_PATH)
        .filter(
            ~pl.col("is_sentinel"),
            pl.col("kind") == "real_operation",
            pl.col("visibility").is_in(["conserved", "one_sided"]),
        )
        .with_columns(
            hours=(pl.col("logged_open_utc") - pl.col("logged_close_utc")).dt.total_minutes() / 60,
            share=pl.col("moved_share_of_receiver").fill_nan(None),
        )
    )
    rows = []
    for row in frame.iter_rows(named=True):
        receivers = [int(r["series"]) for r in json.loads(row["receivers"])]
        series = sorted(
            {
                int(s)
                for s in (row["donor"], row["side_a"], row["side_b"], *receivers)
                if s is not None
            }
        )
        # a conserved call that rests on a mostly missing side is treated as one-sided
        visibility = "one_sided" if row["max_side_missing_share"] > 0.5 else row["visibility"]
        rows.append(
            {
                "card_id": row["card_id"],
                "visibility": visibility,
                "onset": stamp_of(when=row["onset_utc"]),
                "end": stamp_of(when=row["end_utc"]) + 1,
                "hours": row["hours"],
                "share": row["share"],
                "series": ",".join(map(str, series)),
            }
        )
    return pl.DataFrame(rows)


def _score_real(*, baseline: dict) -> pl.DataFrame:
    real = _real_events()
    rows = []
    for (name, variant), (_, intervals) in baseline["results"].items():
        grouped = by_series(intervals=intervals)
        for row in real.iter_rows(named=True):
            series = tuple(int(s) for s in row["series"].split(","))
            match = match_event(
                series=series, onset=row["onset"], end=row["end"], intervals=grouped
            )
            rows.append(
                {
                    **row,
                    "setting": name,
                    "variant": variant,
                    "onset_found": match.onset_found,
                    "found": match.found,
                    "end_error": match.end_error,
                }
            )
    return pl.DataFrame(rows)


def _alarm_sample() -> pl.DataFrame:
    """SW2-0b's labelled alarm sample: visible counts per operating point."""
    meta = json.loads(LABELLED_META_PATH.read_text())
    labelled = pl.read_parquet(LABELLED_PATH).select("card_id", "visibility")
    alarm_cards = pl.DataFrame(
        [{"card_id": c, "setting": m["detector"]} for c, m in meta.items() if m["kind"] == "alarm"]
    )
    return (
        alarm_cards.join(labelled, on="card_id")
        .group_by("setting")
        .agg(
            sampled=pl.len(),
            visible=pl.col("visibility").is_in(["conserved", "one_sided"]).sum(),
        )
    )


def main() -> None:
    """Write the baseline, injected-event, real-event, and alarm-sample tables."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    scales = _scales(residual=panel["residual"])
    baseline = _baseline(panel=panel, scales=scales)
    baseline["summary"].write_parquet(WORK_DIR / "baseline_summary.parquet")
    print(baseline["summary"])
    quiet = near_log_masks(pad_stamps=QUIET_STAMPS)
    for table in (
        pl.read_parquet(WORK_DIR / f"baseline_alarms_{n}.parquet") for n in OPERATING_POINTS
    ):
        for series, stamp in table.select("series", "stamp").iter_rows():
            quiet[series][max(stamp - 2 * 48, 0) : stamp + 2 * 48] = True
    events = draw_events(quiet=quiet, power=panel["power"], seed=SEED)
    events_frame(events=events).write_parquet(WORK_DIR / "events.parquet")
    groups = batches(events=events)
    print(len(events), "events in", len(groups), "batches")
    state = {"panel": panel, "scales": scales}
    with ProcessPoolExecutor(
        N_WORKERS,
        mp_context=multiprocessing.get_context("fork"),
        initializer=_init,
        initargs=(state,),
    ) as pool:
        rows = [r for part in pool.map(_score_batch, list(enumerate(groups))) for r in part]
    pl.DataFrame(rows).write_parquet(WORK_DIR / "event_results.parquet")
    _score_real(baseline=baseline).write_parquet(WORK_DIR / "real_results.parquet")
    _alarm_sample().write_parquet(WORK_DIR / "alarm_sample.parquet")
    print("done", len(rows), "event rows")


if __name__ == "__main__":
    main()
