r"""Score every detector arm on the 66 labelled real transfers, from stored models, with no refits.

Re-runs experiment 82's real-history stage with more output. The logistic scores of LOGIT_SAME and
RO-NC (stored as `XL` in experiment 82's pickle) are loaded, not refitted, and every threshold is
experiment 82's, read from its `thresholds.json`. Adds the baseline D0 (experiment 43's 24-hour
window at 4.0 noise units). Writes two private tables:

- `event_arms.parquet`: one row per labelled event and arm, with whether an interval found the
  event, the alarm's onset error and detection lag, and the half-hours of the event that the arm
  flagged;
- `alarm_counts.parquet`: one row per arm with all alarms, those inside a labelled event, those
  within 24 hours of a logged closure, and the unexplained rest.

    OMP_NUM_THREADS=1 uv run python ml-experiments/00101-*/e00130_real.py
"""

from __future__ import annotations

import io
import json
import pickle
from typing import Final

import numpy as np
import polars as pl
from e00130_common import EXPERIMENT_43_DIR, LABELLED_PATH, WORK_DIR, near_log_masks
from e00130_detect import EXPLAIN_PAD, load, real_events
from e00130_detectors import alarms
from e00130_intervals import build_intervals, relative_residual
from e00130_run import ARMS, all_scales, build_detectors
from e00130_score import by_series, match_event
from e00130_variants import load_both, variant_inputs

EXPERIMENT_82_DIR: Final = (
    WORK_DIR.parent / "00082-sw2-18-19-combined-refit-offset-and-same-sign-count"
)
"""Experiment 82's private folder: the fitted logistic scores and the thresholds."""
D0_THRESHOLD: Final[float] = 4.0
"""Noise units; experiment 43's D0_48 operating point."""
D0_WINDOW: Final[int] = 48
LABELS: Final[dict[str, str]] = {"XL": "RO-NC"}
"""Experiment 82 called RO-NC `XL`; the output tables use the new name."""
WINDOW_STAMPS: Final[int] = 48
"""The 24-hour window that every arm's step statistic looks forward over."""


class _Renamed(pickle.Unpickler):
    """Unpickler that finds experiment 82's classes in this folder's copy of the module."""

    def find_class(self, module: str, name: str) -> type:
        return super().find_class(module.replace("e00082", "e00130"), name)


def load_models() -> dict:
    """Experiment 82's fitted logistic scores, unpickled against this folder's modules."""
    return _Renamed(io.BytesIO((EXPERIMENT_82_DIR / "models.pkl").read_bytes())).load()


def arm_label(arm: str) -> str:
    """The name used in tables and figures."""
    return LABELS.get(arm, arm)


def d0_result(*, panel: dict) -> tuple[pl.DataFrame, list]:
    """The D0 alarms and intervals at experiment 43's 24-hour, 4.0-unit operating point."""
    scales = load()[1]
    inputs = load_inputs_d0(panel=panel, scales=scales)
    table = alarms(inputs=inputs, detector="D0", threshold=D0_THRESHOLD)
    relative = {
        s: relative_residual(residual=panel["residual"][s], expected=panel["expected"][s])
        for s in panel["residual"]
    }
    return table, build_intervals(
        alarm_table=table, relative=relative, window=D0_WINDOW, detector="D0"
    )


def load_inputs_d0(*, panel: dict, scales: dict):  # noqa: ANN201
    """D0's level shifts on the stored residual, with the 24-hour odd-month noise scale."""
    from e00130_detectors import prepare

    return prepare(residuals=panel["residual"], window=D0_WINDOW, scale=scales[D0_WINDOW])


def event_rows(
    *,
    events: pl.DataFrame,
    arm: str,
    intervals: list,
    noise: dict[int, float],
    steps: dict[int, np.ndarray],
) -> list[dict]:
    """One row per event for one arm."""
    grouped = by_series(intervals=intervals)
    rows = []
    for event in events.iter_rows(named=True):
        series = tuple(int(s) for s in event["series"].split(","))
        match = match_event(
            series=series, onset=event["onset"], end=event["end"], intervals=grouped
        )
        near = [
            iv for s in series for iv in grouped.get(s, []) if abs(iv.onset - event["onset"]) <= 4
        ]
        best = min(near, key=lambda iv: abs(iv.onset - event["onset"]), default=None)
        flagged = np.zeros(event["end"] - event["onset"], dtype=bool)
        for s in series:
            for iv in grouped.get(s, []):
                lo, hi = max(iv.onset, event["onset"]), min(iv.end, event["end"])
                if hi > lo:
                    flagged[lo - event["onset"] : hi - event["onset"]] = True
        window = slice(max(event["onset"] - 4, 0), event["onset"] + 5)
        peak = max(
            (
                float(np.nanmax(np.abs(steps[s][window]))) / noise[s]
                for s in series
                if s in steps and np.isfinite(steps[s][window]).any()
            ),
            default=float("nan"),
        )
        rows.append(
            {
                "card_id": event["card_id"],
                "arm": arm_label(arm),
                "found": match.found,
                "onset_found": match.onset_found,
                "onset_error_h": (best.onset - event["onset"]) / 2 if best else None,
                "detection_lag_h": (best.onset + WINDOW_STAMPS - event["onset"]) / 2
                if best
                else None,
                "flagged_half_hours": int(flagged.sum()),
                "event_half_hours": int(flagged.size),
                "peak_step_z": peak,
            }
        )
    return rows


def interval_rows(*, events: pl.DataFrame, arm: str, intervals: list) -> list[dict]:
    """Every interval of `arm` that overlaps a labelled event on one of its series."""
    grouped = by_series(intervals=intervals)
    return [
        {
            "card_id": event["card_id"],
            "arm": arm_label(arm),
            "series": s,
            "start_h": (iv.onset - event["onset"]) / 2,
            "end_h": (iv.end - event["onset"]) / 2,
        }
        for event in events.iter_rows(named=True)
        for s in (int(x) for x in event["series"].split(","))
        for iv in grouped.get(s, [])
        if iv.end > event["onset"] and iv.onset < event["end"]
    ]


def alarm_counts(*, tables: dict[str, pl.DataFrame], events: pl.DataFrame) -> pl.DataFrame:
    """Per arm: all alarms, those inside a labelled event, logged-explained, and unexplained."""
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    spans = [
        (tuple(int(s) for s in e["series"].split(",")), e["onset"] - 4, e["end"])
        for e in events.iter_rows(named=True)
    ]
    rows = []
    for arm, table in tables.items():
        inside = explained = unexplained = 0
        for series, stamp in table.select("series", "stamp").iter_rows():
            in_event = any(series in s and lo <= stamp <= hi for s, lo, hi in spans)
            is_logged = bool(logged[series][stamp])
            inside += in_event
            explained += is_logged and not in_event
            unexplained += not (in_event or is_logged)
        rows.append(
            {
                "arm": arm_label(arm),
                "alarms": table.height,
                "in_labelled_event": inside,
                "logged_only": explained,
                "unexplained": unexplained,
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    """Score all arms on the real history and write the two tables."""
    panel, offset_panel = load_both()
    scales = all_scales(panel=panel, offset_panel=offset_panel)
    thresholds = json.loads((EXPERIMENT_82_DIR / "thresholds.json").read_text())
    results = build_detectors(
        panel=panel,
        offset_panel=offset_panel,
        residual=panel["residual"],
        offset=offset_panel["residual"],
        scales=scales,
        thresholds=thresholds,
        models=load_models(),
    )
    d0_table, d0_intervals = d0_result(panel=panel)
    events = real_events(power=panel["power"]).join(
        pl.read_parquet(LABELLED_PATH).select("card_id", "moved_mw_p50", "donor", "receivers"),
        on="card_id",
        validate="1:1",
    )
    plain = variant_inputs(
        variant="PF14",
        residual=panel["residual"],
        offset=offset_panel["residual"],
        scale=scales["PF14"],
    )
    noise = plain.scale
    steps = dict(plain.shift.items())
    rows = event_rows(events=events, arm="D0", intervals=d0_intervals, noise=noise, steps=steps)
    for arm in ARMS:
        rows += event_rows(
            events=events, arm=arm, intervals=results[arm][1], noise=noise, steps=steps
        )
    spans = interval_rows(events=events, arm="D0", intervals=d0_intervals)
    for arm in ARMS:
        spans += interval_rows(events=events, arm=arm, intervals=results[arm][1])
    pl.DataFrame(spans, infer_schema_length=None).write_parquet(
        WORK_DIR / "event_intervals.parquet"
    )
    out = pl.DataFrame(rows, infer_schema_length=None)
    out.write_parquet(WORK_DIR / "event_arms.parquet")
    events.write_parquet(WORK_DIR / "events_real.parquet")
    tables = {"D0": d0_table} | {arm: results[arm][0] for arm in ARMS}
    counts = alarm_counts(tables=tables, events=events)
    counts.write_parquet(WORK_DIR / "alarm_counts.parquet")
    print(counts)
    print(out.group_by("arm").agg(pl.col("found").sum(), pl.col("onset_found").sum()))
    print(f"D0 intervals {len(d0_intervals)}; stored 43 count check: {EXPERIMENT_43_DIR.name}")


if __name__ == "__main__":
    main()
